'use strict';
// Hugging Face dataset discovery tests: search validation/mapping,
// info splits mapping, failure handling, and config passthrough.
// Upstream HF calls are stubbed — no network, no datasets library needed.

const assert = require('assert');
const path = require('path');

const ml = require('../ml_experiment_backend');

const REAL_FETCH = global.fetch;

function stubFetch(handler) {
  global.fetch = async (url, init) => handler(String(url), init || {});
}

function jsonResponse(body, { status = 200, link = null } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (n) => (String(n).toLowerCase() === 'link' ? link : 'application/json') },
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  };
}

function expectErr(fn, code) {
  let err = null;
  try {
    const r = fn();
    if (r && typeof r.then === 'function') throw new Error('expected sync throw, got promise');
  } catch (e) { err = e; }
  assert(err, 'expected an error to be thrown');
  if (code) assert.strictEqual(err.code, code, `expected code ${code}, got ${err.code}: ${err.message}`);
  return err;
}

async function expectErrAsync(fn, code, messageMatch) {
  let err = null;
  try { await fn(); } catch (e) { err = e; }
  assert(err, 'expected an async error to be thrown');
  if (code) assert.strictEqual(err.code, code, `expected code ${code}, got ${err.code}: ${err.message}`);
  if (messageMatch) assert(messageMatch.test(err.message), `message mismatch: ${err.message}`);
  return err;
}

async function run() {
  // ── 1. query validation ──
  const def = ml.validateHfSearch({});
  assert.strictEqual(def.sort, 'downloads');
  assert.strictEqual(def.limit, 20);
  expectErr(() => ml.validateHfSearch({ sort: 'nope' }), 'bad_request');
  expectErr(() => ml.validateHfSearch({ task: 'nope' }), 'bad_request');
  expectErr(() => ml.validateHfSearch({ language: 'english-toolong' }), 'bad_request');
  expectErr(() => ml.validateHfSearch({ limit: 0 }), 'bad_request');
  expectErr(() => ml.validateHfSearch({ limit: 'abc' }), 'bad_request');
  assert.strictEqual(ml.validateHfSearch({ limit: 500 }).limit, 50, 'limit capped at 50');
  assert.strictEqual(ml.validateHfSearch({ limit: '7' }).limit, 7);
  console.log('✓ search query validation + limit cap');

  // ── 2. metadata mapping (contents never fetched) ──
  const seen = [];
  stubFetch(async (url) => {
    seen.push(url);
    assert(url.startsWith('https://huggingface.co/api/datasets?'), `fixed Hub base, got ${url}`);
    assert(!url.includes('http://evil'), 'no client-controlled hosts');
    return jsonResponse([
      { id: 'stanfordnlp/imdb', author: 'stanfordnlp', likes: 100, downloads: 5000,
        lastModified: '2024-01-01', tags: ['task_categories:text-classification', 'language:en'],
        cardData: { description: 'Movie reviews for sentiment.' } },
      { id: 'bad-entry-no-slash-should-filter', likes: 1 },
    ], { link: '<https://huggingface.co/api/datasets?cursor=abc123>; rel="next"' });
  });
  try {
    const out = await ml.searchHfDatasets(ml.validateHfSearch({ q: 'imdb', task: 'text-classification', language: 'en' }));
    assert.strictEqual(out.datasets.length, 1, 'entries without owner/name filtered');
    const d = out.datasets[0];
    assert.strictEqual(d.id, 'stanfordnlp/imdb');
    assert.strictEqual(d.author, 'stanfordnlp');
    assert.strictEqual(d.name, 'imdb');
    assert(d.description.includes('Movie reviews'), 'description mapped');
    assert.deepStrictEqual(d.tasks, ['text-classification']);
    assert.deepStrictEqual(d.languages, ['en']);
    assert.strictEqual(out.next_cursor, 'abc123');
    assert(seen[0].includes('search=imdb'), 'query forwarded');
    assert(seen[0].includes('filter='), 'task/language filter forwarded');
    // Cache: second identical call performs no fetch.
    seen.length = 0;
    await ml.searchHfDatasets(ml.validateHfSearch({ q: 'imdb', task: 'text-classification', language: 'en' }));
    assert.strictEqual(seen.length, 0, 'identical search served from cache');
  } finally {
    global.fetch = REAL_FETCH;
  }
  console.log('✓ search metadata mapping + pagination + cache');

  // ── 3. upstream failures → friendly errors, no tracebacks ──
  stubFetch(async () => { throw new Error('socket hang up'); });
  try {
    await expectErrAsync(() => ml.searchHfDatasets(ml.validateHfSearch({ q: 'x' })),
      'hf_unavailable', /temporarily unavailable/);
  } finally {
    global.fetch = REAL_FETCH;
  }
  stubFetch(async () => jsonResponse({ message: 'Not Found' }, { status: 404 }));
  try {
    await expectErrAsync(() => ml.getHfDatasetInfo('no/suchds'), 'not_found', /could not be found/);
  } finally {
    global.fetch = REAL_FETCH;
  }
  stubFetch(async () => jsonResponse({ message: 'Gated' }, { status: 401 }));
  try {
    await expectErrAsync(() => ml.getHfDatasetInfo('gated/ds'), 'gated_dataset', /gated/);
  } finally {
    global.fetch = REAL_FETCH;
  }
  await expectErrAsync(() => ml.getHfDatasetInfo('not-an-id'), 'bad_request');
  console.log('✓ upstream failure handling (network/404/gated/invalid id)');

  // ── 4. info splits/configs mapping ──
  stubFetch(async (url) => {
    if (url.includes('/api/datasets/')) {
      return jsonResponse({ id: 'a/b', author: 'a', likes: 5, downloads: 50,
        lastModified: '2024-05-05', tags: ['task_categories:tabular-classification'],
        cardData: { description: 'Tabular data.' } });
    }
    if (url.includes('datasets-server.huggingface.co/splits')) {
      assert(url.includes('dataset=a%2Fb'), `dataset param encoded, got ${url}`);
      return jsonResponse({ splits: [
        { config: 'default', split: 'train' },
        { config: 'default', split: 'test' },
        { config: 'v2', split: 'train' },
      ] });
    }
    throw new Error(`unexpected upstream: ${url}`);
  });
  try {
    const info = await ml.getHfDatasetInfo('a/b');
    assert.deepStrictEqual(info.configs, ['default', 'v2']);
    assert.deepStrictEqual(info.splits, { default: ['train', 'test'], v2: ['train'] });
    assert.strictEqual(info.author, 'a');
    assert.deepStrictEqual(info.tasks, ['tabular-classification']);
  } finally {
    global.fetch = REAL_FETCH;
  }
  // Splits endpoint down → configs empty, loader can still try.
  stubFetch(async (url) => {
    if (url.includes('/api/datasets/')) return jsonResponse({ id: 'c/d' });
    return jsonResponse({}, { status: 500 });
  });
  try {
    const info = await ml.getHfDatasetInfo('c/d');
    assert.deepStrictEqual(info.configs, []);
    assert.deepStrictEqual(info.splits, {});
  } finally {
    global.fetch = REAL_FETCH;
  }
  console.log('✓ info configs/splits mapping + degraded splits');

  // ── 5. config passthrough validation ──
  assert.strictEqual(ml.validateHfDataset({ dataset_id: 'a/b', config: 'default' }).config, 'default');
  assert.strictEqual(ml.validateHfDataset({ dataset_id: 'a/b' }).config, null);
  expectErr(() => ml.validateHfDataset({ dataset_id: 'nope' }), 'bad_request');
  console.log('✓ hf ingest validation accepts optional config');

  console.log('\nAll HF dataset discovery backend tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
