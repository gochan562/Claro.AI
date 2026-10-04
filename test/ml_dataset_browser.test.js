'use strict';
// Hugging Face dataset browser tests: discovery UI, selection state,
// split/config handling, load via the existing ingest path, upload-mode
// preservation, and persistence. Upstream HF calls are stubbed — no network.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
const ML_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'ml_experiment.js'), 'utf8');

const SEARCH_RESULT = {
  datasets: [{
    id: 'stanfordnlp/imdb', name: 'imdb', author: 'stanfordnlp',
    description: 'Movie reviews for sentiment classification.',
    downloads: 5000, likes: 100, last_modified: '2024-01-01',
    tasks: ['text-classification'], languages: ['en'],
  }],
  next_cursor: null,
};
const INFO_RESULT = {
  id: 'stanfordnlp/imdb', name: 'imdb', author: 'stanfordnlp',
  description: 'Movie reviews.', downloads: 5000, likes: 100,
  last_modified: '2024-01-01', tasks: ['text-classification'], languages: ['en'],
  configs: ['default', 'plain_text'], splits: { default: ['train', 'test'], plain_text: ['train'] },
};
const LOADED = { workspace_id: 'mlw_aaaaaaaaaaaa',
  profile: { row_count: 10, column_count: 2, columns: ['text', 'label'], duplicate_rows: 0 } };

function makeWindow() {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>` +
    `<div class="notebook-toolbar"></div>` +
    `<div id="notebook-cells"></div>` +
    `<div id="gpu-terminal"></div>` +
    `</body></html>`,
    { url: 'http://localhost:5000/', runScripts: 'dangerously' }
  );
  const w = dom.window;
  w.CodeMirror = function (host, opts) {
    const ed = {
      _value: opts.value || '', _focused: false,
      getValue: () => ed._value, setValue: (v) => { ed._value = v; },
      on: () => {}, focus: () => { ed._focused = true; },
      refresh: () => {}, getWrapperElement: () => w.document.createElement('div'),
    };
    const el = w.document.createElement('div');
    el.className = 'CodeMirror';
    host.appendChild(el);
    return ed;
  };
  w.marked = { parse: (s) => `<p>${s}</p>` };
  w.DOMPurify = { sanitize: (s) => s };
  w.PyodideLocal = { run: async () => ({ ok: true, stdout: '', stderr: '', error: null }) };
  const calls = [];
  w.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    const u = String(url);
    let body = {};
    if (u.includes('/api/ml/datasets/search')) body = SEARCH_RESULT;
    else if (u.includes('/api/ml/datasets/info')) body = INFO_RESULT;
    else if (u.includes('/api/ml/datasets/hf')) body = LOADED;
    else if (u.includes('/api/ml/datasets/mlw_')) {
      body = { workspace_id: 'mlw_aaaaaaaaaaaa', profile: LOADED.profile, manifest: {} };
    } else throw new Error(`unexpected fetch in test: ${u}`);
    return { ok: true, status: 200, json: async () => body };
  };
  w.prompt = () => null;
  w.confirm = () => false;
  for (const src of [ML_SRC, APP_SRC]) {
    const script = w.document.createElement('script');
    script.textContent = src;
    w.document.body.appendChild(script);
  }
  for (const fn of ['addCell', 'mlHfSearch', 'mlHfSelect', 'mlLoadHfDataset',
    'mlHfEnsureInfo', 'getActiveNotebook', 'defaultMlCell']) {
    assert.strictEqual(typeof w[fn], 'function', `must expose global ${fn}()`);
  }
  return { dom, w, calls };
}

function newDatasetCell(w) {
  const id = w.createNotebook('HF Test');
  w.setActiveNotebook(id);
  w.addCell('dataset');
  const nb = w.getActiveNotebook();
  return nb.cells[nb.cells.length - 1];
}

const tick = (ms) => new Promise((r) => setTimeout(r, ms || 20));

let passed = 0, failed = 0;
function test(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => { console.log(`✓ ${name}`); passed++; })
    .catch((e) => { console.error(`✗ ${name}: ${e.message}`); failed++; });
}

async function main() {
  console.log('=== HF Dataset Browser Tests ===\n');

  await test('1. new Dataset Cell defaults to Hugging Face browser', () => {
    const { w } = makeWindow();
    const cell = newDatasetCell(w);
    assert.strictEqual(cell.ml.source, 'huggingface');
    const html = w.document.getElementById(`cell-${cell.id}`).innerHTML;
    assert(html.includes('Search datasets'), 'search input present');
    assert(html.includes('mlHfSearch'), 'search wired');
    assert(html.includes('Load more') === false, 'no load-more without cursor');
  });

  await test('2. popular datasets auto-load on first render', async () => {
    const { w, calls } = makeWindow();
    newDatasetCell(w);
    await tick(50);
    const hit = calls.find((c) => c.url.includes('/api/ml/datasets/search?sort=downloads'));
    assert(hit, 'popular auto-fetch issued');
    assert(w.document.getElementById('notebook-cells').innerHTML.includes('stanfordnlp/imdb'),
      'popular result rendered');
  });

  await test('3. search renders results with metadata + select', async () => {
    const { w } = makeWindow();
    const cell = newDatasetCell(w);
    await tick(50);
    const input = w.document.querySelector(`#cell-${cell.id} [data-ml-hf="q"]`);
    input.value = 'imdb';
    await w.mlHfSearch(cell.id, true);
    const html = w.document.getElementById(`cell-${cell.id}`).innerHTML;
    assert(html.includes('Movie reviews for sentiment'), 'description shown');
    assert(html.includes('text-classification'), 'task tag shown');
    assert(html.includes('mlHfSelect'), 'select button wired');
  });

  await test('4. select stores state and loads splits/configs', async () => {
    const { w, calls } = makeWindow();
    const cell = newDatasetCell(w);
    await tick(50);
    await w.mlHfSelect(cell.id, 'stanfordnlp/imdb');
    const nb = w.getActiveNotebook();
    const updated = nb.cells.find((c) => c.id === cell.id);
    assert.strictEqual(updated.ml.dataset_id, 'stanfordnlp/imdb');
    const html = w.document.getElementById(`cell-${cell.id}`).innerHTML;
    assert(html.includes('plain_text'), 'config options shown');
    assert(html.includes('value="test"') || html.includes('>test<'), 'split options shown');
    assert(calls.some((c) => c.url.includes('/api/ml/datasets/info?dataset_id=stanfordnlp%2Fimdb')),
      'info fetched for selection');
  });

  await test('5. load uses existing ingest with config + split', async () => {
    const { w, calls } = makeWindow();
    const cell = newDatasetCell(w);
    await tick(50);
    await w.mlHfSelect(cell.id, 'stanfordnlp/imdb');
    // choose non-default config + split through the rendered selects
    const el = w.document.getElementById(`cell-${cell.id}`);
    const cfgSel = el.querySelector('[data-ml-hf="hf_config_sel"]');
    if (cfgSel) cfgSel.value = 'plain_text';
    const splitSel = el.querySelector('[data-ml-hf="hf_split_sel"]');
    if (splitSel) splitSel.value = 'test';
    await w.mlLoadHfDataset(cell.id);
    const post = calls.find((c) => c.url === '/api/ml/datasets/hf');
    assert(post, 'ingest endpoint called');
    const body = JSON.parse(post.init.body);
    assert.strictEqual(body.dataset_id, 'stanfordnlp/imdb');
    assert.strictEqual(body.config, 'plain_text');
    assert.strictEqual(body.split, 'test');
    const nb = w.getActiveNotebook();
    const updated = nb.cells.find((c) => c.id === cell.id);
    assert.strictEqual(updated.ml.workspace_id, 'mlw_aaaaaaaaaaaa');
    assert.strictEqual(updated.ml.hf_config, 'plain_text');
    assert(updated.ml.profile && updated.ml.profile.row_count === 10, 'profile stored');
  });

  await test('6. legacy upload-mode state still renders', () => {
    const { w } = makeWindow();
    newDatasetCell(w);
    const nb = w.getActiveNotebook();
    const cell = nb.cells[nb.cells.length - 1];
    cell.ml.source = 'upload';
    cell.ml.filename = 'old.csv';
    w.renderNotebookEditor();
    const html = w.document.getElementById(`cell-${cell.id}`).innerHTML;
    assert(html.includes('type="file"'), 'upload input present');
    assert(!html.includes('Search datasets'), 'no browser in upload mode');
  });

  await test('7. legacy manual HF state still renders and loads', async () => {
    const { w, calls } = makeWindow();
    newDatasetCell(w);
    const nb = w.getActiveNotebook();
    const cell = nb.cells[nb.cells.length - 1];
    cell.ml.dataset_id = 'a/b';
    cell.ml.hf_split = 'train';
    w.renderNotebookEditor();
    const html = w.document.getElementById(`cell-${cell.id}`).innerHTML;
    assert(html.includes('owner/name') || html.includes('Manual ID') || html.includes('dataset_id_manual'),
      'manual entry present');
    await w.mlLoadHfDataset(cell.id);
    const post = calls.find((c) => c.url === '/api/ml/datasets/hf');
    assert(post, 'manual load posts to existing ingest');
    assert.strictEqual(JSON.parse(post.init.body).dataset_id, 'a/b');
  });

  await test('8. persistence keeps discovery state', () => {
    const first = makeWindow();
    newDatasetCell(first.w);
    const nb = first.w.getActiveNotebook();
    const cell = nb.cells[nb.cells.length - 1];
    cell.ml.dataset_id = 'stanfordnlp/imdb';
    cell.ml.hf_config = 'plain_text';
    cell.ml.hf_title = 'IMDb';
    first.w.updateNotebook();
    const saved = first.w.localStorage.getItem('claro-notebooks-registry');
    const second = makeWindow();
    second.w.localStorage.setItem('claro-notebooks-registry', saved);
    second.w.loadAllNotebooks();
    const nb2 = second.w.getActiveNotebook ? second.w.getActiveNotebook() : null;
    const reg = JSON.parse(second.w.localStorage.getItem('claro-notebooks-registry'));
    const id = Object.keys(reg.notebooks)[0];
    const ds = reg.notebooks[id].cells.find((c) => c.ml && c.ml.kind === 'dataset');
    assert.strictEqual(ds.ml.dataset_id, 'stanfordnlp/imdb');
    assert.strictEqual(ds.ml.hf_config, 'plain_text');
    assert.strictEqual(ds.ml.hf_title, 'IMDb');
    assert(nb2 !== undefined, 'registry loads');
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
