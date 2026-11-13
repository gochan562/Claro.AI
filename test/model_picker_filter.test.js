'use strict';
// Model Picker eligibility tests: resource-conscious, text-only selection.
//
// Covers the spec matrix:
//   - 7B accepted / 8B accepted / >8B rejected / unknown params rejected
//   - image-text-to-text rejected, text-generation with vision tags rejected,
//     video model rejected
//   - ordinary text-only model accepted
//   - a text model whose ID *mentions* images is NOT rejected (metadata
//     checks, never name/card wording)
//   - pagination still works after filtering (auto-advance past empty pages)
//   - search & sorting params still reach the API
//   - the "Text-only models · Up to 8B parameters" label renders
//
// Drives the REAL dashboard-app.js in jsdom with a stubbed Hugging Face API.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
const HTML_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard.html'), 'utf8');

function makeWindow() {
  // Extract the model picker markup from the real dashboard.html so the
  // harness always matches production DOM shape.
  const start = HTML_SRC.indexOf('<div id="model-picker-modal"');
  const end = HTML_SRC.indexOf('</main>', start);
  const pickerHtml = HTML_SRC.slice(start, end);

  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
      <div class="notebook-toolbar"></div>
      <div id="notebook-cells"></div>
      <div id="gpu-terminal"></div>
      ${pickerHtml}
    </body></html>`,
    { url: 'http://localhost:5000/', runScripts: 'outside-only' }
  );
  const w = dom.window;
  w.CodeMirror = function (host, opts) {
    const ed = { getValue: () => '', setValue: () => {}, on: () => {}, refresh: () => {},
      getWrapperElement: () => w.document.createElement('div') };
    return ed;
  };
  w.marked = { parse: (s) => `<p>${s}</p>` };
  w.DOMPurify = { sanitize: (s) => s };
  w.PyodideLocal = { run: async () => ({ ok: true, stdout: '', stderr: '', error: null }) };
  w.fetch = async () => { throw new Error('fetch not stubbed'); };
  w.prompt = () => null;
  w.confirm = () => false;
  w.eval(APP_SRC);
  return { dom, w };
}

// ── model fixtures ──────────────────────────────────────────────────────────
const B = 1e9;
function textModel(overrides = {}) {
  return Object.assign({
    id: 'owner/text-model',
    pipeline_tag: 'text-generation',
    tags: ['transformers', 'safetensors', 'text-generation'],
    library_name: 'transformers',
    downloads: 1000, likes: 50,
    // explicitly public + open (fail-closed eligibility requires both)
    gated: false, private: false,
    safetensors: { total: 7 * B, parameters: { F16: 7 * B } },
  }, overrides);
}

let passed = 0, failed = 0;
function test(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => { console.log(`✓ ${name}`); passed++; })
    .catch((e) => { console.error(`✗ ${name}: ${e.message}`); failed++; });
}

async function main() {
  console.log('=== Model Picker Eligibility Tests ===\n');

  const { w } = makeWindow();
  const elig = w.hfModelEligibility;
  assert.strictEqual(typeof elig, 'function', 'hfModelEligibility must be global');
  assert.strictEqual(typeof w.applyModelFilters, 'function', 'applyModelFilters must be global');

  await test('1. 7B text-generation model is accepted', () => {
    const r = elig(textModel({ safetensors: { total: 7 * B } }));
    // property assertions (the result object comes from the jsdom realm)
    assert.strictEqual(r.eligible, true, r.reason);
    assert.strictEqual(r.reason, 'ok');
  });

  await test('2. exactly 8B text-generation model is accepted ("up to 8B")', () => {
    const r = elig(textModel({ safetensors: { total: 8 * B } }));
    assert.strictEqual(r.eligible, true, r.reason);
    const r2 = elig(textModel({ safetensors: { total: 8 * B + 1 } }));
    assert.strictEqual(r2.eligible, false, '8B + 1 byte of parameters must be rejected');
  });

  await test('3. model above 8B is rejected', () => {
    const r = elig(textModel({ safetensors: { total: 9 * B } }));
    assert.strictEqual(r.eligible, false);
    assert.match(r.reason, /8B/);
  });

  await test('4. unknown parameter count is rejected (missing/null/malformed)', () => {
    for (const m of [
      textModel({ safetensors: undefined }),
      textModel({ safetensors: null }),
      textModel({ safetensors: {} }),
      textModel({ safetensors: { total: null } }),
      textModel({ safetensors: { total: 'about seven billion' } }),
      textModel({ safetensors: { total: 0 } }),
    ]) {
      const r = elig(m);
      assert.strictEqual(r.eligible, false, `expected rejection for ${JSON.stringify(m.safetensors)}`);
      assert.match(r.reason, /unknown parameter count/);
    }
    // GGUF-style repo with no safetensors metadata at all
    assert.strictEqual(elig({ id: 'o/gguf', pipeline_tag: 'text-generation', tags: [] }).eligible, false);
  });

  await test('5. image-text-to-text (vision-language) model is rejected', () => {
    const r = elig(textModel({ pipeline_tag: 'image-text-to-text' }));
    assert.strictEqual(r.eligible, false);
    // also when it only appears as a tag on a text-generation model
    const r2 = elig(textModel({ tags: ['transformers', 'image-text-to-text'] }));
    assert.strictEqual(r2.eligible, false);
  });

  await test('6. text-generation model with vision tags is rejected (multimodal)', () => {
    for (const tags of [
      ['transformers', 'vision'],
      ['transformers', 'multimodal'],
      ['transformers', 'image'],
      ['transformers', 'vlm'],
      ['transformers', 'image-something-else'],
    ]) {
      const r = elig(textModel({ tags }));
      assert.strictEqual(r.eligible, false, `tags ${JSON.stringify(tags)} must be rejected`);
    }
    // diffusers library is non-text
    assert.strictEqual(elig(textModel({ library_name: 'diffusers' })).eligible, false);
  });

  await test('7. video model is rejected', () => {
    assert.strictEqual(elig(textModel({ pipeline_tag: 'text-to-video' })).eligible, false);
    assert.strictEqual(elig(textModel({ pipeline_tag: 'video-classification' })).eligible, false);
    assert.strictEqual(elig(textModel({ tags: ['transformers', 'video'] })).eligible, false);
    assert.strictEqual(elig(textModel({ tags: ['transformers', 'image-to-video'] })).eligible, false);
  });

  await test('8. ordinary text-only model is accepted (all allowed tasks)', () => {
    for (const task of ['text-generation', 'text2text-generation', 'text-classification',
      'token-classification', 'question-answering', 'summarization', 'translation',
      'fill-mask', 'sentence-similarity', 'feature-extraction', 'zero-shot-classification']) {
      const r = elig(textModel({ pipeline_tag: task }));
      assert.strictEqual(r.eligible, true, `${task} must be allowed`);
    }
  });

  await test('9. metadata checks only — ID mentioning images does NOT reject a text model', () => {
    const r = elig(textModel({ id: 'owner/text-model-for-image-captions-discussion' }));
    assert.strictEqual(r.eligible, true,
      'ordinary text models must not be rejected on name wording');
    // and a model with no tags array at all passes on pipeline+params alone
    const r2 = elig({ id: 'owner/min', pipeline_tag: 'text-generation', tags: undefined,
      gated: false, private: false, safetensors: { total: 3 * B } });
    assert.strictEqual(r2.eligible, true);
  });

  await test('10. malformed/missing metadata fails safe without throwing', () => {
    assert.strictEqual(elig(null).eligible, false);
    assert.strictEqual(elig(undefined).eligible, false);
    assert.strictEqual(elig('not-a-model').eligible, false);
    assert.strictEqual(elig({}).eligible, false);
    assert.strictEqual(elig({ id: 'x', pipeline_tag: 'not-a-real-task',
      safetensors: { total: 1 * B }, tags: 'not-an-array' }).eligible, false);
    assert.strictEqual(elig({ id: 'x', pipeline_tag: null,
      safetensors: { total: 1 * B }, tags: [] }).eligible, false);
  });

  // ── pagination / search / sorting through the real loadHFModels ──
  function page(models, nextCursor) {
    return {
      status: 200,
      headers: nextCursor ? { 'X-Next-Cursor': nextCursor } : {},
      json: models,
    };
  }

  await test('11. pagination auto-advances past filtered-out pages and renders eligible models', async () => {
    const { w: w2 } = makeWindow();
    const bigVision = textModel({ id: 'owner/vision-llm', safetensors: { total: 7 * B },
      tags: ['transformers', 'image-text-to-text'] });
    const pages = [
      page([bigVision, textModel({ id: 'owner/over-8b', safetensors: { total: 70 * B } })], 'CUR-2'),
      page([textModel({ id: 'owner/good-1' }), textModel({ id: 'owner/good-2' })], null),
    ];
    const urls = [];
    w2.fetch = async (url) => {
      urls.push(String(url));
      const idx = urls.filter((u) => u.includes('/api/models')).length - 1;
      if (idx >= pages.length) throw new Error('too many fetches');
      const resp = pages[idx];
      return {
        ok: true, status: 200,
        headers: { get: (n) => resp.headers[n] || null },
        json: async () => resp.json,
      };
    };
    await w2.loadHFModels(true);
    const rendered = w2.document.getElementById('model-list').innerHTML;
    assert(rendered.includes('owner/good-1') && rendered.includes('owner/good-2'),
      'eligible models from the second page rendered');
    assert(!rendered.includes('owner/vision-llm') && !rendered.includes('owner/over-8b'),
      'filtered models never render');
    assert.strictEqual(
      w2.document.querySelectorAll('#model-list .model-item').length, 2,
      'only eligible models render as cards');
    assert(!rendered.includes('hf-load-more'), 'no Load more when the cursor is exhausted');
    // server-side pre-filter present
    assert(urls[0].includes('num_parameters=max%3A8B'), `num_parameters param: ${urls[0]}`);
    assert(urls[0].includes('expand%5B%5D=safetensors'), 'safetensors expansion requested');
  });

  await test('12. search and sorting still reach the API; empty state is helpful', async () => {
    const { w: w3 } = makeWindow();
    const urls = [];
    w3.fetch = async (url) => {
      urls.push(String(url));
      return { ok: true, status: 200, headers: { get: () => null },
        json: async () => [] };
    };
    w3.document.getElementById('model-search').value = 'tiny llama';
    w3.document.getElementById('sort-filter').value = 'downloads';
    w3.document.getElementById('task-filter').value = 'text-generation';
    await w3.loadHFModels(true);
    const u = urls[0];
    assert(u.includes('search=tiny+llama'), `search param: ${u}`);
    assert(u.includes('sort=downloads') && u.includes('direction=-1'), `sort params: ${u}`);
    assert(u.includes('filter=text-generation'), `task filter: ${u}`);
    // empty result → helpful message, not a generic empty state
    const html = w3.document.getElementById('model-list').innerHTML;
    assert(html.includes('No text-only models up to 8B parameters matched'), html);
    assert(html.includes('Model Cell'), 'points users to the custom-ID workflow');
    // the label from spec 5 renders next to the list
    const hint = w3.document.getElementById('model-list-hint');
    assert(hint && hint.textContent.includes('Text-only models · Up to 8B parameters'),
      'picker label present');
  });

  await test('13. Load more appends the next eligible page (cursor preserved)', async () => {
    const { w: w4 } = makeWindow();
    const pages = {
      '': page([textModel({ id: 'owner/p1' })], 'CUR-A'),
      'CUR-A': page([textModel({ id: 'owner/p2' })], 'CUR-B'),
      'CUR-B': page([textModel({ id: 'owner/vision-only', tags: ['transformers', 'image'] }),
                     textModel({ id: 'owner/p3' })], null),
    };
    w4.fetch = async (url) => {
      const cursor = /[?&]cursor=([^&]*)/.exec(String(url));
      const key = cursor ? decodeURIComponent(cursor[1]) : '';
      const resp = pages[key] || page([], null);
      return { ok: true, status: 200, headers: { get: (n) => resp.headers[n] || null },
        json: async () => resp.json };
    };
    await w4.loadHFModels(true);
    await w4.loadHFModels(false); // Load more → page 2
    await w4.loadHFModels(false); // Load more → page 3 (half filtered, auto-advances to exhaust)
    const rendered = w4.document.getElementById('model-list').innerHTML;
    assert(rendered.includes('owner/p1') && rendered.includes('owner/p2') && rendered.includes('owner/p3'),
      'all eligible pages accumulated in order');
    assert(!rendered.includes('owner/vision-only'), 'mid-pagination ineligible model filtered');
    assert.strictEqual(
      w4.document.querySelectorAll('#model-list .model-item').length, 3,
      'three eligible cards after three pages');
    assert(!rendered.includes('hf-load-more'), 'cursor exhausted after auto-advance');
  });

  await test('14. drag-and-drop and selection remain wired to filtered models', async () => {
    const { w: w5 } = makeWindow();
    w5.fetch = async () => ({ ok: true, status: 200, headers: { get: () => null },
      json: async () => [textModel({ id: 'owner/draggable' })] });
    await w5.loadHFModels(true);
    const evt = { dataTransfer: { setData: (t, v) => { evt.payload = v; } } };
    w5.handleModelDragStart(evt, 'owner/draggable');
    assert(evt.payload && JSON.parse(evt.payload).id === 'owner/draggable',
      'drag payload carries the full model object');
  });

  // ── 15. gating / accessibility: fail-closed public-only selection ──
  // Regression: google/embeddinggemma-300m (public repo, gated "manual")
  // appeared in the selector and produced a 403 GatedRepoError at load time.
  await test('15. only publicly accessible models are selectable', () => {
    // public + open → eligible
    assert.strictEqual(elig(textModel()).eligible, true);
    // manual gate (the EmbeddingGemma case) → excluded
    const manual = elig(textModel({ id: 'google/embeddinggemma-300m', gated: 'manual' }));
    assert.strictEqual(manual.eligible, false);
    assert.match(manual.reason, /gated/);
    // auto gate (accept-conditions style) → excluded too
    const auto = elig(textModel({ gated: 'auto' }));
    assert.strictEqual(auto.eligible, false);
    // truthy non-string gate value → excluded
    assert.strictEqual(elig(textModel({ gated: true })).eligible, false);
    // unknown/missing gating status → FAIL CLOSED
    assert.strictEqual(elig(textModel({ gated: undefined })).eligible, false);
    assert.strictEqual(elig(textModel({ gated: null })).eligible, false);
    const noGate = elig({ id: 'owner/mystery', pipeline_tag: 'text-generation',
      tags: [], gated: undefined, private: false, safetensors: { total: 1 * B } });
    assert.strictEqual(noGate.eligible, false);
    assert.match(noGate.reason, /unknown access status/);
    // private repos excluded
    assert.strictEqual(elig(textModel({ private: true })).eligible, false);
    // missing private flag → excluded (cannot establish openness)
    const noPrivate = elig(textModel({ private: undefined }));
    assert.strictEqual(noPrivate.eligible, false);
    assert.match(noPrivate.reason, /private/);
    // task compatibility is independent of access: an OPEN model of a
    // non-text task is still excluded by the task rule
    assert.strictEqual(elig(textModel({ pipeline_tag: 'text-to-video' })).eligible, false);
  });

  await test('16. gating excludes models across task categories and pagination', async () => {
    const { w: w6 } = makeWindow();
    const gated = (id) => textModel({ id, gated: 'manual' });
    // page 1 mixes two task categories; every model in it is gated
    const page1 = page([
      gated('owner/gated-gen'),
      gated('owner/gated-embed'),
    ], 'CUR-G');
    // page 2 has one open model of each category
    const page2 = page([
      textModel({ id: 'owner/open-gen' }),
      textModel({ id: 'owner/open-embed', pipeline_tag: 'feature-extraction' }),
    ], null);
    const urls = [];
    w6.fetch = async (url) => {
      urls.push(String(url));
      const cursor = /[?&]cursor=([^&]*)/.exec(String(url));
      const key = cursor ? decodeURIComponent(cursor[1]) : '';
      const resp = key === 'CUR-G' ? page2 : page1;
      return { ok: true, status: 200, headers: { get: (n) => resp.headers[n] || null },
        json: async () => resp.json };
    };
    await w6.loadHFModels(true);
    await w6.loadHFModels(false);
    const rendered = w6.document.getElementById('model-list').innerHTML;
    assert(!rendered.includes('gated-gen') && !rendered.includes('gated-embed'),
      'gated models never render, from either task category or page');
    assert(rendered.includes('owner/open-gen') && rendered.includes('owner/open-embed'),
      'open models from both categories render');
    // server-side gated filter was requested
    assert(urls[0].includes('gated=false'), `server-side gated filter: ${urls[0]}`);
    assert(urls[0].includes('expand%5B%5D=gated') && urls[0].includes('expand%5B%5D=private'),
      'gated/private metadata requested');
  });

  await test('17. cached/rendered lists cannot bypass the eligibility check', async () => {
    const { w: w7 } = makeWindow();
    // the cached list is poisoned with a gated entry at fetch time; the
    // fetch-time filter drops it AND the render-time defensive re-filter
    // would drop it again -- the gate is structural, not path-dependent
    w7.fetch = async () => ({ ok: true, status: 200, headers: { get: () => null },
      json: async () => [textModel({ id: 'owner/fresh' }),
                         textModel({ id: 'owner/gated-sneaky', gated: 'manual' })] });
    await w7.loadHFModels(true);
    // re-render twice from the cached list (modal reopen path)
    w7.renderModelList();
    w7.renderModelList();
    const rendered = w7.document.getElementById('model-list').innerHTML;
    assert(rendered.includes('owner/fresh'), 'legit cached model renders');
    assert(!rendered.includes('owner/gated-sneaky'),
      'gated entry cannot bypass eligibility from a cached list');
    // render-time defensive filter is structural
    const appSrc = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
    assert(/applyModelFilters\(filteredModels\)/.test(appSrc),
      'renderModelList must re-run eligibility over the cached list');
  });

  await test('18. API failures never present unverified models as accessible', async () => {
    const { w: w8 } = makeWindow();
    w8.fetch = async () => ({ ok: false, status: 503, json: async () => ({}), headers: { get: () => null } });
    await w8.loadHFModels(true);
    const rendered = w8.document.getElementById('model-list').innerHTML;
    assert(rendered.includes('Failed to load models'), 'error surfaced to the user');
    assert.strictEqual(w8.document.querySelectorAll('#model-list .model-item').length, 0,
      'no models rendered from a failed metadata request');
    assert(!rendered.includes('select-model-btn'), 'nothing selectable');
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
