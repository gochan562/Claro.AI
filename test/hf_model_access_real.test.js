'use strict';
// REAL Hugging Face Hub integration tests for the Model Picker's
// public-access-only selection. These are NOT mocked: they call the live
// https://huggingface.co/api/models endpoint.
//
//   - No GPU is consumed and no ZeroGPU quota is touched (metadata only).
//   - No HF token is required or sent.
//   - Every check is skippable when the Hub is unreachable (⊘), so an
//     offline machine still passes the suite. Failure INSIDE a live check
//     (reachable Hub, wrong verdict) is a hard failure.
//
// Mocked counterparts live in test/model_picker_filter.test.js (checks
// 15–18); this file only asserts what only the live Hub can prove.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');

// Extract the PRODUCTION eligibility function (no re-implementation here).
function loadEligibility() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>',
    { url: 'http://localhost:5000/', runScripts: 'outside-only' });
  dom.window.CodeMirror = function () { return { getValue: () => '', setValue: () => {}, on: () => {}, refresh: () => {}, getWrapperElement: () => dom.window.document.createElement('div') }; };
  dom.window.marked = { parse: (s) => s };
  dom.window.DOMPurify = { sanitize: (s) => s };
  dom.window.PyodideLocal = { run: async () => ({ ok: true, stdout: '', stderr: '', error: null }) };
  dom.window.fetch = async () => { throw new Error('fetch not stubbed'); };
  dom.window.prompt = () => null;
  dom.window.confirm = () => false;
  // The app's init path targets notebook markup this minimal document does
  // not contain; neutering window timers keeps its deferred callbacks from
  // ever being queued (this file only exercises hfModelEligibility).
  dom.window.setTimeout = () => 0;
  dom.window.clearTimeout = () => {};
  dom.window.setInterval = () => 0;
  dom.window.clearInterval = () => {};
  dom.window.requestAnimationFrame = () => 0;
  dom.window.eval(APP_SRC);
  return dom.window;
}

// The EXACT query the Model Picker builds (public/dashboard-app.js).
function pickerUrl(extra = {}) {
  const params = new URLSearchParams({ limit: '30', full: 'false' });
  params.set('num_parameters', 'max:8B');
  params.set('gated', 'false');
  for (const f of ['safetensors', 'pipeline_tag', 'tags', 'library_name',
                   'downloads', 'likes', 'lastModified', 'gated', 'private']) {
    params.append('expand[]', f);
  }
  for (const [k, v] of Object.entries(extra)) params.set(k, v);
  return `https://huggingface.co/api/models?${params}`;
}

let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

async function main() {
  let w = null;
  try {
    const res = await fetch(pickerUrl({ limit: '5' }), { signal: AbortSignal.timeout(20000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await res.json();
    w = loadEligibility();
  } catch (e) {
    console.log(`⊘ real HF integration tests skipped (Hub unreachable: ${e.message})`);
    return;
  }
  const elig = w.hfModelEligibility;
  assert.strictEqual(typeof elig, 'function', 'hfModelEligibility must be global');

  // NOTE: hfModelEligibility enforces MORE than public access (task,
  // modality, parameter-count rules). A live "trending" page legitimately
  // contains models rejected for those reasons. The access invariant below
  // is therefore only about gating/privacy — never about task/size rules.
  const ACCESS_RE = /gated|private|access status/;

  // ── 1. LIVE invariant: every model the picker query returns carries
  // explicit public+open metadata, and NO model is rejected for access.
  {
    const res = await fetch(pickerUrl({ limit: '30' }), { signal: AbortSignal.timeout(20000) });
    assert.strictEqual(res.status, 200, `picker query shape must be accepted: ${res.status}`);
    const models = await res.json();
    assert(Array.isArray(models) && models.length > 0, 'live query must return models');
    const notOpen = models.filter((m) => m.private !== false || m.gated !== false)
      .map((m) => `${m.id} (gated=${JSON.stringify(m.gated)}, private=${JSON.stringify(m.private)})`);
    assert.strictEqual(notOpen.length, 0,
      `the gated=false server filter must keep only explicitly public+open models; offenders: ${notOpen.join(', ')}`);
    const unknown = models.filter((m) => m.gated === undefined || m.private === undefined);
    assert.strictEqual(unknown.length, 0,
      `expand[]=gated,expand[]=private must materialize on every live entry; missing on: ${unknown.map((m) => m.id).join(', ')}`);
    const accessRejected = models.filter((m) => !elig(m).eligible && ACCESS_RE.test(elig(m).reason))
      .map((m) => `${m.id} (${elig(m).reason})`);
    assert.strictEqual(accessRejected.length, 0,
      `no live result may be rejected for access status; offenders: ${accessRejected.join(', ')}`);
    ok(`live picker page: ${models.length} models, all explicitly public+open, zero access rejections`);
  }

  // ── 2. LIVE regression: google/embeddinggemma-300m is gated "manual" and
  // the production eligibility function rejects it (the 403 GatedRepoError
  // that motivated this rule can never be reached from the selector).
  {
    const res = await fetch(
      'https://huggingface.co/api/models/google/embeddinggemma-300m?expand[]=gated&expand[]=private',
      { signal: AbortSignal.timeout(20000) });
    assert.strictEqual(res.status, 200, 'regression model metadata must be fetchable');
    const meta = await res.json();
    if (meta.gated === false && meta.private === false) {
      // The repo may become ungated someday; then the selector MUST show it.
      assert.strictEqual(elig(meta).eligible, true,
        'an ungated google/embeddinggemma-300m would be selectable');
      ok('regression model is now open — selector would show it (assertion updated automatically)');
    } else {
      assert.strictEqual(meta.gated, 'manual',
        `expected gated:"manual" on the regression model, got ${JSON.stringify(meta.gated)}`);
      const verdict = elig(meta);
      assert.strictEqual(verdict.eligible, false, 'gated model must be ineligible');
      assert.match(verdict.reason, /gated/, `reason must say why: ${verdict.reason}`);
      ok('live regression: google/embeddinggemma-300m (gated "manual") rejected by production eligibility');
    }
  }

  // ── 3. LIVE search path: server-side gated=false applies to search too —
  // the exact gated regression ID can never appear in search results.
  {
    const res = await fetch(pickerUrl({ limit: '30', search: 'embeddinggemma-300m' }),
      { signal: AbortSignal.timeout(20000) });
    assert.strictEqual(res.status, 200, 'search query shape must be accepted');
    const models = await res.json();
    assert(Array.isArray(models), 'search must return a list');
    const hit = models.find((m) => m.id === 'google/embeddinggemma-300m');
    assert(!hit, `gated regression model must not appear in live search results (got ${models.length} results)`);
    const accessRejected = models.filter((m) => !elig(m).eligible && ACCESS_RE.test(elig(m).reason));
    assert.strictEqual(accessRejected.length, 0,
      `no live search result may be rejected for access status; offenders: ${accessRejected.map((m) => m.id).join(', ')}`);
    ok(`live search "embeddinggemma-300m": ${models.length} results, none access-rejected, regression ID absent`);
  }

  console.log(`\nAll REAL HF model-access tests passed (${passed} checks).`);
}

main().catch((e) => { console.error(e); process.exit(1); });
