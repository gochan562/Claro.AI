'use strict';
// Storage Manager frontend tests: drives the REAL dashboard-app.js +
// dashboard.html in jsdom with stubbed fetch. Covers:
//   - stats render on open (loading state → numbers), manual refresh
//   - confirmation dialog shows category, estimated freed, what, warning
//   - successful cleanup: reclaimed amount, stats refreshed, success note
//   - failed cleanup: error shown, NO success message
//   - duplicate cleanup disabled while running
//   - warning banner at ≥80% volume usage; no invented capacity when absent

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
const HTML_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard.html'), 'utf8');

function makeWindow() {
  const start = HTML_SRC.indexOf('<div id="storage-manager-modal"');
  const end = HTML_SRC.indexOf('</main>', start);
  const pickerHtml = HTML_SRC.slice(HTML_SRC.indexOf('<div id="model-picker-modal"'), end);
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
  w.CodeMirror = function () { return { getValue: () => '', setValue: () => {}, on: () => {}, refresh: () => {}, getWrapperElement: () => w.document.createElement('div') }; };
  w.marked = { parse: (s) => `<p>${s}</p>` };
  w.DOMPurify = { sanitize: (s) => s };
  w.PyodideLocal = { run: async () => ({ ok: true, stdout: '', stderr: '', error: null }) };
  w.prompt = () => null; w.confirm = () => false;
  w.__fetchLog = [];
  w.eval(APP_SRC);
  return { dom, w };
}

const STATS = {
  categories: {
    models: { label: 'Model cache', bytes: 40 * 1024 ** 3, files: 10, entries: 3, protected_entries: 1 },
    datasets: { label: 'Dataset cache', bytes: 338 * 1024 ** 2, files: 5, entries: 2, protected_entries: 0 },
    temp: { label: 'Temporary files', bytes: 5 * 1024 ** 2, files: 0, entries: 4, protected_entries: 0 },
  },
  totals: { managed_bytes: 40 * 1024 ** 3 + 338 * 1024 ** 2 + 5 * 1024 ** 2,
            available_bytes: 60 * 1024 ** 3, total_bytes: 494 * 1024 ** 3,
            volume_percent: 82.3, warning: true,
            warning_reason: 'Volume usage at 82.3% (threshold 80%)' },
  protected_roots: ['ml_workspaces', 'ml_experiments', 'training_outputs'],
  active: { training_jobs: 1, dataset_ingestions_in_flight: 0 },
  generated_at: new Date().toISOString(),
};

function statsJson(overrides = {}) {
  const s = JSON.parse(JSON.stringify(STATS));
  Object.assign(s, overrides);
  return s;
}

let passed = 0, failed = 0;
function test(name, fn) {
  return Promise.resolve().then(fn)
    .then(() => { console.log(`✓ ${name}`); passed++; })
    .catch((e) => { console.error(`✗ ${name}: ${e.message}`); failed++; });
}

async function main() {
  console.log('=== Storage Manager Frontend Tests ===\n');

  await test('1. opening fetches stats, renders numbers, bar, and warning banner', async () => {
    const { w } = makeWindow();
    w.fetch = async (url) => {
      w.__fetchLog.push(String(url));
      if (String(url).includes('/api/storage/stats')) {
        return { ok: true, status: 200, json: async () => STATS };
      }
      throw new Error('unexpected fetch ' + url);
    };
    await w.openStorageManager();
    assert(w.document.getElementById('storage-manager-modal').classList.contains('active'));
    const summary = w.document.getElementById('storage-summary').innerHTML;
        const expectedManaged = (40 * 1024 ** 3 + 338 * 1024 ** 2 + 5 * 1024 ** 2) / 1024 ** 3;
    assert(summary.includes(`${expectedManaged.toFixed(2)} GB`), `managed bytes rendered: ${summary.slice(0, 200)}`);
    assert(summary.includes('60.00 GB'), 'available bytes rendered');
    assert(summary.includes('82.3%'), 'volume percent rendered');
    assert(w.document.querySelector('#storage-summary div[style*="width:82.3%"]'), 'progress bar rendered');
    const warn = w.document.getElementById('storage-warning');
    assert(warn.style.display !== 'none' && warn.textContent.includes('82.3%'), 'warning banner shown');
    const cats = w.document.getElementById('storage-categories').innerHTML;
    assert(cats.includes('Model cache') && cats.includes('338.0 MB') && cats.includes('Dataset cache'));
    assert(cats.includes('1 protected (active)'), 'protected entries surfaced');
  });

  await test('2. no invented capacity: without volume stats, only available is shown', async () => {
    const { w } = makeWindow();
    const s = statsJson();
    s.totals.available_bytes = 60 * 1024 ** 3;
    s.totals.total_bytes = null; s.totals.volume_percent = null; s.totals.warning = false;
    s.totals.warning_reason = null;
    w.fetch = async () => ({ ok: true, status: 200, json: async () => s });
    await w.openStorageManager();
    const summary = w.document.getElementById('storage-summary').innerHTML;
    assert(summary.includes('60.00 GB'), 'available still shown');
    assert(!summary.includes('width:'), 'no progress bar without capacity');
    assert(summary.includes('not reliably exposed'), 'honest capacity note shown');
    assert(w.document.getElementById('storage-warning').style.display === 'none');
  });

  await test('3. confirmation dialog shows category, estimate, what, warning', async () => {
    const { w } = makeWindow();
    w.fetch = async (url) => {
      w.__fetchLog.push(String(url));
      return { ok: true, status: 200, json: async () => STATS };
    };
    await w.openStorageManager();
    await w.requestStorageCleanup('models');
    const box = w.document.getElementById('storage-confirm');
    assert(box.style.display !== 'none', 'confirm dialog visible');
    const body = w.document.getElementById('storage-confirm-body').innerHTML;
    assert(body.includes('Clean model cache'), 'category named');
    assert(body.includes('Estimated space to free:'), 'estimate label present');
    assert(body.includes('models--*'), 'what will be deleted');
    assert(body.includes('downloaded again'), 're-download warning');
    assert(body.includes('protected (skipped)'), 'protected count surfaced');
    // manual refresh works
    const countStats = () => w.__fetchLog.filter((u) => u.includes('/api/storage/stats')).length;
    const before = countStats();
    await w.refreshStorageStats();
    assert(countStats() > before, 'refresh refetches stats');
  });

  await test('4. successful cleanup shows reclaimed amount, hides dialog, refreshes stats', async () => {
    const { w } = makeWindow();
    let cleanups = 0;
    w.fetch = async (url, init = {}) => {
      w.__fetchLog.push(String(url));
      if (String(url).includes('/api/storage/cleanup')) {
        cleanups += 1;
        assert.strictEqual(init.method, 'POST');
        assert.deepStrictEqual(JSON.parse(init.body), { category: 'datasets' });
        return { ok: true, status: 200, json: async () => ({
          category: 'datasets', removed_count: 2, freed_bytes: 336 * 1024 ** 2,
          skipped: [], errors: [], dry_run: false }) };
      }
      return { ok: true, status: 200, json: async () => STATS };
    };
    await w.openStorageManager();
    await w.requestStorageCleanup('datasets');
    await w.confirmStorageCleanup();
    const result = w.document.getElementById('storage-result').innerHTML;
    assert(result.includes('✅'), 'success notification shown');
    assert(result.includes('2 item(s)'), 'removed count shown');
    assert(result.includes('336.0 MB'), 'reclaimed bytes shown');
    assert(!result.includes('❌'), 'no error on success');
    assert.strictEqual(cleanups, 1);
    assert(w.document.getElementById('storage-confirm').style.display === 'none');
    // stats refreshed after cleanup (log contains 3+ stats fetches)
    const statsFetches = w.__fetchLog.filter((u) => u.includes('/api/storage/stats')).length;
    assert(statsFetches >= 3, `stats refetched after cleanup (${statsFetches})`);
  });

  await test('5. failed cleanup shows error and NO success message', async () => {
    const { w } = makeWindow();
    w.fetch = async (url, init = {}) => {
      if (String(url).includes('/api/storage/cleanup')) {
        return { ok: false, status: 500, json: async () => ({ error: 'disk busy', code: 'storage_error' }) };
      }
      return { ok: true, status: 200, json: async () => STATS };
    };
    await w.openStorageManager();
    await w.requestStorageCleanup('temp');
    await w.confirmStorageCleanup();
    const result = w.document.getElementById('storage-result').innerHTML;
    assert(result.includes('❌'), 'error shown');
    assert(result.includes('disk busy'), 'meaningful message shown');
    assert(!result.includes('✅'), 'no success on failure');
  });

  await test('6. duplicate cleanups disabled while an operation runs', async () => {
    const { w } = makeWindow();
    let release;
    const gate = new Promise((r) => { release = r; });
    w.fetch = async (url, init = {}) => {
      w.__fetchLog.push(String(url));
      if (String(url).includes('/api/storage/cleanup')) { await gate; return { ok: true, status: 200, json: async () => ({ category: 'models', removed_count: 1, freed_bytes: 10, skipped: [], errors: [] }) }; }
      return { ok: true, status: 200, json: async () => STATS };
    };
    await w.openStorageManager();
    await w.requestStorageCleanup('models');
    const first = w.confirmStorageCleanup(); // in flight
    const second = w.confirmStorageCleanup(); // must be a no-op (storageBusy)
    release();
    await Promise.all([first, second]);
    const cleanups = w.__fetchLog.filter((u) => u.includes('/api/storage/cleanup')).length;
    assert.strictEqual(cleanups, 1, `duplicate suppressed (got ${cleanups})`);
    const btn = w.document.getElementById('storage-confirm-btn');
    assert.strictEqual(btn.textContent, 'Yes, clean it', 'button label restored');
    assert.strictEqual(btn.disabled, false, 'button re-enabled');
  });

  await test('7. small-screen friendly: modal capped at viewport width', async () => {
    const { w } = makeWindow();
    w.fetch = async () => ({ ok: true, status: 200, json: async () => STATS });
    await w.openStorageManager();
    const content = w.document.querySelector('#storage-manager-modal .modal-content');
    assert(content.getAttribute('style').includes('min(600px'), 'inline width cap present');
  });

  await test('8. notebook editor toolbar keeps the Storage button across re-renders', async () => {
    const { w } = makeWindow();
    w.fetch = async (url) => {
      w.__fetchLog.push(String(url));
      if (String(url).includes('/api/storage/stats')) {
        return { ok: true, status: 200, json: async () => STATS };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    };
    // create a notebook through the app's own API and enter the editor
    const nbId = w.createNotebook('Storage Test');
    w.setActiveNotebook(nbId);
    w.renderNotebookEditor();
    const toolbar = w.document.querySelector('.notebook-toolbar');
    const findBtn = () => [...toolbar.querySelectorAll('button')]
      .find((b) => b.getAttribute('onclick') === 'openStorageManager()');
    const btn = findBtn();
    assert(btn, 'Storage button present in the editor toolbar');
    assert.strictEqual(btn.textContent.trim(), '💾 Storage');
    // existing toolbar functionality unchanged: neighbors intact
    const onclk = (sel) => [...toolbar.querySelectorAll('button')].map((b) => b.getAttribute('onclick'));
    const all = onclk();
    assert(all.includes('runAllCells()') && all.includes('runMarkdownCells()') &&
           all.includes('toggleMlExperiments()') && all.includes('backToList()'),
      'existing toolbar buttons unchanged');
    // re-render must NOT remove the button (the exact regression concern)
    w.renderNotebookEditor();
    w.renderNotebookEditor();
    assert(findBtn(), 'Storage button survives editor re-render');
    // The button's wiring is the inline onclick attribute pointing at the
    // SINGLE global implementation (jsdom outside-only does not execute
    // inline handlers, so invoke the global the attribute references).
    assert.strictEqual(typeof w.openStorageManager, 'function',
      'openStorageManager must be a global the onclick attribute resolves to');
    await w.openStorageManager();
    assert(w.document.getElementById('storage-manager-modal').classList.contains('active'),
      'button target opens the storage manager modal');
    assert(w.__fetchLog.some((u) => u.includes('/api/storage/stats')),
      'stats fetched on open');
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
