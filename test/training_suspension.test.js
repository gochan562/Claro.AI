'use strict';
// Fine-tuning suspension regression tests.
//
// Training Cells are TEMPORARILY removed from the user-facing product
// (TRAINING_SUSPENDED = true in public/dashboard-app.js). These tests pin
// the suspension contract:
//   1. no cell picker (static HTML or JS-rendered) offers a Training Cell
//   2. addCell('training') refuses to create a cell
//   3. a restored notebook containing Training Cells loads without crashing,
//      renders an unavailable notice, and PRESERVES the cell + its config
//      data for future restoration
//   4. no training job lifecycle can be triggered from the UI (start/stop/
//      SSE are no-ops; no network calls)
//   5. code / markdown / model / parameter cells still create and render
//
// The training backend, routes, and TrainingUI module are intentionally
// preserved and are covered by their own suites (training_backend.test.js,
// maxsteps.test.js, zerogpu_training.test.js, …), which must keep passing.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
const HTML_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard.html'), 'utf8');

let passed = 0, failed = 0;
async function test(name, fn) {
  try { await fn(); console.log(`✓ ${name}`); passed++; }
  catch (e) { console.log(`✗ ${name}: ${e.message}`); failed++; }
}

function makeWindow() {
  const start = HTML_SRC.indexOf('<div id="model-picker-modal"');
  const pickerHtml = start >= 0 ? HTML_SRC.slice(start, HTML_SRC.indexOf('</main>', start)) : '';
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
      <div class="notebook-toolbar"></div>
      <div id="notebook-cells"></div>
      <div id="gpu-terminal"></div>
      ${pickerHtml}
    </body></html>`,
    { url: 'http://localhost:5000/', runScripts: 'outside-only' });
  const w = dom.window;
  w.CodeMirror = function (host, opts) {
    const ed = { getValue: () => '', setValue: () => {}, on: () => {}, refresh: () => {},
      getWrapperElement: () => w.document.createElement('div') };
    return ed;
  };
  w.marked = { parse: (s) => `<p>${s}</p>` };
  w.DOMPurify = { sanitize: (s) => s };
  w.PyodideLocal = { run: async () => ({ ok: true, stdout: '', stderr: '', error: null }) };
  // Network tripwire: any fetch during these tests means the suspended
  // feature (or the suspension change) tried to reach the backend.
  const netCalls = [];
  w.fetch = async (url, init) => {
    const u = String(url);
    // the app's own GPU-provider init is unrelated to training
    if (!u.includes('/api/gpu-config')) netCalls.push(u);
    throw new Error(`fetch not allowed in suspension tests: ${url}`);
  };
  w.prompt = () => 'test-nb';
  w.confirm = () => false;
  w.eval(APP_SRC);
  return { dom, w, netCalls };
}

async function main() {
  // ── 1. the suspension gate is present and ON ──
  await test('1. TRAINING_SUSPENDED gate exists and is enabled', () => {
    assert(/const TRAINING_SUSPENDED = true;/.test(APP_SRC),
      'dashboard-app.js must declare const TRAINING_SUSPENDED = true');
  });

  // ── 2. no Training Cell entry in either add-cell menu ──
  await test('2. cell pickers (static HTML and JS) offer no Training Cell', () => {
    for (const [file, src] of [['dashboard.html', HTML_SRC], ['dashboard-app.js', APP_SRC]]) {
      assert(!src.includes("addCell('training')"), `${file} must not contain an addCell('training') entry`);
      assert(!/Training Cell<\/div>/.test(src), `${file} must not render a "Training Cell" menu label`);
      assert(!/Fine-tune a model on a dataset/.test(src), `${file} must not describe fine-tuning in a menu`);
    }
  });

  // ── 3. addCell('training') is refused ──
  await test('3. addCell("training") refuses to create a cell', async () => {
    const { w } = makeWindow();
    w.createNewNotebook();
    const before = w.getActiveNotebook().cells.length;
    w.addCell('training');
    assert.strictEqual(w.getActiveNotebook().cells.length, before,
      'no cell may be created for the suspended type');
    w.addCell('code');
    assert.strictEqual(w.getActiveNotebook().cells.length, before + 1,
      'other cell types must still be creatable');
  });

  // ── 4. restored notebook with Training Cells: loads, notice, data kept ──
  await test('4. existing notebooks with Training Cells load with a notice and keep their data', async () => {
    const { w, netCalls } = makeWindow();
    w.createNewNotebook();
    // Simulate an EXISTING notebook saved before the suspension, with a
    // fully configured training cell mid-job.
    const nb = w.getActiveNotebook();
    nb.cells.push({
      id: nb.nextCellId++, type: 'training', content: '', output: '', status: 'idle',
      outputFormat: 'text', modelInfo: null,
      training: {
        model_preset: 'distilbert-base', dataset_preset: 'imdb',
        model_id: 'distilbert-base-uncased', dataset_id: 'stanfordnlp/imdb',
        task_type: 'text-classification', epochs: 2, batch_size: 8,
        learning_rate: 0.00002, max_steps: '', validation_split: 10,
        training_method: 'lora', lora_r: 8, lora_alpha: 16, lora_dropout: 0.05,
        target_modules: 'auto', job_id: 'job_legacy_1', status: 'finished',
        progress: {}, metrics: [], logs: []
      }
    });
    // re-render from the registry the way a page load restores notebooks
    w.renderNotebookEditor();
    const html = w.document.getElementById('notebook-cells').innerHTML;
    assert(html.includes('temporarily unavailable'), 'the cell must show the unavailability notice');
    assert(!html.includes('Start Training'), 'no Start Training control may render');
    assert(!html.includes('Stop'), 'no Stop control may render');
    assert(html.includes('Delete'), 'the cell remains deletable');
    // data preservation: the cell and its config survive a save/load round-trip
    w.saveAllNotebooks();
    const saved = JSON.parse(w.localStorage.getItem('claro-notebooks-registry'));
    const restoredNb = Object.values(saved.notebooks)[0];
    const tc = restoredNb.cells.find((c) => c.type === 'training');
    assert(tc, 'the Training Cell must persist in the notebook');
    assert.strictEqual(tc.training.model_id, 'distilbert-base-uncased');
    assert.strictEqual(tc.training.training_method, 'lora');
    assert.strictEqual(tc.training.job_id, 'job_legacy_1');
    assert.strictEqual(tc.training.epochs, 2, 'configuration must be preserved untouched');
    // no network attempts during restore/render
    assert.strictEqual(netCalls.length, 0, `no backend calls during restore (got ${netCalls.join(', ')})`);
  });

  // ── 5. lifecycle no-ops: no job start, no stop request, no SSE ──
  await test('5. start/stop/SSE entry points are no-ops with zero network activity', async () => {
    const { w, netCalls } = makeWindow();
    w.createNewNotebook();
    const nb = w.getActiveNotebook();
    nb.cells.push({
      id: nb.nextCellId++, type: 'training', content: '', output: '', status: 'idle',
      outputFormat: 'text', modelInfo: null,
      training: { model_id: 'distilbert-base-uncased', dataset_id: 'stanfordnlp/imdb',
        task_type: 'text-classification', epochs: 2, batch_size: 8, learning_rate: 0.00002,
        validation_split: 10, job_id: null, status: 'idle', progress: {}, metrics: [], logs: [] }
    });
    const tc = nb.cells.find((c) => c.type === 'training');
    await w.startTrainingCell(tc.id);   // must not throw and must not POST
    await w.stopTrainingCell(tc.id, { caller: 'test' });
    w.attachTrainingSSE(tc.id, 'job_x'); // must not open an EventSource
    await w.runCell(tc.id);              // Run must not dispatch to training
    assert.strictEqual(netCalls.length, 0, `no network calls allowed (got ${netCalls.join(', ')})`);
    assert.strictEqual(tc.training.status, 'idle', 'cell state must be untouched');
    assert.strictEqual(tc.training.job_id, null, 'no job id may be attached');
  });

  // ── 6. other cell types still render ──
  await test('6. code, markdown, model, parameter cells still create and render', async () => {
    const { w } = makeWindow();
    w.createNewNotebook();
    w.addCell('code'); w.addCell('markdown'); w.addCell('model'); w.addCell('parameter');
    const nb = w.getActiveNotebook();
    // createNotebook seeds one starter code cell; the four added types follow
    // (string compare avoids jsdom cross-realm array identity issues)
    assert.strictEqual(nb.cells.map((c) => c.type).join(','),
      'code,code,markdown,model,parameter');
    w.renderNotebookEditor();
    const html = w.document.getElementById('notebook-cells').innerHTML;
    for (const marker of ['💻 Code', '📝 Markdown', '🤖 Model Cell', '🎛️ Parameters']) {
      assert(html.includes(marker), `cell type label must render: ${marker}`);
    }
    assert(html.includes('Select Model'), 'model picker button must still render');
  });

  // ── 7. README: no fine-tuning capability claims ──
  await test('7. README no longer claims fine-tuning as a capability', () => {
    const readme = fs.readFileSync(path.join(__dirname, '..', 'README.md'), 'utf8');
    assert(!/Training Cell provides/.test(readme), 'no Training Cell feature section');
    assert(!/LoRA fine-tuning/.test(readme), 'no LoRA capability claim');
    assert(!/^- Training Cells :/m.test(readme), 'no Training Cells bullet in the feature list');
    assert(!/more training presets/.test(readme), 'no training presets roadmap item');
    assert(!/Configure training/.test(readme), 'no Configure training workflow step');
    assert(/temporarily unavailable|temporarily suspended/.test(readme),
      'README should state the suspension');
    // preserved-implementation descriptions remain accurate
    assert(readme.includes('training_backend.js - Training job orchestration'),
      'file table stays accurate — the implementation is preserved');
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
