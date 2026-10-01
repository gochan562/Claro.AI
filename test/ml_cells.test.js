'use strict';
// Classical-ML cell tests: creation, rendering, persistence, migration,
// and config helpers — driven against the real dashboard-app.js plus
// ml_experiment.js in jsdom. No network.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
const ML_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'ml_experiment.js'), 'utf8');

function makeWindow(savedRegistry) {
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
  w.fetch = async () => { throw new Error('no network in tests'); };
  w.prompt = () => null;
  w.confirm = () => false;
  if (savedRegistry) w.localStorage.setItem('claro-notebooks-registry', savedRegistry);
  const ml = w.document.createElement('script');
  ml.textContent = ML_SRC;
  w.document.body.appendChild(ml);
  const app = w.document.createElement('script');
  app.textContent = APP_SRC;
  w.document.body.appendChild(app);
  for (const fn of ['addCell', 'insertCellAfter', 'deleteCell', 'getActiveNotebook',
    'defaultMlCell', 'mlBuildExperimentConfig', 'mlParseSearchText', 'mlCoerceParams',
    'buildMlCellElement', 'mlMetricLine']) {
    assert.strictEqual(typeof w[fn], 'function', `must expose global ${fn}()`);
  }
  return { dom, w };
}

function newNotebook(w) {
  const id = w.createNotebook('ML Test');
  w.setActiveNotebook(id);
  return id;
}

let passed = 0, failed = 0;
function test(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => { console.log(`✓ ${name}`); passed++; })
    .catch((e) => { console.error(`✗ ${name}: ${e.message}`); failed++; });
}

async function main() {
  console.log('=== Classical-ML Cell Tests ===\n');

  await test('1. all six ML cell types create with serializable state', () => {
    const { w } = makeWindow();
    newNotebook(w);
    for (const t of ['dataset', 'split', 'preprocess', 'predictor', 'evaluate', 'predict']) {
      w.addCell(t);
    }
    const nb = w.getActiveNotebook();
    // seed code cell id 1 + 6 ML cells
    assert.strictEqual(nb.cells.length, 7);
    const kinds = nb.cells.slice(1).map((c) => c.ml && c.ml.kind);
    assert.deepStrictEqual(Array.from(kinds),
      ['dataset', 'split', 'preprocess', 'predictor', 'evaluate', 'predict']);
    for (const c of nb.cells.slice(1)) {
      assert.strictEqual(c.ml.version, 1);
      JSON.stringify(c.ml); // must serialize
    }
  });

  await test('2. ML cells render with headers, controls, and bodies', () => {
    const { w } = makeWindow();
    newNotebook(w);
    for (const t of ['dataset', 'split', 'preprocess', 'predictor', 'evaluate', 'predict']) {
      w.addCell(t);
    }
    const bodies = w.document.querySelectorAll('.ml-cell-body');
    assert.strictEqual(bodies.length, 6);
    const html = w.document.getElementById('notebook-cells').innerHTML;
    for (const label of ['Dataset', 'Split', 'Preprocess', 'Predictor', 'Evaluate', 'Predict']) {
      assert(html.includes(label), `missing ${label} UI`);
    }
    assert(html.includes('mlRunCell('), 'Run buttons wired to mlRunCell');
  });

  await test('3. notebook persistence keeps ML state across reload', () => {
    const first = makeWindow();
    newNotebook(first.w);
    first.w.addCell('predictor');
    const nb = first.w.getActiveNotebook();
    nb.cells[1].ml.task = 'regression';
    nb.cells[1].ml.model_family = 'Ridge';
    first.w.updateNotebook();
    const saved = first.w.localStorage.getItem('claro-notebooks-registry');
    assert(saved.includes('Ridge'), 'registry must persist ML config');
    const second = makeWindow(saved);
    // Simulate app boot reading the saved registry.
    const reg = JSON.parse(second.w.localStorage.getItem('claro-notebooks-registry'));
    assert(reg.notebooks[nb.id], 'notebook survives reload');
    const pc = reg.notebooks[nb.id].cells.find((c) => c.ml && c.ml.kind === 'predictor');
    assert.strictEqual(pc.ml.model_family, 'Ridge');
    assert.strictEqual(pc.ml.task, 'regression');
  });

  await test('4. legacy notebooks without ML cells still load', () => {
    const legacy = JSON.stringify({
      notebooks: { notebook_1: { id: 'notebook_1', name: 'Old', cells: [
        { id: 1, type: 'code', content: 'print(1)', output: '', status: 'idle' },
        { id: 2, type: 'training', content: '{}', output: '', status: 'idle' },
      ], nextCellId: 3 } },
      notebookOrder: ['notebook_1'], activeNotebookId: 'notebook_1', nextNotebookId: 2,
    });
    const { w } = makeWindow(legacy);
    w.loadAllNotebooks();
    w.setActiveNotebook('notebook_1');
    const nb = w.getActiveNotebook();
    assert.strictEqual(nb.cells.length, 2);
    assert.strictEqual(nb.cells[0].type, 'code');
    w.addCell('dataset');
    assert.strictEqual(nb.cells.length, 3);
    assert.strictEqual(nb.cells[2].ml.kind, 'dataset');
  });

  await test('5. search-text parsing (grid lists + ranges)', () => {
    const { w } = makeWindow();
    const p = w.mlParseSearchText('n_estimators: 50, 100\nmax_depth: 3-5:3:int\nkernel: rbf, linear');
    assert.deepStrictEqual(Array.from(p.n_estimators), [50, 100]);
    assert.deepStrictEqual(Array.from(p.max_depth), [3, 4, 5]);
    assert.deepStrictEqual(Array.from(p.kernel), ['rbf', 'linear']);
    assert.throws(() => w.mlParseSearchText('oops-no-colon'), /Bad search line/);
  });

  await test('6. param coercion follows schema', () => {
    const { w } = makeWindow();
    const out = w.mlCoerceParams('RandomForestClassifier',
      { param_n_estimators: '50', param_max_depth: '0', param_max_features: '' });
    assert.strictEqual(out.n_estimators, 50);
    assert.strictEqual(out.max_depth, 0);
    assert(!('max_features' in out), 'blank choice omitted (server default applies)');
    assert.throws(() => w.mlCoerceParams('Ridge', { param_alpha: 'abc' }), /must be a number/);
  });

  await test('7. experiment config builder validates references', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('predictor');
    assert.throws(() => w.mlBuildExperimentConfig(w.getActiveNotebook().cells[1]), /Dataset Cell/);
  });

  await test('8. existing cell types unaffected by ML integration', () => {
    const { w } = makeWindow();
    newNotebook(w);
    w.addCell('code');
    w.addCell('markdown');
    w.addCell('training');
    const nb = w.getActiveNotebook();
    assert(nb.cells[1].ml === undefined && nb.cells[2].ml === undefined);
    assert(nb.cells[3].training && nb.cells[3].training.task_type === 'text-classification');
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
