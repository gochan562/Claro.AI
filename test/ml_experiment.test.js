'use strict';
// Classical-ML backend tests: validation, catalog parity with the Python
// catalog, and a full local end-to-end (upload CSV -> split preview ->
// train -> metrics -> predict -> compare -> rerun -> rename -> delete).
// No network (HF paths only validated, never fetched).

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const childProcess = require('child_process');

const ml = require('../ml_experiment_backend');

function expectErr(fn, code) {
  let err = null;
  try { fn(); } catch (e) { err = e; }
  assert(err, 'expected an error to be thrown');
  if (code) assert.strictEqual(err.code, code, `expected code ${code}, got ${err.code}: ${err.message}`);
  return err;
}

function tinyCsv(n = 120, seed = 11) {
  // Deterministic pseudo-random CSV (no external RNG needed).
  let s = 'age,income,city,bought\n';
  let x = seed;
  const rnd = () => { x = (x * 1103515245 + 12345) & 0x7fffffff; return x / 0x7fffffff; };
  const cities = ['a', 'b', 'c'];
  for (let i = 0; i < n; i++) {
    const age = 18 + Math.floor(rnd() * 52);
    const income = Math.round(30000 + rnd() * 60000);
    const city = cities[Math.floor(rnd() * 3)];
    const bought = (income > 55000) !== (city === 'a') ? 'yes' : 'no';
    s += `${age},${income},${city},${bought}\n`;
  }
  return s;
}

function waitForTerminal(expId, timeoutMs = 120000) {
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const t = setInterval(() => {
      let rec;
      try { rec = ml.getExperiment(expId); } catch (e) { clearInterval(t); reject(e); return; }
      if (rec.status === 'finished' || rec.status === 'failed') { clearInterval(t); resolve(rec); }
      else if (Date.now() - t0 > timeoutMs) { clearInterval(t); reject(new Error(`timeout waiting for ${expId} (status=${rec.status})`)); }
    }, 500);
  });
}

async function run() {
  // ── 1. validation rejects bad configs with codes ──
  expectErr(() => ml.validateDatasetUpload({}), 'bad_request');
  expectErr(() => ml.validateDatasetUpload({ filename: 'x.exe', content: 'a' }), 'bad_request');
  expectErr(() => ml.validateDatasetUpload({ filename: 'a.csv', content: '' }), 'bad_request');
  expectErr(() => ml.validateHfDataset({ dataset_id: 'not-an-id' }), 'bad_request');
  expectErr(() => ml.validateSplitConfig({ strategy: 'nope' }), 'bad_request');
  expectErr(() => ml.validateSplitConfig({ strategy: 'train_val_test', train_frac: 0.5, val_frac: 0.5, test_frac: 0.5 }), 'bad_request');
  expectErr(() => ml.validateModelConfig({ family: 'Nope' }, 'classification'), 'bad_request');
  expectErr(() => ml.validateModelConfig({ family: 'LinearRegression' }, 'classification'), 'bad_request');
  expectErr(() => ml.validateExperimentConfig({}), 'bad_request');
  expectErr(() => ml.validatePredictRequest({}), 'bad_request');
  expectErr(() => ml.validateSearchConfig({ method: 'grid', params: {} }), 'bad_request');
  expectErr(() => ml.validateCvConfig({ method: 'kfold', folds: 99 }), 'bad_request');
  console.log('✓ validation rejects bad configs with codes');

  // ── 2. JS catalog parity with the Python catalog ──
  {
    const out = childProcess.execFileSync(process.env.PYTHON_BIN || 'python3',
      ['-c', 'import json; from ml.models import CATALOG; print(json.dumps(sorted(CATALOG.keys())))'],
      { cwd: path.join(__dirname, '..'), timeout: 60000, encoding: 'utf8' });
    const pyFams = JSON.parse(out.trim());
    const jsFams = [...ml.CLASSIFICATION_MODELS, ...ml.REGRESSION_MODELS].sort();
    assert.deepStrictEqual(jsFams, pyFams, 'JS family lists must match ml/models.py CATALOG');
    console.log('✓ JS/Python model catalog parity');
  }

  // ── 3. upload + profile ──
  process.env.PYTHON_BIN = process.env.PYTHON_BIN || 'python3';
  const csv = tinyCsv(150);
  const up = ml.validateDatasetUpload({ filename: 'customers.csv', content: csv });
  const ing = await ml.ingestUpload(up);
  assert(/^mlw_[a-f0-9]{12}$/.test(ing.workspace_id), 'workspace id shape');
  assert.strictEqual(ing.profile.row_count, 150);
  assert(ing.profile.columns.includes('bought'), 'columns present');
  assert(ing.manifest.sha256 && ing.manifest.sha256.length === 64, 'sha256 recorded');
  console.log('✓ upload + profile (150 rows)');

  // ── 4. split preview ──
  const prev = await ml.splitPreview(ing.workspace_id,
    { strategy: 'train_val_test', train_frac: 0.7, val_frac: 0.15, test_frac: 0.15, seed: 42, shuffle: true, stratify: true },
    {}, 'bought', 'classification');
  assert.strictEqual(prev.counts.train + prev.counts.validation + prev.counts.test, 150);
  assert(prev.classes.train && prev.classes.train.length === 2, 'class balance shown');
  console.log('✓ split preview 70/15/15 stratified');

  // ── 5. full experiment: train + CV + evaluate artifacts ──
  const config = ml.validateExperimentConfig({
    workspace_id: ing.workspace_id,
    task: 'classification',
    target_column: 'bought',
    feature_columns: null,
    dataset_view: {},
    split: { strategy: 'train_val_test', train_frac: 0.7, val_frac: 0.15, test_frac: 0.15, seed: 42, shuffle: true, stratify: true },
    preprocessing: {},
    model: { family: 'LogisticRegression', params: {} },
    seed: 42,
    cv: { method: 'stratified', folds: 5, scoring: 'f1_macro', seed: 42 },
    search: null,
    name: 'e2e-logreg',
  });
  const started = ml.runExperiment(config);
  assert(/^exp_[a-f0-9]{12}$/.test(started.experiment_id), 'experiment id shape');
  const done = await waitForTerminal(started.experiment_id);
  assert.strictEqual(done.status, 'finished', `experiment failed: ${done.error}`);
  assert(done.metrics && done.metrics.test && typeof done.metrics.test.accuracy === 'number', 'test metrics present');
  assert(done.metrics.cv && done.metrics.cv.mean != null, 'cv summary present');
  assert(done.metrics.importance && done.metrics.importance.features.length, 'importance present');
  const dir = path.join(__dirname, '..', 'ml_experiments', done.experiment_id);
  for (const f of ['config.json', 'metrics.json', 'model.joblib', 'split.joblib', 'experiment.json']) {
    assert(fs.existsSync(path.join(dir, f)), `artifact missing: ${f}`);
  }
  // Reload from disk (durability across restarts).
  const reloaded = JSON.parse(fs.readFileSync(path.join(dir, 'experiment.json'), 'utf8'));
  assert.strictEqual(reloaded.status, 'finished');
  console.log(`✓ train finished (test acc ${done.metrics.test.accuracy.toFixed(3)}) + artifacts`);

  // ── 6. second model + compare ──
  const config2 = { ...config, model: { family: 'RandomForestClassifier', params: { n_estimators: 10 } }, name: 'e2e-rf' };
  const started2 = ml.runExperiment(ml.validateExperimentConfig(config2));
  const done2 = await waitForTerminal(started2.experiment_id);
  assert.strictEqual(done2.status, 'finished', `rf failed: ${done2.error}`);
  const cmp = ml.compareExperiments([done.experiment_id, done2.experiment_id]);
  assert.strictEqual(cmp.length, 2);
  assert(cmp[0].model_family && cmp[0].metrics && cmp[0].config === undefined, 'compare rows are summaries');
  assert(cmp[1].model_params.n_estimators === 10, 'hyperparameters compared');
  console.log('✓ second model + comparison');

  // ── 7. predict on new rows + export shape ──
  const pred = await ml.predictWithExperiment(done.experiment_id, {
    experiment_id: done.experiment_id,
    rows: [{ age: 30, income: 80000, city: 'b' }, { age: 60, income: 20000, city: 'a' }],
  });
  assert.strictEqual(pred.count, 2, 'two predictions');
  assert(pred.rows[0].prediction !== undefined, 'labels present');
  assert(Array.isArray(pred.rows[0].probabilities), 'probabilities present');
  assert(fs.existsSync(path.join(dir, 'predictions.csv')), 'predictions.csv stored');
  console.log('✓ batch prediction with probabilities + stored CSV');

  // ── 8. rerun / rename / delete lifecycle ──
  const rerun = ml.rerunExperiment(done.experiment_id);
  assert.notStrictEqual(rerun.experiment_id, done.experiment_id, 'rerun gets a new id');
  const redone = await waitForTerminal(rerun.experiment_id);
  assert.strictEqual(redone.status, 'finished');
  const renamed = ml.renameExperiment(done.experiment_id, 'renamed-exp');
  assert.strictEqual(renamed.name, 'renamed-exp');
  const listed = ml.listExperiments().map((e) => e.experiment_id);
  for (const id of [done.experiment_id, done2.experiment_id, rerun.experiment_id]) {
    assert(listed.includes(id), `list must include ${id}`);
  }
  const del = ml.deleteExperiment(rerun.experiment_id);
  assert(del.deleted && !fs.existsSync(path.join(__dirname, '..', 'ml_experiments', rerun.experiment_id)));
  console.log('✓ rerun / rename / list / delete');

  // ── 9. limits enforced ──
  const tooMany = { ...config };
  tooMany.feature_columns = Array.from({ length: 500 }, (_, i) => `f${i}`);
  expectErr(() => ml.validateExperimentConfig(tooMany), 'bad_request');
  console.log('✓ limits enforced');

  // cleanup test artifacts (keep the uploaded workspace? remove all)
  for (const id of [done.experiment_id, done2.experiment_id]) {
    try { ml.deleteExperiment(id); } catch (_) {}
  }
  try { fs.rmSync(path.join(__dirname, '..', 'ml_workspaces', ing.workspace_id), { recursive: true, force: true }); } catch (_) {}
  console.log('\nAll ML experiment backend tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
