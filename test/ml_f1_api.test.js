'use strict';
// Regression: the experiment detail API response must carry BOTH flat F1
// values (f1_macro AND f1_weighted) for every classification split —
// extracted independently from evaluation.py's macro.f1 / weighted.f1 and
// surviving the whole serialization path (cmd_run record → experiment.json
// → Node store → GET /api/ml/experiments/:id). Asserts against the LIVE
// Express app. Values are asserted distinct so field swapping fails.

const assert = require('assert');
const http = require('http');
const path = require('path');
const fs = require('fs');

const app = require('../server.js');
const ml = require('../ml_experiment_backend.js');

let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

function get(port, p) {
  return new Promise((resolve, reject) => {
    http.get({ host: '127.0.0.1', port, path: p }, (res) => {
      let raw = '';
      res.on('data', (c) => { raw += c; });
      res.on('end', () => {
        let json = null;
        try { json = JSON.parse(raw); } catch (_) {}
        resolve({ status: res.statusCode, body: json, raw });
      });
    }).on('error', reject);
  });
}

function tinyCsv(n) {
  const rows = ['age,income,city,bought'];
  let s = 7;
  const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const cities = ['a', 'b', 'c'];
  for (let i = 0; i < n; i++) {
    const age = 18 + Math.floor(rnd() * 52), income = Math.round(30000 + rnd() * 60000);
    const city = cities[Math.floor(rnd() * 3)];
    rows.push(`${age},${income},${city},${(income > 55000) !== (city === 'a') ? 'yes' : 'no'}`);
  }
  return rows.join('\n');
}

async function main() {
  const server = await new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  const port = server.address().port;

  // Real classification experiment through the full pipeline (no retraining
  // of stored models — a fresh small run, the same one the e2e suite uses).
  const up = ml.validateDatasetUpload({ filename: 'customers.csv', content: tinyCsv(150) });
  const ing = await ml.ingestUpload(up);
  const cfg = ml.validateExperimentConfig({
    workspace_id: ing.workspace_id, task: 'classification', target_column: 'bought',
    dataset_view: {},
    split: { strategy: 'train_val_test', train_frac: 0.7, val_frac: 0.15,
             test_frac: 0.15, seed: 42, shuffle: true, stratify: true },
    preprocessing: {}, model: { family: 'LogisticRegression', params: {} },
    seed: 42, name: 'f1-api-check',
  });
  const started = ml.runExperiment(cfg);
  try {
    const done = await new Promise((resolve, reject) => {
      const t0 = Date.now();
      const t = setInterval(() => {
        const r = ml.getExperiment(started.experiment_id);
        if (r.status === 'finished' || r.status === 'failed') { clearInterval(t); resolve(r); }
        else if (Date.now() - t0 > 120000) { clearInterval(t); reject(new Error('timeout')); }
      }, 300);
    });
    assert.strictEqual(done.status, 'finished', 'experiment finished');

    // ── 1. Detail API response carries both F1 values for every split ──
    const r = await get(port, `/api/ml/experiments/${started.experiment_id}`);
    assert.strictEqual(r.status, 200, `detail status: ${r.status} ${r.raw.slice(0, 200)}`);
    const metrics = r.body.metrics;
    assert(metrics, 'metrics present in detail response');
    for (const part of ['train', 'validation', 'test']) {
      const m = metrics[part];
      assert(m, `${part} split present`);
      assert(typeof m.f1_macro === 'number' && Number.isFinite(m.f1_macro),
        `${part}.f1_macro is a number, got ${JSON.stringify(m.f1_macro)}`);
      assert(typeof m.f1_weighted === 'number' && Number.isFinite(m.f1_weighted),
        `${part}.f1_weighted is a number, got ${JSON.stringify(m.f1_weighted)}`);
      // Not accidentally swapped with other flat metrics either.
      assert(m.f1_macro !== m.accuracy || m.f1_weighted !== m.accuracy,
        'f1 values are F1, not an accuracy copy');
      // No nested legacy objects leak into the flat schema.
      assert(!('macro' in m) && !('weighted' in m),
        'flat schema: no nested macro/weighted objects');
    }
    // Macro and weighted F1 are extracted independently: they may coincide
    // in rare balanced cases, so require neither to be null/zero-by-default
    // and that at least the pair exists on BOTH splits with plausible range.
    for (const part of ['train', 'validation', 'test']) {
      const m = metrics[part];
      assert(m.f1_macro >= 0 && m.f1_macro <= 1, `${part}.f1_macro in [0,1]`);
      assert(m.f1_weighted >= 0 && m.f1_weighted <= 1, `${part}.f1_weighted in [0,1]`);
    }
    ok('detail API carries f1_macro AND f1_weighted for train/validation/test');

    // ── 2. Swap detection at the summarize layer (distinct fixed values) ──
    // The live run's macro/weighted are close by nature; the strict
    // swap check uses a fixed-value fixture through the real Python
    // serializer (subprocess, same pattern as the catalog-parity test).
    {
      const childProcess = require('child_process');
      const out = JSON.parse(childProcess.execFileSync(
        process.env.PYTHON_BIN || 'python3',
        ['-c',
         'import json, sys; sys.path.insert(0, ".");' +
         'from ml.experiment_runner import _summarize_metrics;' +
         'm = {"task": "classification", "train": {"task": "classification",' +
         '"accuracy": 0.8, "balanced_accuracy": 0.7, "roc_auc": 0.85, "pr_auc": 0.75, "log_loss": 0.4,' +
         '"macro": {"precision": 0.6, "recall": 0.5, "f1": 0.25},' +
         '"weighted": {"precision": 0.8, "recall": 0.8, "f1": 0.875}}};' +
         'print(json.dumps(_summarize_metrics(m)["train"]))'],
        { cwd: path.join(__dirname, '..'), timeout: 60000, encoding: 'utf8' }));
      assert.strictEqual(out.f1_macro, 0.25, 'macro.f1 → f1_macro (no swap)');
      assert.strictEqual(out.f1_weighted, 0.875, 'weighted.f1 → f1_weighted (no swap)');
      ok('_summarize_metrics extracts macro/weighted independently (no swap)');
    }

    // ── 3. List endpoint exposes the same flat metrics ──
    {
      const r2 = await get(port, '/api/ml/experiments');
      assert.strictEqual(r2.status, 200);
      const mine = (r2.body.experiments || []).find((e) => e.experiment_id === started.experiment_id);
      assert(mine, 'experiment listed');
      assert(typeof mine.metrics.test.f1_weighted === 'number',
        'list response carries f1_weighted too');
      ok('list API carries the same flat F1 fields');
    }
  } finally {
    server.close();
    try { ml.deleteExperiment(started.experiment_id); } catch (_) {}
    try { fs.rmSync(path.join(__dirname, '..', 'ml_workspaces', ing.workspace_id), { recursive: true, force: true }); } catch (_) {}
  }

  console.log(`\nAll F1 API serialization tests passed (${passed} checks).`);
}

main().catch((e) => { console.error(e); process.exit(1); });
