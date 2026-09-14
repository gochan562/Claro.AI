'use strict';
// Curated Model + Dataset catalog tests.
//  1. valid preset combinations resolve server-side
//  2. incompatible model/dataset combinations rejected
//  3. unknown preset IDs rejected
//  4. manually supplied (smuggled) oversized models ignored when preset present
//  5. attempts to exceed preset limits rejected
//  6. old notebook migration (known values, aliases, unknown, task mismatch)
//  7. correct ZeroGPU request resolution (resolved IDs hit the wire)
const assert = require('assert');
const http = require('http');
const fs = require('fs');
const path = require('path');
const childProcess = require('child_process');

const tb = require('../training_backend');
const ui = require('../training_ui');

const ORIG_ENV = { ...process.env };
const ORIG_SPAWN = childProcess.spawn;
const ORIG_SPAWNSYNC = childProcess.spawnSync;
let spawnCalls = [];
function stubSpawn() {
  spawnCalls = [];
  childProcess.spawn = (...a) => { spawnCalls.push(a); throw new Error('spawn must not be called'); };
  childProcess.spawnSync = (...a) => { spawnCalls.push(a); return { status: 1, stdout: '', stderr: '' }; };
}
function restoreSpawn() {
  childProcess.spawn = ORIG_SPAWN;
  childProcess.spawnSync = ORIG_SPAWNSYNC;
}
function cleanupJobs() {
  for (const j of tb.jobs.values()) {
    try { fs.rmSync(path.join(__dirname, '..', 'training_outputs', j.job_id), { recursive: true, force: true }); } catch (_) {}
  }
  tb.clearAllJobs();
}
function expect400(fn, snippet) {
  let err = null;
  try { fn(); } catch (e) { err = e; }
  assert(err, 'expected a validation error');
  assert.strictEqual(err.status, 400, `expected 400, got ${err.status}: ${err.message}`);
  if (snippet) assert.ok(err.message.includes(snippet), `error should mention '${snippet}', got: ${err.message}`);
  return err;
}

async function run() {
  // ── 1. valid preset combinations resolve to catalog IDs ──
  {
    const cases = [
      [{ model_preset: 'distilbert-base', dataset_preset: 'imdb', task_type: 'text-classification', epochs: 2 },
        'distilbert-base-uncased', 'stanfordnlp/imdb', 5000],
      [{ model_preset: 'distilbert-base', dataset_preset: 'ag-news', task_type: 'text-classification', epochs: 3 },
        'distilbert-base-uncased', 'fancyzhx/ag_news', 5000],
      [{ model_preset: 'mobilenet-v2', dataset_preset: 'beans', task_type: 'image-classification', epochs: 2 },
        'google/mobilenet_v2_1.0_224', 'beans', 2000],
      [{ model_preset: 'resnet-18', dataset_preset: 'beans', task_type: 'image-classification', epochs: 5 },
        'microsoft/resnet-18', 'beans', 2000],
    ];
    for (const [body, modelId, datasetId, samples] of cases) {
      const cfg = tb.validateTrainingRequest(body);
      assert.strictEqual(cfg.model_id, modelId);
      assert.strictEqual(cfg.dataset_id, datasetId);
      assert.strictEqual(cfg.max_samples, samples);
      assert.ok(cfg.model_preset && cfg.dataset_preset, 'preset keys echoed in config');
    }
    console.log('✓ 1. valid preset combinations resolve server-side (4 combos)');
  }

  // ── 2. incompatible combinations rejected ──
  {
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'distilbert-base', dataset_preset: 'beans', task_type: 'text-classification' }), 'beans');
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'mobilenet-v2', dataset_preset: 'imdb', task_type: 'image-classification' }), 'imdb');
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'distilbert-base', dataset_preset: 'imdb', task_type: 'image-classification' }), 'distilbert-base');
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'mobilenet-v2', dataset_preset: 'beans', task_type: 'image-classification', training_method: 'lora' }), 'lora');
    console.log('✓ 2. incompatible model/dataset/task/method combinations rejected (400)');
  }

  // ── 3. unknown preset IDs rejected ──
  {
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'gpt-99', dataset_preset: 'imdb', task_type: 'text-classification' }), 'gpt-99');
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'distilbert-base', dataset_preset: 'cifar10', task_type: 'text-classification' }), 'cifar10');
    console.log('✓ 3. unknown preset IDs rejected (400)');
  }

  // ── 4. smuggled oversized models ignored when a preset is present ──
  {
    const cfg = tb.validateTrainingRequest({
      model_preset: 'distilbert-base', dataset_preset: 'imdb',
      model_id: 'meta-llama/Llama-2-70b-hf', dataset_id: 'evil/backdoor',
      task_type: 'text-classification', epochs: 2,
    });
    assert.strictEqual(cfg.model_id, 'distilbert-base-uncased', 'frontend model_id must not override the preset');
    assert.strictEqual(cfg.dataset_id, 'stanfordnlp/imdb', 'frontend dataset_id must not override the preset');
    // legacy raw path (no presets) keeps working exactly as before
    const leg = tb.validateTrainingRequest({ model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification' });
    assert.strictEqual(leg.model_id, 'a/b');
    assert.strictEqual(leg.model_preset, null);
    assert.strictEqual(leg.dataset_preset, null);
    console.log('✓ 4. smuggled model/dataset IDs ignored; legacy raw path unchanged');
  }

  // ── 5. preset ceilings enforced on top of global limits ──
  {
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'distilbert-base', dataset_preset: 'imdb', task_type: 'text-classification', epochs: 5 }), '1..3');
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'distilbert-base', dataset_preset: 'imdb', task_type: 'text-classification', batch_size: 32 }), '1..16');
    expect400(() => tb.validateTrainingRequest(
      { model_preset: 'mobilenet-v2', dataset_preset: 'beans', task_type: 'image-classification', max_steps: 5000 }), '1..2000');
    // at-ceiling values pass
    const ok = tb.validateTrainingRequest(
      { model_preset: 'distilbert-base', dataset_preset: 'imdb', task_type: 'text-classification', epochs: 3, batch_size: 16, max_steps: 2000 });
    assert.strictEqual(ok.epochs, 3);
    // global limits still apply to raw path
    expect400(() => tb.validateTrainingRequest({ model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification', epochs: 99 }), '1..5');
    console.log('✓ 5. preset ceilings enforced (epochs/batch/steps); globals intact for raw path');
  }

  // ── 6. old notebook migration ──
  {
    const t1 = { model_id: 'distilbert-base-uncased', dataset_id: 'stanfordnlp/imdb', task_type: 'text-classification' };
    ui.migrateLegacyTrainingToPresets(t1);
    assert.strictEqual(t1.model_preset, 'distilbert-base');
    assert.strictEqual(t1.dataset_preset, 'imdb');
    const t2b = { model_id: 'distilbert-base-uncased', dataset_id: 'ag_news', task_type: 'text-classification' };
    ui.migrateLegacyTrainingToPresets(t2b);
    assert.strictEqual(t2b.model_preset, 'distilbert-base');
    assert.strictEqual(t2b.dataset_preset, 'ag-news');
    assert.strictEqual(t2b.dataset_id, 'fancyzhx/ag_news', 'canonical dataset ID synced');
    const t3 = { model_id: 'microsoft/resnet-18', dataset_id: 'beans', task_type: 'image-classification' };
    ui.migrateLegacyTrainingToPresets(t3);
    assert.strictEqual(t3.model_preset, 'resnet-18');
    assert.strictEqual(t3.dataset_preset, 'beans');
    const t4 = { model_id: 'my-org/huge-model', dataset_id: 'my-org/private-ds', task_type: 'text-generation' };
    ui.migrateLegacyTrainingToPresets(t4);
    assert.strictEqual(t4.model_preset, 'custom');
    assert.strictEqual(t4.dataset_preset, 'custom');
    assert.strictEqual(t4.model_id, 'my-org/huge-model', 'unknown raw IDs preserved');
    // task mismatch demotes the preset instead of corrupting the task
    const t5 = { model_id: 'distilbert-base-uncased', dataset_id: 'stanfordnlp/imdb', task_type: 'text-generation' };
    ui.migrateLegacyTrainingToPresets(t5);
    assert.strictEqual(t5.model_preset, 'custom');
    assert.strictEqual(t5.task_type, 'text-generation', 'explicit task never rewritten');
    // already-migrated cells untouched
    const t6 = { model_preset: 'mobilenet-v2', dataset_preset: 'beans', model_id: 'google/mobilenet_v2_1.0_224', dataset_id: 'beans', task_type: 'image-classification' };
    ui.migrateLegacyTrainingToPresets(t6);
    assert.strictEqual(t6.model_preset, 'mobilenet-v2');
    assert.strictEqual(t6.dataset_preset, 'beans');
    // preview shows friendly names + advanced IDs
    const p = ui.getPreviewData({ ...t6, training_method: 'full', epochs: 2, batch_size: 8, learning_rate: 2e-5, validation_split: 10, lora_r: 8, lora_alpha: 16, lora_dropout: 0.05, target_modules: 'auto', max_steps: '' });
    assert.strictEqual(p.model, 'MobileNetV2');
    assert.strictEqual(p.dataset, 'Beans (leaf images)');
    assert.ok(p.advancedDetails.includes('google/mobilenet_v2_1.0_224'), p.advancedDetails);
    assert.ok(p.advancedDetails.includes('beans'), p.advancedDetails);
    console.log('✓ 6. migration maps known values, preserves unknown, never rewrites task');
  }

  // ── 7. ZeroGPU request carries resolved catalog IDs ──
  {
    const bodies = [];
    const server = http.createServer((req, res) => {
      let raw = '';
      req.on('data', (c) => { raw += c; });
      req.on('end', () => {
        if (req.method === 'POST' && req.url === '/gradio_api/call/v2/train') {
          try { bodies.push(JSON.parse(raw)); } catch (_) {}
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ event_id: 'mock-ev-1' }));
          return;
        }
        // hang the stream: the test only needs the POST capture
        if (req.method === 'GET') return;
        res.writeHead(404);
        res.end('nope');
      });
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    process.env.TRAINING_PROVIDER = 'zerogpu';
    process.env.ZEROGPU_TRAIN_API = `http://127.0.0.1:${server.address().port}/gradio_api`;
    process.env.ZEROGPU_TRAIN_TIMEOUT_MS = '60000';
    stubSpawn();
    tb.clearAllJobs();
    try {
      // attacker-style body: preset + smuggled raw IDs
      const cfg = tb.validateTrainingRequest({
        model_preset: 'distilbert-base', dataset_preset: 'ag-news',
        model_id: 'meta-llama/Llama-2-70b-hf', dataset_id: 'evil/ds',
        task_type: 'text-classification', epochs: 2,
      });
      const job = tb.createJob(cfg, 'local-user');
      tb.startJob(job);
      const t0 = Date.now();
      while (bodies.length === 0 && Date.now() - t0 < 10000) {
        await new Promise((r) => setTimeout(r, 50));
      }
      assert.strictEqual(bodies.length, 1, 'exactly one train POST');
      const sent = bodies[0].train_request_json;
      assert.strictEqual(sent.model_id, 'distilbert-base-uncased');
      assert.strictEqual(sent.dataset_id, 'fancyzhx/ag_news');
      assert.strictEqual(spawnCalls.length, 0, 'no local spawn');
      console.log('✓ 7. ZeroGPU request carries resolved catalog IDs (smuggled IDs never hit the wire)');
    } finally {
      delete process.env.ZEROGPU_TRAIN_TIMEOUT_MS;
      restoreSpawn();
      server.close();
      cleanupJobs();
    }
  }

  process.env = ORIG_ENV;
  console.log('\nAll catalog tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
