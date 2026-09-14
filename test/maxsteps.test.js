'use strict';
// Max Steps control tests:
//  1. missing -> Auto (null)
//  2. max_steps=5 -> exactly 5, total_steps=5
//  3. max_steps=0 -> 400
//  4. non-integer -> 400
//  5. > TRAIN_MAX_STEPS -> 400
//  6. local provider passes --max_steps to training_runner.py
//  8. preview displays Auto / value, estimate follows explicit value
//  9. progress total_steps equals max_steps when explicitly set
// (7. ZeroGPU passthrough is covered in zerogpu_training.test.js.)
const assert = require('assert');
const childProcess = require('child_process');
const tb = require('../training_backend');
const ui = require('../training_ui');

const ORIG_SPAWN = childProcess.spawn;
const ORIG_SPAWNSYNC = childProcess.spawnSync;

function base(over = {}) {
  return Object.assign({
    model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification',
    epochs: 2, batch_size: 8, learning_rate: 2e-5, validation_split: 10,
  }, over);
}

async function run() {
  tb.clearAllJobs();

  // 1. missing / blank / null -> Auto (null), never 0 or NaN
  for (const v of [undefined, null, '', '   ']) {
    const cfg = tb.validateTrainingRequest(base({ max_steps: v }));
    assert.strictEqual(cfg.max_steps, null, `max_steps=${JSON.stringify(v)} should normalize to null`);
  }
  const cfgNoKey = tb.validateTrainingRequest(base());
  assert.ok(!('max_steps' in base()), 'precondition: key truly absent');
  assert.strictEqual(cfgNoKey.max_steps, null);
  console.log('✓ 1. max_steps missing/blank -> Auto (null)');

  // 2. explicit value preserved exactly
  const cfg5 = tb.validateTrainingRequest(base({ max_steps: 5 }));
  assert.strictEqual(cfg5.max_steps, 5);
  const cfgStr = tb.validateTrainingRequest(base({ max_steps: '20' }));
  assert.strictEqual(cfgStr.max_steps, 20, 'numeric strings convert');
  console.log('✓ 2. max_steps=5 -> exactly 5 (total_steps follows below)');

  // 3. zero -> 400
  for (const v of [0, '0', -3]) {
    let err = null;
    try { tb.validateTrainingRequest(base({ max_steps: v })); } catch (e) { err = e; }
    assert(err && err.status === 400, `max_steps=${v} must be 400`);
    assert.strictEqual(err.code, 'bad_request');
  }
  console.log('✓ 3. max_steps=0 (and negatives) -> 400');

  // 4. non-integer -> 400
  for (const v of [2.5, 'abc', '5steps', NaN]) {
    let err = null;
    try { tb.validateTrainingRequest(base({ max_steps: v })); } catch (e) { err = e; }
    assert(err && err.status === 400, `max_steps=${JSON.stringify(v)} must be 400`);
  }
  console.log('✓ 4. non-integer max_steps -> 400');

  // 5. above TRAIN_MAX_STEPS -> 400 (do not weaken the server limit)
  const limit = tb.MAX_STEPS_LIMIT;
  assert.ok(Number.isFinite(limit) && limit > 0, 'limit exported');
  let err = null;
  try { tb.validateTrainingRequest(base({ max_steps: limit + 1 })); } catch (e) { err = e; }
  assert(err && err.status === 400, `max_steps=${limit + 1} must be 400`);
  const cfgLim = tb.validateTrainingRequest(base({ max_steps: limit }));
  assert.strictEqual(cfgLim.max_steps, limit, 'boundary value allowed');
  console.log(`✓ 5. max_steps > TRAIN_MAX_STEPS (${limit}) -> 400, boundary ok`);

  // 6 + 9. local provider: --max_steps reaches training_runner.py, total_steps == max_steps
  const savedProv = process.env.TRAINING_PROVIDER;
  process.env.TRAINING_PROVIDER = 'local';
  let spawnArgs = null;
  childProcess.spawnSync = () => ({ status: 1, stdout: '', stderr: '' });
  childProcess.spawn = (...a) => {
    spawnArgs = a;
    const { EventEmitter } = require('events');
    const proc = new EventEmitter();
    proc.stdout = new EventEmitter();
    proc.stderr = new EventEmitter();
    proc.kill = () => true;
    setImmediate(() => proc.emit('close', 1, null));
    return proc;
  };
  try {
    const job = tb.createJob(tb.validateTrainingRequest(base({ max_steps: 5 })), 'local-user');
    assert.strictEqual(job.progress.total_steps, 5, 'createJob total_steps must equal explicit max_steps');
    tb.startJob(job);
    await new Promise((r) => setTimeout(r, 400));
    assert.ok(spawnArgs, 'local provider must spawn');
    const args = spawnArgs[1];
    assert.ok(String(args[0]).endsWith('training_runner.py'), 'must run training_runner.py');
    const i = args.indexOf('--max_steps');
    assert.ok(i !== -1 && args[i + 1] === '5', `args must contain --max_steps 5, got: ${args.join(' ')}`);
    assert.strictEqual(job.progress.total_steps, 5, 'progress total_steps stays 5 (Step x / 5)');
    assert.ok(!job._remote, 'local jobs must not have remote state');
    console.log('✓ 6+9. local passes --max_steps 5 to training_runner.py; total_steps=5');
  } finally {
    childProcess.spawn = ORIG_SPAWN;
    childProcess.spawnSync = ORIG_SPAWNSYNC;
    if (savedProv === undefined) delete process.env.TRAINING_PROVIDER;
    else process.env.TRAINING_PROVIDER = savedProv;
    for (const j of tb.jobs.values()) {
      try { require('fs').rmSync(require('path').join(__dirname, '..', 'training_outputs', j.job_id), { recursive: true, force: true }); } catch (_) {}
    }
    tb.clearAllJobs();
  }

  // 8. preview display
  const tBase = { model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification', training_method: 'full', epochs: 2, batch_size: 8, learning_rate: 2e-5, validation_split: 10, lora_r: 8, lora_alpha: 16, lora_dropout: 0.05, target_modules: 'auto' };
  let p = ui.getPreviewData({ ...tBase, max_steps: 20 });
  assert.strictEqual(p.maxSteps, '20');
  assert.strictEqual(p.estimatedSteps, '20', 'explicit value is the primary step count');
  assert.strictEqual(p.estimatedTime, 'Estimate unavailable', 'no fake time estimate');
  p = ui.getPreviewData({ ...tBase, max_steps: '' });
  assert.strictEqual(p.maxSteps, 'Auto');
  p = ui.getPreviewData({ ...tBase });
  assert.strictEqual(p.maxSteps, 'Auto', 'legacy cells without the field show Auto');
  console.log('✓ 8. preview shows Max steps Auto/20; estimate follows explicit value; no fake time');

  console.log('\nAll Max Steps tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
