'use strict';
// ZeroGPU remote training tests — mock Space implementing the exact v2 contract:
//   POST {api}/call/v2/train        -> {"event_id": ...}
//   GET  {api}/call/train/<event>   -> SSE generating frames + complete frame
//   POST {api}/call/v2/cancel_train -> {"event_id": ...}
//   GET  {api}/call/cancel_train/<event> -> complete frame with {"ok":true}
//   GET  <artifact url>              -> artifacts zip download
//
// Proves: zerogpu provider NEVER spawns local Python; real metric/progress
// events drive the job; artifacts land on disk inference-ready; stop cancels
// remotely; invalid configs are 400s (never 401s); estimate math is exact.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const childProcess = require('child_process');

const tb = require('../training_backend');

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

// ── Mock Space ────────────────────────────────────────────────────────────
function sseFrame(event, data) {
  return `event: ${event}\ndata: ${data}\n\n`;
}

function startMockSpace({ mode = 'success', zipPath = null } = {}) {
  const hits = { trainPost: 0, cancelPost: 0, artifactGet: 0, bodies: [] };
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://mock');
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      if (req.method === 'POST' && url.pathname === '/gradio_api/call/v2/train') {
        hits.trainPost++;
        try { hits.bodies.push(JSON.parse(body)); } catch { hits.bodies.push(body); }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ event_id: 'mock-ev-1' }));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/gradio_api/call/train/mock-ev-1') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
        if (mode === 'hang') return; // never completes; client must time out / be stopped
        const ev = (obj) => sseFrame('generating', JSON.stringify([JSON.stringify(obj), null]));
        res.write(ev({ type: 'status', status: 'loading', line: '[TRAIN] loading model=a/b on ZeroGPU worker job=train_abc' }));
        res.write(ev({ type: 'log', line: '[TRAIN] dataset loaded train=4500' }));
        res.write(ev({ type: 'progress', progress: { total_steps: 1126 } }));
        res.write(ev({ type: 'metric', metric: { step: 563, epoch: 1.0, train_loss: 0.42, learning_rate: 0.00002 } }));
        res.write(ev({ type: 'metric', metric: { step: 1126, epoch: 2.0, train_loss: 0.21, eval_loss: 0.25, learning_rate: 0.00002 } }));
        res.write(ev({ type: 'log', line: '[TRAIN] finished' }));
        const manifest = JSON.stringify({ job_id: 'JOBID', status: 'finished', metrics_count: 2, files: ['config.json'] });
        const file = { path: '/tmp/mock-artifacts.zip', url: '/gradio_api/file=/tmp/mock-artifacts.zip', orig_name: 'artifacts.zip' };
        res.end(sseFrame('complete', JSON.stringify([manifest, file])));
        return;
      }
      if (req.method === 'POST' && url.pathname === '/gradio_api/call/v2/cancel_train') {
        hits.cancelPost++;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ event_id: 'mock-cancel-1' }));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/gradio_api/call/cancel_train/mock-cancel-1') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        res.end(sseFrame('complete', JSON.stringify([JSON.stringify({ ok: true })])));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/gradio_api/file=/tmp/mock-artifacts.zip') {
        hits.artifactGet++;
        const data = fs.readFileSync(zipPath);
        res.writeHead(200, { 'Content-Type': 'application/zip', 'Content-Length': data.length });
        res.end(data);
        return;
      }
      res.writeHead(404);
      res.end('not found');
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      resolve({ server, hits, port: server.address().port });
    });
  });
}

function buildArtifactZip(kind = 'full') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'claro-art-'));
  const files = kind === 'lora'
    ? { 'adapter_config.json': '{}', 'adapter_model.safetensors': 'fake-weights', 'tokenizer.json': '{}', 'metrics.json': '[]' }
    : { 'config.json': '{}', 'model.safetensors': 'fake-weights', 'tokenizer.json': '{}', 'metrics.json': '[]' };
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(dir, name), content);
  }
  const zipPath = path.join(dir, 'artifacts.zip');
  childProcess.spawnSync('python3', ['-c',
    `import zipfile,sys; z=zipfile.ZipFile(sys.argv[1],'w',zipfile.ZIP_DEFLATED); ` +
    `import os; [z.write(os.path.join(sys.argv[2],f),f) for f in os.listdir(sys.argv[2]) if f!='artifacts.zip']; z.close()`,
    zipPath, dir], { timeout: 15000 });
  assert(fs.existsSync(zipPath), 'test zip must exist');
  return { dir, zipPath };
}

function waitFor(jobId, want, timeoutMs = 15000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const t = setInterval(() => {
      const job = tb.getJob(jobId);
      if (job && want.includes(job.status)) { clearInterval(t); resolve(job); }
      else if (Date.now() - start > timeoutMs) { clearInterval(t); reject(new Error(`timeout waiting for ${want} (now=${job && job.status})`)); }
    }, 50);
  });
}

function validConfig() {
  return tb.validateTrainingRequest({
    model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification',
    epochs: 2, batch_size: 8, learning_rate: 2e-5, validation_split: 10,
  });
}

function cleanupJobDirs() {
  for (const j of tb.jobs.values()) {
    try { fs.rmSync(path.join(__dirname, '..', 'training_outputs', j.job_id), { recursive: true, force: true }); } catch (_) {}
  }
}

async function run() {
  // ── 1. zerogpu success path: no spawn, real events, artifacts on disk ──
  {
    const { dir, zipPath } = buildArtifactZip('full');
    const mock = await startMockSpace({ zipPath });
    process.env.TRAINING_PROVIDER = 'zerogpu';
    process.env.ZEROGPU_TRAIN_API = `http://127.0.0.1:${mock.port}/gradio_api`;
    delete process.env.ZEROGPU_API_TOKEN;
    stubSpawn();
    tb.clearAllJobs();
    try {
      const job = tb.createJob(validConfig(), 'local-user');
      tb.startJob(job);
      assert.strictEqual(job.config.provider, 'zerogpu');
      assert.strictEqual(job.progress.gpu_status, 'zerogpu');
      const done = await waitFor(job.job_id, ['finished', 'failed']);
      assert.strictEqual(done.status, 'finished', `job failed: ${done.logs.slice(-5).join(' | ')}`);
      assert.strictEqual(spawnCalls.length, 0, 'zerogpu must NEVER spawn local Python');
      assert.strictEqual(done._pythonProc, null);
      assert.ok(done._remote && done._remote.eventId === 'mock-ev-1');
      // remote events applied
      assert.strictEqual(done.progress.current_step, 1126);
      assert.strictEqual(done.progress.total_steps, 1126);
      assert.strictEqual(done.metrics.length, 2);
      assert.strictEqual(done.metrics[1].eval_loss, 0.25);
      // structured request wrapped under the single declared v2 input, no code
      assert.strictEqual(mock.hits.trainPost, 1);
      const sent = mock.hits.bodies[0];
      assert.ok(sent && typeof sent.train_request_json === 'object', 'config must be wrapped as {train_request_json} (flat objects are rejected by the v2 gateway)');
      assert.strictEqual(sent.train_request_json.model_id, 'a/b');
      assert.strictEqual(sent.train_request_json.job_id, job.job_id);
      assert.ok(!('code' in sent) && !('code' in sent.train_request_json), 'must not send arbitrary code');
      // artifacts downloaded + extracted, inference-ready
      assert.strictEqual(mock.hits.artifactGet, 1);
      const meta = tb.getArtifactMetadata(job.job_id);
      assert.strictEqual(meta.ready, true);
      assert.strictEqual(meta.training_method, 'full');
      const names = fs.readdirSync(meta.artifact_dir);
      for (const f of ['config.json', 'model.safetensors', 'tokenizer.json', 'metrics.json', 'job.json', 'training_logs.txt']) {
        assert.ok(names.includes(f), `missing artifact ${f}: ${names}`);
      }
      console.log('✓ zerogpu success: no spawn, real events, inference-ready artifacts');
    } finally {
      restoreSpawn();
      mock.server.close();
      fs.rmSync(dir, { recursive: true, force: true });
      cleanupJobDirs();
      tb.clearAllJobs();
    }
  }

  // ── 2. stop cancels remotely, no orphaned local process ──
  {
    const mock = await startMockSpace({ mode: 'hang' });
    process.env.TRAINING_PROVIDER = 'zerogpu';
    process.env.ZEROGPU_TRAIN_API = `http://127.0.0.1:${mock.port}/gradio_api`;
    process.env.ZEROGPU_TRAIN_TIMEOUT_MS = '60000';
    stubSpawn();
    tb.clearAllJobs();
    try {
      const job = tb.createJob(validConfig(), 'local-user');
      tb.startJob(job);
      await new Promise((r) => setTimeout(r, 800)); // let the stream attach
      const stopped = tb.stopJob(job.job_id, { caller: 'test', userInitiated: true });
      assert.strictEqual(stopped.status, 'failed');
      assert.strictEqual(stopped.error, 'stopped_by_user');
      assert.strictEqual(spawnCalls.length, 0);
      await new Promise((r) => setTimeout(r, 800)); // let cancel round-trip land
      assert.strictEqual(mock.hits.cancelPost, 1, 'remote cancel_train must be called');
      assert.ok(stopped.logs.join('\n').includes('remote cancel'), 'cancel outcome must be logged honestly');
      console.log('✓ zerogpu stop: remote cancel requested, honestly logged, no local proc');
    } finally {
      delete process.env.ZEROGPU_TRAIN_TIMEOUT_MS;
      restoreSpawn();
      mock.server.close();
      tb.clearAllJobs();
    }
  }

  // ── 3. invalid configs are 400s, never 401s ──
  {
    const bad = [
      [{ model_id: '!!!', dataset_id: 'c/d', task_type: 'text-classification' }, 'invalid_model_id'],
      [{ model_id: 'a/b', dataset_id: 'c/d', task_type: 'nope' }, 'bad_request'],
      [{ model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification', epochs: 99 }, 'bad_request'],
      [{ model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification', training_method: 'lora', lora_r: 999 }, 'bad_request'],
      [null, 'bad_request'],
    ];
    for (const [body, code] of bad) {
      let err = null;
      try { tb.validateTrainingRequest(body); } catch (e) { err = e; }
      assert(err, `expected throw for ${JSON.stringify(body)}`);
      assert.strictEqual(err.status, 400, `expected 400, got ${err.status}`);
      assert.strictEqual(err.code, code);
      assert.notStrictEqual(err.status, 401);
    }
    console.log('✓ invalid training configs return 400 validation errors (never 401)');
  }

  // ── 4. estimate math ──
  {
    const fakeFetch = async (url) => (String(url).includes('/splits?')
      ? { ok: true, json: async () => ({ splits: [{ config: 'plain_text', split: 'train' }] }) }
      : { ok: true, json: async () => ({ num_rows_total: 25000 }) });
    // max_samples cap 5000, 10% val -> 4500 train rows; ceil(4500/8)=563 *2 epochs = 1126
    const est = await tb.estimateTrainingSteps({ dataset_id: 'c/d', epochs: 2, batch_size: 8, validation_split: 10 }, fakeFetch);
    assert.strictEqual(est.estimated_total_steps, 1126, JSON.stringify(est));
    assert.strictEqual(est.train_rows, 4500);
    assert.strictEqual(est.total_rows, 25000);
    assert.strictEqual(est.basis, 'dataset-rows');
    const ms = await tb.estimateTrainingSteps({ dataset_id: 'c/d', epochs: 2, batch_size: 8, max_steps: 50 }, fakeFetch);
    assert.strictEqual(ms.estimated_total_steps, 50);
    assert.strictEqual(ms.basis, 'max_steps');
    const unknown = await tb.estimateTrainingSteps({ dataset_id: 'zzz/no-such-dataset', epochs: 2, batch_size: 8 },
      async () => ({ ok: false }));
    assert.strictEqual(unknown.estimated_total_steps, null);
    const ui = require('../training_ui');
    assert.strictEqual(ui.estimateStepsFromSamples({ trainRows: 4500, batchSize: 8, epochs: 2 }), 1126);
    assert.strictEqual(ui.estimateStepsFromSamples({ trainRows: 0, batchSize: 8, epochs: 2 }), null);
    assert.strictEqual(ui.estimateStepsFromSamples({ trainRows: 100, batchSize: 8, epochs: 1, maxSteps: 7 }), 7);
    console.log('✓ estimate math exact; unknown stays null (never invented)');
  }

  // ── 5. local path still spawns training_runner.py locally ──
  {
    process.env.TRAINING_PROVIDER = 'local';
    delete process.env.ZEROGPU_TRAIN_API;
    spawnCalls = [];
    childProcess.spawn = (...a) => {
      spawnCalls.push(a);
      const { EventEmitter } = require('events');
      const proc = new EventEmitter();
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      proc.kill = () => true;
      setImmediate(() => proc.emit('close', 1, null));
      return proc;
    };
    childProcess.spawnSync = () => ({ status: 1, stdout: '', stderr: '' });
    tb.clearAllJobs();
    try {
      const job = tb.createJob(validConfig(), 'local-user');
      tb.startJob(job);
      assert.strictEqual(job.config.provider, 'local');
      await new Promise((r) => setTimeout(r, 300));
      assert.ok(spawnCalls.length > 0, 'local provider must spawn local Python');
      assert.ok(String(spawnCalls[0][1][0]).endsWith('training_runner.py'), `must run training_runner.py, got ${spawnCalls[0][1][0]}`);
      assert.ok(!job._remote, 'local jobs must not have remote state');
      console.log('✓ local provider still runs training_runner.py locally');
    } finally {
      restoreSpawn();
      cleanupJobDirs();
      tb.clearAllJobs();
    }
  }

  // ── 6. modal fails loudly instead of silently running locally ──
  {
    process.env.TRAINING_PROVIDER = 'modal';
    stubSpawn();
    tb.clearAllJobs();
    try {
      const job = tb.createJob(validConfig(), 'local-user');
      tb.startJob(job);
      assert.strictEqual(job.status, 'failed');
      assert.strictEqual(spawnCalls.length, 0, 'modal must not silently run locally');
      assert.ok(job.logs.join('\n').includes('Modal training is not implemented'));
      console.log('✓ modal provider fails loudly (no silent local run)');
    } finally {
      restoreSpawn();
      tb.clearAllJobs();
    }
  }

  // ── 7. zip traversal guard ──
  {
    const { dir, zipPath } = buildArtifactZip('full');
    const evil = path.join(dir, 'evil.zip');
    childProcess.spawnSync('python3', ['-c',
      `import zipfile,sys; z=zipfile.ZipFile(sys.argv[1],'w'); z.writestr('../../evil.txt','x'); z.writestr('ok.txt','y'); z.close()`, evil], { timeout: 15000 });
    const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'claro-zip-'));
    const names = tb._extractZipBuffer(fs.readFileSync(evil), dest);
    assert.deepStrictEqual(names, ['ok.txt']);
    assert.ok(!fs.existsSync(path.join(os.tmpdir(), 'evil.txt')));
    console.log('✓ zip extraction rejects traversal paths');
    fs.rmSync(dir, { recursive: true, force: true });
    fs.rmSync(dest, { recursive: true, force: true });
  }

  // ── 8. scheduler denials classify as zerogpu_quota (never model error) ──
  {
    const cls = tb._classifyZeroGpuErrorFrame;
    assert.strictEqual(typeof cls, 'function');
    let e = cls('AcceleratorError: You have exceeded your ZeroGPU runs limit');
    assert.strictEqual(e.code, 'zerogpu_quota', e.code);
    assert.ok(/quota|retry later/i.test(e.message), e.message);
    e = cls(JSON.stringify({ error: 'quota exceeded for this Space', title: 'Something else' }));
    assert.strictEqual(e.code, 'zerogpu_quota', e.code);
    e = cls('CUDA out of memory. Tried to allocate 1GB');
    assert.strictEqual(e.code, 'gpu_oom', e.code);
    e = cls('plain worker boom');
    assert.strictEqual(e.code, 'zerogpu_runtime', e.code);
    console.log('✓ scheduler denials classify as zerogpu_quota with actionable text');
  }

  process.env = ORIG_ENV;
  console.log('\nAll ZeroGPU training tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
