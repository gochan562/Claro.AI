'use strict';
// Durable artifact rehydration tests: the local _safeArtifactDir(job_id)
// directory is a CACHE; manifest.hub_path is the durable reference.
//
// Simulates (with a mock Space + mock Hub, no network):
//   1. training completes, local artifact exists, inference-ready;
//   2. local artifact directory is deleted;
//   3. job still has hub_path → inference-ready check rehydrates from Hub;
//   4. repeat check uses the local cache (no second Hub download);
//   5. no hub_path + missing dir → clear artifact-unavailable error;
//   6. Hub failure → clear artifact-unavailable error (job NOT marked failed).
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const childProcess = require('child_process');

const tb = require('../training_backend');

const ORIG_ENV = { ...process.env };
const ORIG_SPAWN = childProcess.spawn;
const REAL_FETCH = global.fetch;
let spawnCalls = [];

const HUB_REPO = 'test-owner/test-artifacts';
const state = { hubPath: 'unset', hubMode: 'ok', hubCalls: 0 };

function sseFrame(event, data) {
  return `event: ${event}\ndata: ${data}\n\n`;
}

function buildZip() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'claro-rehyd-'));
  for (const [n, c] of Object.entries({
    'config.json': '{}',
    'model.safetensors': 'fake-weights',
    'tokenizer.json': '{}',
  })) fs.writeFileSync(path.join(dir, n), c);
  const zp = path.join(dir, 'a.zip');
  childProcess.spawnSync('python3', ['-c',
    `import zipfile,sys,os; z=zipfile.ZipFile(sys.argv[1],'w',zipfile.ZIP_DEFLATED); ` +
    `[z.write(os.path.join(sys.argv[2],f),f) for f in sorted(os.listdir(sys.argv[2])) if f!='a.zip']; z.close()`,
    zp, dir], { timeout: 15000 });
  assert(fs.existsSync(zp), 'test zip must exist');
  return { dir, data: fs.readFileSync(zp) };
}

function startMockSpace(zipData) {
  let lastJobId = 'train_aaaaaaaaaaaaaaaa';
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://mock');
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      if (req.method === 'POST' && url.pathname === '/gradio_api/call/v2/train') {
        try { lastJobId = JSON.parse(body).train_request_json.job_id || lastJobId; } catch (_) {}
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ event_id: 'rehyd-ev-1' }));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/gradio_api/call/train/rehyd-ev-1') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        const ev = (o) => sseFrame('generating', JSON.stringify([JSON.stringify(o), null]));
        res.write(ev({ type: 'log', line: '[TRAIN] loading model=a/b' }));
        res.write(ev({ type: 'log', line: '[TRAIN] finished artifacts=3' }));
        const manifest = JSON.stringify({
          job_id: lastJobId, status: 'finished',
          metrics_count: 0, files: ['config.json', 'model.safetensors', 'tokenizer.json'],
          hub_path: state.hubPath,
        });
        const file = { path: '/tmp/artifacts.zip', url: '/gradio_api/file=/tmp/artifacts.zip', orig_name: 'artifacts.zip' };
        res.end(sseFrame('complete', JSON.stringify([manifest, file])));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/gradio_api/file=/tmp/artifacts.zip') {
        res.writeHead(200, { 'Content-Type': 'application/zip', 'Content-Length': zipData.length });
        res.end(zipData);
        return;
      }
      res.writeHead(404);
      res.end('not found');
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

function installHubMock(hubZipData) {
  global.fetch = async (url, init) => {
    if (String(url).startsWith('https://huggingface.co/datasets/')) {
      state.hubCalls++;
      if (state.hubMode === 'missing') {
        return new Response('not found', { status: 404, headers: { 'Content-Type': 'text/plain' } });
      }
      return new Response(hubZipData, { status: 200, headers: { 'Content-Type': 'application/zip' } });
    }
    return REAL_FETCH(url, init);
  };
}

function waitFor(jobId, want, timeoutMs = 15000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const t = setInterval(() => {
      const job = tb.getJob(jobId);
      if (job && want.includes(job.status)) { clearInterval(t); resolve(job); }
      else if (Date.now() - start > timeoutMs) { clearInterval(t); reject(new Error(`timeout (now=${job && job.status})`)); }
    }, 50);
  });
}

function validConfig() {
  return tb.validateTrainingRequest({
    model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification',
    epochs: 1, batch_size: 8, learning_rate: 2e-5, validation_split: 10,
  });
}

function rmJobDir(jobId) {
  try { fs.rmSync(path.join(__dirname, '..', 'training_outputs', jobId), { recursive: true, force: true }); } catch (_) {}
}

async function runJobToFinished() {
  const job = tb.createJob(validConfig(), 'local-user');
  state.hubPath = `${HUB_REPO}:${job.job_id}/artifacts.zip`;
  tb.startJob(job);
  const done = await waitFor(job.job_id, ['finished', 'failed']);
  assert.strictEqual(done.status, 'finished', `setup job failed: ${done.logs.slice(-3).join(' | ')}`);
  const meta = tb.getArtifactMetadata(job.job_id);
  assert.strictEqual(meta.ready, true, 'setup: must start inference-ready');
  assert.strictEqual(done.hub_path, state.hubPath, 'setup: job must retain hub_path');
  return done;
}

async function run() {
  const { dir: tmpDir, data: zipData } = buildZip();
  const mock = await startMockSpace(zipData);
  process.env.TRAINING_PROVIDER = 'zerogpu';
  process.env.ZEROGPU_TRAIN_API = `http://127.0.0.1:${mock.port}/gradio_api`;
  delete process.env.ZEROGPU_API_TOKEN;
  delete process.env.HF_TOKEN;
  delete process.env.HUGGING_FACE_HUB_TOKEN;
  childProcess.spawn = (...a) => { spawnCalls.push(a); throw new Error('local spawn forbidden'); };
  installHubMock(zipData);
  tb.clearAllJobs();
  try {
    // ── 1-4. Delete cache → async check rehydrates once, then cache-hit. ──
    {
      const job = await runJobToFinished();
      const expected = tb._safeArtifactDir(job.job_id);
      rmJobDir(job.job_id);
      assert(!fs.existsSync(expected), 'setup: cache must be gone');

      let syncErr = null;
      try { tb.getArtifactMetadata(job.job_id); } catch (e) { syncErr = e; }
      assert(syncErr && /Artifact directory not found/.test(syncErr.message),
        'sync check keeps its exact legacy behavior');

      state.hubCalls = 0;
      const meta = await tb.getArtifactMetadataAsync(job.job_id);
      assert.strictEqual(meta.ready, true, 'rehydrated check must be inference-ready');
      assert.strictEqual(meta.artifact_dir, expected);
      assert.strictEqual(state.hubCalls, 1, 'exactly one Hub download');
      for (const f of ['config.json', 'model.safetensors', 'tokenizer.json']) {
        assert(fs.readdirSync(expected).includes(f), `rehydrated file missing: ${f}`);
      }
      assert.strictEqual(tb.getJob(job.job_id).status, 'finished', 'job must NOT be marked failed');
      assert.strictEqual(tb.getJob(job.job_id).artifacts_dir, expected);

      const meta2 = await tb.getArtifactMetadataAsync(job.job_id);
      assert.strictEqual(meta2.ready, true);
      assert.strictEqual(state.hubCalls, 1, 'second check must use the local cache (no re-download)');
      console.log('✓ rehydrate-on-missing-cache, single download, cache-hit after');
      rmJobDir(job.job_id);
    }

    // ── 5. No hub_path + missing dir → clear artifact-unavailable error. ──
    {
      state.hubPath = null;
      const job = tb.createJob(validConfig(), 'local-user');
      tb.startJob(job);
      const done = await waitFor(job.job_id, ['finished', 'failed']);
      assert.strictEqual(done.status, 'finished');
      assert.strictEqual(done.hub_path, null);
      rmJobDir(job.job_id);
      let err = null;
      try { await tb.getArtifactMetadataAsync(job.job_id); } catch (e) { err = e; }
      assert(err, 'must throw');
      assert.strictEqual(err.code, 'missing_artifact');
      assert(/no Hub reference \(hub_path\)/.test(err.message), `unclear error: ${err.message}`);
      assert.strictEqual(tb.getJob(job.job_id).status, 'finished', 'job must NOT be marked failed');
      console.log('✓ no-hub-path gives clear artifact-unavailable error');
      rmJobDir(job.job_id);
    }

    // ── 6. Hub failure → clear error, job still finished. ──
    {
      const job = tb.createJob(validConfig(), 'local-user');
      state.hubPath = `${HUB_REPO}:${job.job_id}/artifacts.zip`;
      tb.startJob(job);
      const done = await waitFor(job.job_id, ['finished', 'failed']);
      assert.strictEqual(done.status, 'finished');
      rmJobDir(job.job_id);
      state.hubMode = 'missing';
      let err = null;
      try { await tb.getArtifactMetadataAsync(job.job_id); } catch (e) { err = e; }
      finally { state.hubMode = 'ok'; }
      assert(err, 'must throw');
      assert(/Hub rehydration failed|Artifact unavailable/.test(err.message), `unclear error: ${err.message}`);
      assert.strictEqual(tb.getJob(job.job_id).status, 'finished', 'job must NOT be marked failed');
      console.log('✓ hub-failure gives clear error without failing the job');
      rmJobDir(job.job_id);
    }
  } finally {
    childProcess.spawn = ORIG_SPAWN;
    global.fetch = REAL_FETCH;
    mock.server.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
    for (const j of [...tb.jobs.values()]) rmJobDir(j.job_id);
    tb.clearAllJobs();
    process.env = ORIG_ENV;
  }
  console.log('\nAll artifact rehydration tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
