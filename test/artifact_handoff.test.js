'use strict';
// Artifact handoff regression test: remote training complete
// → artifact ZIP downloaded → extracted into the expected LOCAL artifact
// directory → inference-ready check succeeds.
//
// Guards the "Not inference-ready / Artifact directory not found" failure:
// the expected directory is ALWAYS derived via _safeArtifactDir(job_id)
// (host-local), never a hard-coded remote path such as
// /home/runner/workspace/training_outputs/<job_id>.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const childProcess = require('child_process');

const tb = require('../training_backend');

const ORIG_ENV = { ...process.env };
const ORIG_SPAWN = childProcess.spawn;
let spawnCalls = [];

function sseFrame(event, data) {
  return `event: ${event}\ndata: ${data}\n\n`;
}

function buildZip() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'claro-handoff-'));
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
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://mock');
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      if (req.method === 'POST' && url.pathname === '/gradio_api/call/v2/train') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ event_id: 'art-ev-1' }));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/gradio_api/call/train/art-ev-1') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        const ev = (o) => sseFrame('generating', JSON.stringify([JSON.stringify(o), null]));
        res.write(ev({ type: 'log', line: '[TRAIN] loading model=a/b' }));
        res.write(ev({ type: 'log', line: '[TRAIN] finished artifacts=3' }));
        const manifest = JSON.stringify({
          job_id: 'train_aaaaaaaaaaaaaaaa', status: 'finished',
          metrics_count: 0, files: ['config.json', 'model.safetensors', 'tokenizer.json'],
          hub_path: null,
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

async function run() {
  const { dir: tmpDir, data: zipData } = buildZip();
  const mock = await startMockSpace(zipData);
  process.env.TRAINING_PROVIDER = 'zerogpu';
  process.env.ZEROGPU_TRAIN_API = `http://127.0.0.1:${mock.port}/gradio_api`;
  delete process.env.ZEROGPU_API_TOKEN;
  childProcess.spawn = (...a) => { spawnCalls.push(a); throw new Error('local spawn forbidden'); };
  tb.clearAllJobs();
  try {
    const cfg = tb.validateTrainingRequest({
      model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification',
      epochs: 1, batch_size: 8, learning_rate: 2e-5, validation_split: 10,
    });
    const job = tb.createJob(cfg, 'local-user');
    tb.startJob(job);
    const done = await waitFor(job.job_id, ['finished', 'failed']);
    assert.strictEqual(done.status, 'finished', `handoff failed: ${done.logs.slice(-3).join(' | ')}`);
    assert.strictEqual(spawnCalls.length, 0, 'remote path must never spawn');

    // Expected dir is DERIVED host-local — never a hard-coded remote path.
    const expected = tb._safeArtifactDir(job.job_id);
    assert(!expected.includes('/home/runner'),
      `derived dir must not hard-code the Replit host path: ${expected}`);
    assert.strictEqual(done.artifacts_dir, expected, 'job must store the local extracted path (not a Space path)');
    assert(fs.existsSync(expected) && fs.statSync(expected).isDirectory(), 'extracted dir must exist');
    for (const f of ['config.json', 'model.safetensors', 'tokenizer.json']) {
      assert(fs.readdirSync(expected).includes(f), `missing extracted file ${f}`);
    }

    // Inference-ready check succeeds against the extracted directory.
    const meta = tb.getArtifactMetadata(job.job_id);
    assert.strictEqual(meta.ready, true);
    assert.strictEqual(meta.artifact_dir, expected);
    assert.strictEqual(meta.training_method, 'full');
    console.log('✓ handoff ready in-process');

    // Simulated server restart: drop in-memory state, rely on job.json on disk.
    tb.jobs.delete(job.job_id);
    const meta2 = tb.getArtifactMetadata(job.job_id);
    assert.strictEqual(meta2.ready, true, 'disk fallback must stay inference-ready');
    assert.strictEqual(meta2.artifact_dir, expected);
    console.log('✓ handoff ready after restart (disk fallback)');

    // Static guard: the artifact handoff must never hard-code the remote
    // host's training_outputs path (determination E guard). NOTE: the
    // unrelated local-Python lookup '/home/runner/workspace/.pythonlibs/...'
    // in _findPythonWithDeps is explicitly allowed — only the artifact
    // directory path is guarded here.
    const root = path.join(__dirname, '..');
    for (const f of ['training_backend.js', 'server.js', 'inference_runner.py']) {
      const src = fs.readFileSync(path.join(root, f), 'utf8');
      assert(!src.includes('/home/runner/workspace/training_outputs'),
        `${f} must not hard-code the remote artifact path (determination E guard)`);
    }
    console.log('✓ no hard-coded remote artifact path in handoff chain');
  } finally {
    childProcess.spawn = ORIG_SPAWN;
    mock.server.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
    for (const j of [...tb.jobs.values()]) {
      try { fs.rmSync(path.join(__dirname, '..', 'training_outputs', j.job_id), { recursive: true, force: true }); } catch (_) {}
    }
    tb.clearAllJobs();
    process.env = ORIG_ENV;
  }
  console.log('\nAll artifact handoff tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
