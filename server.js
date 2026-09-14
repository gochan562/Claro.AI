require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const helmet  = require('helmet');
const rateLimit = require('express-rate-limit');
const { ipKeyGenerator } = require('express-rate-limit');
const path    = require('path');
const { Readable } = require('stream');
const { execFile } = require('child_process');
const { promisify } = require('util');

const app  = express();
const PORT = process.env.PORT || 5000;
const execFileAsync = promisify(execFile);

// ── Auth removed ─────────────────────────────────────────────────────────
// Claro.AI runs as a single-user local app with no login. There is no
// Firebase Admin, no session, and no user store. Training/GPU endpoints that
// were previously gated behind requireAuth/requireAdmin are intentionally
// open. NOTE: /api/run-cell and /api/stream-logs forward caller-supplied code
// to a Modal backend and therefore allow remote code execution by anyone who
// can reach this server — see the route comments below. Only expose this
// server to trusted networks (localhost by default).

// Single anonymous owner id used for training jobs (per-user quota and job
// ownership collapse to one local user; no account system).
const ANON_USER_ID = 'local-user';

// GPU provider abstraction.  GPU_PROVIDER controls which backend resolves:
//   "zerogpu" (default) → Hugging Face ZeroGPU Space (Gradio API)
//   "modal"             → existing Modal endpoints (forwarded unchanged)
const {
  resolveBackend,
  ZeroGPUBackend,
  GpuError,
} = require('./gpu_backends');
const gpuBackend = resolveBackend(process.env);

// Training backend — generic engine (inference vs training are independent)
const trainingBackend = require('./training_backend');

// ── Security headers ───────────────────────────────────────────────────
app.use(helmet({
  crossOriginOpenerPolicy: { policy: "same-origin-allow-popups" },
  contentSecurityPolicy: {
    directives: {
      ...helmet.contentSecurityPolicy.getDefaultDirectives(),
      "script-src": [
        "'self'",
        "https://cdn.jsdelivr.net",      // marked, dompurify, xterm, chart.js
        "https://cdnjs.cloudflare.com",  // codemirror
      ],
      "img-src": [
        "'self'",
        "data:",
        "https://api.dicebear.com",      // dashboard avatar
      ],
      "connect-src": [
        "'self'",
      ],
      "script-src-attr": ["'unsafe-inline'"],
    },
  },
}));

// ── CORS allow-list ────────────────────────────────────────────────────
const allowedOrigins = (process.env.FRONTEND_ORIGIN || 'http://localhost:5000')
  .split(',').map(s=>s.trim()).filter(Boolean);
app.use(cors({
  origin: function(origin, cb){
    // allow non-browser requests (no origin) and allow-listed origins
    if (!origin) return cb(null, true);
    if (allowedOrigins.includes(origin) || allowedOrigins.includes('*')) return cb(null, true);
    // For development, allow localhost any port
    if (origin.startsWith('http://localhost:') || origin.startsWith('http://127.0.0.1:')) return cb(null, true);
    return cb(null, false); // reject cleanly — no thrown Error, no 500
  },
  credentials: true
}));

// ── Body parsing ───────────────────────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ── Rate limiting ───────────────────────────────────────────────────────
const globalLimiter = rateLimit({
  windowMs: 15*60*1000,
  max: 300,
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(globalLimiter);

const strictLimiter = rateLimit({
  windowMs: 60*1000,
  max: 10,
  keyGenerator: (req) => ipKeyGenerator(req.ip),
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req,res)=> res.status(429).json({ error: 'too many requests', code: 'rate_limited' })
});

// ── Static serving (restricted to public/) ─────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));
// Explicitly handle root
app.get('/', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── GET /api/hf-loader (public, used for model picker) ─────────────────
app.get('/api/hf-loader', async (req, res) => {
  const modelId = String(req.query.modelId || '').trim();
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(modelId)) {
    return res.status(400).json({ error: 'A valid Hugging Face model id is required.' });
  }

  const script = [
    'import json, sys',
    'from hf_loader.notebook_builder import build_loader_cell',
    'try:',
    '    print(json.dumps({"loader": build_loader_cell(sys.argv[1])}))',
    'except Exception as exc:',
    '    print(json.dumps({"error": str(exc)}))',
    '    raise',
  ].join('\n');

  try {
    const { stdout } = await execFileAsync(
      process.env.PYTHON_BIN || 'python3',
      ['-c', script, modelId],
      { cwd: __dirname, timeout: 10000, maxBuffer: 256 * 1024 }
    );
    const payload = JSON.parse(stdout.trim());
    if (payload.error || !payload.loader) {
      return res.status(500).json({ error: payload.error || 'Could not generate loader cell.' });
    }
    return res.json(payload);
  } catch (err) {
    const detail = (err.stderr || err.message || 'Python loader generation failed').trim();
    console.error('HF loader generation error:', detail);
    return res.status(500).json({ error: 'Could not generate the Hugging Face loader cell.' });
  }
});

// ── Helper: forward a request to a Modal endpoint ────────────────────────────
async function callModal(url, body) {
  const { MODAL_AUTH_SECRET } = process.env;
  const res  = await fetch(url, {
    method:  'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${MODAL_AUTH_SECRET || ''}`
    },
    body: JSON.stringify(body)
  });

  const text = await res.text();          // always safe to read as text
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    // Modal returned a non-JSON body (auth error, cold-start page, etc.)
    data = { error: text.slice(0, 500) }; // surface the raw message
  }
  return { status: res.status, data };
}

// ── GET /api/gpu-status ────────────────────────────────────────────────────────
app.get('/api/gpu-status', async (req, res) => {
  try {
    const info = await gpuBackend.status();
    return res.json(info);
  } catch (err) {
    return res.status(502).json({
      state: 'disconnected',
      provider: gpuBackend.kind,
      error: err && err.message ? err.message : String(err),
    });
  }
});

// ── GET /api/gpu-config ──────────────────────────────────────────────────────
app.get('/api/gpu-config', (req, res) => {
  res.json({
    provider: gpuBackend.kind,
    zerogpu: {
      space: gpuBackend.kind === 'zerogpu' ? gpuBackend.spaceId : (process.env.ZEROGPU_SPACE || 'Gochan562/claro_ai_gpu'),
    },
    modal: {
      configured: !!(process.env.MODAL_RUN_URL),
    },
  });
});

// ── GET /api/stream-logs ─────────────────────────────────────────────────────
// NOTE: This endpoint forwards caller-supplied code to a Modal backend, so
// anyone who can reach this server can execute code via the configured Modal
// app. It was previously gated behind admin auth; with login removed it is
// open. Only expose this server to trusted networks (localhost by default).
app.get('/api/stream-logs', async (req, res) => {
  const { MODAL_STREAM_URL, MODAL_AUTH_SECRET } = process.env;
  const code = req.query.code || '';

  res.set({
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    Connection:           'keep-alive',
    'X-Accel-Buffering': 'no'
  });
  res.flushHeaders();

  const sendLine = (line) => res.write(`data: ${JSON.stringify({ line })}\n\n`);
  const sendDone = (ok, message) => {
    res.write(`event: done\ndata: ${JSON.stringify({ ok, message: message || '' })}\n\n`);
    res.end();
  };

  if (!MODAL_STREAM_URL) {
    sendLine('❌ MODAL_STREAM_URL is not configured. Deploy trainer.py to Modal then set the env var.');
    return sendDone(false);
  }
  if (!code.trim()) {
    sendLine('❌ No code provided.');
    return sendDone(false);
  }

  const controller = new AbortController();
  req.on('close', () => controller.abort());

  try {
    const url = `${MODAL_STREAM_URL}?code=${encodeURIComponent(code)}`;
    const modalRes = await fetch(url, {
      method:  'GET',
      headers: { Authorization: `Bearer ${MODAL_AUTH_SECRET || ''}` },
      signal:  controller.signal
    });

    if (!modalRes.ok || !modalRes.body) {
      const text = await modalRes.text();
      sendLine(`❌ Modal error (${modalRes.status}): ${text.slice(0, 400)}`);
      return sendDone(false);
    }

    const nodeStream = Readable.fromWeb(modalRes.body);
    let buffer = '';

    for await (const chunk of nodeStream) {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        sendLine(buffer.slice(0, idx));
        buffer = buffer.slice(idx + 1);
      }
    }
    if (buffer) sendLine(buffer);

    sendDone(true);
  } catch (err) {
    if (err.name !== 'AbortError') {
      sendLine(`❌ Connection error: ${err.message}`);
      sendDone(false);
    }
  }
});

// ── POST /api/zerogpu-run ─────────────────────────────────────────────────────
app.post('/api/zerogpu-run', strictLimiter, async (req, res) => {
  if (gpuBackend.kind !== 'zerogpu') {
    return res.status(400).json({
      error: `Not on ZeroGPU backend (current provider: ${gpuBackend.kind}). Set GPU_PROVIDER=zerogpu to use this endpoint.`,
    });
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(process.env.ZEROGPU_TIMEOUT_MS || 120000));
  const authToken = req.headers.authorization?.replace(/^Bearer\s+/i, '') || '';
  try {
    const result = await gpuBackend.run(req.body || {}, controller.signal, null, authToken);
    return res.json(result);
  } catch (err) {
    const code = err && err.code ? err.code : 'zerogpu_runtime';
    const status = err && err.status ? err.status : 502;
    if (err instanceof GpuError) {
      return res.status(status).json({ error: err.message, code });
    }
    return res.status(502).json({ error: `ZeroGPU call failed: ${err.message || err}`, code });
  } finally {
    clearTimeout(timer);
  }
});

// ── GET /api/zerogpu-stream ──────────────────────────────────────────────────
app.get('/api/zerogpu-stream', async (req, res) => {
  res.set({
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    Connection:           'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders();

  const sendLine = (line) => res.write(`data: ${JSON.stringify({ line })}\n\n`);
  const sendDone = (ok, message) => {
    res.write(`event: done\ndata: ${JSON.stringify({ ok, message: message || '' })}\n\n`);
    res.end();
  };

  if (gpuBackend.kind !== 'zerogpu') {
    sendLine(`❌ Not on ZeroGPU backend (current provider: ${gpuBackend.kind}).`);
    return sendDone(false);
  }
  let body;
  try {
    const raw = req.query.req || '{}';
    body = JSON.parse(decodeURIComponent(raw));
  } catch (err) {
    sendLine(`❌ Malformed ?req= JSON: ${err.message}`);
    return sendDone(false);
  }

  const controller = new AbortController();
  req.on('close', () => controller.abort());

  try {
    await gpuBackend.run(body, controller.signal, sendLine);
    sendDone(true);
  } catch (err) {
    const code = err && err.code ? err.code : 'zerogpu_runtime';
    sendLine(`❌ ${err.message || err}  (code: ${code})`);
    sendDone(false);
  }
});

// ── POST /api/run-cell ────────────────────────────────────────────────────────
// NOTE: This endpoint executes caller-supplied code on the Modal backend, so
// anyone who can reach this server can run code via the configured Modal app.
// It was previously gated behind admin auth; with login removed it is open
// (rate-limited). Only expose this server to trusted networks.
app.post('/api/run-cell', strictLimiter, async (req, res) => {
  const { code } = req.body || {};
  if (!code || !String(code).trim()) {
    return res.status(400).json({ error: 'No code provided.' });
  }

  if (gpuBackend.kind === 'zerogpu') {
    return res.status(400).json({
      error:
        'ZeroGPU backend only accepts structured { model_id, task, inputs } ' +
        'requests via POST /api/zerogpu-run (or GET /api/zerogpu-stream). ' +
        'Plain notebook-cell Python is not executed on ZeroGPU.',
      code: 'zerogpu_no_python_cells',
    });
  }

  try {
    const { status, data } = await gpuBackend.run({ code });
    return res.status(status).json(data);
  } catch (err) {
    const status = err && err.status ? err.status : 502;
    return res.status(status).json({ error: err.message || String(err) });
  }
});

// ── Training Engine ────────────────────────────────────────────────────────
// Real step-count estimate from dataset size (no fake numbers). Used by the
// Training Cell preview; estimated_total_steps is null when the dataset size
// cannot be resolved. Time estimates are intentionally NOT invented.
app.get('/api/train/estimate', async (req, res) => {
  try {
    const est = await trainingBackend.estimateTrainingSteps({
      dataset_id: req.query.dataset_id || req.query.datasetId || '',
      epochs: req.query.epochs,
      batch_size: req.query.batch_size !== undefined ? req.query.batch_size : req.query.batchSize,
      max_steps: req.query.max_steps !== undefined ? req.query.max_steps : req.query.maxSteps,
      validation_split: req.query.validation_split !== undefined ? req.query.validation_split : req.query.validationSplit,
    });
    return res.json(est);
  } catch (err) {
    return res.json({ estimated_total_steps: null, train_rows: null, total_rows: null, basis: 'estimate-failed' });
  }
});

app.post('/api/train/start', strictLimiter, async (req, res) => {
  try {
    // Single-user quota: 1 concurrent job for the local user
    const active = trainingBackend.getActiveJobCountForUser(ANON_USER_ID);
    const limit = parseInt(process.env.TRAIN_MAX_CONCURRENT_PER_USER || '1',10);
    if (active >= limit) {
      return res.status(429).json({ error: `Concurrent job limit reached (${limit}). Wait for your current job to finish.`, code: 'quota_exceeded' });
    }
    const config = trainingBackend.validateTrainingRequest(req.body);
    const job = trainingBackend.createJob(config, ANON_USER_ID);
    trainingBackend.startJob(job);
    // Best-effort: replace the epochs*100 progress fallback with the real
    // estimate once the dataset size resolves (broadcasts to SSE clients).
    trainingBackend.refreshTotalStepsEstimate(job).catch(() => {});
    return res.json({
      job_id: job.job_id,
      status: job.status,
      config: job.config,
      message: 'Training job queued',
    });
  } catch (err) {
    const status = err.status || 400;
    const code = err.code || 'bad_request';
    return res.status(status).json({ error: err.message, code });
  }
});

app.get('/api/train/status/:job_id', (req, res) => {
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({
    job_id: job.job_id,
    status: job.status,
    progress: job.progress,
    config: job.config,
    error: job.error,
    created_at: job.created_at,
    metrics_count: job.metrics.length,
    artifacts_dir: job.artifacts_dir,
  });
});

app.get('/api/train/metrics/:job_id', (req, res) => {
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });

  const wantsSSE = (req.headers.accept || '').includes('text/event-stream') || req.query.stream === '1' || req.query.stream === 'true';
  if (wantsSSE) {
    res.set({
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.flushHeaders();
    trainingBackend.attachSSE(job.job_id, res);
    const ping = setInterval(() => {
      try { res.write(': ping\n\n'); } catch (_) { clearInterval(ping); }
    }, 15000);
    req.on('close', () => {
      clearInterval(ping);
    });
    return;
  }

  const limit = Math.min(parseInt(req.query.limit || '100', 10) || 100, 500);
  const slice = job.metrics.slice(-limit);
  return res.json({
    job_id: job.job_id,
    status: job.status,
    progress: job.progress,
    metrics: slice,
  });
});

app.get('/api/train/stream/:job_id', (req, res) => {
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders();
  trainingBackend.attachSSE(job.job_id, res);
  const ping = setInterval(() => {
    try { res.write(': ping\n\n'); } catch (_) { clearInterval(ping); }
  }, 15000);
  req.on('close', () => clearInterval(ping));
});

app.get('/api/train/list', (req, res) => {
  return res.json({ jobs: trainingBackend.listJobs() });
});

app.post('/api/train/stop', (req, res) => {
  const job_id = String(req.body.job_id || req.body.jobId || req.query.job_id || '').trim();
  const diagCaller = req.body._diag_caller || req.headers['x-diag-caller'] || 'unknown';
  const diagCellId = req.body._diag_cellId || req.body.cellId || 'unknown';
  const diagUserInitiated = req.body._diag_userInitiated;
  const diagStack = req.body._diag_stack || '';
  console.log(`[TRAIN-DIAG] POST /api/train/stop job_id=${job_id} caller=${diagCaller} cellId=${diagCellId} userInitiated=${diagUserInitiated} ts=${new Date().toISOString()} ip=${req.ip} stack=${String(diagStack).slice(0,500)}`);
  if (!job_id) return res.status(400).json({ error: 'job_id is required', code: 'bad_request' });
  const job = trainingBackend.getJob(job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  const stopped = trainingBackend.stopJob(job_id, { caller: diagCaller, cellId: diagCellId, userInitiated: diagUserInitiated, stack: diagStack });
  if (!stopped) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({ job_id, status: stopped.status, message: 'Job stopped' });
});

app.post('/api/train/stop/:job_id', (req, res) => {
  console.log(`[TRAIN-DIAG] POST /api/train/stop/:job_id job_id=${req.params.job_id} ts=${new Date().toISOString()} ip=${req.ip}`);
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  const stopped = trainingBackend.stopJob(req.params.job_id, { caller: 'stop/:job_id', userInitiated: false });
  if (!stopped) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({ job_id: stopped.job_id, status: stopped.status, message: 'Job stopped' });
});

app.get('/api/train/artifacts/:job_id', (req, res) => {
  try {
    const meta = trainingBackend.getArtifactMetadata(req.params.job_id);
    return res.json(meta);
  } catch (err) {
    const status = err.status || 500;
    const code = err.code || 'artifact_error';
    return res.status(status).json({ error: err.message, code });
  }
});

app.post('/api/inference/trained', strictLimiter, async (req, res) => {
  const { job_id, prompt, max_new_tokens, task, image, image_base64, imageBase64 } = req.body || {};
  if (!job_id) return res.status(400).json({ error: 'job_id is required', code: 'bad_request' });
  let meta;
  try {
    meta = trainingBackend.getArtifactMetadata(String(job_id).trim());
  } catch (err) {
    const status = err.status || 500;
    const code = err.code || 'artifact_error';
    return res.status(status).json({ error: err.message, code });
  }

  // Use artifact metadata to derive paths — never use client-provided paths
  const jobId = meta.job_id;
  const taskType = String(task || meta.task_type || 'text-generation').toLowerCase();
  const maxTokens = Math.min(parseInt(max_new_tokens, 10) || 100, 2048);
  const isImageTask = taskType === 'image-classification';
  const imagePayload = image_base64 || image || imageBase64 || null;
  let promptStr = null;
  if (!isImageTask) {
    if (!prompt || typeof prompt !== 'string' || !prompt.trim()) return res.status(400).json({ error: 'prompt is required', code: 'bad_request' });
    promptStr = String(prompt).slice(0, 4000);
  } else {
    if (imagePayload) {
      promptStr = String(prompt || '').slice(0, 4000);
    } else if (prompt && typeof prompt === 'string' && prompt.startsWith('data:image')) {
      promptStr = prompt.slice(0, 5000000);
    } else if (prompt && typeof prompt === 'string' && prompt.trim()) {
      promptStr = String(prompt).slice(0, 4000);
    } else {
      return res.status(400).json({ error: 'image is required for image-classification (send image_base64)', code: 'bad_request' });
    }
  }

  const pythonBin = process.env.PYTHON_BIN || 'python3';
  const runnerPath = path.join(__dirname, 'inference_runner.py');
  if (!require('fs').existsSync(runnerPath)) {
    return res.status(500).json({ error: 'inference_runner.py not found', code: 'inference_error' });
  }

  const args = [
    runnerPath,
    '--job_id', jobId,
    '--task_type', taskType,
    '--max_new_tokens', String(maxTokens),
  ];
  if (isImageTask) {
    if (imagePayload) {
      args.push('--image_base64', String(imagePayload).slice(0, 8000000));
    } else if (promptStr && promptStr.startsWith('data:image')) {
      args.push('--image_base64', promptStr);
    } else if (promptStr) {
      args.push('--prompt', promptStr);
    }
  } else {
    args.push('--prompt', promptStr);
  }

  const { spawn } = require('child_process');
  let proc;
  try {
    proc = spawn(pythonBin, args, { cwd: __dirname, env: process.env });
  } catch (e) {
    return res.status(500).json({ error: `Failed to spawn inference: ${e.message}`, code: 'inference_error' });
  }

  let stdout = '';
  let stderr = '';
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    try { proc.kill('SIGKILL'); } catch (_) {}
  }, 60000);

  proc.stdout.on('data', (chunk) => { stdout += chunk.toString('utf8'); });
  proc.stderr.on('data', (chunk) => { stderr += chunk.toString('utf8'); });

  proc.on('close', (code) => {
    clearTimeout(timer);
    if (timedOut) {
      return res.status(504).json({ error: 'Inference timed out', code: 'inference_timeout' });
    }
    if (code !== 0) {
      try {
        const parsed = JSON.parse(stdout.trim().split('\n').pop());
        if (parsed && parsed.error) {
          return res.status(500).json({ error: parsed.error, code: 'inference_error', stderr: stderr.slice(0, 1000) });
        }
      } catch (_) {}
      return res.status(500).json({ error: `Inference failed (exit ${code}): ${stderr.slice(0, 1000) || stdout.slice(0, 1000)}`, code: 'inference_error' });
    }
    try {
      const lines = stdout.trim().split('\n');
      let data = null;
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith('{') && line.endsWith('}')) {
          try { data = JSON.parse(line); break; } catch (_) {}
        }
      }
      if (!data || (data.output === undefined && data.label === undefined && data.scores === undefined)) {
        return res.status(500).json({ error: `No output from inference: ${stdout.slice(0, 1000)}`, code: 'inference_error', stderr: stderr.slice(0, 500) });
      }
      return res.json({
        task: data.task || taskType,
        output: data.output !== undefined ? data.output : (data.label || JSON.stringify(data)),
        label: data.label,
        confidence: data.confidence,
        prediction: data.prediction,
        scores: data.scores,
        tokens: data.tokens,
        entities: data.entities,
        raw: data
      });
    } catch (e) {
      return res.status(500).json({ error: `Failed to parse inference output: ${e.message}`, code: 'inference_error', stdout: stdout.slice(0, 1000), stderr: stderr.slice(0, 500) });
    }
  });

  proc.on('error', (err) => {
    clearTimeout(timer);
    return res.status(500).json({ error: `Inference spawn error: ${err.message}`, code: 'inference_error' });
  });
});

// ── POST /api/train (legacy) ──────────────────────────────────────────────────
app.post('/api/train', strictLimiter, async (req, res) => {
  const { MODAL_URL } = process.env;

  if (!MODAL_URL) {
    return res.status(500).json({ error: 'MODAL_URL is not configured.' });
  }

  const { modelId, datasetId, epochs, lr } = req.body;
  if (!modelId || !datasetId) {
    return res.status(400).json({ error: 'modelId and datasetId are required.' });
  }

  try {
    const { status, data } = await callModal(MODAL_URL, {
      model_id:      modelId,
      dataset_id:    datasetId,
      epochs:        parseInt(epochs, 10) || 3,
      learning_rate: parseFloat(lr)       || 2e-4
    });
    return res.status(status).json(data);
  } catch (err) {
    return res.status(502).json({ error: `Could not reach Modal: ${err.message}` });
  }
});

// Fallback for client-side routing: serve index.html for unknown non-API routes
app.get('/*splat', (req,res,next)=>{
  if (req.path.startsWith('/api/')) return next();
  // Let static handle if file exists, otherwise 404
  return res.status(404).send('Not found');
});

// ── Generic error handler (must be last) ────────────────────────────────
app.use((err, req, res, next) => {
  console.error(err); // full detail stays in server logs
  if (res.headersSent) return next(err);
  res.status(err.status || 500).json({
    error: process.env.NODE_ENV === 'production' ? 'internal server error' : err.message,
    code: 'internal_error'
  });
});

if (require.main === module) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Claro.AI server listening on port ${PORT}`);
  });
}
module.exports = app;

