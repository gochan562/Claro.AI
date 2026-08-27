const express = require('express');
const cors    = require('cors');
const path    = require('path');
const { Readable } = require('stream');
const { execFile } = require('child_process');
const { promisify } = require('util');

const app  = express();
const PORT = process.env.PORT || 5000;
const execFileAsync = promisify(execFile);

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

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname)));

// ── GET /api/hf-loader ───────────────────────────────────────────────────────
// The browser asks the Python loader package for Cell 1.  The package emits
// runtime AutoConfig-based code; it never downloads or inspects a model on the
// web server.  Keeping this adapter here lets the existing static dashboard
// consume the new Python implementation without bundling Python into JS.
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
// Lightweight reachability check for the active GPU backend.  When the backend
// is ZeroGPU this probes the Space's /gradio_api/config without queueing a GPU
// call.  When Modal is selected this hits MODAL_RUN_URL with a GET.
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
// Returns which backend is active + Space info so the dashboard can route
// inference/model-cell runs to the right endpoint (structured ZeroGPU vs.
// Python-cell Modal) without guessing.
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
// Server-Sent Events endpoint. Streams a notebook cell's stdout/stderr to the
// browser line-by-line as it's produced on the Modal GPU, so students watch
// downloads/training print out in real time instead of waiting for one big
// JSON blob. EventSource only supports GET, so the code is passed as a query
// parameter.
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
// Structured request to the ZeroGPU Space.  Accepts:
//   { model_id: str, task: str, inputs: { prompt, max_new_tokens, ... } }
// and returns { output, stderr }.  Never forwards arbitrary Python; the model
// loads and runs inside the Space.
app.post('/api/zerogpu-run', async (req, res) => {
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
// Server-Sent Events: opens a ZeroGPU Space call and streams the resulting
// lines into the dashboard GPU terminal as they arrive.  Accepts the same
// structured payload as /api/zerogpu-run but as URL-safe JSON in ?req=
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
// Executes a notebook cell.  Behaviour depends on the active backend:
//   • ZeroGPU (default) → rejects Python-cell requests for the inference/model
//     cells; the dashboard must send structured requests to /api/zerogpu-run.
//     Other code/parameter/markdown cells never reach here from a ZeroGPU
//     model cell, but we keep the endpoint available for plain Python cells
//     by forwarding to Modal *only if* Modal is configured.
//   • Modal → runs arbitrary Python on the remote A10G (unchanged behaviour).
app.post('/api/run-cell', async (req, res) => {
  const { code } = req.body || {};
  if (!code || !String(code).trim()) {
    return res.status(400).json({ error: 'No code provided.' });
  }

  if (gpuBackend.kind === 'zerogpu') {
    // The local Claro.AI should only communicate with the Space; it does not
    // run arbitrary Python anywhere when ZeroGPU is the active backend.
    return res.status(400).json({
      error:
        'ZeroGPU backend only accepts structured { model_id, task, inputs } ' +
        'requests via POST /api/zerogpu-run (or GET /api/zerogpu-stream). ' +
        'Plain notebook-cell Python is not executed on ZeroGPU.',
      code: 'zerogpu_no_python_cells',
    });
  }

  // Modal path (unchanged from the original implementation).
  try {
    const { status, data } = await gpuBackend.run({ code });
    return res.status(status).json(data);
  } catch (err) {
    const status = err && err.status ? err.status : 502;
    return res.status(status).json({ error: err.message || String(err) });
  }
});

// ── Training Engine — generic backend (independent from inference) ──────────
// Supports ZeroGPU | Local Python | Modal (future). Uses HF Trainer + task-aware
// preprocessing (AutoTokenizer / AutoImageProcessor) and auto-detects model class
// via hf_loader logic (inside training_runner.py).
//
// Jobs: { job_id, status, progress, metrics } with statuses:
//   queued | loading | training | evaluating | finished | failed

// POST /api/train/start — create and launch a background training job
app.post('/api/train/start', async (req, res) => {
  try {
    const config = trainingBackend.validateTrainingRequest(req.body);
    const job = trainingBackend.createJob(config);
    trainingBackend.startJob(job);
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

// GET /api/train/status/:job_id — current job snapshot (polling fallback)
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

// GET /api/train/metrics/:job_id — return recent metrics as JSON, or SSE stream if requested
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
    // replay + attach
    trainingBackend.attachSSE(job.job_id, res);
    // keep-alive ping every 15s
    const ping = setInterval(() => {
      try { res.write(': ping\n\n'); } catch (_) { clearInterval(ping); }
    }, 15000);
    req.on('close', () => {
      clearInterval(ping);
      // attachSSE handles removal on res close
    });
    return;
  }

  // polling JSON
  const limit = Math.min(parseInt(req.query.limit || '100', 10) || 100, 500);
  const slice = job.metrics.slice(-limit);
  return res.json({
    job_id: job.job_id,
    status: job.status,
    progress: job.progress,
    metrics: slice,
  });
});

// GET /api/train/stream/:job_id — dedicated SSE stream (alias for metrics SSE)
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

// GET /api/train/list — list all jobs (debug)
app.get('/api/train/list', (req, res) => {
  return res.json({ jobs: trainingBackend.listJobs() });
});

// POST /api/train/stop — stop a running job
app.post('/api/train/stop', (req, res) => {
  const job_id = String(req.body.job_id || req.body.jobId || req.query.job_id || '').trim();
  const diagCaller = req.body._diag_caller || req.headers['x-diag-caller'] || 'unknown';
  const diagCellId = req.body._diag_cellId || req.body.cellId || 'unknown';
  const diagUserInitiated = req.body._diag_userInitiated;
  const diagStack = req.body._diag_stack || '';
  console.log(`[TRAIN-DIAG] POST /api/train/stop job_id=${job_id} caller=${diagCaller} cellId=${diagCellId} userInitiated=${diagUserInitiated} ts=${new Date().toISOString()} ip=${req.ip} stack=${String(diagStack).slice(0,500)}`);
  if (!job_id) return res.status(400).json({ error: 'job_id is required', code: 'bad_request' });
  const job = trainingBackend.stopJob(job_id, { caller: diagCaller, cellId: diagCellId, userInitiated: diagUserInitiated, stack: diagStack });
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({ job_id, status: job.status, message: 'Job stopped' });
});

// also support POST /api/train/stop/:job_id
app.post('/api/train/stop/:job_id', (req, res) => {
  console.log(`[TRAIN-DIAG] POST /api/train/stop/:job_id job_id=${req.params.job_id} ts=${new Date().toISOString()} ip=${req.ip}`);
  const job = trainingBackend.stopJob(req.params.job_id, { caller: 'stop/:job_id', userInitiated: false });
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({ job_id: job.job_id, status: job.status, message: 'Job stopped' });
});

// GET /api/train/artifacts/:job_id — inference-ready artifact metadata
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

// POST /api/inference/trained — run inference with a trained model (full or LoRA)
// Supports text tasks via prompt and image tasks via image_base64 / image
app.post('/api/inference/trained', async (req, res) => {
  const { job_id, prompt, max_new_tokens, task, image, image_base64, imageBase64 } = req.body || {};
  if (!job_id) return res.status(400).json({ error: 'job_id is required', code: 'bad_request' });
  // Validate job and artifacts server-side (never trust client path)
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
  // For image tasks, allow image payload; for text tasks, prompt required
  const isImageTask = taskType === 'image-classification';
  const imagePayload = image_base64 || image || imageBase64 || null;
  let promptStr = null;
  if (!isImageTask) {
    if (!prompt || typeof prompt !== 'string' || !prompt.trim()) return res.status(400).json({ error: 'prompt is required', code: 'bad_request' });
    promptStr = String(prompt).slice(0, 4000);
  } else {
    // image task: need image, allow prompt to carry image if provided as data URL
    if (imagePayload) {
      promptStr = String(prompt || '').slice(0, 4000); // not used but keep
    } else if (prompt && typeof prompt === 'string' && prompt.startsWith('data:image')) {
      // allow prompt to be data URL
      promptStr = prompt.slice(0, 5000000); // allow large data URL (up to 5MB, limit enforced by json limit)
    } else if (prompt && typeof prompt === 'string' && prompt.trim()) {
      // If image task but prompt looks like text, treat as error to guide UI
      // But allow text prompt if user mistakenly sends text; we will handle as image path failure
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
      // fallback for text misuse
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
  }, 60000); // 60s timeout for inference

  proc.stdout.on('data', (chunk) => { stdout += chunk.toString('utf8'); });
  proc.stderr.on('data', (chunk) => { stderr += chunk.toString('utf8'); });

  proc.on('close', (code) => {
    clearTimeout(timer);
    if (timedOut) {
      return res.status(504).json({ error: 'Inference timed out', code: 'inference_timeout' });
    }
    if (code !== 0) {
      // Try to parse JSON error from stdout
      try {
        const parsed = JSON.parse(stdout.trim().split('\n').pop());
        if (parsed && parsed.error) {
          return res.status(500).json({ error: parsed.error, code: 'inference_error', stderr: stderr.slice(0, 1000) });
        }
      } catch (_) {}
      return res.status(500).json({ error: `Inference failed (exit ${code}): ${stderr.slice(0, 1000) || stdout.slice(0, 1000)}`, code: 'inference_error' });
    }
    // stdout should be a JSON line with {"output": ...} plus structured fields
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
      // Forward full structured response (task, prediction, scores, tokens etc.) but ensure output exists
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
// Kicks off a fine-tuning job (kept for programmatic use).
app.post('/api/train', async (req, res) => {
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

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Claro.AI server listening on port ${PORT}`);
});
