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

app.use(cors());
app.use(express.json());
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

// ── POST /api/train ───────────────────────────────────────────────────────────
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
