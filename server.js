const express = require('express');
const cors    = require('cors');
const path    = require('path');
const { Readable } = require('stream');
const { execFile } = require('child_process');
const { promisify } = require('util');

const app  = express();
const PORT = process.env.PORT || 5000;
const execFileAsync = promisify(execFile);

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
// Lightweight reachability check for the Modal GPU endpoint (does not execute
// any code — just confirms the server can be reached).
app.get('/api/gpu-status', async (req, res) => {
  const { MODAL_RUN_URL } = process.env;
  const gpuType = process.env.GPU_TYPE || 'A10G';

  if (!MODAL_RUN_URL) {
    return res.json({ state: 'not-configured', gpuType });
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 6000);
    const modalRes = await fetch(MODAL_RUN_URL, { method: 'GET', signal: controller.signal });
    clearTimeout(timer);
    // Any HTTP response (even 4xx/405 for a GET on a POST-only endpoint)
    // proves the Modal server is up and reachable.
    return res.json({ state: 'connected', gpuType, httpStatus: modalRes.status });
  } catch (err) {
    return res.json({ state: 'disconnected', gpuType, error: err.message });
  }
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

// ── POST /api/run-cell ────────────────────────────────────────────────────────
// Executes a notebook cell's Python code on the Modal A10G GPU and returns
// stdout + stderr so the notebook can display real output.
app.post('/api/run-cell', async (req, res) => {
  const { MODAL_RUN_URL } = process.env;

  if (!MODAL_RUN_URL) {
    return res.status(500).json({
      error: 'MODAL_RUN_URL is not configured. Deploy trainer.py to Modal then set the env var.'
    });
  }

  const { code } = req.body;
  if (!code || !code.trim()) {
    return res.status(400).json({ error: 'No code provided.' });
  }

  try {
    const { status, data } = await callModal(MODAL_RUN_URL, { code });
    return res.status(status).json(data);
  } catch (err) {
    return res.status(502).json({ error: `Could not reach Modal: ${err.message}` });
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
