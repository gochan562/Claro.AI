'use strict';

// GPU provider abstraction for Claro.AI.
//
//   GPUBackend (interface)
//   ├── ZeroGPUBackend (default)   → Hugging Face ZeroGPU Space Gradio API
//   └── ModalBackend              → existing Modal endpoints (unchanged)
//
// The frontend notebook currently builds a combined Python cell (loader +
// inference) and sends it to POST /api/run-cell (or to the SSE
// /api/stream-logs).  Modal expects arbitrary Python (`{ code }`) because it
// runs the notebook verbatim on a remote A10G.
//
// The ZeroGPU Space, by contrast, only exposes a narrow Gradio endpoint:
//
//   POST   {space}/gradio_api/call/v2/generate        {"prompt": str, "max_new_tokens": int}
//   GET    {space}/gradio_api/call/generate/<event_id>   → SSE stream ending in
//                                                         event: complete
//                                                         data: ["<output str>"]
//
// So for ZeroGPU we do NOT forward arbitrary notebook Python.  Instead we
// accept a structured request { model_id, task, inputs } and map it to the
// Space's { prompt, max_new_tokens } surface.  The model itself is loaded and
// executes inside the Space (Gochan562/claro_ai_gpu hard-codes
// HuggingFaceTB/SmolLM3-3B today).  The local client only communicates with
// the Space; it never imports torch/transformers.
//
// Configuration (env vars):
//   GPU_PROVIDER   = "zerogpu" (default) | "modal"
//   ZEROGPU_SPACE  = "Gochan562/claro_ai_gpu"
//   ZEROGPU_API    = full base URL override
//                     (default: https://{owner}-{repo-underscores}.hf.space)
//   ZEROGPU_TIMEOUT_MS = per-call timeout, default 120000
//
//   MODAL_RUN_URL / MODAL_STREAM_URL / MODAL_AUTH_SECRET (existing, unchanged)

const { Readable } = require('stream');

// ────────────────────────────────────────────────────────────────────────────
// Shared errors
// ────────────────────────────────────────────────────────────────────────────
class GpuError extends Error {
  constructor(message, code, status = 502, cause) {
    super(message);
    this.name = 'GpuError';
    this.code = code;       // machine-readable: 'space_unavailable' | 'zerogpu_quota' | ...
    this.status = status;
    if (cause) this.cause = cause;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

// `Gochan562/claro_ai_gpu` → `https://gochan562-claro-ai-gpu.hf.space`
function defaultSpaceUrl(spaceId) {
  if (!spaceId || !/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(spaceId)) {
    throw new GpuError(
      `Invalid ZEROGPU_SPACE id: ${JSON.stringify(spaceId)}`,
      'space_unavailable',
      500
    );
  }
  const [owner, repo] = spaceId.split('/');
  const slug = `${owner.toLowerCase()}-${repo.replace(/[._]/g, '-').toLowerCase()}`;
  return `https://${slug}.hf.space`;
}

function safeReadSpaceResolvedUrl(env) {
  const space = env.ZEROGPU_SPACE || 'Gochan562/claro_ai_gpu';
  const base = env.ZEROGPU_API ? env.ZEROGPU_API.replace(/\/+$/, '') : defaultSpaceUrl(space);
  return { spaceId: space, base };
}

// Validate the structured request.  Returns a normalized object or throws.
function normalizeStructuredRequest(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new GpuError('Malformed request body (expected an object)', 'bad_request', 400);
  }
  const modelId = String(body.model_id || '').trim();
  const task = String(body.task || 'text-generation').toLowerCase().trim();
  const inputs = body.inputs && typeof body.inputs === 'object' ? body.inputs : {};

  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(modelId)) {
    throw new GpuError(
      `Invalid model_id: ${modelId || '(empty)'}`,
      'invalid_model_id',
      400
    );
  }
  if (!task) {
    throw new GpuError('Invalid task: (empty)', 'bad_request', 400);
  }

  // The Space exposes one text-generation surface today.  We accept a
  // permissive `inputs` shape and coerce it to { prompt, max_new_tokens }.
  let prompt;
  if (typeof inputs.prompt === 'string') {
    prompt = inputs.prompt;
  } else if (typeof inputs.text === 'string') {
    prompt = inputs.text;
  } else if (Array.isArray(inputs.prompts) && inputs.prompts.length) {
    prompt = String(inputs.prompts[0]);
  } else if (Array.isArray(inputs.messages)) {
    // Render a tiny chat→prompt mapping so the chat/conversational task can
    // reuse the same Space.  No model-specific templating is assumed.
    prompt = inputs.messages
      .map((m) => {
        const role = String(m.role || 'user').toLowerCase();
        const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '');
        return `${role}: ${content}`;
      })
      .join('\n');
  } else {
    throw new GpuError(
      'inputs.prompt (or inputs.text/messages) is required for the ZeroGPU backend',
      'bad_request',
      400
    );
  }

  let maxNewTokens = 100;
  const raw = inputs.max_new_tokens ?? inputs.maxTokens ?? inputs.max_tokens;
  if (raw !== undefined && raw !== null) {
    const n = Number(raw);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      throw new GpuError('inputs.max_new_tokens must be an integer', 'bad_request', 400);
    }
    maxNewTokens = n;
  }
  // The Space's slider is 1..500.  Clamp before sending so we don't trigger a
  // Gradio validation error, but still flag wildly out-of-range values.
  if (maxNewTokens < 1) maxNewTokens = 1;
  if (maxNewTokens > 500) {
    // Soft cap rather than hard reject — long generations still work.
    maxNewTokens = 500;
  }

  return { modelId, task, prompt, max_new_tokens: maxNewTokens };
}

// ────────────────────────────────────────────────────────────────────────────
// ZeroGPUBackend
// ────────────────────────────────────────────────────────────────────────────
class ZeroGPUBackend {
  constructor(env) {
    const { spaceId, base } = safeReadSpaceResolvedUrl(env);
    this.spaceId = spaceId;
    this.base = base;
    this.timeoutMs = Number(env.ZEROGPU_TIMEOUT_MS || 120000);
  }

  get kind() { return 'zerogpu'; }

  async status() {
    // Lightweight reachability probe.  We hit the Gradio API info endpoint,
    // which is cheap and does not queue a GPU call.  Some Spaces do not
    // expose /gradio_api/config but always expose /gradio_api/info.
    const probePaths = ['/gradio_api/info', '/gradio_api/config'];
    let lastErr;
    for (const p of probePaths) {
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 6000);
        const r = await fetch(`${this.base}${p}`, { signal: ctrl.signal });
        clearTimeout(timer);
        if (r.ok) {
          return {
            state: 'connected',
            provider: 'zerogpu',
            space: this.spaceId,
            base: this.base,
            endpoint: p,
            httpStatus: r.status,
          };
        }
        lastErr = new Error(`HTTP ${r.status} on ${p}`);
      } catch (err) {
        lastErr = err.name === 'AbortError' ? new Error('timeout') : err;
      }
    }
    return {
      state: 'disconnected',
      provider: 'zerogpu',
      space: this.spaceId,
      base: this.base,
      error: lastErr ? lastErr.message : 'unknown',
    };
  }

  // POST → returns {event_id}.  Pure JSON, never SSE here.
  // The Space's `/generate(prompt, max_new_tokens, model_id, task)` is the
  // generic backend endpoint: it uses `model_id` + `task` to load and run any
  // supported repo (Transformers / Diffusers / GGUF) instead of hard-coding
  // a single model.  We forward the validated model_id/task alongside the
  // prompt so the Space actually does the requested work.
  async _postGenerate(prompt, maxNewTokens, signal, modelId, task, authToken) {
    const url = `${this.base}/gradio_api/call/v2/generate`;
    const body = { prompt, max_new_tokens: maxNewTokens };
    if (modelId) body.model_id = modelId;
    if (task) body.task = task;
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
    });
    if (res.status === 404) {
      throw new GpuError(
        `ZeroGPU Space endpoint not found at ${url}. The Space may be misconfigured or the /generate API renamed.`,
        'space_unavailable',
        502
      );
    }
    if (res.status === 502 || res.status === 503 || res.status === 429) {
      // HF Spaces Sleeping/Error state or rate-limited.
      const detail = await res.text().catch(() => '');
      const sleepMsg = /sleeping|paused|building|loading/i.test(detail);
      throw new GpuError(
        sleepMsg
          ? `ZeroGPU Space is unavailable (HTTP ${res.status}). It may be sleeping or building.`
          : `ZeroGPU Space rejected the request (HTTP ${res.status}): ${detail.slice(0, 200)}`,
        'space_unavailable',
        res.status
      );
    }
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new GpuError(
        `ZeroGPU Space call failed (HTTP ${res.status}): ${detail.slice(0, 200)}`,
        'space_unavailable',
        res.status
      );
    }
    const ctype = res.headers?.get('content-type') || 'unknown';
    const raw = await res.text();
    console.error('[ZeroGPU POST] status:', res.status, 'content-type:', ctype, 'body:', raw.slice(0, 1000));
    let payload;
    try {
      payload = JSON.parse(raw);
    } catch (err) {
      throw new GpuError(`ZeroGPU returned non-JSON response: ${err.message}`, 'malformed_response', 502);
    }
    const eventId = payload && payload.event_id;
    if (!eventId || typeof eventId !== 'string') {
      throw new GpuError(`ZeroGPU did not return an event_id: ${JSON.stringify(payload).slice(0, 200)}`, 'malformed_response', 502);
    }
    return eventId;
  }

  // GET the SSE stream for an event_id and return the final `complete` payload.
  // `onLine` is optional and receives every raw SSE line (for live terminal).
  async _streamResult(eventId, signal, onLine, authToken) {
    const url = `${this.base}/gradio_api/call/generate/${encodeURIComponent(eventId)}`;
    const headers = {};
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const res = await fetch(url, { signal, headers });
    if (res.status === 404) {
      throw new GpuError('ZeroGPU event was not found (event expired or Space restarted).', 'space_unavailable', 502);
    }
    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => '');
      throw new GpuError(`ZeroGPU stream open failed (HTTP ${res.status}): ${detail.slice(0, 200)}`, 'space_unavailable', res.status || 502);
    }
    const streamCtype = res.headers?.get('content-type') || 'unknown';
    console.error('[ZeroGPU STREAM] status:', res.status, 'content-type:', streamCtype);

    const nodeStream = Readable.fromWeb(res.body);
    let buffer = '';
    let eventType = '';
    let dataBuf = '';
    let resolved = false;
    let hadError = false;
    let errMsg = '';

    for await (const chunk of nodeStream) {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).replace(/\r$/, '');
        buffer = buffer.slice(idx + 1);
        // Parse SSE framing fields (no multi-line data folding expected here).
        if (line === '' || line === '\r') {
          // frame boundary
          if (eventType && dataBuf !== '' && onLine) {
            // Emit the parsed line so the UI can mirror what the GPU produced.
            onLine(`event: ${eventType}\ndata: ${dataBuf}`);
          }
          if (eventType) {
            if (eventType === 'complete') {
              resolved = true;
              console.error('[ZeroGPU STREAM] complete frame:', dataBuf.slice(0, 3000));
              try {
                const parsed = JSON.parse(dataBuf);
                // Gradio returns an array: ["<generated text>"]
                const text = Array.isArray(parsed) ? parsed[0] : parsed;
                const out = typeof text === 'string' ? text : JSON.stringify(text);
                // The Space now returns structured error strings of the form
                // "[CLARO:<code>] <message>" when a model fails to load (or the
                // task is incompatible).  Detect them so the local provider can
                // surface typed error codes instead of presenting an error as
                // though it were generated text.
                const m = typeof out === 'string' ? out.match(/^\[CLARO:([a-z_]+)\]\s*(.*)$/s) : null;
                if (m) {
                  const code = m[1];
                  const message = m[2];
                  const status =
                    code === 'invalid_model_id'   ? 400 :
                    code === 'bad_request'         ? 400 :
                    code === 'task_model_mismatch'  ? 422 :
                    code === 'task_not_exposed'     ? 422 :
                    code === 'model_load_error'     ? 502 :
                    code === 'gguf_multiple_files'  ? 422 :
                    502;
                  throw new GpuError(`ZeroGPU model error: ${message}`, code, status);
                }
                return out;
              } catch (err) {
                if (err && err instanceof GpuError) throw err;
                throw new GpuError(`ZeroGPU returned malformed complete frame: ${dataBuf.slice(0, 200)}`, 'malformed_response', 502);
              }
            }
            if (eventType === 'error') {
              hadError = true;
              errMsg = dataBuf;
              console.error('[ZeroGPU STREAM] error frame:', dataBuf.slice(0, 3000));
            }
          }
          eventType = '';
          dataBuf = '';
          continue;
        }
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataBuf = line.slice(5).trim();
        } else if (line.endsWith(': heartbeat')) {
          // Gradio Keep-alive — intentionally ignored.
        }
      }
    }

    // Flush any pending frame if the stream ended without a trailing blank line
    if (eventType && dataBuf !== '') {
      if (eventType === 'complete') {
        resolved = true;
        try {
          const parsed = JSON.parse(dataBuf);
          const text = Array.isArray(parsed) ? parsed[0] : parsed;
          const out = typeof text === 'string' ? text : JSON.stringify(text);
          const m = typeof out === 'string' ? out.match(/^\[CLARO:([a-z_]+)\]\s*(.*)$/s) : null;
          if (m) {
            const code = m[1];
            const message = m[2];
            const status =
              code === 'invalid_model_id'   ? 400 :
              code === 'bad_request'         ? 400 :
              code === 'task_model_mismatch'  ? 422 :
              code === 'task_not_exposed'     ? 422 :
              code === 'model_load_error'     ? 502 :
              code === 'gguf_multiple_files'  ? 422 :
              502;
            throw new GpuError(`ZeroGPU model error: ${message}`, code, status);
          }
          return out;
        } catch (err) {
          if (err && err instanceof GpuError) throw err;
          throw new GpuError(`ZeroGPU returned malformed complete frame: ${dataBuf.slice(0, 200)}`, 'malformed_response', 502);
        }
      }
      if (eventType === 'error') {
        hadError = true;
        errMsg = dataBuf;
        console.error('[ZeroGPU STREAM] error frame (flush):', dataBuf.slice(0, 3000));
      }
    }

    if (hadError) {
      // Gradio error frames come in two shapes:
      //   {"error": "...", "title": "ZeroGPU worker error"}
      //   "404: Not Found"
      let message = errMsg;
      let code = 'zerogpu_runtime';
      let status = 502;
      try {
        const parsed = JSON.parse(errMsg);
        if (parsed && typeof parsed === 'object' && parsed.error) message = parsed.error;
        if (parsed && parsed.title === 'ZeroGPU worker error') code = 'zerogpu_quota';
        if (parsed && parsed.title === 'ZeroGPU queue timeout') code = 'zerogpu_timeout';
      } catch {
        // already a string
      }
      if (/No GPU available|quota|GHA seconds|GPU is not available/i.test(message)) code = 'zerogpu_quota';
      if (/No GPU was available after \d+s/i.test(message)) code = 'zerogpu_timeout';
      if (/Not Found/i.test(message)) code = 'space_unavailable';
      if (/Unrecognized|does not have|model_type|from_pretrained/i.test(message)) code = 'invalid_model_id';
      let displayMessage = message;
      if (code === 'zerogpu_timeout') {
        displayMessage = 'ZeroGPU timeout: this model took too long to initialize or execute within ZeroGPU\'s execution limit. Try a smaller/quantized model or switch GPU provider.';
      }
      throw new GpuError(`ZeroGPU error: ${displayMessage}`, code, status);
    }
    if (!resolved) {
      throw new GpuError('ZeroGPU stream closed without a complete frame.', 'malformed_response', 502);
    }
    return '';
  }

  // run(body, signal, onLine, authToken): { output, stderr }
  async run(body, signal, onLine, authToken) {
    const { modelId, task, prompt, max_new_tokens } = normalizeStructuredRequest(body);
    if (onLine) onLine(`↗ ZeroGPU → ${this.spaceId}  model=${modelId} task=${task} max_new_tokens=${max_new_tokens}`);
    let eventId;
    try {
      eventId = await this._postGenerate(prompt, max_new_tokens, signal, modelId, task, authToken);
    } catch (err) {
      if (err.name === 'AbortError') throw new GpuError('ZeroGPU call was cancelled.', 'cancelled', 499);
      if (err.name === 'TypeError') {
        // fetch() throws TypeError on DNS/network failure.
        throw new GpuError(`Could not reach ZeroGPU Space ${this.spaceId}: ${err.message}`, 'space_unavailable', 504);
      }
      throw err;
    }
    let text = '';
    try {
      text = await this._streamResult(eventId, signal, onLine, authToken);
    } catch (err) {
      if (err.name === 'AbortError') throw new GpuError('ZeroGPU stream was cancelled.', 'cancelled', 499);
      if (err.name === 'TypeError') {
        throw new GpuError(`Lost connection to ZeroGPU stream: ${err.message}`, 'space_unavailable', 504);
      }
      throw err;
    }
    return {
      output: `=== ZeroGPU output (${modelId}, task=${task}) ===\n${text}`,
      stderr: '',
    };
  }
}

// ────────────────────────────────────────────────────────────────────────────
// ModalBackend — wraps the existing Modal forwarding behaviour unchanged.
// ────────────────────────────────────────────────────────────────────────────
class ModalBackend {
  constructor(env) {
    this.env = env;
    this.runUrl = env.MODAL_RUN_URL || '';
    this.streamUrl = env.MODAL_STREAM_URL || '';
    this.authSecret = env.MODAL_AUTH_SECRET || '';
  }

  get kind() { return 'modal'; }
  get state() { return this.runUrl ? 'configured' : 'not-configured'; }

  async status() {
    if (!this.runUrl) {
      return { state: 'not-configured', provider: 'modal', gpuType: this.env.GPU_TYPE || 'A10G' };
    }
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 6000);
      const modalRes = await fetch(this.runUrl, { method: 'GET', signal: controller.signal });
      clearTimeout(timer);
      return {
        state: 'connected',
        provider: 'modal',
        gpuType: this.env.GPU_TYPE || 'A10G',
        httpStatus: modalRes.status,
      };
    } catch (err) {
      return {
        state: 'disconnected',
        provider: 'modal',
        gpuType: this.env.GPU_TYPE || 'A10G',
        error: err.name === 'AbortError' ? 'timeout' : err.message,
      };
    }
  }

  async callModal(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.authSecret || ''}`,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch { data = { error: text.slice(0, 500) }; }
    return { status: res.status, data };
  }

  // Modal expects arbitrary Python code in `{ code }` and runs the notebook
  // verbatim on an A10G.  This is intentionally unchanged from the original
  // behaviour so existing model-cell generation keeps working.
  async run(body, signal) {
    if (!this.runUrl) {
      throw new GpuError('MODAL_RUN_URL is not configured. Deploy trainer.py to Modal then set the env var.', 'modal_unconfigured', 500);
    }
    const code = typeof body === 'string' ? body : (body && body.code);
    if (!code || !String(code).trim()) {
      throw new GpuError('No code provided.', 'bad_request', 400);
    }
    const { status, data } = await this.callModal(this.runUrl, { code });
    if (data && data.error && status >= 500) throw new GpuError(data.error, 'modal_runtime', status);
    return { status, data };
  }

  async streamRun(code, signal, onLine) {
    const streamUrl = this.streamUrl || this.runUrl?.replace('-run-code.modal.run', '-run-code-stream.modal.run') || null;
    if (!streamUrl) {
      throw new GpuError('MODAL_STREAM_URL is not configured.', 'modal_unconfigured', 500);
    }
    const url = `${streamUrl}?code=${encodeURIComponent(code)}`;
    const modalRes = await fetch(url, {
      method: 'GET',
      headers: { Authorization: `Bearer ${this.authSecret || ''}` },
      signal,
    });
    if (!modalRes.ok || !modalRes.body) {
      const text = await modalRes.text().catch(() => '');
      throw new GpuError(`Modal error (${modalRes.status}): ${text.slice(0, 400)}`, 'modal_runtime', modalRes.status);
    }
    const nodeStream = Readable.fromWeb(modalRes.body);
    let buffer = '';
    for await (const chunk of nodeStream) {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        if (onLine) onLine(line);
      }
    }
    if (buffer && onLine) onLine(buffer);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Resolver
// ────────────────────────────────────────────────────────────────────────────
function resolveBackend(env) {
  const provider = String(env.GPU_PROVIDER || 'zerogpu').toLowerCase().trim();
  if (provider === 'modal') return new ModalBackend(env);
  if (provider === 'zerogpu') return new ZeroGPUBackend(env);
  // Unknown provider → explicit error so misconfiguration is visible.
  throw new GpuError(
    `Unknown GPU_PROVIDER "${env.GPU_PROVIDER}". Use "zerogpu" or "modal".`,
    'bad_config',
    500
  );
}

module.exports = {
  GpuError,
  ZeroGPUBackend,
  ModalBackend,
  resolveBackend,
  normalizeStructuredRequest,
  defaultSpaceUrl,
};
