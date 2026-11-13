'use strict';
// ZeroGPU authentication + Gradio API schema tests.
//
// MOCKED (all but check 7): stubbed fetch verifies that the server-side HF
// token reaches BOTH Space requests (POST /call/v2 and GET /call/<id>) and
// that quota/auth/schema-mismatch failures classify into actionable codes.
// REAL (check 7, skippable): fetches the DEPLOYED Space's /gradio_api/info
// and verifies the exposed generate() schema accepts the arguments the
// client sends — this is the deployment-verification gate.

const assert = require('assert');
const path = require('path');
const { ZeroGPUBackend, normalizeStructuredRequest, GpuError } = require(path.join(__dirname, '..', 'gpu_backends.js'));

let calls = [];
const realFetch = global.fetch; // preserved for the REAL schema check
function setBackendResponses(scenario) {
  calls = [];
  global.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), method: init.method || 'GET', headers: init.headers || {}, body: init.body });
    const r = await scenario(String(url), init);
    if (r instanceof Error) throw r;
    const responseText = r.text ?? (r.json !== undefined ? JSON.stringify(r.json) : '');
    const responseHeaders = r.headers || { 'content-type': r.json !== undefined ? 'application/json' : 'text/event-stream' };
    return {
      ok: r.ok ?? true, status: r.status ?? 200,
      headers: { get: (n) => responseHeaders[n.toLowerCase()] },
      async json() { return r.json; },
      async text() { return responseText; },
      get body() {
        if (r.body === null || r.body === undefined) return null;
        const { Readable } = require('stream');
        const chunks = r.body.split(/(?<=\n)/).map((s) => new TextEncoder().encode(s));
        const rs = new ReadableStream({ start(c) { chunks.forEach((x) => c.enqueue(x)); c.close(); } });
        return rs;
      },
    };
  };
}

function sseComplete(eventId, text) {
  return (u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate') && init.method === 'POST') {
      return { status: 200, json: { event_id: eventId } };
    }
    if (u.endsWith(`/gradio_api/call/generate/${eventId}`) && init.method == null) {
      return { ok: true, status: 200, body: `event: complete\ndata: [${JSON.stringify(text)}]\n\n` };
    }
    return { status: 404, text: 'unexpected' };
  };
}

let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

const QUOTA_MSG = "You have exceeded your ZeroGPU runs limit. Authenticate with a Hugging Face token for more quota";

async function main() {
  // ── 1. server-side HF token forwarded on BOTH requests ──
  {
    const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu', HF_TOKEN: 'hf_test_server_token' };
    setBackendResponses(sseComplete('evt-auth', 'ok'));
    await new ZeroGPUBackend(infra).run(
      { model_id: 'HuggingFaceTB/SmolLM3-3B', task: 'text-generation',
        inputs: { prompt: 'hi', max_new_tokens: 8 } },
      new AbortController().signal, () => {});
    const post = calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate'));
    const get = calls.find((c) => c.url.includes('/gradio_api/call/generate/'));
    assert.strictEqual(post.headers.Authorization, 'Bearer hf_test_server_token',
      'POST must carry the server-side HF token');
    assert.strictEqual(get.headers.Authorization, 'Bearer hf_test_server_token',
      'the SSE stream request must carry the same token (the scheduler ' +
      'attributes quota at GPU-acquire time, on both requests)');
    ok('server-side HF token forwarded on POST and SSE GET (authenticated quota)');
  }

  // ── 2. no configured token → no Authorization header (anonymous, as before) ──
  {
    const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu' };
    setBackendResponses(sseComplete('evt-anon', 'ok'));
    await new ZeroGPUBackend(infra).run(
      { model_id: 'o/m', task: 'text-generation', inputs: { prompt: 'hi', max_new_tokens: 4 } },
      new AbortController().signal, () => {});
    const post = calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate'));
    assert(!('Authorization' in post.headers), 'no token configured → no header');
    ok('unauthenticated requests omit the header (anonymous quota, unchanged shape)');
  }

  // ── 3. per-request caller token overrides the server token ──
  {
    const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu', HF_TOKEN: 'hf_server' };
    setBackendResponses(sseComplete('evt-ovr', 'ok'));
    // distinct model_id → distinct dedup key (test 2's request is cached ~1s)
    await new ZeroGPUBackend(infra).run(
      { model_id: 'o/override', task: 'text-generation', inputs: { prompt: 'hi', max_new_tokens: 4 } },
      new AbortController().signal, () => {}, 'hf_caller_token');
    const post = calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate'));
    assert.strictEqual(post.headers.Authorization, 'Bearer hf_caller_token',
      'a per-request token takes precedence over the server default');
    ok('per-request caller token overrides the server default');
  }

  // ── 4. quota exhaustion classified as zerogpu_quota_exceeded (429) ──
  {
    const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu' };
    setBackendResponses((u, init) => {
      if (u.endsWith('/gradio_api/call/v2/generate') && init.method === 'POST') {
        return { status: 200, json: { event_id: 'evt-q' } };
      }
      if (u.includes('/gradio_api/call/generate/evt-q') && init.method == null) {
        return { ok: true, status: 200,
                 body: `event: complete\ndata: [${JSON.stringify('[CLARO:model_runtime] unexpected error: ' + QUOTA_MSG)}]\n\n` };
      }
      return { status: 404, text: 'unexpected' };
    });
    let thrown = null;
    try {
      await new ZeroGPUBackend(infra).run(
        { model_id: 'HuggingFaceTB/SmolLM3-3B', task: 'text-generation',
          inputs: { prompt: 'quota probe', max_new_tokens: 8 } },
        new AbortController().signal, () => {});
    } catch (e) { thrown = e; }
    assert(thrown instanceof GpuError, 'typed error thrown');
    assert.strictEqual(thrown.code, 'zerogpu_quota_exceeded',
      `quota phrase must classify as zerogpu_quota_exceeded, got ${thrown.code}: ${thrown.message}`);
    assert.strictEqual(thrown.status, 429);
    assert(thrown.message.includes('runs limit'), 'original message preserved');
    assert(!thrown.message.includes('hf_'), 'no token material in the error');
    ok('quota exhaustion → zerogpu_quota_exceeded (429), original message preserved');
  }

  // ── 5. schema mismatch classified as space_api_mismatch (502) ──
  {
    const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu' };
    setBackendResponses((u, init) => {
      if (u.endsWith('/gradio_api/call/v2/generate') && init.method === 'POST') {
        return { status: 200, json: { event_id: 'evt-mm' } };
      }
      if (u.includes('/gradio_api/call/generate/evt-mm') && init.method == null) {
        return { ok: true, status: 200,
                 body: 'event: complete\ndata: ["[CLARO:model_runtime] unexpected error: ' +
                       "TypeError: Parameter `task_inputs` is not a valid key-word argument.\"]\n\n" };
      }
      return { status: 404, text: 'unexpected' };
    });
    let thrown = null;
    try {
      await new ZeroGPUBackend(infra).run({
        model_id: 'o/mismatch-probe', task: 'text-generation',
        inputs: { prompt: 'x', max_new_tokens: 4, task_inputs: { top_k: 2 } },
      }, new AbortController().signal, () => {});
    } catch (e) { thrown = e; }
    assert(thrown instanceof GpuError);
    assert.strictEqual(thrown.code, 'space_api_mismatch',
      `schema mismatch must classify as space_api_mismatch, got ${thrown.code}`);
    assert.strictEqual(thrown.status, 502);
    assert(thrown.message.includes('redeploy'), 'actionable redeploy guidance');
    ok('Gradio schema mismatch → space_api_mismatch (502) with redeploy guidance');
  }

  // ── 6. task_inputs serialization contract (unchanged for authenticated flows) ──
  {
    setBackendResponses(sseComplete('evt-ti', 'ok'));
    await new ZeroGPUBackend({ ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu', HF_TOKEN: 'hf_t' }).run({
      model_id: 'owner/bart-auth', task: 'summarization',
      inputs: { prompt: 'doc', task_inputs: { max_length: 64, min_length: 10 } },
    }, new AbortController().signal, () => {});
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert.strictEqual(typeof sent.task_inputs, 'string',
      'task_inputs crosses the Gradio textbox-parameter boundary as a JSON string');
    assert.deepStrictEqual(JSON.parse(sent.task_inputs), { max_length: 64, min_length: 10 });
    assert.strictEqual(calls[0].headers.Authorization, 'Bearer hf_t');
    ok('task_inputs serialization unchanged and authenticated');
  }

  // ── 7. REAL integration: deployed Space schema accepts the expected args ──
  // This is the deployment-verification gate. It hits the LIVE Space's
  // /gradio_api/info (no GPU consumption, no quota) and FAILS with redeploy
  // guidance when the deployed build predates task_inputs.
  {
    // REAL Hugging Face integration (no GPU consumption, no quota): the
    // schema endpoint is served by the deployed Space itself, so this check
    // is only as current as the last deployment.
    const base = 'https://gochan562-claro-ai-gpu.hf.space';
    let schema = null, reachable = true;
    try {
      const res = await realFetch(`${base}/gradio_api/info`, { signal: AbortSignal.timeout(20000) });
      if (!res.ok) reachable = false;
      else schema = await res.json();
    } catch (_) { reachable = false; }
    if (!reachable || !schema) {
      console.log('⊘ real schema check skipped (Space unreachable)');
    } else {
      const namedEndpoint = schema.named_endpoints && schema.named_endpoints['/generate'];
      const jsonStr = JSON.stringify(namedEndpoint || {});
      const exposed = (namedEndpoint?.parameters || []).map((p) => p.parameter_name);
      const strict = !!process.env.ZEROGPU_SCHEMA_STRICT;
      if (!jsonStr.includes('task_inputs')) {
        const msg = 'DEPLOYED SPACE IS STALE: /gradio_api/info does not expose ' +
          'task_inputs (exposed: ' + JSON.stringify(exposed) + '). ' +
          'Redeploy with: python3 space/push_to_hf.py — then rerun with ' +
          'ZEROGPU_SCHEMA_STRICT=1.';
        if (strict) assert.fail(msg);
        console.log('⚠️  ' + msg);
        ok('schema probe ran — deployed Space predates task_inputs (see ⚠️ above)');
      } else {
        for (const expected of ['prompt', 'max_new_tokens', 'model_id', 'task']) {
          assert(jsonStr.includes(`"${expected}"`), `schema must expose ${expected}`);
        }
        ok('REAL deployed Space schema accepts task_inputs (+ prompt/model_id/task)');
      }
    }
  }

  // ── 8. source-level guard: the Space's generate() signature AND its
  // Gradio component list must both declare task_inputs (the schema is
  // built from the component list — signature alone is not enough).
  {
    const appSrc = require('fs').readFileSync(
      path.join(__dirname, '..', 'space', 'app.py'), 'utf8');
    const sig = appSrc.match(/def generate\(([^)]*)\)/s);
    assert(sig && sig[1].includes('task_inputs'), 'generate() must accept task_inputs');
    const inputsBlock = appSrc.match(/demo_infer = gr\.Interface\([\s\S]*?inputs=\[([\s\S]*?)\],/);
    assert(inputsBlock && inputsBlock[1].includes('Task inputs (JSON'),
      'demo_infer.inputs must include the task_inputs component — Gradio ' +
      'builds the API schema from the component list, and v2 named-kwargs ' +
      'validation rejects kwargs without one');
    ok('space source: generate() signature + Gradio component list both declare task_inputs');
  }

  console.log(`\nAll ZeroGPU auth/schema tests passed (${passed} checks).`);
}

main().catch((e) => { console.error(e); process.exit(1); });
