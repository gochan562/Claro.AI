'use strict';

// End-to-end test for the ZeroGPU provider integration.
//
// Runs WITHOUT the real Hugging Face Space (which subjects us to ZeroGPU's
// anonymous quota).  We stub `global.fetch` with an httplib-style mock that
// answers the two endpoints we hit:
//   1. POST {base}/gradio_api/call/v2/generate           → {"event_id": "..."}
//   2. GET  {base}/gradio_api/call/generate/<event_id>    → SSE stream ending
//                                                            in `event: complete\ndata: ["<text>"]`
//
// Then we drive the same ZeroGPUBackend.run() that the express handler uses,
// assert the structured request maps correctly to {prompt, max_new_tokens},
// assert the SSE is parsed correctly, and assert error codes for the various
// failure modes the Space can return (quota, Not Found, malformed, …).

const assert = require('assert');
const path = require('path');
const { ZeroGPUBackend, normalizeStructuredRequest, GpuError } = require(path.join(__dirname, '..', 'gpu_backends.js'));

let calls = [];
function setBackendResponses(scenario) {
  calls = [];
  global.fetch = async (url, init = {}) => {
    const u = String(url);
    calls.push({ url: u, method: init.method || 'GET', body: init.body });
    const r = scenario(u, init);
    if (r instanceof Error) throw r;
    const responseText = r.text ?? (r.json !== undefined ? JSON.stringify(r.json) : '');
    const responseHeaders = r.headers || { 'content-type': r.json !== undefined ? 'application/json' : 'text/event-stream' };
    return {
      ok: r.ok ?? true,
      status: r.status ?? 200,
      headers: {
        get: (name) => responseHeaders[name.toLowerCase()],
      },
      async json() { return r.json; },
      async text() { return responseText; },
      get body() {
        if (r.body === null || r.body === undefined) return null;
        const { Readable } = require('stream');
        const chunks = r.body.split(/(?<=\n)/).map((s) => new TextEncoder().encode(s));
        const rs = new ReadableStream({
          start(controller) { chunks.forEach((c) => controller.enqueue(c)); controller.close(); },
        });
        return rs;
      },
    };
  };
}

async function run() {
  const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu' };

  // ─── 1. Normal text-generation call ├──
  setBackendResponses((u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate') && init.method === 'POST') {
      const b = JSON.parse(init.body);
      assert.strictEqual(typeof b.prompt, 'string');
      assert.strictEqual(typeof b.max_new_tokens, 'number');
      return { status: 200, json: { event_id: 'evt-1' } };
    }
    if (u.endsWith('/gradio_api/call/generate/evt-1') && init.method == null) {
      return {
        ok: true,
        status: 200,
        body:
          'event: complete\ndata: ["Once upon a time, there was a king who loved horses."]\n\n',
      };
    }
    return { status: 404, text: 'unexpected' };
  });
  const backend = new ZeroGPUBackend(infra);
  const result = await backend.run(
    { model_id: 'HuggingFaceTB/SmolLM3-3B', task: 'text-generation', inputs: { prompt: 'Once upon a time', max_new_tokens: 40 } },
    new AbortController().signal,
    () => {}
  );
  assert(result.output && result.output.includes('Once upon a time'), 'output should be a string');
  console.log('✓ test 1: normal text-generation succeeds');

  // verify request mapping — the Space now consumes model_id+task so the
  // backend actually uses the incoming structured request instead of the
  // hard-coded default.
  assert.deepStrictEqual(
    JSON.parse(calls[0].body),
    {
      prompt: 'Once upon a time',
      max_new_tokens: 40,
      model_id: 'HuggingFaceTB/SmolLM3-3B',
      task: 'text-generation',
    }
  );
  console.log('✓ test 1b: structured request maps to {prompt, max_new_tokens, model_id, task}');

  // ─── 2. Quota exhausted ├──
  setBackendResponses((u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate')) return { status: 200, json: { event_id: 'evt-q' } };
    if (u.endsWith('/gradio_api/call/generate/evt-q')) {
      return {
        ok: true,
        status: 200,
        body: 'event: error\ndata: {"error": "You have exceeded your ZeroGPU runs limit. Authenticate with a Hugging Face token for more quota - https://huggingface.co/settings/tokens", "title": "ZeroGPU quota exceeded"}\n\n',
      };
    }
    return { status: 404, text: '?' };
  });
  let err = null;
  try {
    await new ZeroGPUBackend(infra).run(
      { model_id: 'HuggingFaceTB/SmolLM3-3B', task: 'text-generation', inputs: { prompt: 'x', max_new_tokens: 5 } },
      new AbortController().signal
    );
  } catch (e) { err = e; }
  assert(err, 'expected an error');
  assert(err instanceof GpuError, 'should be GpuError');
  assert.strictEqual(err.code, 'zerogpu_quota', 'code should be zerogpu_quota');
  console.log('✓ test 2: ZeroGPU quota error mapped to code=zerogpu_quota');

  // ─── 3. Space unavailable (404 on POST) ├──
  setBackendResponses(() => ({ status: 404, text: '404: Not Found' }));
  err = null;
  try {
    await new ZeroGPUBackend(infra).run(
      { model_id: 'm/m', task: 'text-generation', inputs: { prompt: 'x', max_new_tokens: 5 } },
      new AbortController().signal
    );
  } catch (e) { err = e; }
  assert(err && err.code === 'space_unavailable', 'code should be space_unavailable');
  console.log('✓ test 3: Space unavailable → code=space_unavailable');

  // ─── 4. Invalid model id (validator level) ├──
  err = null;
  try {
    normalizeStructuredRequest({ model_id: 'bad', task: 'text-generation', inputs: { prompt: 'x' } });
  } catch (e) { err = e; }
  assert(err && err.code === 'invalid_model_id');
  console.log('✓ test 4: invalid model_id raised before any network call');

  // ─── 5. Missing prompt ├──
  err = null;
  try {
    normalizeStructuredRequest({ model_id: 'm/m', task: 'text-generation', inputs: {} });
  } catch (e) { err = e; }
  assert(err && err.code === 'bad_request');
  console.log('✓ test 5: missing prompt → bad_request');

  // ─── 6. max_new_tokens clamp + out-of-range error from Space ├──
  const norm = normalizeStructuredRequest({
    model_id: 'm/m', task: 'text-generation',
    inputs: { prompt: 'x', max_new_tokens: 99999 },
  });
  assert.strictEqual(norm.max_new_tokens, 500, 'over-limit value should be soft-capped to 500');
  console.log('✓ test 6: max_new_tokens=99999 is clamped to 500 (not silently rejected)');

  // Gradio can also reject with the slider-max error; surface it.
  setBackendResponses((u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate')) return { status: 200, json: { event_id: 'evt-999' } };
    if (u.endsWith('/gradio_api/call/generate/evt-999')) {
      return {
        ok: true,
        status: 200,
        body: 'event: error\ndata: {"error": "Value 999 is greater than maximum value 500.", "title": "Error"}\n\n',
      };
    }
    return { status: 404, text: '?' };
  });
  err = null;
  try {
    await new ZeroGPUBackend(infra).run(
      { model_id: 'm/m', task: 'text-generation', inputs: { prompt: 'x', max_new_tokens: 999 } },
      new AbortController().signal
    );
  } catch (e) { err = e; }
  assert(err && err instanceof GpuError);
  assert(err.message.includes('greater than maximum value 500'));
  console.log('✓ test 7: Gradio validation error surfaced');

  // ─── 8. Malformed response (stream ends without complete frame) ├──
  setBackendResponses((u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate')) return { status: 200, json: { event_id: 'evt-mal' } };
    if (u.endsWith('/gradio_api/call/generate/evt-mal')) return { ok: true, status: 200, body: 'event: heartbeat\ndata: ...\n\n' };
    return { status: 404, text: '?' };
  });
  err = null;
  try {
    await new ZeroGPUBackend(infra).run(
      { model_id: 'm/m', task: 'text-generation', inputs: { prompt: 'x', max_new_tokens: 5 } },
      new AbortController().signal
    );
  } catch (e) { err = e; }
  assert(err && err.code === 'malformed_response');
  console.log('✓ test 8: stream without complete frame → malformed_response');

  // ─── 9. Status probe discovers the Space via /gradio_api/info ├──
  setBackendResponses((u) => {
    if (u.endsWith('/gradio_api/info')) return { ok: true, status: 200, text: '{}' };
    if (u.endsWith('/gradio_api/config')) return { ok: false, status: 404, text: '404' };
    return { ok: false, status: 404, text: '?' };
  });
  const st = await new ZeroGPUBackend(infra).status();
  assert.strictEqual(st.state, 'connected');
  assert.strictEqual(st.space, 'Gochan562/claro_ai_gpu');
  console.log('✓ test 9: status() probes /gradio_api/info and reports connected');

  // ─── 10. Space returns [CLARO:invalid_model_id] inside the success frame ├──
  // The Space, after becoming a generic backend, surfaces model-load failures
  // as structured strings in the `event: complete` data rather than falling
  // back to SmolLM3-3B.  The local provider must convert that into a typed
  // GpuError so the dashboard can show a real error instead of fake text.
  setBackendResponses((u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate')) return { status: 200, json: { event_id: 'evt-claro' } };
    if (u.endsWith('/gradio_api/call/generate/evt-claro')) {
      return {
        ok: true, status: 200,
        body: 'event: complete\ndata: "[CLARO:model_load_error] Could not read AutoConfig for some/bad-repo"\n\n',
      };
    }
    return { status: 404, text: '?' };
  });
  err = null;
  try {
    await new ZeroGPUBackend(infra).run(
      { model_id: 'some/bad-repo', task: 'text-generation', inputs: { prompt: 'x', max_new_tokens: 5 } },
      new AbortController().signal
    );
  } catch (e) { err = e; }
  assert(err instanceof GpuError, 'should be GpuError');
  assert.strictEqual(err.code, 'model_load_error', 'code parsed from [CLARO:...] prefix');
  assert(err.message.includes('Could not read AutoConfig for some/bad-repo'));
  console.log('✓ test 10: structured [CLARO:<code>] error from complete frame mapped to typed code');

  delete global.fetch;
  console.log('\nAll ZeroGPU integration tests passed.');
}

run().catch((e) => {
  console.error('TEST FAILED:', e);
  process.exit(1);
});
