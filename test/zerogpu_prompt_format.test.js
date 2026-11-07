'use strict';
// prompt_format pass-through tests for the ZeroGPU backend.
//
// prompt_format = auto | chat | raw:
//   - normalizeStructuredRequest defaults to 'auto' when omitted
//     (backward compatibility with requests that never send it),
//   - accepts explicit chat/raw and rejects invalid values,
//   - run() forwards explicit chat/raw to the Space POST body,
//   - 'auto' is OMITTED from the wire body so older Spaces (without the
//     generate(prompt_format=...) parameter) never see an unknown kwarg.
// Uses the same fetch-stub harness as zerogpu_provider.test.js — no real
// Space, no GPU.

const assert = require('assert');
const path = require('path');
const { ZeroGPUBackend, normalizeStructuredRequest, GpuError } = require(path.join(__dirname, '..', 'gpu_backends.js'));

let calls = [];
function setBackendResponses(scenario) {
  calls = [];
  global.fetch = async (url, init = {}) => {
    const u = String(url);
    calls.push({ url: u, method: init.method || 'GET', body: init.body });
    const r = await scenario(u, init);
    if (r instanceof Error) throw r;
    const responseText = r.text ?? (r.json !== undefined ? JSON.stringify(r.json) : '');
    const responseHeaders = r.headers || { 'content-type': r.json !== undefined ? 'application/json' : 'text/event-stream' };
    return {
      ok: r.ok ?? true,
      status: r.status ?? 200,
      headers: { get: (name) => responseHeaders[name.toLowerCase()] },
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

function sseComplete(eventId, text) {
  return (u, init) => {
    if (u.endsWith('/gradio_api/call/v2/generate') && init.method === 'POST') {
      return { status: 200, json: { event_id: eventId } };
    }
    if (u.endsWith(`/gradio_api/call/generate/${eventId}`) && init.method == null) {
      return { ok: true, status: 200, body: `event: complete\ndata: ["${text}"]\n\n` };
    }
    return { status: 404, text: 'unexpected' };
  };
}

async function main() {
  const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu' };

  // ── 1. normalizeStructuredRequest: default + validation ──
  {
    const n = normalizeStructuredRequest({
      model_id: 'owner/model', task: 'text-generation',
      inputs: { prompt: 'hi', max_new_tokens: 16 },
    });
    assert.strictEqual(n.prompt_format, 'auto', 'omitted prompt_format defaults to auto (backward compat)');

    const n2 = normalizeStructuredRequest({
      model_id: 'owner/model', task: 'text-generation',
      inputs: { prompt: 'hi', max_new_tokens: 16, prompt_format: 'CHAT' },
    });
    assert.strictEqual(n2.prompt_format, 'chat', 'case-insensitive normalization');

    // whitespace/case are normalized (matching the Space's strip().lower());
    // genuinely invalid values throw.
    const nSpace = normalizeStructuredRequest({
      model_id: 'owner/model', task: 'text-generation',
      inputs: { prompt: 'hi', max_new_tokens: 16, prompt_format: ' chat ' },
    });
    assert.strictEqual(nSpace.prompt_format, 'chat', 'whitespace normalized');
    for (const bad of ['template', 'chatty', 5, {}]) {
      let threw = null;
      try {
        normalizeStructuredRequest({
          model_id: 'owner/model', task: 'text-generation',
          inputs: { prompt: 'hi', max_new_tokens: 16, prompt_format: bad },
        });
      } catch (e) { threw = e; }
      assert(threw instanceof GpuError && threw.code === 'bad_request',
        `invalid prompt_format ${JSON.stringify(bad)} must throw bad_request`);
    }
    // body-level prompt_format accepted too (not only inputs.*)
    const n3 = normalizeStructuredRequest({
      model_id: 'owner/model', task: 'text-generation', prompt_format: 'raw',
      inputs: { prompt: 'hi', max_new_tokens: 16 },
    });
    assert.strictEqual(n3.prompt_format, 'raw');
    console.log('✓ normalize: default auto, chat/raw accepted, invalid rejected, body-level accepted');
  }

  // ── 2. chat format is forwarded to the Space POST body ──
  setBackendResponses(sseComplete('evt-chat', '4'));
  {
    const backend = new ZeroGPUBackend(infra);
    const result = await backend.run(
      { model_id: 'MuyaoYuan/LoCo-7B', task: 'text-generation',
        inputs: { prompt: 'What is 2 + 2? Give a short answer.', max_new_tokens: 64, prompt_format: 'chat' } },
      new AbortController().signal, () => {});
    assert(result.output, 'output present');
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert.strictEqual(sent.prompt_format, 'chat', 'explicit chat reaches the wire body');
    assert.strictEqual(sent.model_id, 'MuyaoYuan/LoCo-7B');
    assert.strictEqual(sent.prompt, 'What is 2 + 2? Give a short answer.');
    console.log('✓ run() forwards prompt_format=chat in the POST body');
  }

  // ── 3. raw format forwarded; auto omitted (old-Space compat) ──
  setBackendResponses(sseComplete('evt-raw', 'ok'));
  {
    await new ZeroGPUBackend(infra).run(
      { model_id: 'owner/model', task: 'text-generation',
        inputs: { prompt: 'hi', max_new_tokens: 8, prompt_format: 'raw' } },
      new AbortController().signal, () => {});
    const sentRaw = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert.strictEqual(sentRaw.prompt_format, 'raw', 'raw forwarded');

    setBackendResponses(sseComplete('evt-auto', 'ok'));
    await new ZeroGPUBackend(infra).run(
      { model_id: 'owner/model', task: 'text-generation',
        inputs: { prompt: 'hi', max_new_tokens: 8, prompt_format: 'auto' } },
      new AbortController().signal, () => {});
    const sentAuto = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert(!('prompt_format' in sentAuto),
      'auto must be OMITTED from the body (older Spaces reject unknown kwargs)');
    console.log('✓ raw forwarded; auto omitted from the wire body (backward compat)');
  }

  // ── 4. requests WITHOUT prompt_format: unchanged wire body, default auto ──
  setBackendResponses(sseComplete('evt-legacy', 'ok'));
  {
    const backend = new ZeroGPUBackend(infra);
    await backend.run(
      { model_id: 'owner/model', task: 'text-generation',
        inputs: { prompt: 'legacy prompt', max_new_tokens: 12 } },
      new AbortController().signal, () => {});
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert.deepStrictEqual(Object.keys(sent).sort(),
      ['max_new_tokens', 'model_id', 'prompt', 'task'],
      'legacy request shape unchanged');
    // dedup: same-shape legacy requests still dedupe (key includes format)
    console.log('✓ legacy requests (no prompt_format) keep the exact old wire shape');
  }

  console.log('\nAll prompt_format backend tests passed (4 checks).');
}

main().catch((e) => { console.error(e); process.exit(1); });
