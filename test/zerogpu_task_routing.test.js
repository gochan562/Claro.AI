'use strict';
// Task-aware routing tests (Node layer): task_inputs validation, wire
// format, and that the frontend's task→internal mapping is consistent.

const assert = require('assert');
const path = require('path');
const { ZeroGPUBackend, normalizeStructuredRequest, GpuError } = require(path.join(__dirname, '..', 'gpu_backends.js'));

let calls = [];
function setBackendResponses(scenario, { keepCalls = false } = {}) {
  if (!keepCalls) calls = [];
  global.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), method: init.method || 'GET', body: init.body });
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
      return { ok: true, status: 200, body: `event: complete\ndata: ["${text}"]\n\n` };
    }
    return { status: 404, text: 'unexpected' };
  };
}

let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

async function main() {
  const infra = { ZEROGPU_SPACE: 'Gochan562/claro_ai_gpu' };

  // ── 1. task_inputs gather + validate ──
  {
    const n = normalizeStructuredRequest({
      model_id: 'owner/bart', task: 'summarization',
      inputs: { prompt: 'document text', task_inputs: { max_length: 64, min_length: 10 } },
    });
    assert.deepStrictEqual(n.taskInputs, { max_length: 64, min_length: 10 });
    // top-level convenience keys gathered too
    const n2 = normalizeStructuredRequest({
      model_id: 'owner/qa', task: 'question-answering',
      inputs: { prompt: 'q', question: 'Where?', context: 'The tower is in Paris.' },
    });
    assert.deepStrictEqual(n2.taskInputs, { question: 'Where?', context: 'The tower is in Paris.' });
    // unknown key rejected
    let err = null;
    try {
      normalizeStructuredRequest({ model_id: 'o/m', task: 't',
        inputs: { prompt: 'x', task_inputs: { hacker_field: 1 } } });
    } catch (e) { err = e; }
    assert(err instanceof GpuError && err.code === 'bad_request');
    // non-scalar values rejected
    err = null;
    try {
      normalizeStructuredRequest({ model_id: 'o/m', task: 't',
        inputs: { prompt: 'x', task_inputs: { context: { nested: true } } } });
    } catch (e) { err = e; }
    assert(err instanceof GpuError && err.code === 'bad_request');
    // out-of-range top_k rejected
    err = null;
    try {
      normalizeStructuredRequest({ model_id: 'o/m', task: 'text-classification',
        inputs: { prompt: 'x', task_inputs: { top_k: 99 } } });
    } catch (e) { err = e; }
    assert(err instanceof GpuError && err.code === 'bad_request');
    ok('task_inputs gathered, allowlisted, and validated (unknown/nested/out-of-range rejected)');
  }

  // ── 2. summarization: task_inputs reach the wire as JSON string ──
  setBackendResponses(sseComplete('evt-sum', '=== Summary ==='));
  {
    await new ZeroGPUBackend(infra).run({
      model_id: 'owner/bart', task: 'summarization',
      inputs: { prompt: 'The transformer architecture revolutionized NLP.',
                task_inputs: { max_length: 64, min_length: 10 } },
    }, new AbortController().signal, () => {});
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert.strictEqual(sent.prompt, 'The transformer architecture revolutionized NLP.',
      'source document is the prompt — not a generic default');
    assert.strictEqual(sent.task, 'summarization');
    assert.deepStrictEqual(JSON.parse(sent.task_inputs), { max_length: 64, min_length: 10 });
    ok('summarization request carries the source document + structured task_inputs');
  }

  // ── 3. QA: question + context both forwarded ──
  setBackendResponses(sseComplete('evt-qa', '=== Question answering ==='));
  {
    await new ZeroGPUBackend(infra).run({
      model_id: 'owner/bert-squad', task: 'question-answering',
      inputs: { prompt: 'Where is the Eiffel Tower?',
                task_inputs: { question: 'Where is the Eiffel Tower?',
                               context: 'The Eiffel Tower is located in Paris, France.' } },
    }, new AbortController().signal, () => {});
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    const ti = JSON.parse(sent.task_inputs);
    assert.strictEqual(ti.question, 'Where is the Eiffel Tower?');
    assert.strictEqual(ti.context, 'The Eiffel Tower is located in Paris, France.');
    ok('question-answering forwards question AND context');
  }

  // ── 4. sentence-similarity: two separate texts ──
  setBackendResponses(sseComplete('evt-sim', '=== Sentence similarity ==='));
  {
    await new ZeroGPUBackend(infra).run({
      model_id: 'owner/embed', task: 'sentence-similarity',
      inputs: { task_inputs: { text1: 'first sentence', text2: 'second sentence' } },
    }, new AbortController().signal, () => {});
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    const ti = JSON.parse(sent.task_inputs);
    assert.strictEqual(ti.text1, 'first sentence');
    assert.strictEqual(ti.text2, 'second sentence');
    ok('sentence-similarity forwards two separate texts');
  }

  // ── 5. no task_inputs → field omitted (old Spaces unaffected) ──
  setBackendResponses(sseComplete('evt-plain', 'ok'));
  {
    await new ZeroGPUBackend(infra).run({
      model_id: 'owner/gpt', task: 'text-generation',
      inputs: { prompt: 'user typed this', max_new_tokens: 32 },
    }, new AbortController().signal, () => {});
    const sent = JSON.parse(calls.find((c) => c.url.endsWith('/gradio_api/call/v2/generate')).body);
    assert(!('task_inputs' in sent), 'no task_inputs field when none provided');
    assert.strictEqual(sent.prompt, 'user typed this', 'user prompt preserved');
    ok('plain generation request unchanged; user prompt passed verbatim');
  }

  // ── 6. dedup separates identical prompts with different task inputs ──
  setBackendResponses(sseComplete('evt-d1', 'a'));
  {
    const backend = new ZeroGPUBackend(infra);
    await backend.run({ model_id: 'o/t5', task: 'translation',
      inputs: { prompt: 'same text', task_inputs: { target_language: 'de' } } },
      new AbortController().signal, () => {});
    setBackendResponses(sseComplete('evt-d2', 'b'), { keepCalls: true });
    // the successful-result cache lives ~1s; outwait it so the second call
    // reaches the wire and the dedup KEY (which includes task_inputs) is
    // what actually differentiates the requests
    await new Promise((r) => setTimeout(r, 1100));
    await backend.run({ model_id: 'o/t5', task: 'translation',
      inputs: { prompt: 'same text', task_inputs: { target_language: 'fr' } } },
      new AbortController().signal, () => {});
    const posts = calls.filter((c) => c.url.endsWith('/gradio_api/call/v2/generate'));
    assert.strictEqual(posts.length, 2, 'different task_inputs must not dedupe together');
    ok('dedup key includes task_inputs (different targets → two requests)');
  }

  // ── 7. frontend mapping consistency (source-level guard) ──
  {
    const appSrc = require('fs').readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
    const m = appSrc.match(/const HF_TASK_TO_INTERNAL = \{([\s\S]*?)\};/);
    assert(m, 'HF_TASK_TO_INTERNAL table exists');
    for (const hf of ['text-generation', 'text2text-generation', 'summarization',
                      'feature-extraction', 'sentence-similarity', 'text-classification',
                      'question-answering', 'translation']) {
      assert(m[1].includes(`'${hf}'`), `mapping covers ${hf}`);
    }
    // no generic default prompt in the ZeroGPU transport
    const parseFn = appSrc.slice(appSrc.indexOf('function parseInferenceInputs'),
                                appSrc.indexOf('function parsePromptAndMaxTokens') === -1
                                  ? appSrc.indexOf('async function runCellViaZeroGPU')
                                  : appSrc.indexOf('function parsePromptAndMaxTokens'));
    assert(!parseFn.includes('Once upon a time'), 'no hardcoded generic prompt in the parser');
    ok('frontend mapping table complete; transport has no generic default prompt');
  }

  console.log(`\nAll task-routing backend tests passed (${passed} checks).`);
}

main().catch((e) => { console.error(e); process.exit(1); });
