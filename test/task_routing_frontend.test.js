'use strict';
// Task-aware frontend tests: HF-task → internal-task mapping, inference-cell
// input parsing, per-task request building, and the absence of generic
// default prompts on the ZeroGPU path.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');
const HTML_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard.html'), 'utf8');

function makeWindow() {
  const start = HTML_SRC.indexOf('<div id="model-picker-modal"');
  const pickerHtml = HTML_SRC.slice(start, HTML_SRC.indexOf('</main>', start));
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
      <div class="notebook-toolbar"></div>
      <div id="notebook-cells"></div>
      <div id="gpu-terminal"></div>
      ${pickerHtml}
    </body></html>`,
    { url: 'http://localhost:5000/', runScripts: 'outside-only' }
  );
  const w = dom.window;
  w.CodeMirror = function () { return { getValue: () => '', setValue: () => {}, on: () => {}, refresh: () => {}, getWrapperElement: () => w.document.createElement('div') }; };
  w.marked = { parse: (s) => `<p>${s}</p>` };
  w.DOMPurify = { sanitize: (s) => s };
  w.PyodideLocal = { run: async () => ({ ok: true, stdout: '', stderr: '', error: null }) };
  w.__fetchLog = [];
  w.prompt = () => null; w.confirm = () => false;
  try { w.eval(APP_SRC); } catch (e) { console.log(`[makeWindow] eval threw: ${e.message}`); }
  console.log(`[makeWindow] createNotebook=${typeof w.createNotebook} openStorageManager=${typeof w.openStorageManager} resolveInternalTask=${typeof w.resolveInternalTask}`);
  return { dom, w };
}

let passed = 0, failed = 0;
function test(name, fn) {
  return Promise.resolve().then(fn)
    .then(() => { console.log(`✓ ${name}`); passed++; })
    .catch((e) => { console.error(`✗ ${name}: ${e.message}`); if (name.startsWith('6')) console.error(e.stack.split('\n').slice(1,4).join('\n')); failed++; });
}

async function main() {
  console.log('=== Task-Aware Routing Frontend Tests ===\n');
  const { w } = makeWindow();

  await test('1. HF pipeline tags map to the internal task table', () => {
    const cases = [
      ['text-generation', 'text_generation'],
      ['text2text-generation', 'text_generation'],
      ['summarization', 'summarization'],
      ['feature-extraction', 'embeddings'],
      ['sentence-similarity', 'sentence_similarity'],
      ['text-classification', 'classification'],
      ['question-answering', 'question_answering'],
      ['translation', 'translation'],
    ];
    for (const [hf, internal] of cases) {
      const r = w.resolveInternalTask(hf);
      assert.strictEqual(r.internal, internal, `${hf} → ${internal}`);
      assert.strictEqual(r.supported, true, `${hf} supported`);
    }
    const unknown = w.resolveInternalTask('fill-mask');
    assert.strictEqual(unknown.internal, null);
    assert.strictEqual(unknown.supported, false);
  });

  await test('2. inference-cell parsing extracts task-specific fields', () => {
    const f = w.parseInferenceInputs([
      'question = "Where is the Eiffel Tower?"',
      'context  = "The Eiffel Tower is located in Paris, France."',
    ].join('\n'));
    assert.strictEqual(f.question, 'Where is the Eiffel Tower?');
    assert.strictEqual(f.context, 'The Eiffel Tower is located in Paris, France.');

    const f2 = w.parseInferenceInputs('texts = ["I love this", "I hate this"]');
    assert.strictEqual(JSON.stringify(f2.texts), JSON.stringify(['I love this', 'I hate this']));

    const f3 = w.parseInferenceInputs('input_text = "The transformer revolutionized NLP."\nmax_length = 64\nmin_length = 10');
    assert.strictEqual(f3.input_text, 'The transformer revolutionized NLP.');
    assert.strictEqual(f3.max_length, 64);
    assert.strictEqual(f3.min_length, 10);

    const f4 = w.parseInferenceInputs('text1 = "first"\ntext2 = "second"');
    assert.strictEqual(f4.text1, 'first');
    assert.strictEqual(f4.text2, 'second');
  });

  await test('3. per-task request building uses the right fields', () => {
    // generation: user prompt verbatim
    const gen = w.buildTaskAwareInputs('text_generation', { prompt: 'What is 2 + 2?', max_new_tokens: 64, prompt_format: 'chat' });
    assert.strictEqual(gen.inputs.prompt, 'What is 2 + 2?');
    assert.strictEqual(gen.inputs.prompt_format, 'chat');

    // summarization: document + length bounds
    const sum = w.buildTaskAwareInputs('summarization', { input_text: 'long document', max_length: 64, min_length: 10 });
    assert.strictEqual(sum.inputs.prompt, 'long document');
    assert.strictEqual(JSON.stringify(sum.inputs.task_inputs), JSON.stringify({ max_length: 64, min_length: 10 }));

    // sentence similarity: two texts
    const sim = w.buildTaskAwareInputs('sentence_similarity', { text1: 'a', text2: 'b' });
    assert.strictEqual(JSON.stringify(sim.inputs.task_inputs), JSON.stringify({ text1: 'a', text2: 'b' }));

    // QA: question + context
    const qa = w.buildTaskAwareInputs('question_answering', { question: 'q?', context: 'c' });
    assert.strictEqual(qa.inputs.prompt, 'q?');
    assert.strictEqual(JSON.stringify(qa.inputs.task_inputs), JSON.stringify({ question: 'q?', context: 'c' }));

    // translation: source + target language
    const tr = w.buildTaskAwareInputs('translation', { input_text: 'hello', target_language: 'fr' });
    assert.strictEqual(tr.inputs.prompt, 'hello');
    assert.strictEqual(JSON.stringify(tr.inputs.task_inputs), JSON.stringify({ target_language: 'fr' }));

    // classification: multi-text joined to lines
    const cls = w.buildTaskAwareInputs('classification', { texts: ['a', 'b'], top_k: 2 });
    assert.strictEqual(cls.inputs.prompt, 'a\nb');
    assert.strictEqual(JSON.stringify(cls.inputs.task_inputs), JSON.stringify({ top_k: 2 }));
  });

  await test('4. missing required fields produce guidance errors, not generic prompts', () => {
    const e1 = w.buildTaskAwareInputs('summarization', {});
    assert(e1.error && e1.error.includes('input_text'), 'summarization guidance');
    const e2 = w.buildTaskAwareInputs('sentence_similarity', { text1: 'only one' });
    assert(e2.error && e2.error.includes('text2'), 'similarity guidance');
    const e3 = w.buildTaskAwareInputs('question_answering', { question: 'q' });
    assert(e3.error && e3.error.includes('context'), 'QA guidance');
    const e4 = w.buildTaskAwareInputs('text_generation', {});
    assert(e4.error && e4.error.includes('prompt'), 'generation guidance');
    assert(!e4.inputs, 'no inputs built on validation failure');
  });

  await test('5. unsupported tasks are surfaced, never routed to text generation', () => {
    for (const hf of ['fill-mask', 'token-classification', 'zero-shot-classification',
                      'text-to-image', 'unknown-future-task']) {
      const r = w.resolveInternalTask(hf);
      assert.strictEqual(r.supported, false, `${hf} unsupported`);
    }
    // source-level guarantee: the ZeroGPU transport has no hardcoded story prompt
    const transport = APP_SRC.slice(APP_SRC.indexOf('async function runCellViaZeroGPU'),
                                    APP_SRC.indexOf('const STREAM_CODE_LENGTH_LIMIT'));
    assert(!transport.includes('Once upon a time'), 'transport never injects a generic prompt');
  });

  await test('6. runCellViaZeroGPU sends task-shaped bodies (integration via stubbed fetch)', async () => {
    const { w: w2 } = makeWindow();
    const bodies = [];
    w2.fetch = async (url, init = {}) => {
      const u = String(url);
      if (u.includes('/api/zerogpu-run')) {
        bodies.push(JSON.parse(init.body));
        return { ok: true, status: 200, text: async () => JSON.stringify({ output: '=== Summary ===\ndone' }) };
      }
      if (u.includes('/api/gpu-config')) return { ok: true, json: async () => ({ provider: 'zerogpu', zerogpu: { space: 'x' } }) };
      return { ok: true, status: 200, text: async () => '{}', json: async () => ({}) };
    };
    // build a notebook with a summarization model + inference cell
    const nbId = w2.createNotebook('Task Routing');
    w2.setActiveNotebook(nbId);
    w2.addCell('model');
    w2.addCell('code');
    const nb = w2.getActiveNotebook();
    const modelCell = nb.cells.find((c) => c.type === 'model');
    const inferCell = nb.cells.find((c) => c.type === 'code');
    modelCell.modelInfo = { id: 'owner/bart', task: 'summarization', role: 'model' };
    inferCell.modelInfo = { role: 'inference', loaderCellId: modelCell.id };
    inferCell.content = [
      '# Summarize',
      'input_text = "The transformer architecture revolutionized NLP by enabling parallel processing."',
      'max_length = 64',
      'min_length = 10',
    ].join('\n');

    await w2.runCellViaZeroGPU(nb, inferCell);
    assert.strictEqual(bodies.length, 1, 'exactly one zerogpu-run call');
    const b = bodies[0];
    assert.strictEqual(b.model_id, 'owner/bart');
    assert.strictEqual(b.task, 'summarization');
    assert.strictEqual(b.inputs.prompt,
      'The transformer architecture revolutionized NLP by enabling parallel processing.',
      'the SOURCE DOCUMENT is sent — not a generic prompt');
    assert.strictEqual(JSON.stringify(b.inputs.task_inputs), JSON.stringify({ max_length: 64, min_length: 10 }));
  });

  await test('7. QA model sends question + context through the real path', async () => {
    const { w: w2 } = makeWindow();
    const bodies = [];
    w2.fetch = async (url, init = {}) => {
      if (String(url).includes('/api/zerogpu-run')) {
        bodies.push(JSON.parse(init.body));
        return { ok: true, status: 200, text: async () => JSON.stringify({ output: 'A: Paris' }) };
      }
      return { ok: true, status: 200, text: async () => '{}', json: async () => ({}) };
    };
    const nbId = w2.createNotebook('QA Routing');
    w2.setActiveNotebook(nbId);
    w2.addCell('model');
    w2.addCell('code');
    const nb = w2.getActiveNotebook();
    const modelCell = nb.cells.find((c) => c.type === 'model');
    const inferCell = nb.cells.find((c) => c.type === 'code');
    modelCell.modelInfo = { id: 'owner/bert-squad', task: 'question-answering' };
    inferCell.modelInfo = { role: 'inference', loaderCellId: modelCell.id };
    inferCell.content = [
      'question = "Where is the Eiffel Tower?"',
      'context = "The Eiffel Tower is located in Paris, France."',
    ].join('\n');
    await w2.runCellViaZeroGPU(nb, inferCell);
    const b = bodies[0];
    assert.strictEqual(b.task, 'question-answering');
    assert.strictEqual(b.inputs.task_inputs.question, 'Where is the Eiffel Tower?');
    assert.strictEqual(b.inputs.task_inputs.context, 'The Eiffel Tower is located in Paris, France.');
  });

  await test('8. missing task inputs block the request with a visible error', async () => {
    const { w: w2 } = makeWindow();
    const bodies = [];
    w2.fetch = async (url, init = {}) => {
      if (String(url).includes('/api/zerogpu-run')) bodies.push(JSON.parse(init.body));
      return { ok: true, status: 200, text: async () => '{}', json: async () => ({}) };
    };
    const nbId = w2.createNotebook('Empty Inputs');
    w2.setActiveNotebook(nbId);
    w2.addCell('model');
    w2.addCell('code');
    const nb = w2.getActiveNotebook();
    const modelCell = nb.cells.find((c) => c.type === 'model');
    const inferCell = nb.cells.find((c) => c.type === 'code');
    modelCell.modelInfo = { id: 'owner/bart', task: 'summarization' };
    inferCell.modelInfo = { role: 'inference', loaderCellId: modelCell.id };
    inferCell.content = '# no inputs filled in';
    await w2.runCellViaZeroGPU(nb, inferCell);
    assert.strictEqual(bodies.length, 0, 'no request sent without required inputs');
  });

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failed) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
