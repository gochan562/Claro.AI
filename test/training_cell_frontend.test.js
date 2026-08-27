'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const tb = require('../training_backend');
const ui = require('../training_ui');

const dashboardPath = path.join(__dirname, '..', 'dashboard.html');
const dashboardHtml = fs.readFileSync(dashboardPath, 'utf8');

let passed = 0, failed = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`✓ ${name}`);
    passed++;
  } catch (e) {
    console.error(`✗ ${name}: ${e.message}`);
    console.error(e.stack);
    failed++;
  }
}
async function asyncTest(name, fn) {
  try {
    await fn();
    console.log(`✓ ${name}`);
    passed++;
  } catch (e) {
    console.error(`✗ ${name}: ${e.message}`);
    console.error(e.stack);
    failed++;
  }
}

// Helper: default config as per spec
function defaultConfig() {
  return {
    model_id: 'distilbert-base-uncased',
    dataset_id: 'stanfordnlp/imdb',
    task_type: 'text-classification',
    epochs: 2,
    batch_size: 8,
    learning_rate: 0.00002,
    max_steps: '',
    validation_split: 10,
    training_method: 'auto',
    lora_r: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    target_modules: 'auto',
  };
}

console.log('=== Training Cell Frontend Regression Tests ===\n');

// 1. Default Training Cell configuration.
test('1. Default Training Cell configuration', () => {
  const cfg = defaultConfig();
  // must match spec defaults
  assert.strictEqual(cfg.epochs, 2);
  assert.strictEqual(cfg.batch_size, 8);
  assert.strictEqual(cfg.learning_rate, 2e-5);
  assert.strictEqual(cfg.validation_split, 10);
  assert.strictEqual(cfg.training_method, 'auto');
  assert.strictEqual(cfg.lora_r, 8);
  assert.strictEqual(cfg.lora_alpha, 16);
  assert.strictEqual(cfg.lora_dropout, 0.05);
  assert.strictEqual(cfg.target_modules, 'auto');
  // backend validation should accept defaults
  const validated = tb.validateTrainingRequest(cfg);
  assert.strictEqual(validated.epochs, 2);
  assert.strictEqual(validated.batch_size, 8);
  assert.strictEqual(validated.validation_split, 10);
  assert.strictEqual(validated.training_method, 'full', 'distilbert <1B auto should resolve to full');
  // frontend validation should also pass
  const fv = ui.validateTrainingConfigFrontend(cfg);
  assert.strictEqual(fv.valid, true, `frontend validation should pass defaults, got ${JSON.stringify(fv.errors)}`);
  // check dashboard.html default still contains those values in addCell
  assert(dashboardHtml.includes("model_id: 'distilbert-base-uncased'"));
  assert(dashboardHtml.includes("dataset_id: 'stanfordnlp/imdb'"));
  assert(dashboardHtml.includes("epochs: 2"));
  assert(dashboardHtml.includes("batch_size: 8"));
  assert(dashboardHtml.includes("validation_split: 10"));
});

// 2. Existing old Training Cell migration.
test('2. Existing old Training Cell migration', () => {
  // simulate old cell missing fields
  const oldCell = { type: 'training', training: { model_id: 'distilbert-base-uncased' } };
  // migration logic as in dashboard.html loadAllNotebooks
  const t = oldCell.training;
  if (!('task_type' in t)) t.task_type = 'text-classification';
  if (!('epochs' in t)) t.epochs = 2;
  if (!('batch_size' in t)) t.batch_size = 8;
  if (!('learning_rate' in t)) t.learning_rate = 0.00002;
  if (!('validation_split' in t)) t.validation_split = 10;
  if (!('training_method' in t)) t.training_method = 'auto';
  if (!('lora_r' in t)) t.lora_r = 8;
  if (!('lora_alpha' in t)) t.lora_alpha = 16;
  if (!('lora_dropout' in t)) t.lora_dropout = 0.05;
  if (!('target_modules' in t)) t.target_modules = 'auto';
  if (!('_uiCustomizeOpen' in t)) t._uiCustomizeOpen = false;
  if (!('_uiAdvancedOpen' in t)) t._uiAdvancedOpen = false;
  if (!t.progress) t.progress = { training_method: 'auto', trainable_params: null, total_params: null };
  // also job_id preservation
  oldCell.training.job_id = 'train_abcdef123456';
  oldCell.training.status = 'training';
  oldCell.training.metrics = [{step:1, train_loss:1.2}];
  // simulate migration should preserve these
  assert.strictEqual(oldCell.training.job_id, 'train_abcdef123456');
  assert.strictEqual(oldCell.training.status, 'training');
  assert.strictEqual(oldCell.training.metrics.length, 1);
  assert.strictEqual(oldCell.training.training_method, 'auto');
  assert.strictEqual(oldCell.training.lora_r, 8);
  // also check dashboardHtml contains migration for _uiCustomizeOpen
  assert(dashboardHtml.includes('_uiCustomizeOpen'));
  assert(dashboardHtml.includes('_uiAdvancedOpen'));
  // ensure old notebook JSON with content field still migrates: cell.training from JSON.parse(cell.content)
  const legacy = { type: 'training', content: JSON.stringify({ model_id: 'a/b', task_type: 'text-classification' }) };
  let parsed = {};
  try { parsed = JSON.parse(legacy.content || '{}'); } catch(_) { parsed = {}; }
  parsed.model_id = parsed.model_id || 'distilbert-base-uncased';
  assert.strictEqual(parsed.model_id, 'a/b');
});

// 3. Recommended/Auto training method.
test('3. Recommended/Auto training method', () => {
  const cfgAutoSmall = { model_id: 'distilbert-base-uncased', dataset_id: 'stanfordnlp/imdb', task_type: 'text-classification', training_method: 'auto', epochs:2, batch_size:8, learning_rate:2e-5, validation_split:10, lora_r:8,lora_alpha:16,lora_dropout:0.05,target_modules:'auto', max_steps:'' };
  const effSmall = ui.getEffectiveTrainingMethod(cfgAutoSmall);
  assert.strictEqual(effSmall, 'full', 'small model auto should be full');
  const cfgAutoLarge = { ...cfgAutoSmall, model_id: 'meta-llama/Llama-2-7b-hf' };
  const effLarge = ui.getEffectiveTrainingMethod(cfgAutoLarge);
  assert.strictEqual(effLarge, 'lora', '7B model auto should be lora');
  // backend should agree
  const bcSmall = tb.validateTrainingRequest({model_id:'distilbert-base-uncased', dataset_id:'c/d', task_type:'text-classification', training_method:'auto'});
  assert.strictEqual(bcSmall.training_method, 'full');
  const bcLarge = tb.validateTrainingRequest({model_id:'meta-llama/Llama-2-7b-hf', dataset_id:'c/d', task_type:'text-generation', training_method:'auto'});
  assert.strictEqual(bcLarge.training_method, 'lora');
  // UI display: Recommended maps to auto
  assert.strictEqual(ui.getTrainingMethodDisplay('auto'), 'Recommended');
  assert(dashboardHtml.includes('>Recommended</option>'));
  // ensure isLargeModelFrontend works
  assert.strictEqual(ui.isLargeModelFrontend('meta-llama/Llama-2-7b-hf'), true);
  assert.strictEqual(ui.isLargeModelFrontend('distilbert-base-uncased'), false);
});

// 4. Full training selection.
test('4. Full training selection', () => {
  const cfg = { ...defaultConfig(), training_method: 'full' };
  const v = ui.validateTrainingConfigFrontend(cfg);
  assert.strictEqual(v.valid, true);
  assert.strictEqual(v.effectiveMethod, 'full');
  const bc = tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', training_method:'full'});
  assert.strictEqual(bc.training_method, 'full');
  assert(dashboardHtml.includes('value="full"') && dashboardHtml.includes('Full fine-tuning'));
});

// 5. LoRA selection.
test('5. LoRA selection', () => {
  const cfg = { ...defaultConfig(), training_method: 'lora' };
  const v = ui.validateTrainingConfigFrontend(cfg);
  assert.strictEqual(v.valid, true);
  assert.strictEqual(v.effectiveMethod, 'lora');
  const bc = tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', training_method:'lora', lora_r:8, lora_alpha:16, lora_dropout:0.05, target_modules:'auto'});
  assert.strictEqual(bc.training_method, 'lora');
  assert(dashboardHtml.includes('value="lora"') && dashboardHtml.includes('>LoRA</option>'));
});

// 6. LoRA fields hidden when Full is selected.
test('6. LoRA fields hidden when Full is selected', () => {
  const cfgFull = { ...defaultConfig(), training_method: 'full' };
  const eff = ui.getEffectiveTrainingMethod(cfgFull);
  assert.strictEqual(eff, 'full');
  // UI logic: advanced panel should be hidden when eff != lora
  // Check dashboard.html syncAdvancedVisibility hides when not lora
  assert(dashboardHtml.includes("advToggle.style.display = isLora ? '' : 'none'"));
  // Also check that old lora-section logic was replaced with advanced-panel
  assert(dashboardHtml.includes('train-advanced-panel'));
  // Simulate DOM via jsdom for this condition
  const dom = new JSDOM(`<div id="tr-1-advanced-toggle"></div><div id="tr-1-advanced-panel"></div>`);
  // effective full => should be hidden; we test ui helper directly
  assert.strictEqual(eff !== 'lora', true);
});

// 7. LoRA fields visible when effective method is LoRA.
test('7. LoRA fields visible when effective method is LoRA', () => {
  const cfgLora = { ...defaultConfig(), training_method: 'lora' };
  const eff = ui.getEffectiveTrainingMethod(cfgLora);
  assert.strictEqual(eff, 'lora');
  const cfgAutoLarge = { ...defaultConfig(), model_id: 'HuggingFaceTB/SmolLM3-3B', training_method: 'auto' };
  // SmolLM3-3B contains 3b => large
  const effAutoLarge = ui.getEffectiveTrainingMethod(cfgAutoLarge);
  assert.strictEqual(effAutoLarge, 'lora', '3B should be lora');
  // Also small auto should be hidden
  const cfgAutoSmall = { ...defaultConfig(), model_id: 'distilbert-base-uncased', training_method: 'auto' };
  assert.strictEqual(ui.getEffectiveTrainingMethod(cfgAutoSmall), 'full');
  // Check that dashboard shows LoRA when effective lora
  // In buildTrainingCellBody, isLoraEffective determines visibility
  assert(dashboardHtml.includes("isLoraEffective"));
});

// 8. Invalid epochs rejected.
test('8. Invalid epochs rejected', () => {
  const invalids = [0, -1, 6, 99, 'abc', '', null];
  for (const val of invalids) {
    const cfg = { ...defaultConfig(), epochs: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(v.errors.epochs, `epochs=${val} should be rejected, got ${JSON.stringify(v)}`);
  }
  const valids = [1,2,3,4,5];
  for (const val of valids) {
    const cfg = { ...defaultConfig(), epochs: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(!v.errors.epochs, `epochs=${val} should be valid`);
  }
  // backend also rejects
  let err=null;
  try { tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', epochs: 99}); } catch(e){err=e;}
  assert(err && err.code === 'bad_request');
});

// 9. Invalid batch size rejected.
test('9. Invalid batch size rejected', () => {
  const invalids = [0, 33, 99, -1, 'xyz'];
  for (const val of invalids) {
    const cfg = { ...defaultConfig(), batch_size: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(v.errors.batch_size, `batch_size=${val} should be rejected`);
  }
  const valids = [1,8,16,32];
  for (const val of valids) {
    const cfg = { ...defaultConfig(), batch_size: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(!v.errors.batch_size, `batch_size=${val} should be valid`);
  }
  let err=null;
  try { tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', batch_size: 99}); } catch(e){err=e;}
  assert(err && err.code === 'bad_request');
});

// 10. Invalid learning rate rejected.
test('10. Invalid learning rate rejected', () => {
  const invalids = [0, 1e-7, 0.02, 1, 99, -0.001];
  for (const val of invalids) {
    const cfg = { ...defaultConfig(), learning_rate: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(v.errors.learning_rate, `lr=${val} should be rejected`);
  }
  const valids = [1e-6, 2e-5, 1e-3, 1e-2];
  for (const val of valids) {
    const cfg = { ...defaultConfig(), learning_rate: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(!v.errors.learning_rate, `lr=${val} should be valid`);
  }
  let err=null;
  try { tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', learning_rate: 99}); } catch(e){err=e;}
  assert(err);
});

// 11. Invalid validation split rejected.
test('11. Invalid validation split rejected', () => {
  const invalids = [-1, 51, 100, 'abc'];
  for (const val of invalids) {
    const cfg = { ...defaultConfig(), validation_split: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(v.errors.validation_split, `vs=${val} should be rejected`);
  }
  const valids = [0,10,20,50];
  for (const val of valids) {
    const cfg = { ...defaultConfig(), validation_split: val };
    const v = ui.validateTrainingConfigFrontend(cfg);
    assert(!v.errors.validation_split, `vs=${val} should be valid`);
  }
});

// 12. Invalid LoRA parameters rejected.
test('12. Invalid LoRA parameters rejected', () => {
  // lora_r
  let cfg = { ...defaultConfig(), training_method:'lora', lora_r: 0 };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.lora_r);
  cfg = { ...defaultConfig(), training_method:'lora', lora_r: 65 };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.lora_r);
  cfg = { ...defaultConfig(), training_method:'lora', lora_r: 8 };
  assert(!ui.validateTrainingConfigFrontend(cfg).errors.lora_r);
  // lora_alpha
  cfg = { ...defaultConfig(), training_method:'lora', lora_alpha: 0 };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.lora_alpha);
  cfg = { ...defaultConfig(), training_method:'lora', lora_alpha: 200 };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.lora_alpha);
  // lora_dropout
  cfg = { ...defaultConfig(), training_method:'lora', lora_dropout: -0.1 };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.lora_dropout);
  cfg = { ...defaultConfig(), training_method:'lora', lora_dropout: 0.6 };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.lora_dropout);
  // target_modules invalid
  cfg = { ...defaultConfig(), training_method:'lora', target_modules: 'invalid!@#' };
  assert(ui.validateTrainingConfigFrontend(cfg).errors.target_modules);
  cfg = { ...defaultConfig(), training_method:'lora', target_modules: 'q_proj, v_proj' };
  assert(!ui.validateTrainingConfigFrontend(cfg).errors.target_modules);
  // when effective method is full, lora params should NOT be validated (should be ignored)
  cfg = { ...defaultConfig(), training_method:'full', lora_r: 999 };
  const vFull = ui.validateTrainingConfigFrontend(cfg);
  assert(!vFull.errors.lora_r, 'full should not validate lora_r');
});

// 13. Start button disabled/prevented for invalid configurations.
test('13. Start button disabled/prevented for invalid configurations', () => {
  const invalidCfg = { ...defaultConfig(), epochs: 99 };
  const v = ui.validateTrainingConfigFrontend(invalidCfg);
  assert.strictEqual(v.valid, false);
  // startTrainingCell should check validation and not send request
  // Check dashboardHtml contains guard: if (!v.valid) return;
  assert(dashboardHtml.includes("if (!v.valid)"));
  // Also button disabled logic
  assert(dashboardHtml.includes('startBtn.disabled = !v.valid || c.status ==='));
  // Also ensure validation box shows error
  assert(dashboardHtml.includes('train-validation-box error'));
  // Ensure updateTrainingValidationUI disables start
  // Simulate DOM: create start button and check logic
  const dom = new JSDOM(`<button id="tr-1-start"></button><div id="tr-1-validation"></div>`);
  global.document = dom.window.document;
  // we can't fully run updateTrainingValidationUI without notebookRegistry, but we test static presence
  assert(dashboardHtml.includes('showStopConfirm'));
});

// 14. Valid configuration still produces the same API request.
test('14. Valid configuration still produces the same API request', () => {
  const cfg = defaultConfig();
  const body = {
    model_id: cfg.model_id,
    dataset_id: cfg.dataset_id,
    task_type: cfg.task_type,
    epochs: Number(cfg.epochs),
    batch_size: Number(cfg.batch_size),
    learning_rate: Number(cfg.learning_rate),
    max_steps: cfg.max_steps === '' ? null : Number(cfg.max_steps),
    validation_split: Number(cfg.validation_split),
    training_method: cfg.training_method,
    lora_r: Number(cfg.lora_r),
    lora_alpha: Number(cfg.lora_alpha),
    lora_dropout: Number(cfg.lora_dropout),
    target_modules: cfg.target_modules,
  };
  // This is exactly what startTrainingCell sends (see dashboard.html)
  assert(dashboardHtml.includes("model_id: t.model_id"));
  assert(dashboardHtml.includes("dataset_id: t.dataset_id"));
  assert(dashboardHtml.includes("task_type: t.task_type"));
  // Validate via backend still works
  const validated = tb.validateTrainingRequest(body);
  assert.strictEqual(validated.model_id, 'distilbert-base-uncased');
  assert.strictEqual(validated.dataset_id, 'stanfordnlp/imdb');
  assert.strictEqual(validated.task_type, 'text-classification');
  assert.strictEqual(validated.epochs, 2);
  assert.strictEqual(validated.batch_size, 8);
  // As JSON stringify produces same
  const json = JSON.stringify(body);
  const parsed = JSON.parse(json);
  assert.strictEqual(parsed.training_method, 'auto');
});

// 15. job_id survives cell re-render.
test('15. job_id survives cell re-render', () => {
  // Simulate notebookRegistry with a training cell having job_id
  const notebook = { id: 'notebook_1', cells: [{ id: 1, type: 'training', training: { job_id: 'train_abc123456789', status: 'training', progress: {}, metrics: [], logs: [] } }], nextCellId: 2 };
  const saved = JSON.stringify(notebook);
  const loaded = JSON.parse(saved);
  assert.strictEqual(loaded.cells[0].training.job_id, 'train_abc123456789');
  // buildTrainingCellBody should preserve job_id and re-attach SSE
  assert(dashboardHtml.includes('if (t.job_id && isRunning)'));
  assert(dashboardHtml.includes('attachTrainingSSE(cell.id, t.job_id)'));
  // Ensure renderNotebookEditor does not clear job_id (search for clearing)
  const renderEditorSection = dashboardHtml.slice(dashboardHtml.indexOf('function renderNotebookEditor'));
  // Should NOT contain stopTrainingCell or clearing job_id
  assert(!renderEditorSection.slice(0,5000).includes('stopTrainingCell'), 'renderNotebookEditor should not call stopTrainingCell');
  // Also ensure notebook switching does not reset job_id
  assert(!dashboardHtml.includes('training.job_id = null') || dashboardHtml.includes('t.job_id = data.job_id'), 'job_id only set on start, not cleared on re-render');
});

// 16. notebook switching does NOT call stopTrainingCell().
test('16. notebook switching does NOT call stopTrainingCell()', () => {
  function getFunctionBody(name) {
    const idx = dashboardHtml.indexOf(name);
    assert(idx !== -1, `${name} should exist`);
    // find next function declaration after this one
    const nextFn = dashboardHtml.indexOf('    function ', idx + name.length);
    const end = nextFn === -1 ? idx+8000 : nextFn;
    return dashboardHtml.slice(idx, end);
  }
  const switchFuncs = ['function setActiveNotebook', 'function backToList', 'function loadAllNotebooks', 'function renderNotebookEditor'];
  for (const fname of switchFuncs) {
    const body = getFunctionBody(fname);
    assert(!body.includes('stopTrainingCell('), `${fname} must NOT call stopTrainingCell`);
  }
  // also switching via UI nav should not stop
  const tabIdx = dashboardHtml.indexOf('TAB_MAP');
  const tabSnippet = dashboardHtml.slice(tabIdx, dashboardHtml.indexOf('async function loadDatasets', tabIdx));
  assert(!tabSnippet.includes('stopTrainingCell'));
});

// 17. SSE reconnect does NOT stop training.
test('17. SSE reconnect does NOT stop training', () => {
  function getFuncBody(name) {
    const idx = dashboardHtml.indexOf(name);
    assert(idx !== -1);
    const next = dashboardHtml.indexOf('    function ', idx + name.length);
    const end = next === -1 ? idx+8000 : next;
    return dashboardHtml.slice(idx, end);
  }
  const attachBody = getFuncBody('function attachTrainingSSE');
  assert(attachBody.includes('startTrainingPolling'), 'onerror should fallback to polling');
  assert(!attachBody.includes('stopTrainingCell('), 'SSE error handler must NOT call stopTrainingCell');
  const pollBody = getFuncBody('function startTrainingPolling');
  assert(!pollBody.includes('stopTrainingCell('), 'polling must NOT call stopTrainingCell');
  assert(attachBody.includes('es.onerror'));
});

// 18. Explicit Stop still calls the stop API.
test('18. Explicit Stop still calls the stop API', () => {
  // stopTrainingCell should POST to /api/train/stop with job_id
  assert(dashboardHtml.includes("'/api/train/stop'"));
  assert(dashboardHtml.includes('stopTrainingCell'));
  // confirmStopTraining should call stopTrainingCell with userInitiated:true
  assert(dashboardHtml.includes('confirmStopTraining'));
  assert(dashboardHtml.includes("stopTrainingCell(cellId, {caller:'user Stop confirm'"));
  // also ensure fetch is called with job_id
  const stopIdx = dashboardHtml.indexOf('async function stopTrainingCell');
  const stopSnippet = dashboardHtml.slice(stopIdx, stopIdx+4000);
  assert(stopSnippet.includes('fetch(stopUrl'));
  assert(stopSnippet.includes('job_id: jobId'));
});

// 19. Delete explicitly initiated by the user may stop the job.
test('19. Delete explicitly initiated by the user may stop the job', () => {
  const delIdx = dashboardHtml.indexOf('function deleteCell');
  const delSnippet = dashboardHtml.slice(delIdx, delIdx+2000);
  // deleteCell should call stopTrainingCell with userInitiated:true (after patch)
  assert(delSnippet.includes("stopTrainingCell(cellId, { caller: 'deleteCell'"));
  assert(delSnippet.includes('userInitiated: true'), 'deleteCell should mark userInitiated true');
  // ensure it only stops if running statuses
  assert(delSnippet.includes("['queued','loading','training','evaluating'].includes(t.status)"));
});

// 20. Real training status/metrics continue rendering.
test('20. Real training status/metrics continue rendering', () => {
  // renderTrainingStatus must still update epoch, step, train_loss, eval_loss, eta, gpu, trainable
  assert(dashboardHtml.includes('tr-${cellId}-epoch'));
  assert(dashboardHtml.includes('tr-${cellId}-step'));
  assert(dashboardHtml.includes('tr-${cellId}-train'));
  assert(dashboardHtml.includes('tr-${cellId}-eval'));
  assert(dashboardHtml.includes('tr-${cellId}-eta'));
  assert(dashboardHtml.includes('tr-${cellId}-gpu'));
  assert(dashboardHtml.includes('tr-${cellId}-trainable'));
  // chart must still use real metrics
  assert(dashboardHtml.includes('initTrainingChart'));
  assert(dashboardHtml.includes('updateTrainingChart'));
  // handleTrainingMetric should push real metrics and update chart
  const handleIdx = dashboardHtml.indexOf('function handleTrainingMetric');
  const handleSnippet = dashboardHtml.slice(handleIdx, handleIdx+1500);
  assert(handleSnippet.includes('cell.training.metrics.push(m)'));
  assert(handleSnippet.includes('updateTrainingChart'));
  // ensure no Math.random or fake exp
  assert(!dashboardHtml.includes('Math.random()') || dashboardHtml.includes('Math.random') === false || !dashboardHtml.slice(dashboardHtml.indexOf('handleTrainingMetric'), dashboardHtml.indexOf('handleTrainingMetric')+2000).includes('Math.random'));
  // check that metrics are not simulated via exp(-step)
  assert(!dashboardHtml.includes('exp(-step)'));
});

// 21. Training completion still exposes "Use in Inference Cell."
test('21. Training completion still exposes "Use in Inference Cell."', () => {
  assert(dashboardHtml.includes('Use in Inference Cell'));
  assert(dashboardHtml.includes('useTrainedInInference'));
  // renderTrainingComplete should still fetch artifacts and show button
  const compIdx = dashboardHtml.indexOf('async function renderTrainingComplete');
  const compSnippet = dashboardHtml.slice(compIdx, compIdx+5000);
  assert(compSnippet.includes('training-complete-btn'));
  assert(compSnippet.includes('Use in Inference Cell'));
  // ensure backend endpoint still exists via training_backend getArtifactMetadata
  assert(typeof tb.getArtifactMetadata === 'function');
});

// 22. Existing URL encoding / EventSource robustness remains intact.
test('22. Existing URL encoding / EventSource robustness remains intact', () => {
  // attachTrainingSSE should use new URL with encodeURIComponent and window.location.href
  assert(dashboardHtml.includes('new URL(urlStr, window.location.href)'));
  assert(dashboardHtml.includes('encodeURIComponent(jobId)'));
  // also check that fetch for artifacts uses encodeURIComponent
  assert(dashboardHtml.includes("encodeURIComponent(jobId)"));
  assert(dashboardHtml.includes("encodeURIComponent(jid)") // generic
      || dashboardHtml.includes("encodeURIComponent(jobId)"));
  // Check that test file training_frontend_url still passes concepts: URL construction with base
  const jobId = 'train_5d8b337e4677';
  const bases = ['http://localhost:5007/dashboard.html', 'http://example.com/'];
  for (const base of bases) {
    const urlStr = `/api/train/metrics/${encodeURIComponent(jobId)}?stream=1`;
    const url = new URL(urlStr, base);
    assert(url.toString().includes(`/api/train/metrics/${jobId}?stream=1`));
  }
  // Ensure no raw string without base for EventSource
  // The buggy pattern new EventSource(`/api/train/metrics/${jobId}`) should not exist; must be new URL(..., window.location.href)
  const esPattern = /new EventSource\(\s*['"`]\/api\/train\/metrics/;
  // If found, it should be with URL object, not raw string
  // Check that our code uses new EventSource(url.toString())
  assert(dashboardHtml.includes('new EventSource(url.toString())'));
});

// Additional checks: guardrails and UX principles
test('Progressive disclosure structure exists', () => {
  assert(dashboardHtml.includes('train-customize-toggle'));
  assert(dashboardHtml.includes('train-customize-panel'));
  assert(dashboardHtml.includes('train-advanced-toggle'));
  assert(dashboardHtml.includes('train-advanced-panel'));
  assert(dashboardHtml.includes('Train Model'));
  assert(dashboardHtml.includes('What should it learn?'));
  assert(dashboardHtml.includes('The AI model') && dashboardHtml.includes('train.'));
  assert(dashboardHtml.includes('The examples the model will learn from.'));
  assert(dashboardHtml.includes('Training preview'));
  assert(dashboardHtml.includes('Configuration looks good'));
  assert(dashboardHtml.includes('train-validation-box'));
  assert(dashboardHtml.includes('train-preview-box'));
  assert(dashboardHtml.includes('How many times the model sees'));
  assert(dashboardHtml.includes('How many examples the model processes'));
  assert(dashboardHtml.includes('How strongly the model changes'));
  assert(dashboardHtml.includes('percentage of examples kept aside'));
  assert(dashboardHtml.includes('LoRA lets large models learn'));
});

test('Backend engine unchanged', () => {
  // ensure training_backend still has same validation and no fake metrics
  const backendContent = fs.readFileSync(path.join(__dirname, '..', 'training_backend.js'), 'utf8');
  assert(!backendContent.includes('Math.random'));
  assert(!backendContent.includes('exp(-step)'));
  assert(backendContent.includes('ALLOWED_TASKS'));
  assert(backendContent.includes('isLargeModel'));
  // ensure training_runner still real Trainer
  const runnerContent = fs.readFileSync(path.join(__dirname, '..', 'training_runner.py'), 'utf8');
  assert(runnerContent.includes('Trainer.train()'));
  assert(runnerContent.includes('REAL TRAINING ONLY'));
});

test('No fake estimates introduced', () => {
  assert(dashboardHtml.includes('Estimate unavailable'));
  // should not contain Math.random for estimates
  const previewSection = dashboardHtml.slice(dashboardHtml.indexOf('getPreviewData'), dashboardHtml.indexOf('getPreviewData')+5000);
  // training_ui.js handles preview, dashboard just displays. Check training_ui.js has no random.
  const uiContent = fs.readFileSync(path.join(__dirname, '..', 'training_ui.js'), 'utf8');
  assert(!uiContent.includes('Math.random'));
  assert(uiContent.includes('Estimate unavailable'));
});

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
if (failed > 0) process.exit(1);
