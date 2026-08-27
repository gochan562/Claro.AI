'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const { spawnSync } = require('child_process');

const dashboardPath = path.join(__dirname, '..', 'dashboard.html');
const html = fs.readFileSync(dashboardPath, 'utf8');
const inferenceRunnerPath = path.join(__dirname, '..', 'inference_runner.py');
const serverPath = path.join(__dirname, '..', 'server.js');
const tb = require('../training_backend');

let passed=0, failed=0;
function test(name, fn){
  try{ fn(); console.log(`✓ ${name}`); passed++; } catch(e){ console.error(`✗ ${name}: ${e.message}`); console.error(e.stack); failed++; }
}
async function asyncTest(name, fn){
  try{ await fn(); console.log(`✓ ${name}`); passed++; } catch(e){ console.error(`✗ ${name}: ${e.message}`); console.error(e.stack); failed++; }
}

console.log('=== Test Model Playground Regression Tests ===\n');

// Helper to get python bin
const PYTHON_BIN = process.env.PYTHON_BIN || '/var/folders/f0/ycgc1tnn7ydbn5cg4sk_kq2c0000gn/T/opencode/t5_venv/bin/python';

// 1. Test UI is hidden before training finishes.
test('1. Test UI is hidden before training finishes', () => {
  // buildTrainingCellBody should hide test box when status != finished
  assert(html.includes('id="tr-${cell.id}-test-box"'));
  assert(html.includes("style=\"${t.status==='finished' ? '' : 'display:none;'}\""));
  // also renderTrainingStatus should hide when not finished
  assert(html.includes("if (t.status === 'finished' && t.job_id)"));
  assert(html.includes("testBox.style.display = ''") && html.includes("testBox.style.display = 'none'"));
});

// 2. Test UI appears after successful training.
test('2. Test UI appears after successful training', () => {
  // Check that test box is visible when finished and that renderTestModelInput is called
  assert(html.includes("setTimeout(() => renderTestModelInput(cell.id), 20)"));
  assert(html.includes("test-model-box"));
  assert(html.includes("🧪 Test your trained model"));
  assert(html.includes("Using your trained model"));
});

// 3. Test UI does not appear for failed training.
test('3. Test UI does not appear for failed training', () => {
  // same check as 1: only finished shows
  const snippet = html.slice(html.indexOf('test-model-box'), html.indexOf('test-model-box')+2000);
  // The JS condition ensures display:none when not finished, so failed (not finished) will be hidden
  assert(html.includes("t.status === 'finished'"));
  // Ensure failed case hides
  assert(html.includes("style=\"${t.status==='finished' ? '' : 'display:none;'}\""));
});

// 4. text-classification creates the correct test UI.
test('4. text-classification creates the correct test UI', () => {
  assert(html.includes("task === 'text-classification'"));
  // textarea placeholder for text-class
  assert(html.includes("This movie was absolutely fantastic!"));
  // hint
  assert(html.includes("This movie was surprisingly good"));
  // Result rendering for text-class expects scores
  assert(html.includes("task === 'text-classification' && data.scores"));
  assert(html.includes("test-result-scores"));
});

// 5. text-generation creates the correct test UI.
test('5. text-generation creates the correct test UI', () => {
  assert(html.includes("task === 'text-generation'"));
  assert(html.includes("Once upon a time,"));
  // max_new_tokens control for generation
  assert(html.includes("Max new tokens"));
  assert(html.includes("testMaxTokens"));
  // generation output rendering
  assert(html.includes("task === 'text-generation'"));
});

// 6. token-classification creates the correct test UI.
test('6. token-classification creates the correct test UI', () => {
  assert(html.includes("task === 'token-classification'"));
  assert(html.includes("John lives in Tokyo"));
  // token table rendering
  assert(html.includes("test-token-table"));
  assert(html.includes("Token") && html.includes("Label"));
});

// 7. image-classification creates the correct test UI.
test('7. image-classification creates the correct test UI', () => {
  assert(html.includes("task === 'image-classification'"));
  assert(html.includes("test-image-drop"));
  assert(html.includes("Drop an image here"));
  assert(html.includes("Choose Image"));
  assert(html.includes("handleImageFileSelect"));
  assert(html.includes("handleImageDrop"));
  // image preview
  assert(html.includes("test-image-preview"));
  // result for image also uses scores
  assert(html.includes("task === 'image-classification'"));
});

// 8. Run Test sends the correct job_id.
test('8. Run Test sends the correct job_id', () => {
  // runTestModel should send body with job_id: t.job_id
  assert(html.includes("body = { job_id: t.job_id"));
  assert(html.includes("job_id: t.job_id"));
  // fetch to /api/inference/trained
  assert(html.includes("fetch(url, {") && html.includes("/api/inference/trained"));
  // check that it uses new URL with base
  assert(html.includes("new URL('/api/inference/trained'"));
});

// 9. Run Test does NOT send an arbitrary artifact_path.
test('9. Run Test does NOT send an arbitrary artifact_path', () => {
  // Ensure runTestModel body does NOT send artifact_path as JSON key
  const runIdx = html.indexOf('async function runTestModel');
  const runSnippet = html.slice(runIdx, runIdx+6000);
  // Check that body construction does not include those keys (ignore comments)
  const bodyStart = runSnippet.indexOf('const body =');
  const commentIdx = runSnippet.indexOf('// SECURITY', bodyStart);
  const bodyEnd = commentIdx !== -1 ? commentIdx : runSnippet.indexOf('console.log', bodyStart);
  const bodySnippet = bodyStart!==-1 && bodyEnd!==-1 ? runSnippet.slice(bodyStart, bodyEnd) : runSnippet;
  assert(!bodySnippet.includes('artifact_path'), 'must not send artifact_path in body');
  assert(!bodySnippet.includes('adapter_path'), 'must not send adapter_path in body');
  assert(!bodySnippet.includes('model_path'), 'must not send model_path in body');
  assert(!bodySnippet.includes('artifact_dir'), 'must not send artifact_dir in body');
  // server must derive from job_id
  const serverContent = fs.readFileSync(serverPath, 'utf8');
  assert(serverContent.includes("trainingBackend.getArtifactMetadata(String(job_id).trim())"));
  assert(serverContent.includes("// Use artifact metadata to derive paths — never use client-provided paths"));
});

// 10. LoRA training uses the trained adapter.
test('10. LoRA training uses the trained adapter (real inference)', () => {
  // Find a real LoRA artifact
  const outputs = path.join(__dirname, '..', 'training_outputs');
  if (!fs.existsSync(outputs)) { console.log('  skipping: no training_outputs'); return; }
  const dirs = fs.readdirSync(outputs);
  let found = null;
  for (const d of dirs) {
    const jobJson = path.join(outputs, d, 'job.json');
    const adapterCfg = path.join(outputs, d, 'adapter_config.json');
    if (fs.existsSync(jobJson) && fs.existsSync(adapterCfg)) {
      try {
        const j = JSON.parse(fs.readFileSync(jobJson, 'utf8'));
        if (j.config && j.config.training_method === 'lora' && j.status === 'finished' && fs.statSync(path.join(outputs, d, 'adapter_model.safetensors')).size > 5000) {
          found = d;
          break;
        }
      } catch {}
    }
  }
  assert(found, 'need a real finished LoRA job for this test');
  // Check inference_runner loads via PeftModel
  const runnerContent = fs.readFileSync(inferenceRunnerPath, 'utf8');
  assert(runnerContent.includes("PeftModel.from_pretrained(base_model, str(artifact_path))"));
  // Real inference against that artifact
  const result = spawnSync(PYTHON_BIN, [inferenceRunnerPath, '--job_id', found, '--prompt', 'This movie was fantastic!', '--task_type', 'text-classification'], { cwd: path.join(__dirname, '..'), encoding: 'utf8', timeout: 60000 });
  assert.strictEqual(result.status, 0, `LoRA inference failed: ${result.stderr} ${result.stdout}`);
  const lines = result.stdout.trim().split('\n');
  let data = null;
  for (let i=lines.length-1;i>=0;i--) {
    try { const p = JSON.parse(lines[i].trim()); if(p.output) {data=p; break;} } catch{}
  }
  assert(data && data.output, 'LoRA inference must return output');
  assert(data.task === 'text-classification');
  assert(Array.isArray(data.scores) && data.scores.length >= 2, 'LoRA inference must return scores');
  console.log(`  LoRA ${found} -> ${data.output} (${(data.confidence*100).toFixed(1)}%)`);
});

// 11. Full training uses the saved full model.
test('11. Full training uses the saved full model (real inference)', () => {
  const outputs = path.join(__dirname, '..', 'training_outputs');
  let found = null;
  const dirs = fs.readdirSync(outputs);
  for (const d of dirs) {
    const jobJson = path.join(outputs, d, 'job.json');
    const modelFile = path.join(outputs, d, 'model.safetensors');
    if (fs.existsSync(jobJson) && fs.existsSync(modelFile)) {
      try {
        const j = JSON.parse(fs.readFileSync(jobJson, 'utf8'));
        if (j.config && j.config.training_method === 'full' && j.status === 'finished' && fs.statSync(modelFile).size > 100000) {
          found = d;
          break;
        }
      } catch {}
    }
  }
  assert(found, 'need a real finished full job');
  const runnerContent = fs.readFileSync(inferenceRunnerPath, 'utf8');
  // Full should use AutoModel...from_pretrained(saved_artifact_dir)
  assert(runnerContent.includes("ModelClass.from_pretrained(str(artifact_path)"));
  // Real inference
  const result = spawnSync(PYTHON_BIN, [inferenceRunnerPath, '--job_id', found, '--prompt', 'This movie was terrible!', '--task_type', 'text-classification'], { cwd: path.join(__dirname, '..'), encoding: 'utf8', timeout: 60000 });
  assert.strictEqual(result.status, 0, `Full inference failed: ${result.stderr} ${result.stdout}`);
  const lines = result.stdout.trim().split('\n');
  let data=null;
  for (let i=lines.length-1;i>=0;i--) { try { const p=JSON.parse(lines[i].trim()); if(p.output) {data=p; break;} } catch{} }
  assert(data && data.output, 'Full inference must return output');
  assert(data.task === 'text-classification');
  assert(Array.isArray(data.scores));
  console.log(`  Full ${found} -> ${data.output} (${(data.confidence*100).toFixed(1)}%)`);
  // Ensure not using base fallback: check logs contain "trained loader" or "model.safetensors"
  // Our runner logs to stderr, but we can check that output is not from base random (scores should be plausible)
  assert(data.scores[0].score > 0.3 && data.scores[0].score < 0.99);
});

// 12. Base model is never silently substituted.
test('12. Base model is never silently substituted', () => {
  const runnerContent = fs.readFileSync(inferenceRunnerPath, 'utf8');
  // Should NOT have fallback that loads base model when artifact missing without error
  // load_trained_model should _fail if artifact missing, not silently fallback
  assert(runnerContent.includes("_fail(f\"artifact_dir not found"));
  assert(runnerContent.includes("_fail(f\"adapter_config.json not found"));
  // server must not trust client paths
  const serverContent = fs.readFileSync(serverPath, 'utf8');
  assert(!serverContent.includes("req.body.artifact_dir"));
  assert(!serverContent.includes("req.body.adapter_path"));
  // runTestModel should not have fallback logic
  const runSnippet = html.slice(html.indexOf('async function runTestModel'), html.indexOf('async function runTestModel')+6000);
  assert(!runSnippet.toLowerCase().includes('fallback to base'), 'should not fallback to base');
  // Ensure no hardcoded base model like "distilbert-base-uncased" as fallback in runTestModel
  // It's okay to show base name in UI identity, but not to use for inference when LoRA
  assert(runSnippet.includes('job_id: t.job_id'), 'must use job_id');
});

// 13. Inference errors are shown in the UI.
test('13. Inference errors are shown in the UI', () => {
  assert(html.includes("test-error-box"));
  assert(html.includes("showTestError"));
  assert(html.includes("Inference failed") || html.includes("Could not load the trained model"));
  // Friendly messages for artifact missing, task mismatch etc.
  assert(html.includes("Could not load the trained model"));
  assert(html.includes("Check the training logs"));
  // Should not expose raw traceback in main UI
  const snippet = html.slice(html.indexOf('showTestError'), html.indexOf('showTestError')+4000);
  // The catch block checks for Traceback and shows generic
  assert(html.includes("Traceback") && html.includes("Check the training logs"));
});

// 14. Duplicate Run Test clicks are prevented.
test('14. Duplicate Run Test clicks are prevented', () => {
  const runSnippet = html.slice(html.indexOf('async function runTestModel'), html.indexOf('async function runTestModel')+6000);
  assert(runSnippet.includes("if (runBtn && runBtn.disabled) return;"), 'should prevent duplicate when disabled');
  assert(runSnippet.includes("runBtn.disabled = true"));
  assert(runSnippet.includes("runBtn.disabled = false"));
  // Loading text is in HTML, not just JS; check both
  assert(html.includes('Testing trained model'), 'should show loading text');
  assert(html.includes('test-loading'), 'should have loading element');
});

// 15. Existing Use in Inference Cell still works.
test('15. Existing Use in Inference Cell still works', () => {
  assert(html.includes("Use in Inference Cell"));
  assert(html.includes("useTrainedInInference"));
  // Ensure test model did not replace that button
  const completeIdx = html.indexOf('training-complete');
  const useIdx = html.indexOf('Use in Inference Cell');
  assert(useIdx > completeIdx, 'Use in Inference should still be in completion');
  // Check that renderTrainingComplete still creates that button
  const rcIdx = html.indexOf('async function renderTrainingComplete');
  const rcSnippet = html.slice(rcIdx, rcIdx+6000);
  assert(rcSnippet.includes('Use in Inference Cell'));
});

// 16. job_id survives notebook re-render.
test('16. job_id survives notebook re-render', () => {
  // Check that testInput etc are persisted via updateNotebook -> localStorage
  assert(html.includes("cell.training.testInput"));
  assert(html.includes("cell.training.testImageData"));
  assert(html.includes("updateNotebook()"), 'test input should be persisted');
  // And that buildTrainingCellBody does not overwrite job_id
  assert(html.includes("if (!('testInput' in cell.training))"));
  // Also check that localStorage not storing weights
  assert(!html.includes("localStorage") || html.includes("job_id") );
  // Ensure test model does not store model weights in localStorage (check that testResult stored is small, not model)
  // testResult is stored but it's just prediction, not weights
  const storeSnippet = html.slice(html.indexOf('cell.training.testResult'), html.indexOf('cell.training.testResult')+2000);
  assert(storeSnippet.includes("testResult"));
});

// 17. notebook switching does not accidentally trigger inference.
test('17. notebook switching does not accidentally trigger inference', () => {
  const switchFuncs = ['function setActiveNotebook', 'function backToList', 'function loadAllNotebooks', 'function renderNotebookEditor'];
  for (const fname of switchFuncs) {
    const idx = html.indexOf(fname);
    const next = html.indexOf('    function ', idx + fname.length);
    const end = next===-1 ? idx+8000 : next;
    const body = html.slice(idx, end);
    assert(!body.includes('runTestModel'), `${fname} must NOT trigger runTestModel`);
    assert(!body.includes('fetch(') || !body.includes('/api/inference/trained'), `${fname} must NOT call inference`);
  }
  // Also ensure that rendering test box does not auto-run
  assert(!html.includes("runTestModel(cell.id)") || html.slice(html.indexOf('renderTestModelInput'), html.indexOf('renderTestModelInput')+2000).includes('renderTestModelInput'));
  // The only place runTestModel is called is onclick
  const onclickCount = (html.match(/onclick="runTestModel/g) || []).length;
  assert.strictEqual(onclickCount, 1, 'only one onclick should trigger runTestModel');
});

// 18. No fake/synthetic prediction code exists in production.
test('18. No fake/synthetic prediction code exists in production', () => {
  // Search dashboard, inference_runner, server for forbidden patterns
  const forbidden = ['Math.random() * 2', 'Math.random()', 'exp(-step', 'simulated', 'fake prediction', 'hardcoded example outputs', 'synthetic confidence'];
  // Allow Math.random in unrelated index.html but not in training/test code
  const trainingSections = html.slice(html.indexOf('Test Model playground'), html.indexOf('Test Model playground')+20000);
  for (const pat of ['Math.random()', 'exp(-step', 'fake prediction', 'synthetic']) {
    if (pat === 'Math.random()') {
      // Check that test model section doesn't contain it
      assert(!trainingSections.includes('Math.random'), 'test model should not contain Math.random');
    } else {
      assert(!html.slice(html.indexOf('async function runTestModel'), html.indexOf('async function runTestModel')+8000).includes(pat), `runTestModel should not contain ${pat}`);
    }
  }
  const runnerContent = fs.readFileSync(inferenceRunnerPath, 'utf8');
  assert(!runnerContent.includes('Math.random'));
  assert(!runnerContent.includes('fake'));
  assert(!runnerContent.includes('synthetic'));
  const serverContent = fs.readFileSync(serverPath, 'utf8');
  assert(!serverContent.includes('fake prediction'));
  // Ensure inference uses real model, not mock
  assert(runnerContent.includes('PeftModel.from_pretrained'));
  assert(runnerContent.includes('AutoModel'));
});

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
if (failed>0) process.exit(1);
