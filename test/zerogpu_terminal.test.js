'use strict';
// GPU-terminal error-line regression tests (public/dashboard-app.js).
//
// The Model Cell terminal must render a GPU-availability failure as the
// friendly retry line — never the opaque backend text. The mapping lives in
// the pure helper formatZeroGpuRunError(data), extracted here from the
// shipped source via its TEST-HOOK markers and executed for real.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const ui = require('../training_ui');

const APP_SRC = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8');

function loadFormatter() {
  const startMarker = '// [TEST-HOOK:zeroGpuRunErrorFormat:START]';
  const endMarker = '// [TEST-HOOK:zeroGpuRunErrorFormat:END]';
  const start = APP_SRC.indexOf(startMarker);
  const end = APP_SRC.indexOf(endMarker);
  assert(start !== -1 && end !== -1 && end > start, 'TEST-HOOK markers must exist in dashboard-app.js');
  const src = APP_SRC.slice(start, end);
  const fn = new Function(`${src}; return formatZeroGpuRunError;`)();
  assert.strictEqual(typeof fn, 'function', 'helper must be a pure function');
  return fn;
}

const FRIENDLY = '⚠️ ZeroGPU: GPU unavailable right now. Please try again later.';

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log(`✓ ${name}`); passed++; }
  catch (e) { console.error(`✗ ${name}: ${e.message}`); failed++; }
}

console.log('=== ZeroGPU Terminal Regression Tests ===\n');

// 1. gpu_unavailable renders the exact friendly yellow line.
test('1. gpu_unavailable renders the exact friendly line', () => {
  const format = loadFormatter();
  const lines = format({ error: 'ZeroGPU model error: whatever', code: 'gpu_unavailable' });
  assert.deepStrictEqual(lines, [{ text: FRIENDLY, color: 'yellow' }]);
});

// 2. No "unexpected error" / "RuntimeError" text in the gpu_unavailable line.
test('2. gpu_unavailable line hides opaque backend text', () => {
  const format = loadFormatter();
  const [{ text }] = format({ error: "ZeroGPU model error: unexpected error: 'RuntimeError: No CUDA GPUs are available'", code: 'gpu_unavailable' });
  assert(!text.includes('unexpected error'), 'must not leak "unexpected error"');
  assert(!text.includes('RuntimeError'), 'must not leak "RuntimeError"');
  assert.strictEqual(text, FRIENDLY);
});

// 3. Other codes keep the existing red format with the code.
test('3. model_runtime keeps the red format with code', () => {
  const format = loadFormatter();
  const lines = format({ error: 'ZeroGPU model error: boom', code: 'model_runtime' });
  assert.deepStrictEqual(lines, [{ text: '❌ ZeroGPU (model_runtime): ZeroGPU model error: boom', color: 'red' }]);
});

// 4. Missing code renders without parens.
test('4. missing code renders without parens', () => {
  const format = loadFormatter();
  const lines = format({ error: 'Non-JSON response from server' });
  assert.deepStrictEqual(lines, [{ text: '❌ ZeroGPU: Non-JSON response from server', color: 'red' }]);
});

// 5. The error branch actually calls the helper (wiring guard).
test('5. error branch wires the helper', () => {
  const idx = APP_SRC.indexOf('if (!res.ok || data.error)');
  assert(idx !== -1, 'error branch must exist');
  const snippet = APP_SRC.slice(idx, idx + 400);
  assert(snippet.includes('formatZeroGpuRunError(data)'), 'error branch must call formatZeroGpuRunError');
  assert(snippet.includes("data.code === 'gpu_unavailable'") || APP_SRC.includes("data.code === 'gpu_unavailable'"),
    'helper must branch on gpu_unavailable');
});

// 6. Training UI translates the code too (shared classification).
test('6. translateBackendError maps gpu_unavailable', () => {
  assert.strictEqual(
    ui.translateBackendError('whatever', 'gpu_unavailable'),
    'ZeroGPU: GPU unavailable right now. Please try again later.'
  );
  assert.strictEqual(
    ui.translateBackendError('GPU unavailable right now', 'training_error'),
    'ZeroGPU: GPU unavailable right now. Please try again later.'
  );
  // Unrelated codes are untouched.
  assert(ui.translateBackendError('training_error: boom', 'training_error').includes('Training failed'),
    'training_error mapping must be unchanged');
});

// 7. Browser copy carries the same mapping (root/public sync guard).
test('7. public training_ui.js carries the mapping', () => {
  const pub = fs.readFileSync(path.join(__dirname, '..', 'public', 'training_ui.js'), 'utf8');
  assert(pub.includes("code === 'gpu_unavailable'"), 'public copy must carry the mapping');
});

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
if (failed) process.exit(1);
