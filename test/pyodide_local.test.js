'use strict';

// Tests for the in-browser ordinary-Python execution layer (pyodide_local.js).
//
// Runs the REAL Pyodide WASM runtime under Node (pyodide npm package), driven
// through the same createRunner() wrapper the dashboard uses in the browser,
// so these tests verify actual Python semantics: stdout/stderr capture,
// namespace persistence between cells, async support, serialization, and
// exception surfacing.  A final static test asserts the dashboard routing
// sends ordinary code cells to Pyodide and never to Modal/ZeroGPU.

const assert = require('assert');
const path = require('path');
const fs = require('fs');

const { createRunner } = require(path.join(__dirname, '..', 'pyodide_local.js'));
const { loadPyodide } = require('pyodide');

async function main() {
  // In Node, pyodide.js is already available locally — no <script> injection needed.
  const runner = createRunner({
    injectScript: async () => {},
    resolveLoader: () => loadPyodide,
    loadOptions: {},
  });

  // 1. print("Hello")
  let r = await runner.run('print("Hello")');
  assert.strictEqual(r.ok, true);
  assert.strictEqual(r.stdout, 'Hello\n');
  console.log('✓ 1. print("Hello") captured on stdout');

  // 2. arithmetic
  r = await runner.run('print(2 + 3)');
  assert.strictEqual(r.stdout, '5\n');
  console.log('✓ 2. arithmetic works');

  // 3. namespace persists across cells
  await runner.run('x = 10');
  r = await runner.run('print(x + 5)');
  assert.strictEqual(r.stdout, '15\n', 'x from the previous cell must be visible');
  console.log('✓ 3. variables persist across cells (x = 10 → x + 5 = 15)');

  // 4. Pyodide-supported package import
  r = await runner.run('import json\nprint(json.dumps([1, 2, 3]))');
  assert.strictEqual(r.ok, true);
  assert.strictEqual(r.stdout, '[1, 2, 3]\n');
  console.log('✓ 4. import of Pyodide-supported package (json) works');

  // 4b. unavailable package → clear error, NOT a remote fallback
  r = await runner.run('import definitely_not_a_pyodide_package_zzz');
  assert.strictEqual(r.ok, false);
  assert(/ModuleNotFoundError/.test(r.error));
  assert(/not available in the in-browser/.test(r.error), 'hint about browser runtime limits');
  console.log('✓ 4b. unavailable package → clear in-browser-runtime error (never sent remotely)');

  // 5. stdout + stderr both captured
  r = await runner.run('import sys\nprint("to-out")\nprint("to-err", file=sys.stderr)');
  assert.strictEqual(r.ok, true);
  assert(r.stdout.includes('to-out'), 'stdout captured');
  assert(r.stderr.includes('to-err'), 'stderr captured');
  console.log('✓ 5. stdout and stderr captured separately');

  // 6. Python exception → cell error with traceback, runtime stays alive
  r = await runner.run('raise ValueError("boom")');
  assert.strictEqual(r.ok, false);
  assert(r.error.includes('ValueError: boom'), 'exception message surfaced');
  assert(r.error.includes('Traceback'), 'traceback surfaced');
  r = await runner.run('print("still alive")');
  assert.strictEqual(r.stdout, 'still alive\n');
  console.log('✓ 6. exceptions surfaced as tracebacks; runtime recovers');

  // 7. async Python
  r = await runner.run('import asyncio\nawait asyncio.sleep(0)\nprint("async ok")');
  assert.strictEqual(r.ok, true);
  assert.strictEqual(r.stdout, 'async ok\n');
  console.log('✓ 7. top-level await / async Python works');

  // 8. repeated execution, including concurrent submissions (queue must serialize)
  for (let i = 0; i < 3; i++) {
    r = await runner.run(`print("rep${i}")`);
    assert.strictEqual(r.stdout, `rep${i}\n`);
  }
  // Reset counter, then fire three runs at once; each appends to a shared
  // Python-side list.  Because runs are serialized the list must end ordered
  // and complete — any overlap would corrupt or interleave.
  await runner.run('counter = []');
  await Promise.all([
    runner.run('import time; counter.append(1)'),
    runner.run('counter.append(2)'),
    runner.run('counter.append(3)'),
  ]);
  r = await runner.run('print(sorted(counter), len(counter))');
  assert.strictEqual(r.stdout, '[1, 2, 3] 3\n', 'concurrent runs must serialize, not corrupt the runtime');
  console.log('✓ 8. repeated + concurrent executions serialize safely');

  // 9. reload reuse: ensureRuntime called repeatedly returns the same instance
  const a = await runner.ensureRuntime();
  const b = await runner.ensureRuntime();
  assert.strictEqual(a, b, 'runtime must be reused, never reinitialized');
  assert.strictEqual(runner.isLoaded(), true);
  console.log('✓ 9. Pyodide runtime is lazily created once and reused');

  // 10. Static routing guard: dashboard sends ordinary code cells to Pyodide
  //     and keeps the AI/model path on the configured remote backend; nothing
  //     ordinary is sent to Modal's /api/run-cell or /api/stream-logs.
  const dash = fs.readFileSync(path.join(__dirname, '..', 'dashboard.html'), 'utf8');
  assert(dash.includes('<script src="/pyodide_local.js"></script>'), 'dashboard loads pyodide_local.js');
  assert(dash.includes("cell.type === 'code' && !cell.modelInfo"),
    'ordinary code cells are routed via the Pyodide branch');
  assert(dash.includes('window.PyodideLocal.run('), 'Pyodide runner invoked for ordinary cells');
  assert(dash.includes('runCellViaZeroGPU(notebook, cell)'), 'model/inference cells still route to ZeroGPU');
  assert(dash.includes('streamCellToTerminal(executionCode)'), 'modal AI path unchanged');
  const pyBranchIdx = dash.indexOf("cell.type === 'code' && !cell.modelInfo");
  const remoteBranchIdx = dash.indexOf("cell.type === 'code' || cell.type === 'model'");
  assert(pyBranchIdx > -1 && remoteBranchIdx > -1 && pyBranchIdx < remoteBranchIdx,
    'Pyodide branch must precede the remote/GPU branch so ordinary code never reaches Modal');
  console.log('✓ 10. dashboard routing: ordinary Python → Pyodide; AI cells → ZeroGPU/Modal');

  console.log('\nAll Pyodide local-execution tests passed.');
}

main().catch((e) => {
  console.error('TEST FAILED:', e);
  process.exit(1);
});
