'use strict';
// HF ingestion subprocess tests: structured-error propagation, crash
// fallback, and live acceptance against a tiny public dataset.
// Hermetic parts always run; live parts skip cleanly without HF network.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ml = require('../ml_experiment_backend');

const REAL_PYTHON_BIN = process.env.PYTHON_BIN;
let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

function writeFakeBin(name, body) {
  const p = path.join(os.tmpdir(), name);
  fs.writeFileSync(p, `#!/usr/bin/env node\n${body}\n`, 'utf8');
  fs.chmodSync(p, 0o755);
  return p;
}

async function hasHfNetwork() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    const res = await fetch('https://huggingface.co/api/datasets/scikit-learn/iris',
      { signal: ctrl.signal });
    clearTimeout(timer);
    return res.ok;
  } catch (_) {
    return false;
  }
}

async function run() {
  // ── 1. Structured _fail() JSON survives a non-zero exit ──
  {
    const bin = writeFakeBin('ml-fake-err.js',
      `console.log(JSON.stringify({ type: 'error', error: 'Dataset could not be found.', code: 'hf_dataset_not_found' })); process.exit(1);`);
    process.env.PYTHON_BIN = bin;
    try {
      await ml.ingestHf({ dataset_id: 'no/such', split: null });
      assert.fail('ingestHf should have thrown');
    } catch (e) {
      assert.strictEqual(e.code, 'hf_dataset_not_found', `code, got ${e.code}: ${e.message}`);
      assert.strictEqual(e.status, 404);
      assert(!e.message.includes('Traceback'), 'no traceback in UI error');
    } finally {
      delete process.env.PYTHON_BIN;
      fs.unlinkSync(bin);
    }
    ok('structured Python error survives exit 1 with code + status');
  }

  // ── 2. Crash without JSON stays generic, traceback stays server-side ──
  {
    const bin = writeFakeBin('ml-fake-crash.js',
      `console.error('Traceback (most recent call last):\\n  boom'); console.log('not json'); process.exit(1);`);
    process.env.PYTHON_BIN = bin;
    try {
      await ml.ingestHf({ dataset_id: 'a/b', split: null });
      assert.fail('ingestHf should have thrown');
    } catch (e) {
      assert.strictEqual(e.code, 'ingest_error');
      assert.strictEqual(e.status, 502);
      assert(!e.message.includes('Traceback'), 'traceback must not leak to UI error');
      assert(!e.message.includes('boom'), 'stderr must not leak to UI error');
    } finally {
      delete process.env.PYTHON_BIN;
      fs.unlinkSync(bin);
    }
    ok('crash without JSON falls back generic without leaking stderr');
  }

  // ── 3. Live acceptance (tiny public dataset; skipped offline) ──
  if (!(await hasHfNetwork())) {
    console.log('⊘ live HF acceptance skipped (no Hugging Face network)');
  } else {
    if (REAL_PYTHON_BIN !== undefined) process.env.PYTHON_BIN = REAL_PYTHON_BIN;
    else delete process.env.PYTHON_BIN;
    // 3a. iris + config=default + split=train succeeds.
    const out = await ml.ingestHf({ dataset_id: 'scikit-learn/iris', split: 'train', config: 'default' });
    assert.strictEqual(out.profile.row_count, 150, 'iris has 150 rows');
    assert(out.profile.columns.includes('Species'), 'iris columns present');
    assert(/^mlw_[a-f0-9]{12}$/.test(out.workspace_id));
    // Config recorded on the manifest (durable reference for reruns).
    const manifest = JSON.parse(fs.readFileSync(
      path.join(__dirname, '..', 'ml_workspaces', out.workspace_id, 'manifest.json'), 'utf8'));
    assert.strictEqual(manifest.config, 'default');
    ok('live iris load (config=default, split=train, 150 rows)');
    try { fs.rmSync(path.join(__dirname, '..', 'ml_workspaces', out.workspace_id), { recursive: true, force: true }); } catch (_) {}

    // 3b. Invalid split/config/id map to structured codes (no generic 502).
    const cases = [
      [{ dataset_id: 'scikit-learn/iris', split: 'frobnicator' }, 'hf_split_not_found', 400],
      [{ dataset_id: 'scikit-learn/iris', split: 'train', config: 'nope-config' }, 'hf_config_not_found', 400],
      [{ dataset_id: 'no-such-owner-xyz/no-such-ds-xyz', split: null }, 'hf_dataset_not_found', 404],
    ];
    for (const [args, code, status] of cases) {
      let err = null;
      try { await ml.ingestHf(args); } catch (e) { err = e; }
      assert(err, `expected throw for ${JSON.stringify(args)}`);
      assert.strictEqual(err.code, code, `got ${err.code}: ${err.message}`);
      assert.strictEqual(err.status, status);
    }
    ok('live invalid split/config/dataset map to structured codes');
  }

  if (REAL_PYTHON_BIN !== undefined) process.env.PYTHON_BIN = REAL_PYTHON_BIN;
  else delete process.env.PYTHON_BIN;
  console.log(`\nAll HF ingest subprocess tests passed (${passed} checks).`);
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
