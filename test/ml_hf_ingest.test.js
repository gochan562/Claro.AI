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

  // ── 1b. Missing Python dependency emits structured JSON, not a traceback ──
  // (exact production symptom: bare `import joblib` killed module import
  // before any command — and its error handler — could run).
  {
    const bin = path.join(os.tmpdir(), 'ml-fake-nodeps.sh');
    fs.writeFileSync(bin, '#!/bin/sh\nexec python3 -S "$@"\n', 'utf8');
    fs.chmodSync(bin, 0o755);
    process.env.PYTHON_BIN = bin;
    try {
      await ml.ingestHf({ dataset_id: 'scikit-learn/iris', split: 'train', config: 'default' });
      assert.fail('ingestHf should have thrown');
    } catch (e) {
      assert.strictEqual(e.code, 'hf_dependency_error', `code, got ${e.code}: ${e.message}`);
      assert(e.message.includes('joblib'), `names the missing package: ${e.message}`);
      assert(!e.message.includes('Traceback'), 'no traceback in UI error');
    } finally {
      delete process.env.PYTHON_BIN;
      fs.unlinkSync(bin);
    }
    ok('missing dependency yields structured hf_dependency_error (import-safe entry)');
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

  // ── 2b. Disk-quota-style failure carries structured diagnostics ──
  // Production symptom: OSError Errno 122 on the HF/tmpfs write. Python's
  // structured error payload must reach Node as err.diagnostics (root
  // cause, paths, free space, failing op) with secrets scrubbed, and the
  // full detail must appear in the SERVER log ([ML-DIAG]/[TRAIN-DIAG]).
  {
    const diag = {
      error_type: 'DatasetError',
      error_message: 'Could not save dataset to workspace: [Errno 122] Disk quota exceeded',
      root_cause_type: 'OSError',
      root_cause_message: '[Errno 122] Disk quota exceeded',
      failing_operation: 'save_workspace:write data.csv',
      errno: 122,
      os_error: 'Disk quota exceeded',
      paths: ['/home/runner/workspace/ml_workspaces/mlw_x/data.csv',
              '/home/runner/.cache/huggingface/datasets'],
      disk: [{ path: '/home/runner/workspace/ml_workspaces/mlw_x',
               free_bytes: 123456789, total_bytes: 220000000000 }],
      // planted secrets — MUST be scrubbed from anything client-visible
      note: 'token hf_FAKE-token-abcdef1234567890abcdef leaked Bearer sk-fake-key-1234567890abcdef',
    };
    const bin = writeFakeBin('ml-fake-quota.js',
      `console.log('[TRAIN-DIAG] WRITE save_workspace:write data.csv path=/home/runner/workspace/ml_workspaces/mlw_x/data.csv free_GB=0.01');\n` +
      `console.log(JSON.stringify({ type: 'error', error: 'Hugging Face dataset load failed: Could not save dataset to workspace: [Errno 122] Disk quota exceeded', code: 'ingest_error', diagnostics: ${JSON.stringify(diag)} }));\n` +
      `process.exit(1);`);
    process.env.PYTHON_BIN = bin;
    const errs = [];
    const origErr = console.error;
    console.error = (...a) => { errs.push(a.map(String).join(' ')); };
    try {
      await ml.ingestHf({ dataset_id: 'Salesforce/wikitext', split: 'train', config: 'wikitext-2-raw-v1' });
      assert.fail('ingestHf should have thrown');
    } catch (e) {
      console.error = origErr;
      const d = e.diagnostics;
      assert(d, 'diagnostics attached to the error');
      assert.strictEqual(d.root_cause_type, 'OSError');
      assert.strictEqual(d.errno, 122);
      assert.strictEqual(d.failing_operation, 'save_workspace:write data.csv');
      assert(Array.isArray(d.paths) && d.paths.some(p => p.includes('data.csv')), 'failing path present');
      assert(Array.isArray(d.disk) && d.disk.length && typeof d.disk[0].free_bytes === 'number',
        'statvfs free space present');
      // sanitization: nothing token-shaped survives to the client-visible object
      const flat = JSON.stringify(d);
      assert(!flat.includes('hf_FAKE-token'), 'hf token scrubbed');
      assert(!flat.includes('sk-fake-key'), 'sk key scrubbed');
      assert(!flat.includes('Bearer sk-fake'), 'bearer secret scrubbed');
      assert(!flat.includes('Traceback'), 'no tracebacks in client diagnostics');
      // server log has the full detail
      const logAll = errs.join('\n');
      assert(logAll.includes('[ML-DIAG]'), 'structured diagnostics in server log');
      assert(logAll.includes('[TRAIN-DIAG]'), 'TRAIN-DIAG write log in server log');
      assert(logAll.includes('hf_FAKE-token'), 'unsanitized detail stays server-side only');
      assert(!e.message.includes('Traceback'), 'no traceback in UI message');
    } finally {
      console.error = origErr;
      delete process.env.PYTHON_BIN;
      fs.unlinkSync(bin);
    }
    ok('structured diagnostics reach Node error + server log, secrets scrubbed');
  }

  // ── 2c. Crash without JSON now synthesizes diagnostics (never opaque) ──
  {
    const bin = writeFakeBin('ml-fake-crash2.js',
      `console.error('Traceback (most recent call last):\\n  MemoryError'); process.exit(1);`);
    process.env.PYTHON_BIN = bin;
    try {
      await ml.ingestHf({ dataset_id: 'a/b', split: null });
      assert.fail('ingestHf should have thrown');
    } catch (e) {
      const d = e.diagnostics;
      assert(d, 'synthesized diagnostics on crash');
      assert.strictEqual(d.error_type, 'SubprocessCrash');
      assert(Array.isArray(d.disk) && d.disk.length, 'local free-space snapshot present');
      assert(Array.isArray(d.paths) && d.paths.some(p => p.includes('ml_workspaces')),
        'workspace path present');
    } finally {
      delete process.env.PYTHON_BIN;
      fs.unlinkSync(bin);
    }
    ok('crash without JSON synthesizes diagnostics (SubprocessCrash + disk)');
  }
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

    // 3a2. WikiText regression: config wikitext-2-raw-v1 + split train.
    // (Previously failed opaquely; the loader must succeed and the manifest
    // must record the exact config used.)
    const wiki = await ml.ingestHf({ dataset_id: 'Salesforce/wikitext', split: 'train', config: 'wikitext-2-raw-v1' });
    assert.strictEqual(wiki.profile.row_count, 36718, 'wikitext-2-raw-v1 train has 36,718 rows');
    assert(wiki.profile.columns.includes('text'), 'wikitext text column present');
    const wikiManifest = JSON.parse(fs.readFileSync(
      path.join(__dirname, '..', 'ml_workspaces', wiki.workspace_id, 'manifest.json'), 'utf8'));
    assert.strictEqual(wikiManifest.config, 'wikitext-2-raw-v1');
    ok('live wikitext load (config=wikitext-2-raw-v1, split=train, 36718 rows)');
    try { fs.rmSync(path.join(__dirname, '..', 'ml_workspaces', wiki.workspace_id), { recursive: true, force: true }); } catch (_) {}

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
