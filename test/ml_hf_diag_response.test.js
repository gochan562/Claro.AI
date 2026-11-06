'use strict';
// Regression: the exact structured `ingest_error` response path for
// POST /api/ml/datasets/hf must carry the backend's sanitized diagnostics
// plus the deployment marker. Asserts against the LIVE Express app
// (real _mlErr JSON), not source text — a stale server process would fail
// the diagnostics_version assertion, which is the point.
//
// Covers:
//   1. Python structured error, exit 1 (classic _fail path)
//   2. Python structured error, exit 0 (returned, not exited — must
//      still be forwarded)
//   3. Python crash without JSON (synthesized diagnostics)
//   4. Success responses stay marker-free (error-only marker)

const assert = require('assert');
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');

const app = require('../server.js');

let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

function writeFakeBin(name, body) {
  const p = path.join(os.tmpdir(), name);
  fs.writeFileSync(p, `#!/usr/bin/env node\n${body}\n`, 'utf8');
  fs.chmodSync(p, 0o755);
  return p;
}

function post(port, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request({
      host: '127.0.0.1', port, path: '/api/ml/datasets/hf',
      method: 'POST', headers: { 'Content-Type': 'application/json' },
    }, (res) => {
      let raw = '';
      res.on('data', (c) => { raw += c; });
      res.on('end', () => {
        let json = null;
        try { json = JSON.parse(raw); } catch (_) {}
        resolve({ status: res.statusCode, body: json, raw });
      });
    });
    req.on('error', reject);
    req.end(data);
  });
}

const QUOTA_DIAG = {
  error_type: 'DatasetError',
  error_message: 'Could not save dataset to workspace: [Errno 122] Disk quota exceeded',
  root_cause_type: 'OSError',
  root_cause_message: '[Errno 122] Disk quota exceeded',
  failing_operation: 'save_workspace:write data.csv',
  errno: 122,
  os_error: 'Disk quota exceeded',
  paths: ['/home/runner/workspace/ml_workspaces/mlw_x/data.csv'],
  disk: [{ path: '/home/runner/workspace/ml_workspaces/mlw_x',
           free_bytes: 123456789, total_bytes: 220000000000 }],
};

async function main() {
  const server = await new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  const port = server.address().port;
  const realBin = process.env.PYTHON_BIN;

  try {
    // ── 1. Structured error, exit 1: diagnostics + marker in the JSON body ──
    {
      const bin = writeFakeBin('ml-http-quota.js',
        `console.log(JSON.stringify({ type: 'error', ` +
        `error: 'Hugging Face dataset load failed: Could not save dataset to workspace: [Errno 122] Disk quota exceeded', ` +
        `code: 'ingest_error', diagnostics: ${JSON.stringify(QUOTA_DIAG)} })); process.exit(1);`);
      process.env.PYTHON_BIN = bin;
      try {
        const r = await post(port, { dataset_id: 'Salesforce/wikitext',
          split: 'train', config: 'wikitext-2-raw-v1' });
        assert.strictEqual(r.status, 502, `status, got ${r.status}: ${r.raw}`);
        assert.strictEqual(r.body.code, 'ingest_error');
        assert(r.body.error.includes('Disk quota exceeded'), 'error message present');
        const d = r.body.diagnostics;
        assert(d, `diagnostics MISSING from response: ${r.raw}`);
        assert.strictEqual(d.root_cause_type, 'OSError');
        assert.strictEqual(d.errno, 122);
        assert.strictEqual(d.failing_operation, 'save_workspace:write data.csv');
        assert(Array.isArray(d.paths) && d.paths.length, 'paths present');
        assert(Array.isArray(d.disk) && typeof d.disk[0].free_bytes === 'number',
          'statvfs free space present');
        assert.strictEqual(r.body.diagnostics_version, 'hf-diag-v1',
          'deployment marker MISSING — running server is STALE');
        assert(!r.raw.includes('Traceback'), 'no traceback in response');
        ok('structured ingest_error response carries diagnostics + hf-diag-v1 marker');
      } finally {
        if (realBin === undefined) delete process.env.PYTHON_BIN; else process.env.PYTHON_BIN = realBin;
        fs.unlinkSync(bin);
      }
    }

    // ── 2. Structured error returned WITHOUT exiting non-zero: forwarded too ──
    {
      const bin = writeFakeBin('ml-http-quota-exit0.js',
        `console.log(JSON.stringify({ type: 'error', ` +
        `error: 'Hugging Face dataset load failed: Could not save dataset to workspace: [Errno 122] Disk quota exceeded', ` +
        `code: 'ingest_error', diagnostics: ${JSON.stringify(QUOTA_DIAG)} })); process.exit(0);`);
      process.env.PYTHON_BIN = bin;
      try {
        const r = await post(port, { dataset_id: 'Salesforce/wikitext', split: 'train' });
        assert.strictEqual(r.status, 502);
        assert.strictEqual(r.body.code, 'ingest_error');
        assert(r.body.diagnostics, 'diagnostics forwarded on exit-0 structured error');
        assert.strictEqual(r.body.diagnostics.errno, 122);
        assert.strictEqual(r.body.diagnostics_version, 'hf-diag-v1');
        ok('exit-0 structured error still forwarded with diagnostics + marker');
      } finally {
        if (realBin === undefined) delete process.env.PYTHON_BIN; else process.env.PYTHON_BIN = realBin;
        fs.unlinkSync(bin);
      }
    }

    // ── 3. Crash without JSON: synthesized diagnostics + marker ──
    {
      const bin = writeFakeBin('ml-http-crash.js',
        `console.error('Traceback (most recent call last):\\n  MemoryError'); process.exit(1);`);
      process.env.PYTHON_BIN = bin;
      try {
        const r = await post(port, { dataset_id: 'a/b', split: null });
        assert.strictEqual(r.status, 502);
        assert.strictEqual(r.body.code, 'ingest_error');
        const d = r.body.diagnostics;
        assert(d, `synthesized diagnostics MISSING: ${r.raw}`);
        assert.strictEqual(d.error_type, 'SubprocessCrash');
        assert(Array.isArray(d.disk) && d.disk.length, 'local disk snapshot present');
        assert.strictEqual(r.body.diagnostics_version, 'hf-diag-v1');
        // Sanitize policy: bare exception TYPE NAME allowed (same class as
        // Python's root_cause_type); raw stderr text and traceback framing
        // must never appear.
        assert.strictEqual(d.root_cause_type, 'MemoryError',
          'bare exception type extracted from crash');
        assert(!r.raw.includes('Traceback'), 'no traceback framing in response');
        assert(!d.error_message.includes('MemoryError') &&
               !d.error_message.includes('Traceback'),
          'raw stderr text stays out of error_message');
        ok('crash response carries synthesized diagnostics + marker, no stderr leak');
      } finally {
        if (realBin === undefined) delete process.env.PYTHON_BIN; else process.env.PYTHON_BIN = realBin;
        fs.unlinkSync(bin);
      }
    }

    // ── 4. Marker is error-path-only (sanity) ──
    {
      // Validation error never reaches the ingest path → no marker, no diagnostics.
      const r = await post(port, { dataset_id: 'not-a-valid-id' });
      assert.strictEqual(r.status, 400);
      assert.strictEqual(r.body.code, 'bad_request');
      assert.strictEqual(r.body.diagnostics_version, undefined);
      assert.strictEqual(r.body.diagnostics, undefined);
      ok('validation errors stay marker-free (marker identifies the ingest path)');
    }
  } finally {
    server.close();
  }

  console.log(`\nAll HF diagnostics-response tests passed (${passed} checks).`);
}

main().catch((e) => { console.error(e); process.exit(1); });
