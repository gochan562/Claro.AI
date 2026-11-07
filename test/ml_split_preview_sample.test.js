'use strict';
// Regression: POST /api/ml/split/preview with the dataset cell's exact
// payload (dataset_view.sample = {}) must pass validation — the request
// proceeds to the workspace/Python stage instead of 400
// 'sample needs n or frac.' Asserts against the LIVE Express app.
//
// Stale-deployment signal: a server running pre-fix validateView returns
// 400 bad_request 'sample needs n or frac.' for these payloads. Any such
// response here means the running process predates the fix.

const assert = require('assert');
const http = require('http');

const app = require('../server.js');
const ml = require('../ml_experiment_backend.js');

let passed = 0;
function ok(name) { console.log(`✓ ${name}`); passed++; }

function post(port, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request({
      host: '127.0.0.1', port, path: '/api/ml/split/preview',
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

function tinyCsv(n) {
  const rows = ['age,income,city,bought'];
  let s = 7;
  const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const cities = ['a', 'b', 'c'];
  for (let i = 0; i < n; i++) {
    const age = 18 + Math.floor(rnd() * 52), income = Math.round(30000 + rnd() * 60000);
    const city = cities[Math.floor(rnd() * 3)];
    rows.push(`${age},${income},${city},${(income > 55000) !== (city === 'a') ? 'yes' : 'no'}`);
  }
  return rows.join('\n');
}

async function main() {
  const server = await new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  const port = server.address().port;

  // Real workspace so a passing request can succeed end-to-end (Python).
  const up = ml.validateDatasetUpload({ filename: 'customers.csv', content: tinyCsv(150) });
  const ing = await ml.ingestUpload(up);
  try {
    const SPLIT = { strategy: 'train_val_test', train_frac: 0.7, val_frac: 0.15,
      test_frac: 0.15, seed: 42, shuffle: true, stratify: true };

    // ── 1. The exact reported payload (no workspace_id, no split): validation passes ──
    {
      const r = await post(port, {
        dataset_view: { drop_columns: [], filters: [], sample: {} },
        target_column: 'Species',
        task: 'classification',
      });
      assert(r.status !== 400 || r.body.code !== 'bad_request' ||
        !String(r.body.error).includes('sample needs n or frac'),
        `stale validateView ran: ${r.status} ${r.raw}`);
      // With no workspace_id the request now fails LATER (workspace stage),
      // which is the correct next step — never the sampling validation.
      assert(!String(r.body.error || '').includes('sample'),
        `sample error leaked: ${r.raw}`);
      ok('reported payload (sample:{}) passes validation — no "sample needs n or frac"');
    }

    // ── 2. Same view with a real workspace + split: HTTP 200 end-to-end ──
    {
      const r = await post(port, {
        workspace_id: ing.workspace_id,
        split: SPLIT,
        dataset_view: { drop_columns: [], filters: [], sample: {} },
        target_column: 'bought',
        task: 'classification',
      });
      assert.strictEqual(r.status, 200, `expected 200, got ${r.status}: ${r.raw}`);
      const total = r.body.counts.train + r.body.counts.validation + r.body.counts.test;
      assert.strictEqual(total, 150, 'unsampled preview covers all rows');
      ok('sample:{} split preview reaches ml/split.py (HTTP 200, 150 rows)');
    }

    // ── 3. Seed-only sample: also no sampling request ──
    {
      const r = await post(port, {
        workspace_id: ing.workspace_id,
        split: SPLIT,
        dataset_view: { drop_columns: [], filters: [], sample: { seed: 42 } },
        target_column: 'bought',
        task: 'classification',
      });
      assert.strictEqual(r.status, 200, `seed-only preview: ${r.status} ${r.raw}`);
      const total = r.body.counts.train + r.body.counts.validation + r.body.counts.test;
      assert.strictEqual(total, 150, 'seed-only preview is unsampled');
      ok('sample:{seed:42} treated as no sampling (HTTP 200, 150 rows)');
    }

    // ── 4. Real sampling still works through the endpoint ──
    {
      const r = await post(port, {
        workspace_id: ing.workspace_id,
        split: SPLIT,
        dataset_view: { drop_columns: [], filters: [], sample: { n: 100, seed: 42 } },
        target_column: 'bought',
        task: 'classification',
      });
      assert.strictEqual(r.status, 200, `n-sampling preview: ${r.status} ${r.raw}`);
      const total = r.body.counts.train + r.body.counts.validation + r.body.counts.test;
      assert.strictEqual(total, 100, 'n:100 sample applies (100 rows)');
      ok('sample:{n:100,seed:42} applies row sampling (HTTP 200, 100 rows)');
    }
  } finally {
    server.close();
    try { fs_rmWorkspace(ing.workspace_id); } catch (_) {}
  }

  console.log(`\nAll split-preview sample tests passed (${passed} checks).`);
}

function fs_rmWorkspace(ws) {
  const fs = require('fs');
  const path = require('path');
  fs.rmSync(path.join(__dirname, '..', 'ml_workspaces', ws), { recursive: true, force: true });
}

main().catch((e) => { console.error(e); process.exit(1); });
