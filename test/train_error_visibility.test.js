'use strict';
// Training-failure visibility tests: no error may disappear silently.
//  1. Fast backend failure is replayed to late SSE attachers (logs + done+error).
//  2. Running jobs get NO replay (no duplication risk).
//  3. Own-timeout of the train POST reports timeout (never user-cancel);
//     an outer abort still reports cancelled.
//  4. UI renders backend failure visibly even with zero prior log frames
//     (error box + reason line in the log box), via jsdom + fake EventSource.
const assert = require('assert');
const fs = require('fs');
const http = require('http');
const path = require('path');

const tb = require('../training_backend');

function cleanupJobs() {
  for (const j of tb.jobs.values()) {
    try { fs.rmSync(path.join(__dirname, '..', 'training_outputs', j.job_id), { recursive: true, force: true }); } catch (_) {}
  }
  tb.clearAllJobs();
}
function validConfig() {
  return tb.validateTrainingRequest({
    model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification',
    epochs: 1, batch_size: 8, learning_rate: 2e-5, validation_split: 10,
  });
}
function fakeRes() {
  return { writes: [], write(s) { this.writes.push(s); }, on() {}, events() {
    return this.writes.join('').split('\n\n').filter(Boolean).map((f) => {
      const ev = (f.match(/^event:\s*(.*)$/m) || [])[1] || '';
      const dm = f.match(/^data:\s*([\s\S]*)$/m);
      return { event: ev.trim(), data: dm ? dm[1].trim() : '' };
    });
  } };
}

async function run() {
  // ── 1. terminal replay: logs + done+error reach a late attacher ──
  {
    tb.clearAllJobs();
    const job = tb.createJob(validConfig(), 'local-user');
    // Simulate a fast failure with NO prior SSE client (failure before attach).
    // Use the real funnel indirectly: mark failed exactly like _failZeroGpuJob does.
    tb._log(job, '[TRAIN] training_error: ZeroGPU call failed (HTTP 500): Internal Server Error');
    tb._setStatus(job, 'failed');
    tb._updateProgress(job, { gpu_status: 'idle', eta: 0 });
    job.error = 'training_error';
    job.end_time = Date.now();
    tb._broadcast(job, 'done', { status: 'failed', error: job.error, message: 'boom' });
    const res = fakeRes();
    assert.strictEqual(tb.attachSSE(job.job_id, res), true);
    const evs = res.events();
    const kinds = evs.map((e) => e.event);
    assert.ok(kinds.includes('status'), 'status replayed');
    assert.ok(kinds.includes('log'), `log frames replayed, got: ${kinds.join(',')}`);
    const dones = evs.filter((e) => e.event === 'done');
    assert.strictEqual(dones.length, 1, 'exactly one terminal done replay');
    const done = JSON.parse(dones[0].data);
    assert.strictEqual(done.status, 'failed');
    assert.strictEqual(done.error, 'training_error', 'error must survive to late attachers');
    const logLines = evs.filter((e) => e.event === 'log').map((e) => JSON.parse(e.data).line);
    assert.ok(logLines.some((l) => l.includes('training_error')), 'failure line replayed');
    console.log('✓ 1. late SSE attach receives logs + done{error} for terminal jobs');
    cleanupJobs();
  }

  // ── 2. running jobs: no replay, no duplication ──
  {
    tb.clearAllJobs();
    const job = tb.createJob(validConfig(), 'local-user');
    tb._log(job, '[TRAIN] loading something');
    const res = fakeRes();
    tb.attachSSE(job.job_id, res);
    const kinds = res.events().map((e) => e.event);
    assert.ok(!kinds.includes('done'), 'running jobs must not get a done replay');
    assert.ok(!kinds.includes('log'), 'running jobs stream live; no history replay');
    console.log('✓ 2. running jobs get no replay (duplication impossible)');
    cleanupJobs();
  }

  // ── 3. timeout vs cancel are distinct ──
  {
    const server = http.createServer(() => {}); // hangs forever
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    const api = `http://127.0.0.1:${server.address().port}/gradio_api`;
    try {
      // own timeout, no outer signal -> timeout (never user-cancel)
      let err = null;
      try {
        await tb._postZeroGpuFn(api, 'train', { job_id: 'train_x' }, undefined, '', 300);
      } catch (e) { err = e; }
      assert(err, 'expected a timeout error');
      assert.strictEqual(err.code, 'timeout', `own timeout must be 'timeout', got '${err && err.code}'`);
      assert.ok(!/cancelled by user/i.test(err.message), 'timeout must not claim user cancellation');
      console.log('✓ 3a. POST timeout reports timeout, not user-cancel');
      // outer abort -> cancelled
      const ctrl = new AbortController();
      const p = tb._postZeroGpuFn(api, 'train', { job_id: 'train_x' }, ctrl.signal, '', 10000);
      ctrl.abort();
      err = null;
      try { await p; } catch (e) { err = e; }
      assert(err && err.code === 'cancelled', `outer abort must be 'cancelled', got '${err && err.code}'`);
      console.log('✓ 3b. outer abort still reports cancelled');
    } finally {
      server.close();
    }
  }

  // ── 4. UI renders backend failure with zero prior logs (jsdom) ──
  {
    const { JSDOM } = require('jsdom');
    const html = fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard.html'), 'utf8');
    const dom = new JSDOM(html, { url: 'http://localhost:5000/dashboard.html', runScripts: 'outside-only' });
    const { window } = dom;
    const instances = [];
    window.fetch = () => Promise.reject(new Error('no network in test'));
    window.EventSource = function (url) {
      this.url = url;
      this.handlers = {};
      this.addEventListener = (t, fn) => { (this.handlers[t] = this.handlers[t] || []).push(fn); };
      this.close = () => {};
      instances.push(this);
    };
    window.eval(fs.readFileSync(path.join(__dirname, '..', 'public', 'training_ui.js'), 'utf8'));
    window.eval(fs.readFileSync(path.join(__dirname, '..', 'public', 'dashboard-app.js'), 'utf8'));
    const registry = { notebooks: { 1: { id: 1, name: 'nb', cells: [
      { id: 7, type: 'training', training: {
        model_preset: 'distilbert-base', dataset_preset: 'imdb',
        model_id: 'distilbert-base-uncased', dataset_id: 'stanfordnlp/imdb',
        task_type: 'text-classification', epochs: 2, batch_size: 8,
        learning_rate: 0.00002, validation_split: 10, max_steps: '',
        training_method: 'auto', lora_r: 8, lora_alpha: 16, lora_dropout: 0.05,
        target_modules: 'auto', job_id: 'train_vis1', status: 'loading',
        progress: { current_epoch: 0, current_step: 0, train_loss: null, eval_loss: null, eta: null, gpu_status: 'zerogpu', training_method: 'full', trainable_params: null, total_params: null },
        metrics: [], logs: [],
      } },
    ] } }, notebookOrder: [1], activeNotebookId: 1 };
    window.localStorage.setItem('claro-notebooks-registry', JSON.stringify(registry));
    window.loadAllNotebooks();
    window.renderNotebookEditor();
    window.attachTrainingSSE(7, 'train_vis1');
    assert.strictEqual(instances.length, 1, 'EventSource constructed');
    const es = instances[0];
    const fire = (type, obj) => {
      for (const fn of es.handlers[type] || []) fn({ data: JSON.stringify(obj) });
    };
    // Backend fails fast with NO prior log frames — the exact silent case.
    fire('done', { status: 'failed', error: 'training_error', message: 'ZeroGPU call failed (HTTP 500)' });
    const errBox = window.document.getElementById('tr-7-errorbox');
    assert.ok(errBox && errBox.style.display !== 'none', 'error box must be visible on backend failure');
    assert.ok(errBox.innerHTML.includes('training_error'), 'error box names the failure');
    const logsEl = window.document.getElementById('tr-7-logs');
    assert.ok(logsEl && logsEl.style.display !== 'none', 'log box must be unhidden');
    assert.ok(logsEl.textContent.includes('Training failed'), `reason line rendered, got: ${logsEl.textContent.slice(0, 120)}`);
    assert.ok(logsEl.textContent.includes('HTTP 500'), 'backend message preserved');
    console.log('✓ 4. failed-with-no-logs renders error box + reason line');
  }

  console.log('\nAll train error visibility tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
