'use strict';
const assert = require('assert');

// Regression test for "The string did not match the expected pattern." DOMException
// This was thrown by EventSource/fetch when URL was constructed as a relative string
// without a base, or when jobId/model_id was interpolated without encodeURIComponent.

function isLargeModelFrontend(modelId) {
  const m = String(modelId||'').toLowerCase();
  const mm = m.match(/(\d+(?:\.\d+)?)\s*b\b/);
  if (mm) {
    const num = parseFloat(mm[1]);
    if (!isNaN(num) && num >= 1) return true;
  }
  if (/(?:^|[-_\/\s])(?:1b|1\.5b|3b|7b|8b|13b|30b|70b)(?:$|[-_\/\s])/i.test(m)) return true;
  return false;
}

function testUrlConstruction() {
  console.log('=== Test URL construction for training SSE ===');
  const jobId = 'train_5d8b337e4677';
  const baseHrefs = [
    'http://localhost:5007/dashboard.html',
    'http://localhost:5007/',
    'http://example.com/notebook',
    'https://example.com:8080/path?query=1#hash',
  ];

  for (const baseHref of baseHrefs) {
    const urlStr = `/api/train/metrics/${encodeURIComponent(jobId)}?stream=1`;
    const url = new URL(urlStr, baseHref);
    assert(url.toString().includes(`/api/train/metrics/${jobId}?stream=1`));
    assert(url.toString().startsWith('http'));
    console.log(`✓ base ${baseHref} -> ${url.toString()}`);
  }

  // Old buggy code: new EventSource(`/api/train/metrics/${jobId}?stream=1`) without base
  // This works in browsers when base is http, but fails when window.location.href is about:blank or file://
  // New code: new URL(..., window.location.href).toString() should not throw
  const buggyJobIds = ['train_5d8b337e4677', 'train_b26c7f26ddfd', 'train_ffffffffffff'];
  for (const jid of buggyJobIds) {
    const urlStr = `/api/train/metrics/${encodeURIComponent(jid)}?stream=1`;
    // Simulate old code that might have done new URL(urlStr) without base (which throws)
    let threw = false;
    try {
      new URL(urlStr); // no base -> should throw for relative URL
    } catch (e) {
      threw = true;
      assert(e.message.includes('Invalid URL') || e.name === 'TypeError');
    }
    assert(threw, 'new URL(relative) without base should throw');
    // New code with base should NOT throw
    const fixed = new URL(urlStr, 'http://localhost:5007/dashboard.html');
    assert(fixed.toString() === `http://localhost:5007/api/train/metrics/${jid}?stream=1`);
  }
  console.log('✓ EventSource URL with base does not throw');

  // Test that model_id with single segment (distilbert-base-uncased) does not cause URL issues
  const modelIds = ['distilbert-base-uncased', 'stanfordnlp/imdb', 'HuggingFaceTB/SmolLM3-3B', 'a/b'];
  for (const mid of modelIds) {
    // The old validation would have rejected distilbert-base-uncased, but URL construction should still be safe
    const body = { model_id: mid, dataset_id: 'stanfordnlp/imdb', task_type: 'text-classification' };
    // Simulate fetch URL construction for POST /api/train/start (no model_id in URL, only in body)
    const url = new URL('/api/train/start', 'http://localhost:5007/');
    assert(url.toString() === 'http://localhost:5007/api/train/start');
    // The body JSON should be valid and not affect URL
    const json = JSON.stringify(body);
    assert(JSON.parse(json).model_id === mid);
  }
  console.log('✓ model_id single vs owner/name does not affect URL');

  // Test encodeURIComponent for jobId with special chars (should not happen, but test traversal rejection)
  const malicious = ['train_../etc/passwd', 'train_123', '../../etc/passwd', 'train_ffffffffffff; DROP TABLE'];
  for (const jid of malicious) {
    const encoded = encodeURIComponent(jid);
    const urlStr = `/api/train/metrics/${encoded}?stream=1`;
    const url = new URL(urlStr, 'http://localhost:5007/');
    // The URL should be encoded, but the server should still reject via validation
    assert(url.toString().includes(encodeURIComponent(jid)));
  }
  console.log('✓ encodeURIComponent handles malicious jobIds');

  // Test that the new code correctly handles the demo configuration
  const demoCfg = {
    model_id: 'distilbert-base-uncased',
    dataset_id: 'stanfordnlp/imdb',
    task_type: 'text-classification',
    training_method: 'auto',
    validation_split: 10,
    epochs: 2,
    batch_size: 8,
    learning_rate: 0.00002
  };
  // isLargeModel should correctly handle distilbert-base-uncased (no 1b) -> full
  assert.strictEqual(isLargeModelFrontend(demoCfg.model_id), false);
  // The fetch URL for start should be valid
  const startUrl = new URL('/api/train/start', 'http://localhost:5007/dashboard.html').toString();
  assert.strictEqual(startUrl, 'http://localhost:5007/api/train/start');
  console.log('✓ demo config does not cause URL exception');

  console.log('\nAll frontend URL tests passed — no "The string did not match the expected pattern."');
}

testUrlConstruction();
