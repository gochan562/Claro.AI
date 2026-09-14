'use strict';
// Classification label display tests (training_ui.resolveDisplayLabel is
// display-only; inference_runner._lookup_label stays raw).
//  1. IMDb LABEL_0 -> Negative
//  2. IMDb LABEL_1 -> Positive
//  3. Meaningful id2label preserved (casing normalized, mixed-case untouched)
//  4. Unknown LABEL_0/LABEL_1 NOT blindly mapped to sentiment
//  5. Raw label remains available internally (no mutation, runner stays raw)
//  6. Non-IMDb classification models unaffected
const assert = require('assert');
const { execFileSync } = require('child_process');
const path = require('path');
const ui = require('../training_ui');

const R = (o) => ui.resolveDisplayLabel(o);
const IMDB = { taskType: 'text-classification', datasetId: 'stanfordnlp/imdb', modelId: 'distilbert-base-uncased', numLabels: 2 };

async function run() {
  // 1 + 2. IMDb sentiment mapping
  let r = R({ label: 'LABEL_0', labelId: 0, ...IMDB });
  assert.strictEqual(r.display, 'Negative');
  assert.strictEqual(r.raw, 'LABEL_0');
  r = R({ label: 'LABEL_1', labelId: 1, ...IMDB });
  assert.strictEqual(r.display, 'Positive');
  assert.strictEqual(r.raw, 'LABEL_1');
  console.log('✓ 1+2. IMDb LABEL_0 -> Negative, LABEL_1 -> Positive (raw kept)');

  // case-insensitive generic form + imdb in model id also qualifies
  r = R({ label: 'label_1', labelId: 1, taskType: 'text-classification', modelId: 'someone/imdb-sentiment', datasetId: '', numLabels: 2 });
  assert.strictEqual(r.display, 'Positive');
  console.log('✓ imdb signal via model id works; generic form is case-insensitive');

  // 3. meaningful id2label preserved, casing normalized
  r = R({ label: 'NEGATIVE', labelId: 0, taskType: 'text-classification', datasetId: 'x/y', numLabels: 2 });
  assert.strictEqual(r.display, 'Negative');
  assert.strictEqual(r.raw, 'NEGATIVE');
  r = R({ label: 'positive', labelId: 1, taskType: 'text-classification', datasetId: 'x/y', numLabels: 2 });
  assert.strictEqual(r.display, 'Positive');
  r = R({ label: 'Sci/Tech', labelId: 2, taskType: 'text-classification', datasetId: 'ag_news', numLabels: 4 });
  assert.strictEqual(r.display, 'Sci/Tech', 'mixed-case labels untouched');
  r = R({ label: 'entailment', labelId: 0, taskType: 'text-classification', datasetId: 'glue', numLabels: 3 });
  assert.strictEqual(r.display, 'Entailment');
  console.log('✓ 3. meaningful id2label preserved (NEGATIVE->Negative, Sci/Tech untouched)');

  // 4. unknown generics NOT mapped
  r = R({ label: 'LABEL_0', labelId: 0, taskType: 'text-classification', datasetId: 'ag_news', modelId: 'bert-base', numLabels: 4 });
  assert.strictEqual(r.display, 'LABEL_0', 'non-IMDb must not become Negative');
  r = R({ label: 'LABEL_2', labelId: 2, ...IMDB, numLabels: 3 });
  assert.strictEqual(r.display, 'LABEL_2', 'non-binary IMDb model must not map');
  r = R({ label: 'LABEL_0', labelId: 0, taskType: 'image-classification', datasetId: 'stanfordnlp/imdb', numLabels: 2 });
  assert.strictEqual(r.display, 'LABEL_0', 'mapping is text-classification only');
  r = R({ label: 'LABEL_0', labelId: 0, taskType: 'text-classification', datasetId: '', modelId: '', numLabels: 2 });
  assert.strictEqual(r.display, 'LABEL_0', 'no IMDb signal -> no mapping');
  console.log('✓ 4. unknown LABEL_X never blindly mapped to sentiment');

  // 5. raw stays available; helper is pure (no input mutation)
  const input = { label: 'LABEL_1', labelId: 1, scores: [{ label: 'x' }], ...IMDB };
  const snap = JSON.parse(JSON.stringify(input));
  r = R(input);
  assert.deepStrictEqual(input, snap, 'helper must not mutate its input');
  assert.strictEqual(r.raw, 'LABEL_1');
  const py = execFileSync('python3', ['-c',
    'import inference_runner as r;'
    + 'assert r._lookup_label({0:"LABEL_0",1:"LABEL_1"}, 0) == "LABEL_0";'
    + 'assert r._lookup_label({"0":"NEGATIVE","1":"POSITIVE"}, 1) == "POSITIVE";'
    + 'assert r._lookup_label({}, 0) == "Label 0";'
    + 'assert r._lookup_label(None, 3) == "Label 3";'
    + 'src = open("inference_runner.py").read();'
    + 'assert src.count(chr(34)+"raw_label"+chr(34)) >= 4, "runner must emit raw_label in classification outputs";'
    + 'print("runner raw contract OK")',
  ], { cwd: path.join(__dirname, '..'), timeout: 30000 }).toString().trim();
  assert.ok(py.includes('runner raw contract OK'), py);
  console.log('✓ 5. raw label available internally (runner raw, helper pure)');

  // 6. non-IMDb models unaffected end-to-end through the helper
  const ag = [
    R({ label: 'LABEL_0', labelId: 0, taskType: 'text-classification', datasetId: 'ag_news', modelId: 'bert-base-uncased', numLabels: 4 }),
    R({ label: 'LABEL_3', labelId: 3, taskType: 'text-classification', datasetId: 'ag_news', modelId: 'bert-base-uncased', numLabels: 4 }),
  ];
  assert.deepStrictEqual(ag.map((x) => x.display), ['LABEL_0', 'LABEL_3']);
  const img = R({ label: 'LABEL_5', labelId: 5, taskType: 'image-classification', datasetId: 'cifar10', modelId: 'google/vit-base-patch16-224', numLabels: 10 });
  assert.strictEqual(img.display, 'LABEL_5');
  console.log('✓ 6. non-IMDb classification models unaffected');

  console.log('\nAll label display tests passed.');
}

run().catch((e) => { console.error('TEST FAILED:', e); process.exit(1); });
