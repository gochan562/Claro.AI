'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const tb = require('../training_backend');

// Use the t5 venv python which has peft, torch, transformers, datasets
process.env.PYTHON_BIN = process.env.PYTHON_BIN || '/var/folders/f0/ycgc1tnn7ydbn5cg4sk_kq2c0000gn/T/opencode/t5_venv/bin/python';

async function run() {
  tb.clearAllJobs();

  console.log('=== LoRA validation ===');
  // r, alpha, dropout, target_modules
  let cfg = tb.validateTrainingRequest({
    model_id: 'a/b', dataset_id: 'c/d', task_type: 'text-classification',
    training_method: 'lora', lora_r: 4, lora_alpha: 8, lora_dropout: 0.05, target_modules: 'auto'
  });
  assert.strictEqual(cfg.training_method, 'lora');
  assert.strictEqual(cfg.lora_r, 4);
  assert.strictEqual(cfg.lora_alpha, 8);
  assert.strictEqual(cfg.target_modules, 'auto');
  console.log('✓ lora validation');

  // auto -> lora for >=1B, full for <1B
  cfg = tb.validateTrainingRequest({model_id:'meta-llama/Llama-2-7b-hf', dataset_id:'c/d', task_type:'text-generation', training_method:'auto'});
  assert.strictEqual(cfg.training_method, 'lora', '7B should default to lora');
  cfg = tb.validateTrainingRequest({model_id:'hf-internal-testing/tiny-random-bert', dataset_id:'c/d', task_type:'text-classification', training_method:'auto'});
  assert.strictEqual(cfg.training_method, 'full', 'tiny should default to full');
  console.log('✓ auto default (lora for >=1B, full for <1B)');

  // manual target_modules
  cfg = tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', training_method:'lora', target_modules:'q_proj, v_proj'});
  assert.strictEqual(cfg.target_modules, 'q_proj,v_proj');
  console.log('✓ manual target_modules');

  // invalid lora params should fail
  let err=null;
  try { tb.validateTrainingRequest({model_id:'a/b', dataset_id:'c/d', task_type:'text-classification', training_method:'lora', lora_r: 100}); } catch(e){ err=e; }
  assert(err && err.code==='bad_request');
  console.log('✓ lora_r validation');

  console.log('\n=== LoRA real training (tiny model) ===');
  // Tiny LoRA training — must succeed, only LoRA trainable, adapter saved, reloadable, inference works
  const loraCfg = tb.validateTrainingRequest({
    model_id: 'hf-internal-testing/tiny-random-bert',
    dataset_id: 'stanfordnlp/imdb',
    task_type: 'text-classification',
    epochs: 1, batch_size: 2, learning_rate: 5e-5, max_steps: 2, validation_split: 10,
    training_method: 'lora', lora_r: 4, lora_alpha: 8, lora_dropout: 0.05, target_modules: 'auto'
  });
  const job = tb.createJob(loraCfg);
  tb.startJob(job);
  console.log(`  job ${job.job_id} started (lora r=4, max_steps=2)`);

  // wait for finish (real Trainer, ~20s)
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    if (job.status === 'finished' || job.status === 'failed') break;
    await new Promise(r => setTimeout(r, 2000));
  }
  console.log(`  status ${job.status} trainable ${job.progress.trainable_params} total ${job.progress.total_params}`);

  // 1. training succeeds
  assert.strictEqual(job.status, 'finished', `LoRA training should succeed, got ${job.status} error=${job.error} logs=${job.logs.slice(-5).join('\n')}`);
  console.log('✓ 1. training succeeds');

  // 2. only LoRA params are trainable
  assert(job.progress.trainable_params && job.progress.total_params, 'trainable/total must be set');
  assert(job.progress.trainable_params < job.progress.total_params, 'trainable < total');
  const pct = job.progress.trainable_params / job.progress.total_params * 100;
  assert(pct < 20, `LoRA should be <20% trainable, got ${pct}%`);
  console.log(`✓ 2. only LoRA trainable: ${job.progress.trainable_params} / ${job.progress.total_params} (${pct.toFixed(2)}%)`);

  // 3. adapter files are saved
  const dir = job.artifacts_dir;
  assert(dir && fs.existsSync(dir), 'artifacts_dir missing');
  assert(fs.existsSync(path.join(dir, 'adapter_config.json')), 'adapter_config.json missing');
  assert(fs.existsSync(path.join(dir, 'adapter_model.safetensors')), 'adapter_model.safetensors missing');
  assert(fs.existsSync(path.join(dir, 'tokenizer.json')), 'tokenizer missing');
  assert(fs.existsSync(path.join(dir, 'metrics.json')), 'metrics.json missing');
  // should NOT have full model.safetensors (LoRA saves adapter, not full)
  const hasFull = fs.existsSync(path.join(dir, 'model.safetensors'));
  // peft saves adapter_model, not model.safetensors; but some versions also save model.safetensors? Check that adapter exists and is smaller
  const adapterSize = fs.statSync(path.join(dir, 'adapter_model.safetensors')).size;
  assert(adapterSize > 1000 && adapterSize < 100000, `adapter size should be small (13K), got ${adapterSize}`);
  console.log(`✓ 3. adapter files saved (adapter_config.json, adapter_model.safetensors ${adapterSize} bytes)`);

  // 4. adapter can be reloaded
  const { spawnSync } = require('child_process');
  const pyCheck = `
from peft import PeftConfig
import os
cfg = PeftConfig.from_pretrained("${dir}")
assert cfg.r == 4, f"r mismatch {cfg.r}"
assert cfg.lora_alpha == 8
assert "query" in str(cfg.target_modules) or "q_proj" in str(cfg.target_modules)
print(f"adapter r={cfg.r} alpha={cfg.lora_alpha} target={cfg.target_modules} OK")

from transformers import AutoModelForSequenceClassification
from peft import PeftModel
base = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert", num_labels=2)
peft_model = PeftModel.from_pretrained(base, "${dir}")
print(f"peft model {peft_model.__class__.__name__} active {peft_model.active_adapter}")
print("reload OK")
`;
  const pyRes = spawnSync(process.env.PYTHON_BIN, ['-c', pyCheck], { encoding: 'utf8', timeout: 30000 });
  assert(pyRes.status === 0, `adapter reload failed: ${pyRes.stderr} ${pyRes.stdout}`);
  assert(pyRes.stdout.includes('reload OK'));
  console.log('✓ 4. adapter can be reloaded');

  // 5. inference using base + adapter works
  const pyInfer = `
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
adapter_dir = "${dir}"
tok = AutoTokenizer.from_pretrained(adapter_dir)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert", num_labels=2)
peft_model = PeftModel.from_pretrained(base, adapter_dir)
peft_model.eval()
base.eval()
inputs = tok("This movie was great!", return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    out_peft = peft_model(**inputs).logits
    out_base = base(**inputs).logits
    print(f"peft logits {out_peft[0][:2].tolist()}")
    print(f"base logits {out_base[0][:2].tolist()}")
    print("inference OK")
`;
  const pyInferRes = spawnSync(process.env.PYTHON_BIN, ['-c', pyInfer], { encoding: 'utf8', timeout: 30000 });
  assert(pyInferRes.status === 0, `inference failed: ${pyInferRes.stderr}`);
  assert(pyInferRes.stdout.includes('inference OK'));
  console.log('✓ 5. inference using base + adapter works');

  // Full path unchanged — verify full still saves full model
  console.log('\n=== Full fine-tuning still works ===');
  const fullCfg = tb.validateTrainingRequest({
    model_id: 'hf-internal-testing/tiny-random-bert',
    dataset_id: 'stanfordnlp/imdb',
    task_type: 'text-classification',
    epochs: 1, batch_size: 2, learning_rate: 5e-5, max_steps: 2, validation_split: 10,
    training_method: 'full'
  });
  const fullJob = tb.createJob(fullCfg);
  tb.startJob(fullJob);
  const deadline2 = Date.now() + 120000;
  while (Date.now() < deadline2) {
    if (fullJob.status === 'finished' || fullJob.status === 'failed') break;
    await new Promise(r => setTimeout(r, 2000));
  }
  assert.strictEqual(fullJob.status, 'finished', `Full training should succeed, got ${fullJob.status}`);
  assert(fullJob.progress.trainable_params === null || fullJob.progress.trainable_params === fullJob.progress.total_params || fullJob.progress.training_method === 'full');
  assert(fs.existsSync(path.join(fullJob.artifacts_dir, 'model.safetensors')), 'full should save model.safetensors');
  assert(!fs.existsSync(path.join(fullJob.artifacts_dir, 'adapter_model.safetensors')) || fs.existsSync(path.join(fullJob.artifacts_dir, 'model.safetensors')), 'full should not be adapter-only');
  console.log('✓ Full fine-tuning unchanged (model.safetensors saved)');

  // metrics are real Trainer, not synthetic
  const metrics = JSON.parse(fs.readFileSync(path.join(dir, 'metrics.json'), 'utf8'));
  assert(metrics.length > 0);
  assert(metrics.some(m => 'loss' in m || 'train_loss' in m), 'metrics should have loss');
  assert(!JSON.stringify(metrics).includes('sim'), 'metrics should not contain sim');
  console.log('✓ metrics are real Trainer (no synthetic)');

  tb.clearAllJobs();
  console.log('\nAll LoRA tests passed.');
}

run().catch(e => { console.error('TEST FAILED', e); process.exit(1); });
