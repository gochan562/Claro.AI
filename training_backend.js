'use strict';

/**
 * Training Backend — generic training engine for Claro.AI
 *
 * Inference Cell → gpu_backends.js (inference)
 * Training Cell → training_backend.js (this file)
 *
 * REAL TRAINING ONLY — no synthetic loss curves.
 * All metrics (step, epoch, train_loss, eval_loss, learning_rate, elapsed_time)
 * must come from Hugging Face Trainer (via training_runner.py).
 * If real training cannot run (missing deps, GPU/quota, unsupported model/task),
 * the job fails with code=training_error and the UI shows the error.
 *
 * Test-only simulation is NOT in this file. Tests that need fast fake data
 * must implement their own mock inside the test file itself.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const childProcess = require('child_process');
// Shared catalog (pure helpers only — the backend re-resolves preset IDs
// server-side and never trusts frontend-supplied IDs or limits).
const trainingUI = require('./training_ui');

// ── Resource limits (env overrides) ──
const MAX_EPOCHS = parseInt(process.env.TRAIN_MAX_EPOCHS || '5', 10);
const MAX_BATCH = parseInt(process.env.TRAIN_MAX_BATCH || '32', 10);
const MAX_SAMPLES = parseInt(process.env.TRAIN_MAX_SAMPLES || '5000', 10);
const MAX_TIME_SEC = parseInt(process.env.TRAIN_MAX_TIME_SEC || '1800', 10); // 30 min
const MAX_STEPS_LIMIT = parseInt(process.env.TRAIN_MAX_STEPS || '10000', 10);

const ALLOWED_TASKS = new Set([
  'text-generation',
  'text-classification',
  'image-classification',
  'token-classification',
]);

const ALLOWED_TRAINING_METHODS = new Set(['full', 'lora', 'auto']);

const TASK_ALIASES = {
  'text generation': 'text-generation',
  'text_generation': 'text-generation',
  'text classification': 'text-classification',
  'text_classification': 'text-classification',
  'image classification': 'image-classification',
  'image_classification': 'image-classification',
  'token classification': 'token-classification',
  'token_classification': 'token-classification',
};

function normalizeTask(t) {
  const raw = String(t || '').trim().toLowerCase();
  if (ALLOWED_TASKS.has(raw)) return raw;
  if (TASK_ALIASES[raw]) return TASK_ALIASES[raw];
  return raw;
}

function isLargeModel(modelId) {
  const m = String(modelId || '').toLowerCase();
  // heuristic: look for e.g. 1b, 1.5b, 3b, 7b, 8b, 13b, 30b, 70b
  const sizeMatch = m.match(/(\d+(?:\.\d+)?)\s*b\b/);
  if (sizeMatch) {
    const num = parseFloat(sizeMatch[1]);
    if (!isNaN(num) && num >= 1) return true;
  }
  // also check explicit patterns like -1b-, _7b, etc.
  if (/(?:^|[-_\/\s])(?:1b|1\.5b|3b|7b|8b|13b|30b|70b)(?:$|[-_\/\s])/i.test(m)) return true;
  return false;
}

function normalizeTrainingMethod(m) {
  const raw = String(m || '').trim().toLowerCase();
  if (ALLOWED_TRAINING_METHODS.has(raw)) return raw;
  if (raw === 'lorra' || raw === 'peft') return 'lora';
  if (raw === 'full_finetune' || raw === 'full-finetune') return 'full';
  return raw;
}

// ── Job store ──
const jobs = new Map(); // job_id -> job

function _now() { return Date.now(); }
function _genId() { return 'train_' + crypto.randomBytes(6).toString('hex'); }

// ── Validation ──
function validateTrainingRequest(body) {
  if (!body || typeof body !== 'object') {
    throw Object.assign(new Error('Request body must be an object'), { code: 'bad_request', status: 400 });
  }
  let model_id = String(body.model_id || body.modelId || '').trim();
  let dataset_id = String(body.dataset_id || body.datasetId || '').trim();
  const task_type = normalizeTask(body.task_type || body.task || body.taskType);

  // ── Preset resolution (server-side, authoritative) ─────────────────────
  // When preset IDs are present, underlying IDs resolve from the catalog —
  // any frontend-supplied model_id/dataset_id is ignored. Absent presets
  // (''/null/undefined) mean legacy raw mode with global limits only.
  const mpKey = trainingUI.normPresetId(body.model_preset !== undefined ? body.model_preset : body.modelPreset);
  const dpKey = trainingUI.normPresetId(body.dataset_preset !== undefined ? body.dataset_preset : body.datasetPreset);
  let modelPreset = null;
  let datasetPreset = null;
  let modelPresetKey = null;
  let datasetPresetKey = null;
  if (mpKey !== null || dpKey !== null) {
    modelPresetKey = mpKey || trainingUI.CUSTOM_PRESET;
    datasetPresetKey = dpKey || trainingUI.CUSTOM_PRESET;
    if (modelPresetKey !== trainingUI.CUSTOM_PRESET) {
      modelPreset = trainingUI.getModelPreset(modelPresetKey);
      if (!modelPreset) {
        throw Object.assign(new Error(`Unknown model preset: ${modelPresetKey}`), { code: 'bad_request', status: 400 });
      }
      model_id = modelPreset.modelId;
    }
    if (datasetPresetKey !== trainingUI.CUSTOM_PRESET) {
      datasetPreset = trainingUI.getDatasetPreset(datasetPresetKey);
      if (!datasetPreset) {
        throw Object.assign(new Error(`Unknown dataset preset: ${datasetPresetKey}`), { code: 'bad_request', status: 400 });
      }
      dataset_id = datasetPreset.datasetId;
    }
    const chk = trainingUI.checkPresetCompatibility({ modelPreset: modelPresetKey, datasetPreset: datasetPresetKey, taskType: task_type });
    if (!chk.ok) {
      throw Object.assign(new Error(chk.error), { code: 'bad_request', status: 400 });
    }
  }

  // Preset resource ceilings apply on top of the global limits.
  const epochCap = Math.min(MAX_EPOCHS, modelPreset ? modelPreset.maxEpochs : MAX_EPOCHS);
  const batchCap = Math.min(MAX_BATCH, modelPreset ? modelPreset.maxBatchSize : MAX_BATCH);
  const stepsCap = Math.min(MAX_STEPS_LIMIT, modelPreset ? modelPreset.maxSteps : MAX_STEPS_LIMIT);
  const samplesCap = Math.min(MAX_SAMPLES, modelPreset ? modelPreset.maxSamples : MAX_SAMPLES);

  if (!model_id || !/^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)?$/.test(model_id)) {
    throw Object.assign(new Error(`Invalid model_id: ${model_id || '(empty)'}`), { code: 'invalid_model_id', status: 400 });
  }
  if (!dataset_id || !/^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)?$/.test(dataset_id)) {
    throw Object.assign(new Error(`Invalid dataset_id: ${dataset_id || '(empty)'}`), { code: 'invalid_dataset_id', status: 400 });
  }
  if (!ALLOWED_TASKS.has(task_type)) {
    throw Object.assign(new Error(`Invalid task_type: ${task_type}. Allowed: ${[...ALLOWED_TASKS].join(', ')}`), { code: 'bad_request', status: 400 });
  }

  let epochs = body.epochs !== undefined ? Number(body.epochs) : 3;
  let batch_size = body.batch_size !== undefined ? Number(body.batch_size) : (body.batchSize !== undefined ? Number(body.batchSize) : 8);
  let learning_rate = body.learning_rate !== undefined ? Number(body.learning_rate) : (body.learningRate !== undefined ? Number(body.learningRate) : 2e-5);
  let max_steps = body.max_steps !== undefined ? body.max_steps : (body.maxSteps !== undefined ? body.maxSteps : null);
  let validation_split = body.validation_split !== undefined ? Number(body.validation_split) : (body.validationSplit !== undefined ? Number(body.validationSplit) : 10);

  if (!Number.isFinite(epochs) || !Number.isInteger(epochs) || epochs < 1 || epochs > epochCap) {
    throw Object.assign(new Error(modelPreset
      ? `epochs must be integer 1..${epochCap} for preset '${modelPreset.id}'`
      : `epochs must be integer 1..${epochCap}`), { code: 'bad_request', status: 400 });
  }
  if (!Number.isFinite(batch_size) || !Number.isInteger(batch_size) || batch_size < 1 || batch_size > batchCap) {
    throw Object.assign(new Error(modelPreset
      ? `batch_size must be integer 1..${batchCap} for preset '${modelPreset.id}'`
      : `batch_size must be integer 1..${batchCap}`), { code: 'bad_request', status: 400 });
  }
  if (!Number.isFinite(learning_rate) || learning_rate < 1e-6 || learning_rate > 1e-2) {
    throw Object.assign(new Error('learning_rate must be between 1e-6 and 1e-2'), { code: 'bad_request', status: 400 });
  }
  if (max_steps !== null && max_steps !== undefined && String(max_steps).trim() !== '') {
    max_steps = Number(max_steps);
    if (!Number.isFinite(max_steps) || !Number.isInteger(max_steps) || max_steps < 1 || max_steps > stepsCap) {
      throw Object.assign(new Error(modelPreset
        ? `max_steps must be integer 1..${stepsCap} for preset '${modelPreset.id}' or empty`
        : `max_steps must be integer 1..${stepsCap} or empty`), { code: 'bad_request', status: 400 });
    }
  } else {
    max_steps = null;
  }
  if (!Number.isFinite(validation_split) || validation_split < 0 || validation_split > 50) {
    throw Object.assign(new Error('validation_split must be 0..50 (%)'), { code: 'bad_request', status: 400 });
  }

  // ── Training method: full | lora | auto (auto → lora for >=1B, else full) ──
  let training_method = normalizeTrainingMethod(
    body.training_method || body.trainingMethod || body.method || body.lora_method || body.trainingMethod || 'auto'
  );
  if (!ALLOWED_TRAINING_METHODS.has(training_method)) training_method = 'auto';
  if (training_method === 'auto') {
    training_method = isLargeModel(model_id) ? 'lora' : 'full';
  }
  // lora params — always validate if provided, but only used when method is lora
  let lora_r = body.lora_r !== undefined ? Number(body.lora_r) : (body.r !== undefined ? Number(body.r) : (body.loraR !== undefined ? Number(body.loraR) : 8));
  let lora_alpha = body.lora_alpha !== undefined ? Number(body.lora_alpha) : (body.alpha !== undefined ? Number(body.alpha) : 16);
  let lora_dropout = body.lora_dropout !== undefined ? Number(body.lora_dropout) : (body.loraDropout !== undefined ? Number(body.loraDropout) : 0.05);
  let target_modules = body.target_modules !== undefined ? body.target_modules : (body.targetModules !== undefined ? body.targetModules : 'auto');

  // normalize target_modules: "auto" or comma-separated string or array
  if (Array.isArray(target_modules)) {
    target_modules = target_modules.map(s => String(s).trim()).filter(Boolean).join(',');
  } else {
    target_modules = String(target_modules || 'auto').trim();
  }
  const isAutoTarget = target_modules.toLowerCase() === 'auto' || target_modules === '';
  if (isAutoTarget) target_modules = 'auto';

  if (training_method === 'lora') {
    if (!Number.isFinite(lora_r) || !Number.isInteger(lora_r) || lora_r < 1 || lora_r > 64) {
      throw Object.assign(new Error('lora_r must be integer 1..64'), { code: 'bad_request', status: 400 });
    }
    if (!Number.isFinite(lora_alpha) || !Number.isInteger(lora_alpha) || lora_alpha < 1 || lora_alpha > 128) {
      throw Object.assign(new Error('lora_alpha must be integer 1..128'), { code: 'bad_request', status: 400 });
    }
    if (!Number.isFinite(lora_dropout) || lora_dropout < 0 || lora_dropout > 0.5) {
      throw Object.assign(new Error('lora_dropout must be 0..0.5'), { code: 'bad_request', status: 400 });
    }
    if (!isAutoTarget) {
      const mods = target_modules.split(',').map(s => s.trim()).filter(Boolean);
      if (mods.length === 0) {
        throw Object.assign(new Error('target_modules must be "auto" or comma-separated module names'), { code: 'bad_request', status: 400 });
      }
      for (const m of mods) {
        if (!/^[A-Za-z0-9_\.]+$/.test(m)) {
          throw Object.assign(new Error(`Invalid target_modules entry: ${m}`), { code: 'bad_request', status: 400 });
        }
      }
      target_modules = mods.join(',');
    }
  } else {
    // for full, keep defaults but don't enforce strict validation on lora params if not lora
    // still normalize for storage
    if (!Number.isFinite(lora_r) || lora_r < 1) lora_r = 8;
    if (!Number.isFinite(lora_alpha) || lora_alpha < 1) lora_alpha = 16;
    if (!Number.isFinite(lora_dropout) || lora_dropout < 0) lora_dropout = 0.05;
  }

  // Preset-supported training modes (checked after auto-resolution).
  if (modelPreset && !modelPreset.trainingMethods.includes(training_method)) {
    throw Object.assign(new Error(`Training method '${training_method}' is not supported by preset '${modelPreset.id}' (${modelPreset.name}). Supported: ${modelPreset.trainingMethods.join(', ')}`), { code: 'bad_request', status: 400 });
  }

  // TEMP-DIAG: provider provenance — requested (body) vs env vs resolved.
  // The Training Cell never sends `provider`; resolution is env-driven here.
  try {
    console.error(`[TRAIN-DIAG] validate requested provider=${body && body.provider !== undefined ? JSON.stringify(body.provider) : '(absent from request)'} TRAINING_PROVIDER=${process.env.TRAINING_PROVIDER || '(unset)'} GPU_PROVIDER=${process.env.GPU_PROVIDER || '(unset)'}`);
  } catch (_) {}
  const validatedProvider = (body.provider || process.env.TRAINING_PROVIDER || 'local').toLowerCase();
  try {
    console.error(`[TRAIN-DIAG] validate resolved provider=${validatedProvider}`);
  } catch (_) {}
  return {
    model_id,
    dataset_id,
    task_type,
    model_preset: modelPresetKey,
    dataset_preset: datasetPresetKey,
    epochs,
    batch_size,
    learning_rate,
    max_steps,
    validation_split,
    provider: validatedProvider,
    max_samples: samplesCap,
    max_time_sec: MAX_TIME_SEC,
    training_method,
    lora_r,
    lora_alpha,
    lora_dropout,
    target_modules,
  };
}

// ── Job lifecycle ──
function createJob(config, ownerUserId = null) {
  const job_id = _genId();
  const job = {
    job_id,
    status: 'queued',
    config,
    user_id: ownerUserId || null,
    owner: ownerUserId || null,
    progress: {
      current_epoch: 0,
      current_step: 0,
      total_steps: config.max_steps || (config.epochs * 100),
      train_loss: null,
      eval_loss: null,
      learning_rate: config.learning_rate,
      elapsed_time: 0,
      eta: null,
      gpu_status: 'idle',
      percent: 0,
      training_method: config.training_method,
      trainable_params: null,
      total_params: null,
    },
    metrics: [],
    logs: [],
    sseClients: new Set(),
    start_time: null,
    end_time: null,
    created_at: new Date().toISOString(),
    error: null,
    artifacts_dir: null,
    _pythonProc: null,
    _aborted: false,
  };
  jobs.set(job_id, job);
  return job;
}

function getJob(job_id) { return jobs.get(job_id) || null; }

function getJobsByUser(userId) {
  if (!userId) return [];
  return [...jobs.values()].filter(j => j.user_id === userId || j.owner === userId);
}

function getActiveJobCountForUser(userId) {
  if (!userId) return 0;
  let count = 0;
  for (const j of jobs.values()) {
    if ((j.user_id === userId || j.owner === userId) && ['queued','loading','training','evaluating'].includes(j.status)) count++;
  }
  return count;
}

function isOwner(job, userId) {
  if (!job) return false;
  // Jobs without owner (legacy) are considered owned by no one; deny unless admin
  if (!job.user_id && !job.owner) return false;
  return job.user_id === userId || job.owner === userId;
}

// ── Artifact registry — machine-readable metadata for inference ──
function _isValidJobId(job_id) {
  return typeof job_id === 'string' && /^train_[a-f0-9]{12}$/.test(job_id);
}

function _safeArtifactDir(job_id) {
  // Derive artifact path server-side from job_id only — never from client-provided paths
  // Reject traversal, absolute paths, unknown IDs
  if (!_isValidJobId(job_id)) {
    throw Object.assign(new Error(`Invalid job_id: ${job_id}`), { code: 'bad_request', status: 400 });
  }
  const dir = path.resolve(path.join(__dirname, 'training_outputs', job_id));
  const base = path.resolve(path.join(__dirname, 'training_outputs'));
  if (!dir.startsWith(base + path.sep) && dir !== base) {
    throw Object.assign(new Error('Invalid artifact path'), { code: 'bad_request', status: 400 });
  }
  return dir;
}

function getArtifactMetadata(job_id) {
  if (!_isValidJobId(job_id)) {
    throw Object.assign(new Error(`Invalid job_id: ${job_id}`), { code: 'bad_request', status: 400 });
  }
  let job = jobs.get(job_id);
  // Fallback to disk if not in memory (survives reload)
  let diskMeta = null;
  const dir = _safeArtifactDir(job_id);
  if (!job) {
    // Try to read from disk: training_outputs/<job_id>/job.json
    const jobJsonPath = path.join(dir, 'job.json');
    if (fs.existsSync(jobJsonPath)) {
      try {
        diskMeta = JSON.parse(fs.readFileSync(jobJsonPath, 'utf8'));
        // Reconstruct minimal job for validation
        job = {
          job_id,
          status: diskMeta.status,
          config: diskMeta.config,
          artifacts_dir: dir,
          progress: diskMeta.progress || {},
          user_id: diskMeta.user_id || null,
          owner: diskMeta.user_id || null,
        };
      } catch (_) {
        // fall through to unknown job
      }
    }
  }
  if (!job) {
    throw Object.assign(new Error(`Unknown job: ${job_id}`), { code: 'not_found', status: 404 });
  }
  if (job.status !== 'finished') {
    throw Object.assign(new Error(`Job not finished (status=${job.status})`), { code: 'job_not_finished', status: 400 });
  }
  // Use job.artifacts_dir if set, but verify it matches derived dir (prevent stale)
  const artifactDir = job.artifacts_dir && path.resolve(job.artifacts_dir) === dir ? job.artifacts_dir : dir;

  if (!fs.existsSync(artifactDir) || !fs.statSync(artifactDir).isDirectory()) {
    throw Object.assign(new Error(`Artifact directory not found: ${artifactDir}`), { code: 'missing_artifact', status: 404 });
  }

  let files = [];
  try { files = fs.readdirSync(artifactDir); } catch (e) {
    throw Object.assign(new Error(`Cannot read artifact dir: ${e.message}`), { code: 'missing_artifact', status: 404 });
  }

  const trainingMethod = (job.config && job.config.training_method) || (diskMeta && diskMeta.config && diskMeta.config.training_method) || 'full';
  const baseModelId = job.config ? job.config.model_id : (diskMeta ? diskMeta.config.model_id : null);
  const taskType = job.config ? job.config.task_type : (diskMeta ? diskMeta.config.task_type : null);

  // Validate expected artifacts exist — do not mark ready if missing
  if (trainingMethod === 'lora') {
    const hasAdapterConfig = files.includes('adapter_config.json');
    const hasAdapterModel = files.includes('adapter_model.safetensors') || files.includes('adapter_model.bin');
    if (!hasAdapterConfig) {
      throw Object.assign(new Error('Missing adapter_config.json — LoRA job not inference-ready'), { code: 'missing_artifact', status: 409 });
    }
    if (!hasAdapterModel) {
      throw Object.assign(new Error('Missing adapter_model.safetensors — LoRA job not inference-ready'), { code: 'missing_artifact', status: 409 });
    }
    // also need tokenizer
    const hasTokenizer = files.some(f => f === 'tokenizer.json' || f === 'vocab.json' || f === 'tokenizer_config.json');
    if (!hasTokenizer) {
      throw Object.assign(new Error('Missing tokenizer files — LoRA job not inference-ready'), { code: 'missing_artifact', status: 409 });
    }
    return {
      job_id,
      ready: true,
      training_method: 'lora',
      base_model_id: baseModelId,
      task_type: taskType,
      artifact_dir: artifactDir,
      adapter_dir: artifactDir,
      model_dir: artifactDir,
      tokenizer_dir: artifactDir,
      metrics_path: path.join(artifactDir, 'metrics.json'),
      files,
      lora_r: job.config.lora_r,
      lora_alpha: job.config.lora_alpha,
      target_modules: job.config.target_modules,
      status: job.status,
    };
  } else {
    // Full fine-tuning: need model.safetensors or pytorch_model.bin and config.json (HF) and tokenizer
    const hasModel = files.includes('model.safetensors') || files.includes('pytorch_model.bin');
    const hasConfig = files.includes('config.json');
    const hasTokenizer = files.some(f => f === 'tokenizer.json' || f === 'vocab.json' || f === 'tokenizer_config.json');
    if (!hasModel) {
      throw Object.assign(new Error('Missing model.safetensors/pytorch_model.bin — full job not inference-ready'), { code: 'missing_artifact', status: 409 });
    }
    if (!hasConfig) {
      throw Object.assign(new Error('Missing config.json — full job not inference-ready'), { code: 'missing_artifact', status: 409 });
    }
    if (!hasTokenizer) {
      throw Object.assign(new Error('Missing tokenizer files — full job not inference-ready'), { code: 'missing_artifact', status: 409 });
    }
    return {
      job_id,
      ready: true,
      training_method: 'full',
      base_model_id: baseModelId,
      task_type: taskType,
      artifact_dir: artifactDir,
      adapter_dir: null,
      model_dir: artifactDir,
      tokenizer_dir: artifactDir,
      metrics_path: path.join(artifactDir, 'metrics.json'),
      files,
      status: job.status,
    };
  }
}

function listJobs() {
  return [...jobs.values()].map(j => ({
    job_id: j.job_id,
    status: j.status,
    config: j.config,
    progress: j.progress,
    created_at: j.created_at,
    error: j.error,
  }));
}

function _broadcast(job, event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of job.sseClients) {
    try { res.write(payload); } catch (_) {}
  }
}

function _pushMetric(job, metric) {
  job.metrics.push(metric);
  if (job.metrics.length > 500) job.metrics.shift();
  _broadcast(job, 'metrics', metric);
}

function _updateProgress(job, patch) {
  Object.assign(job.progress, patch);
  _broadcast(job, 'progress', job.progress);
}

function _setStatus(job, status) {
  job.status = status;
  _broadcast(job, 'status', { status, job_id: job.job_id });
}

function _log(job, line) {
  job.logs.push(line);
  if (job.logs.length > 1000) job.logs.shift();
  _broadcast(job, 'log', { line });
}

// Shape one Trainer metric line {step, epoch, train_loss?, eval_loss?, learning_rate?}
// into job progress + metrics. Shared by the local stdout parser and the
// ZeroGPU remote event stream so both providers behave identically.
// Returns true when the line carried a real Trainer metric.
function _applyTrainerMetric(job, m) {
  if (!m || typeof m.step !== 'number' ||
      (typeof m.train_loss !== 'number' && typeof m.eval_loss !== 'number')) {
    return false;
  }
  const cfg = job.config;
  const elapsed = job.start_time ? Math.round((_now() - job.start_time) / 1000) : 0;
  const metric = {
    step: m.step,
    epoch: typeof m.epoch === 'number' ? m.epoch : job.progress.current_epoch,
    train_loss: m.train_loss,
    eval_loss: m.eval_loss,
    learning_rate: m.learning_rate !== undefined ? m.learning_rate : job.progress.learning_rate,
    elapsed_time: elapsed,
  };
  // do not fabricate missing fields — only plot what Trainer emitted
  if (metric.train_loss === undefined) delete metric.train_loss;
  if (metric.eval_loss === undefined) delete metric.eval_loss;

  const total = job.progress.total_steps || (cfg.max_steps || cfg.epochs * 100);
  // Remote providers keep their provider tag in the GPU badge; local runs
  // flip loading/idle -> training exactly as before.
  const keepStatus = job.progress.gpu_status;
  const nextGpu = (keepStatus === 'zerogpu' || keepStatus === 'modal' || keepStatus === 'local')
    ? keepStatus : 'training';
  _updateProgress(job, {
    current_step: metric.step,
    current_epoch: metric.epoch,
    train_loss: metric.train_loss !== undefined ? metric.train_loss : job.progress.train_loss,
    eval_loss: metric.eval_loss !== undefined ? metric.eval_loss : job.progress.eval_loss,
    learning_rate: metric.learning_rate,
    elapsed_time: elapsed,
    eta: Math.max(0, Math.round((total - metric.step) * 1.2)),
    percent: total > 0 ? Math.min(100, Math.round((metric.step / total) * 100)) : 0,
    gpu_status: nextGpu,
  });
  _pushMetric(job, metric);
  // status: if eval_loss present, it's evaluation step
  if (m.eval_loss !== undefined) {
    _setStatus(job, 'evaluating');
    setTimeout(() => { if (job.status === 'evaluating' && !job._aborted) _setStatus(job, 'training'); }, 600);
  } else {
    if (job.status !== 'training') _setStatus(job, 'training');
  }
  return true;
}

function resolveTrainingProvider(env) {
  const p = String(env.TRAINING_PROVIDER || env.GPU_PROVIDER || 'local').toLowerCase().trim();
  if (['local', 'zerogpu', 'modal'].includes(p)) return p;
  return 'local';
}

// ── Real Trainer via Python ──
function _findPythonWithDeps() {
  const candidates = [
    process.env.PYTHON_BIN,
    'python3',
    'python',
    '/usr/bin/python3',
    '/opt/python/bin/python',
    path.join(__dirname, '.venv', 'bin', 'python'),
    path.join(__dirname, 'venv', 'bin', 'python'),
  ].filter(Boolean);
  // Also check common Replit/venv locations
  const extra = [
    '/home/runner/workspace/.pythonlibs/bin/python',
    '/opt/homebrew/bin/python3',
  ];
  candidates.push(...extra);
  const spawnSync = childProcess.spawnSync;
  for (const bin of candidates) {
    try {
      const r = spawnSync(bin, ['-c', 'import torch, transformers, datasets, accelerate; print("ok")'], { timeout: 5000, encoding: 'utf8' });
      if (r.status === 0 && r.stdout && r.stdout.includes('ok')) {
        return bin;
      }
    } catch (_) {}
  }
  // Fallback to env or python3 even if check fails (let training_runner report the real error)
  return process.env.PYTHON_BIN || 'python3';
}

function _startPythonTraining(job) {
  const cfg = job.config;
  // TEMP-DIAG: proves the LOCAL Python path executed (spawns training_runner.py).
  console.error(`[TRAIN-DIAG] _startPythonTraining CALLED job=${job.job_id} (local training_runner.py will spawn)`);
  // Prefer env, but verify it has deps; otherwise try to find one that does
  let pythonBin = process.env.PYTHON_BIN || 'python3';
  // Quick check: if the requested bin doesn't have torch, try to find one that does
  try {
    const spawnSync = childProcess.spawnSync;
    const chk = spawnSync(pythonBin, ['-c', 'import torch'], { timeout: 3000 });
    if (chk.status !== 0) {
      const alt = _findPythonWithDeps();
      if (alt && alt !== pythonBin) {
        // Verify alt actually has torch before switching
        const chk2 = spawnSync(alt, ['-c', 'import torch'], { timeout: 3000 });
        if (chk2.status === 0) {
          pythonBin = alt;
        }
      }
    }
  } catch (_) {}
  const runnerPath = path.join(__dirname, 'training_runner.py');

  if (!fs.existsSync(runnerPath)) {
    const msg = `training_runner.py not found at ${runnerPath}`;
    _log(job, `[TRAIN] training_error: ${msg}`);
    _setStatus(job, 'failed');
    _updateProgress(job, { gpu_status: 'idle', eta: 0 });
    job.error = 'training_error';
    job.end_time = _now();
    _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
    _saveArtifacts(job);
    return;
  }

  job.start_time = _now();
  _setStatus(job, 'loading');
  _log(job, `[TRAIN] loading model=${cfg.model_id} dataset=${cfg.dataset_id} task=${cfg.task_type} provider=${cfg.provider} method=${cfg.training_method}`);
  if (cfg.training_method === 'lora') {
    _log(job, `[TRAIN] lora r=${cfg.lora_r} alpha=${cfg.lora_alpha} dropout=${cfg.lora_dropout} target_modules=${cfg.target_modules}`);
  }
  _log(job, `[TRAIN] config epochs=${cfg.epochs} batch_size=${cfg.batch_size} lr=${cfg.learning_rate} max_steps=${cfg.max_steps || 'auto'} validation_split=${cfg.validation_split}%`);
  _updateProgress(job, { gpu_status: 'loading', total_steps: cfg.max_steps || (cfg.epochs * 100) });

  const args = [
    runnerPath,
    '--model_id', cfg.model_id,
    '--dataset_id', cfg.dataset_id,
    '--task_type', cfg.task_type,
    '--epochs', String(cfg.epochs),
    '--batch_size', String(cfg.batch_size),
    '--learning_rate', String(cfg.learning_rate),
    '--validation_split', String(cfg.validation_split),
    '--max_samples', String(cfg.max_samples),
    '--max_time_sec', String(cfg.max_time_sec),
    '--training_method', cfg.training_method,
    '--lora_r', String(cfg.lora_r),
    '--lora_alpha', String(cfg.lora_alpha),
    '--lora_dropout', String(cfg.lora_dropout),
    '--target_modules', cfg.target_modules,
  ];
  if (cfg.max_steps) args.push('--max_steps', String(cfg.max_steps));
  args.push('--job_id', job.job_id);
  args.push('--output_dir', path.join(__dirname, 'training_outputs', job.job_id));

  let proc;
  try {
    proc = childProcess.spawn(pythonBin, args, { cwd: __dirname, env: process.env });
  } catch (e) {
    const msg = `Failed to spawn Python (${pythonBin}): ${e.message}`;
    _log(job, `[TRAIN] training_error: ${msg}`);
    _setStatus(job, 'failed');
    _updateProgress(job, { gpu_status: 'idle', eta: 0 });
    job.error = 'training_error';
    job.end_time = _now();
    _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
    _saveArtifacts(job);
    return;
  }
  job._pythonProc = proc;

  let buffer = '';
  let stderrBuf = '';

  proc.stdout.on('data', (chunk) => {
    buffer += chunk.toString('utf8');
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const rawLine = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 1);
      const line = rawLine.trim();
      if (!line) continue;

      // Trainer metrics are JSON lines with step
      if (line.startsWith('{') && line.includes('"step"')) {
        try {
          const m = JSON.parse(line);
          if (_applyTrainerMetric(job, m)) continue;
        } catch (_) {
          // not a metric JSON, treat as log
        }
      }

      // status markers from runner
      if (line.startsWith('[TRAIN]')) {
        _log(job, line);
        // Real total optimizer steps reported by the runner — replaces the
        // epochs*100 progress fallback with the actual denominator.
        const totalMatch = line.match(/total_steps=(\d+)/);
        if (totalMatch) {
          const t = parseInt(totalMatch[1], 10);
          if (Number.isFinite(t) && t > 0) _updateProgress(job, { total_steps: t });
        }
        const low = line.toLowerCase();
        // LoRA trainable info: e.g. [TRAIN] lora trainable=5186 total=93181 (5.57%)
        if (low.includes('trainable') && low.includes('total')) {
          const trMatch = line.match(/trainable[^0-9]*([\d,]+)/i);
          const totMatch = line.match(/total[^0-9]*([\d,]+)/i);
          if (trMatch && totMatch) {
            const trainable = parseInt(trMatch[1].replace(/,/g, ''), 10);
            const total = parseInt(totMatch[1].replace(/,/g, ''), 10);
            if (!isNaN(trainable) && !isNaN(total) && total > 0) {
              _updateProgress(job, { trainable_params: trainable, total_params: total });
            }
          }
        }
        if (low.includes('training_error')) {
          // runner reported fatal error
          _setStatus(job, 'failed');
          _updateProgress(job, { gpu_status: 'idle', eta: 0 });
          job.error = 'training_error';
        } else if (low.includes('loading')) {
          _setStatus(job, 'loading');
        } else if (low.includes('evaluating')) {
          _setStatus(job, 'evaluating');
        } else if (low.includes('training') && !low.includes('training_error')) {
          if (job.status === 'loading' || job.status === 'queued') _setStatus(job, 'training');
        } else if (low.includes('finished') && !low.includes('training_error')) {
          _setStatus(job, 'finished');
          job.end_time = _now();
          _broadcast(job, 'done', { status: 'finished' });
          _saveArtifacts(job);
        }
        // training_error with finished should stay failed
        if (job.status === 'failed') {
          // ensure error propagation
        }
      } else {
        _log(job, line);
      }
    }
  });

  proc.stderr.on('data', (chunk) => {
    stderrBuf += chunk.toString('utf8');
    const lines = chunk.toString('utf8').split('\n');
    for (const l of lines) {
      const t = l.trim();
      if (t) _log(job, `[stderr] ${t}`);
    }
  });

  // watchdog for max time — kill proc if exceeds cfg.max_time_sec
  const watchdog = setInterval(() => {
    if (job._aborted || !job._pythonProc) { clearInterval(watchdog); return; }
    const elapsed = (_now() - job.start_time) / 1000;
    if (elapsed > cfg.max_time_sec + 30) { // 30s grace
      _log(job, `[TRAIN] training_error: max training time ${cfg.max_time_sec}s exceeded (elapsed ${Math.round(elapsed)}s)`);
      _setStatus(job, 'failed');
      _updateProgress(job, { gpu_status: 'idle', eta: 0 });
      job.error = 'training_error';
      job.end_time = _now();
      _broadcast(job, 'done', { status: 'failed', error: job.error, message: 'max_training_time_exceeded' });
      try { job._pythonProc.kill('SIGKILL'); } catch (_) {}
      clearInterval(watchdog);
    }
  }, 5000);

  proc.on('close', (code, signal) => {
    clearInterval(watchdog);
    buffer = '';
    job._pythonProc = null;
    if (job._aborted) {
      if (!job.artifacts_dir) _saveArtifacts(job);
      return;
    }

    if (job.status === 'failed' && !job.artifacts_dir) {
      // already marked failed via training_error in stdout — ensure artifacts saved and GPU idle
      _updateProgress(job, { gpu_status: 'idle', eta: 0 });
      _saveArtifacts(job);
      return;
    }

    if (code === 0 && job.status === 'finished') {
      // already finished via [TRAIN] finished marker
      if (!job.artifacts_dir) _saveArtifacts(job);
      return;
    }
    if (code === 0 && job.status !== 'failed') {
      // exited cleanly but no explicit finished marker — check if we have metrics
      if (job.metrics.length > 0) {
        _setStatus(job, 'finished');
        job.end_time = _now();
        _updateProgress(job, { gpu_status: 'idle', eta: 0, percent: 100 });
        _broadcast(job, 'done', { status: 'finished' });
      } else {
        // no metrics at all — likely model/dataset error but exit 0
        const msg = 'Trainer produced no metrics';
        _log(job, `[TRAIN] training_error: ${msg} (stderr: ${stderrBuf.slice(0,500)})`);
        _setStatus(job, 'failed');
        _updateProgress(job, { gpu_status: 'idle', eta: 0 });
        job.error = 'training_error';
        job.end_time = _now();
        _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
      }
      _saveArtifacts(job);
    } else if (code !== 0 || signal) {
      if (job.status !== 'failed' && job.status !== 'finished') {
        const msg = signal ? `killed by ${signal}` : `exit code ${code}`;
        const tail = stderrBuf.slice(-800).trim();
        _log(job, `[TRAIN] training_error: Python ${msg} ${tail ? '— ' + tail : ''}`);
        _setStatus(job, 'failed');
        _updateProgress(job, { gpu_status: 'idle', eta: 0 });
        job.error = 'training_error';
        job.end_time = _now();
        _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
      } else if (job.status === 'failed') {
        _updateProgress(job, { gpu_status: 'idle', eta: 0 });
      }
      if (!job.artifacts_dir) _saveArtifacts(job);
    } else {
      // edge: code 0 but status already failed, ensure artifacts and idle
      _updateProgress(job, { gpu_status: 'idle', eta: 0 });
      if (!job.artifacts_dir) _saveArtifacts(job);
    }
  });

  proc.on('error', (err) => {
    clearInterval(watchdog);
    const msg = `Python spawn error: ${err.message}`;
    _log(job, `[TRAIN] training_error: ${msg}`);
    job._pythonProc = null;
    if (!job._aborted && job.status !== 'finished') {
      _setStatus(job, 'failed');
      _updateProgress(job, { gpu_status: 'idle', eta: 0 });
      job.error = 'training_error';
      job.end_time = _now();
      _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
      _saveArtifacts(job);
    }
  });
}

// ── ZeroGPU remote training ─────────────────────────────────────────────
// provider === 'zerogpu' runs the SAME training_runner.py Trainer logic on the
// ZeroGPU Space through its structured /train endpoint (space/train_api.py).
// This path NEVER spawns a local Python process: no child_process use below.
// Progress arrives as SSE `generating` frames and is fed into the exact same
// _log / _pushMetric / _updateProgress / _setStatus helpers as local training,
// so the UI, SSE fan-out, lifecycle and artifacts behave identically.

class ZeroGpuTrainError extends Error {
  constructor(message, code, status = 502) {
    super(message);
    this.name = 'ZeroGpuTrainError';
    this.code = code;       // machine-readable, mirrors gpu_backends codes
    this.status = status;   // HTTP-ish status for API responses
  }
}

function _zeroGpuTrainConfig(env) {
  const space = env.ZEROGPU_SPACE || 'Gochan562/claro_ai_gpu';
  let apiBase;
  if (env.ZEROGPU_TRAIN_API) {
    apiBase = String(env.ZEROGPU_TRAIN_API).replace(/\/+$/, '');
  } else {
    const { defaultSpaceUrl } = require('./gpu_backends');
    apiBase = `${defaultSpaceUrl(space)}/gradio_api`;
  }
  const token = env.ZEROGPU_API_TOKEN || '';
  return { spaceId: space, apiBase, token };
}

function _zeroGpuHeaders(token) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

function _zeroGpuSpaceOrigin(apiBase) {
  const m = String(apiBase).match(/^(https?:\/\/[^/]+)\/gradio_api\/?$/);
  if (m) return m[1];
  return String(apiBase).replace(/\/+$/, '');
}

// ── [TRAIN-HTTP] response-parsing diagnostics ─────────────────────────────
// Treats `Unexpected token '<'` as a response-parsing failure, never as a
// model/training failure. Every ZeroGPU fetch below reads the body as TEXT
// first, logs method + URL + status + content-type + first 500 chars of the
// raw body, and only then attempts JSON parsing. Secrets/tokens are NEVER
// logged: the Authorization header is never printed and URLs never carry it.
function _trainHttpSnippet(raw) {
  return String(raw || '').slice(0, 500);
}

function _logTrainHttp(method, url, status, contentType, raw) {
  try {
    const ct = contentType || 'unknown';
    console.error(`[TRAIN-HTTP] method=${method}`);
    console.error(`[TRAIN-HTTP] URL=${url}`);
    console.error(`[TRAIN-HTTP] status=${status}`);
    console.error(`[TRAIN-HTTP] content-type=${ct}`);
    console.error(`[TRAIN-HTTP] raw body=${_trainHttpSnippet(raw)}`);
  } catch (_) {}
}

function _looksLikeHtml(raw, contentType) {
  const ct = String(contentType || '').toLowerCase();
  if (ct.includes('text/html')) return true;
  const s = String(raw || '').trimStart();
  return s.startsWith('<');
}

async function _postZeroGpuFn(apiBase, fnName, body, signal, token, timeoutMs = 30000) {
  const url = `${apiBase}/call/v2/${fnName}`;
  const ctrl = new AbortController();
  // TEMP-DIAG: distinguish our own timeout abort from an outer (user/stop)
  // abort so timeouts are never misreported as user cancellation.
  let ownTimeoutFired = false;
  const timer = setTimeout(() => { ownTimeoutFired = true; ctrl.abort(); }, timeoutMs);
  const onAbort = () => ctrl.abort();
  if (signal) {
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  try {
    const res = await fetch(url, { method: 'POST', headers: _zeroGpuHeaders(token), body: JSON.stringify(body), signal: ctrl.signal });
    // Diagnostic: read as TEXT first, log before any JSON parsing.
    // (Authorization header is never logged.)
    const contentType = (res.headers && typeof res.headers.get === 'function' && res.headers.get('content-type')) || 'unknown';
    let raw = '';
    try { raw = await res.text(); }
    catch (_) { raw = ''; }
    _logTrainHttp('POST', url, res.status, contentType, raw);
    const snippet = _trainHttpSnippet(raw);
    if (res.status === 404) {
      throw new ZeroGpuTrainError(
        `ZeroGPU endpoint not found at ${url} (HTTP 404, content-type=${contentType}, body=${snippet}). ` +
        `Deploy the latest space/app.py (with the /${fnName} endpoint) to the Space. ` +
        `An HTML body here usually means a Gradio endpoint/path mismatch (D) or a 404 page (B).`,
        'space_unavailable', 502);
    }
    if (res.status === 502 || res.status === 503 || res.status === 429) {
      const detail = raw;
      const sleepy = /sleeping|paused|building|loading/i.test(detail);
      throw new ZeroGpuTrainError(
        sleepy ? `ZeroGPU Space is unavailable (HTTP ${res.status}, content-type=${contentType}, body=${snippet}). It may be sleeping or building.`
               : `ZeroGPU Space rejected the request (HTTP ${res.status}, content-type=${contentType}): ${snippet}`,
        'space_unavailable', res.status);
    }
    if (!res.ok) {
      const detail = raw;
      throw new ZeroGpuTrainError(
        `ZeroGPU call failed (HTTP ${res.status}, content-type=${contentType}): ${snippet || detail.slice(0, 500)}`,
        'space_unavailable', res.status);
    }
    let payload;
    try { payload = JSON.parse(raw); }
    catch (err) {
      const htmlHint = _looksLikeHtml(raw, contentType)
        ? ' Response looks like HTML (starts with \'<\'). Possible sources: (A) HF Space page, (B) 404/405/500/502 page, (C) Replit/proxy page, (D) Gradio endpoint/path mismatch, (E) app startup/restart page.'
        : '';
      throw new ZeroGpuTrainError(
        `ZeroGPU returned non-JSON response (HTTP ${res.status}, content-type=${contentType}, body=${snippet}): ${err.message}.${htmlHint}`,
        'malformed_response', 502);
    }
    const eventId = payload && payload.event_id;
    if (!eventId || typeof eventId !== 'string') {
      throw new ZeroGpuTrainError(`ZeroGPU did not return an event_id: ${JSON.stringify(payload).slice(0, 200)}`, 'malformed_response', 502);
    }
    return eventId;
  } catch (err) {
    if (err instanceof ZeroGpuTrainError) throw err;
    if (err.name === 'AbortError') {
      if (ownTimeoutFired && !(signal && signal.aborted)) {
        throw new ZeroGpuTrainError(`ZeroGPU ${fnName} request timed out after ${timeoutMs}ms (no event_id received).`, 'timeout', 504);
      }
      throw new ZeroGpuTrainError(`ZeroGPU ${fnName} request was cancelled.`, 'cancelled', 499);
    }
    throw new ZeroGpuTrainError(`Could not reach ZeroGPU training API ${url}: ${err.message}`, 'space_unavailable', 504);
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

// Read the SSE event stream for one ZeroGPU call. `onGenerating(dataArray)`
// is invoked for every `generating` frame. Resolves with the `complete`
// frame's data array. Throws ZeroGpuTrainError on `error` frames or a stream
// that closes without completing.
async function _fetchZeroGpuSse(apiBase, fnName, eventId, signal, token, onGenerating) {
  const { Readable } = require('stream');
  const url = `${apiBase}/call/${fnName}/${encodeURIComponent(eventId)}`;
  const headers = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(url, { signal, headers });
  } catch (err) {
    if (err.name === 'AbortError') throw new ZeroGpuTrainError('ZeroGPU stream was cancelled.', 'cancelled', 499);
    throw new ZeroGpuTrainError(`Lost connection to ZeroGPU stream: ${err.message}`, 'space_unavailable', 504);
  }
  const sseContentType = (res.headers && typeof res.headers.get === 'function' && res.headers.get('content-type')) || 'unknown';
  if (res.status === 404) {
    const detail404 = await res.text().catch(() => '');
    _logTrainHttp('GET', url, res.status, sseContentType, detail404);
    throw new ZeroGpuTrainError(
      `ZeroGPU event was not found (event expired or Space restarted) (HTTP 404, content-type=${sseContentType}, body=${_trainHttpSnippet(detail404)}).`,
      'space_unavailable', 502);
  }
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '');
    _logTrainHttp('GET', url, res.status, sseContentType, detail);
    throw new ZeroGpuTrainError(
      `ZeroGPU stream open failed (HTTP ${res.status}, content-type=${sseContentType}): ${_trainHttpSnippet(detail)}`,
      'space_unavailable', res.status || 502);
  }
  // SSE streams cannot be read fully as text first without breaking the
  // stream, so log the response head now and the first-chunk raw body below.
  // (Authorization header is never logged.)
  _logTrainHttp('GET', url, res.status, sseContentType, '(SSE stream open — first-chunk raw body follows)');
  const nodeStream = Readable.fromWeb(res.body);
  let buffer = '';
  let eventType = '';
  let dataBuf = '';
  let hadError = false;
  let errMsg = '';
  let streamHead = '';
  let streamHeadLogged = false;

  const handleFrame = () => {
    if (!eventType) return null;
    if (eventType === 'complete') {
      try {
        const parsed = JSON.parse(dataBuf);
        // TEMP-DIAG: every terminal frame, with manifest status + artifact presence.
        try {
          const arr = Array.isArray(parsed) ? parsed : [parsed];
          let man = null;
          try { man = typeof arr[0] === 'string' ? JSON.parse(arr[0]) : arr[0]; } catch (_) {}
          console.error(`[TRAIN-DIAG] ZeroGPU complete frame event=${eventId} manifest=${man ? man.status : 'unparseable'} hasFile=${!!(arr[1] && typeof arr[1] === 'object')}`);
        } catch (_) {}
        return { done: true, data: parsed };
      }
      catch (parseErr) {
        const htmlHint = _looksLikeHtml(dataBuf, sseContentType)
          ? ' Complete frame looks like HTML (starts with \'<\'). Possible sources: (A) HF Space page, (B) 404/405/500/502 page, (C) Replit/proxy page, (D) Gradio endpoint/path mismatch, (E) app startup/restart page.'
          : '';
        throw new ZeroGpuTrainError(
          `ZeroGPU returned malformed complete frame (HTTP ${res.status}, content-type=${sseContentType}, body=${_trainHttpSnippet(dataBuf)}): ${parseErr.message}.${htmlHint}`,
          'malformed_response', 502);
      }
    }
    if (eventType === 'error') {
      hadError = true;
      errMsg = dataBuf;
      // TEMP-DIAG: error frames are terminal for the stream — log immediately.
      console.error(`[TRAIN-DIAG] ZeroGPU error frame event=${eventId} body=${String(dataBuf).slice(0, 300)}`);
      return null;
    }
    if (eventType === 'generating' && onGenerating) {
      try { onGenerating(JSON.parse(dataBuf)); }
      catch (genErr) {
        // Diagnostic only — behavior unchanged (malformed progress frames are ignored).
        try {
          if (_looksLikeHtml(dataBuf, sseContentType)) {
            console.error(`[TRAIN-HTTP] method=GET (SSE generating frame non-JSON)`);
            console.error(`[TRAIN-HTTP] URL=${url}`);
            console.error(`[TRAIN-HTTP] status=${res.status}`);
            console.error(`[TRAIN-HTTP] content-type=${sseContentType}`);
            console.error(`[TRAIN-HTTP] raw body=${_trainHttpSnippet(dataBuf)}`);
          }
        } catch (_) {}
      }
    }
    return null;
  };

  for await (const chunk of nodeStream) {
    const text = chunk.toString('utf8');
    // Capture the first 500 chars of the raw stream for the diagnostic.
    if (!streamHeadLogged) {
      streamHead += text;
      if (streamHead.length >= 500) {
        streamHeadLogged = true;
        _logTrainHttp('GET', url, res.status, sseContentType, streamHead);
        if (_looksLikeHtml(streamHead, sseContentType) && !streamHead.includes('event:')) {
          throw new ZeroGpuTrainError(
            `ZeroGPU stream returned non-SSE HTML (HTTP ${res.status}, content-type=${sseContentType}, body=${_trainHttpSnippet(streamHead)}). ` +
            `Possible sources: (A) HF Space page, (B) 404/405/500/502 page, (C) Replit/proxy page, (D) Gradio endpoint/path mismatch, (E) app startup/restart page.`,
            'malformed_response', 502);
        }
      }
    }
    buffer += text;
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).replace(/\r$/, '');
      buffer = buffer.slice(idx + 1);
      if (line === '' || line === '\r') {
        const r = handleFrame();
        if (r && r.done) return r.data;
        eventType = '';
        dataBuf = '';
        continue;
      }
      if (line.startsWith('event:')) eventType = line.slice(6).trim();
      else if (line.startsWith('data:')) dataBuf = line.slice(5).trim();
      else if (line.endsWith(': heartbeat')) { /* keep-alive, ignored */ }
    }
  }
  // Flush a final frame if the stream ended without a trailing blank line.
  if (eventType && dataBuf !== '') {
    const r = handleFrame();
    if (r && r.done) return r.data;
  }
  // Short streams (<500 chars) never triggered the head log above — log what we saw.
  if (!streamHeadLogged) {
    _logTrainHttp('GET', url, res.status, sseContentType, streamHead || buffer);
  }
  if (hadError) throw _classifyZeroGpuErrorFrame(errMsg);
  {
    const tailSnippet = _trainHttpSnippet(streamHead || buffer || dataBuf);
    const htmlHint = _looksLikeHtml(streamHead || buffer || dataBuf, sseContentType)
      ? ' Body looks like HTML (starts with \'<\'). Possible sources: (A) HF Space page, (B) 404/405/500/502 page, (C) Replit/proxy page, (D) Gradio endpoint/path mismatch, (E) app startup/restart page.'
      : '';
    throw new ZeroGpuTrainError(
      `ZeroGPU stream closed without a complete frame (HTTP ${res.status}, content-type=${sseContentType}, body=${tailSnippet}).${htmlHint}`,
      'malformed_response', 502);
  }
}

function _classifyZeroGpuErrorFrame(errMsg) {
  let message = errMsg;
  let code = 'zerogpu_runtime';
  try {
    const parsed = JSON.parse(errMsg);
    if (parsed && typeof parsed === 'object' && parsed.error) message = parsed.error;
    if (parsed && parsed.title === 'ZeroGPU worker error') code = 'zerogpu_quota';
    if (parsed && parsed.title === 'ZeroGPU queue timeout') code = 'zerogpu_timeout';
  } catch { /* already a string */ }
  if (/No GPU available|quota|runs.?limit|credits|exceed|GHA seconds|GPU is not available/i.test(message)) code = 'zerogpu_quota';
  if (/No GPU was available after \d+s/i.test(message)) code = 'zerogpu_timeout';
  if (/Not Found/i.test(message)) code = 'space_unavailable';
  if (/out of memory/i.test(message)) code = 'gpu_oom';
  if (code === 'zerogpu_timeout') {
    message = 'ZeroGPU timeout: the Space could not start the job within its queue/execution limit. Try again later or use a smaller configuration.';
  }
  if (code === 'zerogpu_quota') {
    message = 'ZeroGPU cannot provide a GPU right now (quota exhausted, runs limit, or no capacity). Check the Space owner\u2019s ZeroGPU quota/billing and retry later.';
  }
  if (code === 'gpu_oom') {
    message = 'GPU out of memory on ZeroGPU: this model does not fit. Try a smaller/quantized model.';
  }
  return new ZeroGpuTrainError(`ZeroGPU error: ${message}`, code, 502);
}

// Apply one remote stream event to the job. Mirrors the local stdout parser
// (markers + metric lines) so both providers drive identical UI updates.
function _applyRemoteTrainEvent(job, ev) {
  if (!ev || typeof ev !== 'object') return;
  // TEMP-DIAG: log every remote event as it is applied to the job, so a
  // failure can never pass through this layer without a server-side trace.
  try {
    const summary = ev.type === 'metric' && ev.metric ? `step=${ev.metric.step}`
      : ev.type === 'progress' && ev.progress ? `total_steps=${ev.progress.total_steps}`
      : ev.type === 'status' ? String(ev.status)
      : ev.type === 'error' ? String(ev.error || '').slice(0, 120)
      : String(ev.line || '').slice(0, 120);
    console.error(`[TRAIN-DIAG] remote event job=${job.job_id} type=${ev.type} ${summary}`);
  } catch (_) {}
  if (ev.type === 'metric' && ev.metric) {
    _applyTrainerMetric(job, ev.metric);
    return;
  }
  if (ev.type === 'progress' && ev.progress && typeof ev.progress === 'object') {
    const patch = {};
    for (const k of ['total_steps', 'trainable_params', 'total_params', 'current_step', 'current_epoch']) {
      if (typeof ev.progress[k] === 'number' && Number.isFinite(ev.progress[k])) patch[k] = ev.progress[k];
    }
    if (Object.keys(patch).length) _updateProgress(job, patch);
    return;
  }
  if (ev.type === 'status') {
    if (ev.line) _log(job, String(ev.line));
    if (ev.status === 'failed') {
      if (job.status !== 'failed') _setStatus(job, 'failed');
      if (!job.error) job.error = 'training_error';
    } else if (ev.status) {
      _setStatus(job, ev.status);
    }
    return;
  }
  if (ev.type === 'error') {
    _log(job, `[TRAIN] training_error: ${ev.error || 'remote worker error'}`);
    if (job.status !== 'failed') _setStatus(job, 'failed');
    job.error = 'training_error';
    return;
  }
  // 'log' and anything else carrying a line
  if (typeof ev.line === 'string') {
    _log(job, ev.line);
    const totalMatch = ev.line.match(/total_steps=(\d+)/);
    if (totalMatch) {
      const t = parseInt(totalMatch[1], 10);
      if (Number.isFinite(t) && t > 0) _updateProgress(job, { total_steps: t });
    }
    const low = ev.line.toLowerCase();
    if (low.includes('training_error')) {
      if (job.status !== 'failed') _setStatus(job, 'failed');
      job.error = 'training_error';
    } else if (low.includes('evaluating')) {
      _setStatus(job, 'evaluating');
    } else if (low.includes('loading')) {
      _setStatus(job, 'loading');
    } else if (low.includes('training') && !low.includes('training_error')) {
      if (job.status === 'loading' || job.status === 'queued') _setStatus(job, 'training');
    }
    // NOTE: a remote '[TRAIN] finished' line alone never finalizes the job —
    // only the manifest + downloaded artifacts do.
  }
}

function _zeroGpuFileCandidates(apiBase, file) {
  const origin = _zeroGpuSpaceOrigin(apiBase);
  const out = [];
  const push = (u) => { if (u && !out.includes(u)) out.push(u); };
  if (file && typeof file.url === 'string' && file.url) {
    push(file.url.startsWith('http') ? file.url : `${origin}${file.url.startsWith('/') ? '' : '/'}${file.url}`);
  }
  if (file && typeof file.path === 'string' && file.path) {
    if (file.path.startsWith('http')) push(file.path);
    else {
      push(`${origin}/gradio_api/file=${file.path}`);
      push(`${origin}/file=${file.path}`);
      push(`${origin}/gradio_api/file/${file.path.replace(/^\/+/, '')}`);
    }
  }
  return out;
}

async function _downloadZeroGpuFile(apiBase, file, signal, token) {
  const maxBytes = Number(process.env.ZEROGPU_TRAIN_MAX_ARTIFACT_BYTES || 2000000000);
  const candidates = _zeroGpuFileCandidates(apiBase, file);
  if (!candidates.length) throw new ZeroGpuTrainError('ZeroGPU finished but returned no downloadable artifact file.', 'missing_artifact', 502);
  let lastErr = null;
  let lastHttp = '';
  for (const url of candidates) {
    try {
      const headers = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const res = await fetch(url, { signal, headers });
      const dlContentType = (res.headers && typeof res.headers.get === 'function' && res.headers.get('content-type')) || 'unknown';
      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => '');
        _logTrainHttp('GET', url, res.status, dlContentType, detail);
        lastHttp = `HTTP ${res.status}, content-type=${dlContentType}, body=${_trainHttpSnippet(detail)}`;
        lastErr = new Error(`HTTP ${res.status} content-type=${dlContentType} body=${_trainHttpSnippet(detail)}`);
        continue;
      }
      const chunks = [];
      let total = 0;
      for await (const chunk of res.body) {
        total += chunk.length;
        if (total > maxBytes) throw new ZeroGpuTrainError(`Artifact exceeds size limit (${maxBytes} bytes).`, 'artifact_error', 502);
        chunks.push(Buffer.from(chunk));
      }
      const buf = Buffer.concat(chunks);
      if (!buf.length) {
        _logTrainHttp('GET', url, res.status, dlContentType, '');
        lastHttp = `HTTP ${res.status}, content-type=${dlContentType}, empty file`;
        lastErr = new Error('empty file');
        continue;
      }
      // Diagnostic: text-first peek (first 500 bytes decoded) before treating as zip.
      const headText = buf.subarray(0, 500).toString('utf8');
      _logTrainHttp('GET', url, res.status, dlContentType, `${headText} (… ${total} bytes total)`);
      if (_looksLikeHtml(headText, dlContentType)) {
        lastHttp = `HTTP ${res.status}, content-type=${dlContentType}, body=${_trainHttpSnippet(headText)}`;
        lastErr = new Error(
          `Artifact URL returned HTML instead of zip (HTTP ${res.status}, content-type=${dlContentType}, body=${_trainHttpSnippet(headText)}). ` +
          `Possible sources: (A) HF Space page, (B) 404/405/500/502 page, (C) Replit/proxy page, (D) Gradio endpoint/path mismatch, (E) app startup/restart page.`);
        continue;
      }
      return buf;
    } catch (err) {
      if (err instanceof ZeroGpuTrainError) throw err;
      lastErr = err;
    }
  }
  throw new ZeroGpuTrainError(
    `Could not download ZeroGPU artifacts: ${lastErr ? lastErr.message : 'unknown'}${lastHttp ? ` [last HTTP: ${lastHttp}]` : ''}`,
    'artifact_error', 502);
}

// Minimal dependency-free ZIP reader (stored + deflate). Archive paths are
// NEVER trusted: absolute paths and `..` segments are skipped, and every
// destination is verified to stay inside destDir.
function _extractZipBuffer(buf, destDir) {
  const zlib = require('zlib');
  if (!Buffer.isBuffer(buf)) buf = Buffer.from(buf);
  if (buf.length < 22) throw new ZeroGpuTrainError('Artifact is not a zip file (too small).', 'artifact_error', 502);
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 0xffff); i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new ZeroGpuTrainError('Artifact is not a zip file (EOCD missing).', 'artifact_error', 502);
  const cdCount = buf.readUInt16LE(eocd + 10);
  const cdOffset = buf.readUInt32LE(eocd + 16);
  const base = path.resolve(destDir);
  const names = [];
  let p = cdOffset;
  for (let i = 0; i < cdCount; i++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new ZeroGpuTrainError('Artifact zip central directory is corrupt.', 'artifact_error', 502);
    const flags = buf.readUInt16LE(p + 8);
    const method = buf.readUInt16LE(p + 10);
    const compSize = buf.readUInt32LE(p + 20);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const localOffset = buf.readUInt32LE(p + 42);
    const rawName = buf.toString('utf8', p + 46, p + 46 + nameLen);
    p += 46 + nameLen + extraLen + commentLen;
    const parts = rawName.split('/').filter((s) => s && s !== '.');
    const unsafe = !parts.length || rawName.startsWith('/') || parts.includes('..');
    if (unsafe) continue;
    const resolved = path.resolve(path.join(destDir, ...parts));
    if (resolved !== base && !resolved.startsWith(base + path.sep)) continue;
    if (rawName.endsWith('/')) { fs.mkdirSync(resolved, { recursive: true }); continue; }
    if (flags & 0x08) throw new ZeroGpuTrainError(`Artifact zip uses streaming entries (${rawName}); cannot extract safely.`, 'artifact_error', 502);
    if (buf.readUInt32LE(localOffset) !== 0x04034b50) throw new ZeroGpuTrainError('Artifact zip local header is corrupt.', 'artifact_error', 502);
    const lhNameLen = buf.readUInt16LE(localOffset + 26);
    const lhExtraLen = buf.readUInt16LE(localOffset + 28);
    const dataStart = localOffset + 30 + lhNameLen + lhExtraLen;
    const comp = buf.subarray(dataStart, dataStart + compSize);
    let data;
    if (method === 0) data = Buffer.from(comp);
    else if (method === 8) data = zlib.inflateRawSync(comp);
    else throw new ZeroGpuTrainError(`Artifact zip uses unsupported method ${method} (${rawName}).`, 'artifact_error', 502);
    fs.mkdirSync(path.dirname(resolved), { recursive: true });
    fs.writeFileSync(resolved, data);
    names.push(parts.join('/'));
  }
  return names;
}

function _failZeroGpuJob(job, message, code = 'training_error') {
  if (job.end_time) return; // already terminal
  // TEMP-DIAG: every backend failure finalization gets a server-side trace.
  console.error(`[TRAIN-DIAG] job failed job=${job.job_id} code=${code} message=${String(message).slice(0, 300)}`);
  _log(job, `[TRAIN] training_error: ${message}`);
  _setStatus(job, 'failed');
  _updateProgress(job, { gpu_status: 'idle', eta: 0 });
  job.error = code === 'stopped_by_user' ? 'stopped_by_user' : 'training_error';
  job.end_time = _now();
  _broadcast(job, 'done', { status: 'failed', error: job.error, message });
  _saveArtifacts(job);
}

async function _requestZeroGpuCancel(apiBase, jobId, token) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  try {
    const eventId = await _postZeroGpuFn(apiBase, 'cancel_train', { job_id: jobId }, ctrl.signal, token, 15000);
    const data = await _fetchZeroGpuSse(apiBase, 'cancel_train', eventId, ctrl.signal, token, null);
    const first = Array.isArray(data) ? data[0] : data;
    try {
      const parsed = typeof first === 'string' ? JSON.parse(first) : first;
      return !!(parsed && parsed.ok);
    } catch { return false; }
  } finally {
    clearTimeout(timer);
  }
}

function _finishZeroGpuJob(job, manifest, file) {
  if (job._remote && job._remote.timeout) { clearTimeout(job._remote.timeout); job._remote.timeout = null; }
  if (job._aborted) {
    _log(job, '[TRAIN] remote worker finished after cancel; results ignored');
    return Promise.resolve();
  }
  if (!manifest || manifest.status !== 'finished') {
    _failZeroGpuJob(job, (manifest && manifest.message) || 'remote training failed without a manifest');
    return Promise.resolve();
  }
  if (!file) {
    _failZeroGpuJob(job, 'remote training finished but returned no artifacts');
    return Promise.resolve();
  }
  const { apiBase, token } = job._remote || {};
  const controller = new AbortController();
  if (job._remote) job._remote.downloadController = controller;
  return _downloadZeroGpuFile(apiBase, file, controller.signal, token).then(
    (zipBuf) => {
      if (job._aborted) {
        _log(job, '[TRAIN] artifacts downloaded after cancel; results ignored');
        return;
      }
      const dir = _safeArtifactDir(job.job_id);
      fs.mkdirSync(dir, { recursive: true });
      const names = _extractZipBuffer(zipBuf, dir);
      _log(job, `[TRAIN] artifacts downloaded (${names.length} files) -> ${dir}`);
      job.artifacts_dir = dir;
      job.end_time = _now();
      _setStatus(job, 'finished');
      _updateProgress(job, { gpu_status: 'idle', eta: 0, percent: 100 });
      _broadcast(job, 'done', { status: 'finished' });
      _saveArtifacts(job); // writes metrics.json / job.json / training_logs.txt alongside model files
    },
    (err) => {
      if (job._aborted) return;
      _failZeroGpuJob(job, err.message || String(err), err.code);
    }
  );
}

function _startZeroGPUTraining(job) {
  const cfg = job.config;
  // TEMP-DIAG: proves the REMOTE path executed (no local Python spawn on this path).
  console.error(`[TRAIN-DIAG] _startZeroGPUTraining CALLED job=${job.job_id} (remote Space, never spawns local Python)`);
  let remote;
  try {
    remote = _zeroGpuTrainConfig(process.env);
  } catch (e) {
    _failZeroGpuJob(job, e.message || String(e));
    return;
  }
  job.start_time = _now();
  _setStatus(job, 'loading');
  _log(job, `[TRAIN] loading model=${cfg.model_id} dataset=${cfg.dataset_id} task=${cfg.task_type} provider=zerogpu space=${remote.spaceId} method=${cfg.training_method}`);
  if (cfg.training_method === 'lora') {
    _log(job, `[TRAIN] lora r=${cfg.lora_r} alpha=${cfg.lora_alpha} dropout=${cfg.lora_dropout} target_modules=${cfg.target_modules}`);
  }
  _log(job, `[TRAIN] config epochs=${cfg.epochs} batch_size=${cfg.batch_size} lr=${cfg.learning_rate} max_steps=${cfg.max_steps || 'auto'} validation_split=${cfg.validation_split}%`);
  // Verify the exact final hostname and path before any HF Space request.
  // No pre-flight request is made on this path: the first Space request is
  // POST {apiBase}/call/v2/train, followed by GET {apiBase}/call/train/{event_id}.
  try {
    const origin = _zeroGpuSpaceOrigin(remote.apiBase);
    let hostname = '(unparseable)';
    try { hostname = new URL(remote.apiBase).hostname; } catch (_) {}
    console.error(`[TRAIN-HTTP] resolved spaceId=${remote.spaceId}`);
    console.error(`[TRAIN-HTTP] resolved apiBase=${remote.apiBase}`);
    console.error(`[TRAIN-HTTP] resolved origin=${origin} hostname=${hostname}`);
    console.error(`[TRAIN-HTTP] POST URL=${remote.apiBase}/call/v2/train`);
    console.error(`[TRAIN-HTTP] GET template=${remote.apiBase}/call/train/{event_id} (GET has no /v2 by Gradio convention; POST has /v2)`);
    console.error(`[TRAIN-HTTP] pre-flight requests before POST: none (direct POST)`);
  } catch (_) {}
  _updateProgress(job, { gpu_status: 'zerogpu', total_steps: cfg.max_steps || (cfg.epochs * 100) });

  // Structured training parameters only — never arbitrary code.
  // The v2 gateway maps the POST body onto the endpoint's DECLARED inputs,
  // so the config must be wrapped under the single `train_request_json`
  // input name (a flat config object is rejected with HTTP 500).
  const payload = {
    train_request_json: {
      model_id: cfg.model_id,
      dataset_id: cfg.dataset_id,
      task_type: cfg.task_type,
      epochs: cfg.epochs,
      batch_size: cfg.batch_size,
      learning_rate: cfg.learning_rate,
      validation_split: cfg.validation_split,
      max_samples: cfg.max_samples,
      max_steps: cfg.max_steps,
      training_method: cfg.training_method,
      lora_r: cfg.lora_r,
      lora_alpha: cfg.lora_alpha,
      lora_dropout: cfg.lora_dropout,
      target_modules: cfg.target_modules,
      job_id: job.job_id,
    },
  };
  const controller = new AbortController();
  const timeoutMs = Number(process.env.ZEROGPU_TRAIN_TIMEOUT_MS) || (cfg.max_time_sec + 180) * 1000;
  job._remote = { eventId: null, controller, timeout: null, apiBase: remote.apiBase, token: remote.token, downloadController: null };
  job._remote.timeout = setTimeout(() => {
    if (job.end_time || ['finished', 'failed'].includes(job.status)) return;
    try { controller.abort(); } catch (_) {}
    _failZeroGpuJob(job, `ZeroGPU training timed out after ${Math.round(timeoutMs / 1000)}s (event=${(job._remote && job._remote.eventId) || 'n/a'}). The Space may still be running; results will be ignored.`);
  }, timeoutMs);
  if (job._remote.timeout.unref) job._remote.timeout.unref();

  (async () => {
    try {
      _log(job, `[TRAIN] POST ${remote.apiBase}/call/v2/train`);
      const eventId = await _postZeroGpuFn(remote.apiBase, 'train', payload, controller.signal, remote.token, 60000);
      if (job._aborted) return;
      job._remote.eventId = eventId;
      _log(job, `[TRAIN] ZeroGPU event=${eventId}; streaming progress`);
      try {
        console.error(`[TRAIN-HTTP] method=GET`);
        console.error(`[TRAIN-HTTP] URL=${remote.apiBase}/call/train/${encodeURIComponent(eventId)}`);
        console.error(`[TRAIN-HTTP] note=opening SSE stream (headers + first-chunk body logged inside _fetchZeroGpuSse)`);
      } catch (_) {}
      const data = await _fetchZeroGpuSse(remote.apiBase, 'train', eventId, controller.signal, remote.token, (frame) => {
        if (job._aborted) return;
        const first = Array.isArray(frame) ? frame[0] : frame;
        if (typeof first !== 'string') return;
        let ev;
        try { ev = JSON.parse(first); }
        catch (frameErr) {
          // Diagnostic only — behavior unchanged (non-JSON progress text is logged as-is).
          try {
            if (String(first).trimStart().startsWith('<')) {
              console.error(`[TRAIN-HTTP] method=GET (SSE generating frame non-JSON)`);
              console.error(`[TRAIN-HTTP] URL=${remote.apiBase}/call/train/${encodeURIComponent(eventId)}`);
              console.error(`[TRAIN-HTTP] status=(stream)`);
              console.error(`[TRAIN-HTTP] content-type=(stream)`);
              console.error(`[TRAIN-HTTP] raw body=${String(first).slice(0, 500)}`);
            }
          } catch (_) {}
          _log(job, first); return;
        }
        _applyRemoteTrainEvent(job, ev);
      });
      // complete frame: [manifestJson, fileObj|null]
      const arr = Array.isArray(data) ? data : [data];
      let manifest = null;
      try { manifest = typeof arr[0] === 'string' ? JSON.parse(arr[0]) : arr[0]; }
      catch (manErr) {
        // Diagnostic only — behavior unchanged (null manifest still fails as before).
        try {
          console.error(`[TRAIN-HTTP] method=GET (SSE complete-frame manifest non-JSON)`);
          console.error(`[TRAIN-HTTP] URL=${remote.apiBase}/call/train/${encodeURIComponent(eventId)}`);
          console.error(`[TRAIN-HTTP] status=(stream complete)`);
          console.error(`[TRAIN-HTTP] content-type=(stream)`);
          console.error(`[TRAIN-HTTP] raw body=${String(arr[0]).slice(0, 500)}`);
        } catch (_) {}
        manifest = null;
      }
      const file = arr.length > 1 && arr[1] && typeof arr[1] === 'object' ? arr[1] : null;
      await _finishZeroGpuJob(job, manifest, file);
    } catch (err) {
      if (job._aborted || job.end_time) return;
      if (err && (err.code === 'cancelled' || err.name === 'AbortError')) {
        _failZeroGpuJob(job, 'cancelled by user', 'stopped_by_user');
        return;
      }
      _failZeroGpuJob(job, (err && err.message) || String(err), err && err.code);
    } finally {
      if (job._remote && job._remote.timeout) { clearTimeout(job._remote.timeout); job._remote.timeout = null; }
    }
  })();
}

// ── Step-count estimation (no fake numbers) ──────────────────────────────
// Resolves the dataset's real row count via the datasets-server API and
// computes optimizer steps = ceil(train_rows / batch_size) * epochs
// (or max_steps when set). Never throws: unknown → estimated_total_steps null.
// Small in-memory cache: dataset row counts don't change between keystrokes.
const _rowCountCache = new Map();

async function _fetchJson(url, fetchImpl, timeoutMs = 8000) {
  const f = fetchImpl || fetch;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await f(url, { signal: ctrl.signal });
    // Diagnostic: text-first when the response supports it (never res.json()
    // directly on real fetch, so an HTML error page surfaces as a clear log
    // instead of `Unexpected token '<'`). Test mocks that only implement
    // .json() fall back to the legacy path with identical return semantics.
    const hasText = res && typeof res.text === 'function';
    let contentType = 'unknown';
    try { contentType = (res.headers && typeof res.headers.get === 'function' && res.headers.get('content-type')) || 'unknown'; } catch (_) {}
    if (hasText) {
      let raw = '';
      try { raw = await res.text(); } catch (_) { raw = ''; }
      try {
        console.error(`[TRAIN-HTTP] method=GET`);
        console.error(`[TRAIN-HTTP] URL=${url}`);
        console.error(`[TRAIN-HTTP] status=${res.status}`);
        console.error(`[TRAIN-HTTP] content-type=${contentType}`);
        console.error(`[TRAIN-HTTP] raw body=${String(raw).slice(0, 500)}`);
      } catch (_) {}
      if (!res.ok) return null;
      try { return JSON.parse(raw); }
      catch {
        try { console.error(`[TRAIN-HTTP] non-JSON response (HTTP ${res.status}, content-type=${contentType}, body=${String(raw).slice(0, 500)})`); } catch (_) {}
        return null;
      }
    }
    // Legacy/mock path: no .text() available (e.g. unit-test fakeFetch).
    if (!res.ok) return null;
    try {
      const data = await res.json();
      try {
        console.error(`[TRAIN-HTTP] method=GET`);
        console.error(`[TRAIN-HTTP] URL=${url}`);
        console.error(`[TRAIN-HTTP] status=${res.status}`);
        console.error(`[TRAIN-HTTP] content-type=${contentType}`);
        console.error(`[TRAIN-HTTP] raw body=${JSON.stringify(data).slice(0, 500)}`);
      } catch (_) {}
      return data;
    } catch { return null; }
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function _fetchDatasetRowCount(datasetId, fetchImpl) {
  if (_rowCountCache.has(datasetId)) return _rowCountCache.get(datasetId);
  // Step 1: resolve a config+split (datasets-server requires all three params).
  const splits = await _fetchJson(
    `https://datasets-server.huggingface.co/splits?dataset=${encodeURIComponent(datasetId)}`, fetchImpl);
  let config = null;
  let split = null;
  const list = splits && Array.isArray(splits.splits) ? splits.splits : null;
  if (list && list.length) {
    const train = list.find((s) => String(s.split).toLowerCase() === 'train') || list[0];
    config = train.config;
    split = train.split;
  }
  let total = null;
  if (config && split) {
    // /rows (length=1) carries num_rows_total for the split; first-rows does not.
    const rows = await _fetchJson(
      `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(datasetId)}` +
      `&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=0&length=1`, fetchImpl);
    const n = Number(rows && rows.num_rows_total);
    if (Number.isFinite(n) && n > 0) total = Math.floor(n);
  }
  if (total) _rowCountCache.set(datasetId, total);
  return total;
}

async function estimateTrainingSteps(opts = {}, fetchImpl) {
  const empty = { estimated_total_steps: null, train_rows: null, total_rows: null, basis: 'insufficient-config' };
  const rawMax = opts.max_steps !== undefined ? opts.max_steps : opts.maxSteps;
  if (rawMax !== null && rawMax !== undefined && String(rawMax).trim() !== '') {
    const n = Number(rawMax);
    if (Number.isFinite(n) && Number.isInteger(n) && n > 0) {
      return { estimated_total_steps: n, train_rows: null, total_rows: null, basis: 'max_steps' };
    }
  }
  const datasetId = String(opts.dataset_id || opts.datasetId || '').trim();
  const epochs = Number(opts.epochs);
  const batch = Number(opts.batch_size !== undefined ? opts.batch_size : opts.batchSize);
  if (!datasetId || !Number.isFinite(epochs) || epochs < 1 || !Number.isFinite(batch) || batch < 1) return empty;
  const total = await _fetchDatasetRowCount(datasetId, fetchImpl);
  if (!total) return { estimated_total_steps: null, train_rows: null, total_rows: null, basis: 'unknown-dataset-size' };
  const maxSamples = Number(opts.max_samples) > 0 ? Number(opts.max_samples) : MAX_SAMPLES;
  const effective = Math.min(total, maxSamples);
  const valFrac = Math.min(50, Math.max(0, Number(opts.validation_split !== undefined ? opts.validation_split : opts.validationSplit) || 0)) / 100;
  const trainRows = Math.max(1, Math.round(effective * (1 - valFrac)));
  const steps = Math.max(1, Math.ceil(trainRows / batch) * Math.floor(epochs));
  return { estimated_total_steps: steps, train_rows: trainRows, total_rows: total, basis: 'dataset-rows' };
}

// Best-effort: patch a queued/running job's progress denominator with the
// real estimate. Never throws; safe to fire-and-forget after train/start.
async function refreshTotalStepsEstimate(job) {
  try {
    if (!job || job.end_time || ['finished', 'failed'].includes(job.status)) return null;
    if (job.config && job.config.max_steps) {
      _updateProgress(job, { total_steps: job.config.max_steps });
      return job.config.max_steps;
    }
    const est = await estimateTrainingSteps(job.config || {});
    if (est.estimated_total_steps && !job.end_time && !['finished', 'failed'].includes(job.status)) {
      _updateProgress(job, { total_steps: est.estimated_total_steps });
      _log(job, `[TRAIN] estimated total_steps=${est.estimated_total_steps} (${est.basis}, train_rows=${est.train_rows})`);
    }
    return est.estimated_total_steps;
  } catch {
    return null;
  }
}

function startJob(job) {
  const provider = resolveTrainingProvider(process.env);
  // TEMP-DIAG: full branch trace — config value in, branch out. There is no
  // fallback: exactly one of the three branches below always runs.
  try {
    console.error(`[TRAIN-DIAG] startJob job=${job.job_id} config.provider=${job.config.provider} resolved provider=${provider} branch=${provider === 'zerogpu' ? '_startZeroGPUTraining' : provider === 'modal' ? 'modal-fail' : '_startPythonTraining'}`);
  } catch (_) {}
  job.config.provider = provider;
  if (provider === 'zerogpu') {
    // Remote execution on the ZeroGPU Space — NEVER spawns local Python.
    job.progress.gpu_status = 'zerogpu';
    _log(job, `[TRAIN] provider=zerogpu starting remote training (ZeroGPU Space)`);
    _startZeroGPUTraining(job);
  } else if (provider === 'modal') {
    // No Modal training runner exists: fail loudly instead of silently
    // running locally while the UI claims "modal".
    job.progress.gpu_status = 'modal';
    const msg = 'Modal training is not implemented (no remote runner). Set TRAINING_PROVIDER=local or zerogpu.';
    _log(job, `[TRAIN] training_error: ${msg}`);
    _setStatus(job, 'failed');
    _updateProgress(job, { gpu_status: 'idle', eta: 0 });
    job.error = 'training_error';
    job.end_time = _now();
    _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
    _saveArtifacts(job);
  } else {
    // Local execution only when explicitly selected.
    job.progress.gpu_status = 'local';
    _log(job, `[TRAIN] provider=local starting real training (Trainer)`);
    _startPythonTraining(job);
  }
  return job;
}

function stopJob(job_id, opts = {}) {
  const job = jobs.get(job_id);
  if (!job) return null;
  if (['finished', 'failed'].includes(job.status)) return job;
  job._aborted = true;
  const caller = opts.caller || 'unknown';
  const cellId = opts.cellId || 'unknown';
  const userInitiated = opts.userInitiated;
  const stack = opts.stack || '';
  const ts = new Date().toISOString();
  console.log(`[TRAIN-DIAG] stopJob job_id=${job_id} caller=${caller} cellId=${cellId} userInitiated=${userInitiated} status=${job.status} step=${job.progress?.current_step}/${job.progress?.total_steps} pid=${job._pythonProc?.pid} ts=${ts} stack=${String(stack).slice(0,800)}`);
  _log(job, `[TRAIN-DIAG] stopJob caller=${caller} cellId=${cellId} userInitiated=${userInitiated} stack=${String(stack).slice(0,500)}`);

  // Terminate Python process and its children if possible
  if (job._remote) {
    const { controller, timeout, apiBase, token, eventId, downloadController } = job._remote;
    if (timeout) { clearTimeout(timeout); job._remote.timeout = null; }
    try { if (controller) controller.abort(); } catch (_) {}
    try { if (downloadController) downloadController.abort(); } catch (_) {}
    // Best-effort remote cancellation: the Space acknowledges via its
    // cancel_train endpoint. If it cannot be reached, the worker may run to
    // completion in the background, but its results are ignored locally
    // (job._aborted) so nothing is ever presented as a stopped-then-finished job.
    _requestZeroGpuCancel(apiBase, job_id, token).then(
      (ok) => _log(job, ok
        ? `[TRAIN] remote cancel acknowledged for ${job_id} (event=${eventId || 'n/a'})`
        : `[TRAIN] remote cancel NOT confirmed for ${job_id} — worker may finish in background; results will be ignored`),
      (e) => _log(job, `[TRAIN] remote cancel request failed (${e.message}) — worker may finish in background; results will be ignored`)
    );
    _log(job, `[TRAIN] stop requested for remote ZeroGPU job (caller=${caller})`);
    job._remote.eventId = null;
  } else if (job._pythonProc) {
    const proc = job._pythonProc;
    job._pythonProc = null;
    try {
      // try graceful SIGTERM first, then SIGKILL after 3s
      proc.kill('SIGTERM');
      _log(job, `[TRAIN] SIGTERM sent to ${proc.pid} (caller=${caller})`);
      const killTimer = setTimeout(() => {
        try {
          if (!proc.killed) proc.kill('SIGKILL');
          _log(job, `[TRAIN] SIGKILL sent to ${proc.pid}`);
        } catch (_) {}
      }, 3000);
      proc.on('close', () => clearTimeout(killTimer));
    } catch (e) {
      _log(job, `[TRAIN] stop failed: ${e.message}`);
    }
  } else {
    _log(job, `[TRAIN] stopJob called but no _pythonProc (already exited?) caller=${caller}`);
  }

  _setStatus(job, 'failed');
  job.error = 'stopped_by_user';
  job.end_time = _now();
  _updateProgress(job, { gpu_status: 'idle', eta: 0 });
  _broadcast(job, 'done', { status: 'failed', error: job.error, message: 'stopped by user' });
  _log(job, `[TRAIN] Stopped by user (caller=${caller})`);
  _saveArtifacts(job);
  return job;
}

function _saveArtifacts(job) {
  try {
    const dir = path.join(__dirname, 'training_outputs', job.job_id);
    fs.mkdirSync(dir, { recursive: true });
    job.artifacts_dir = dir;
    fs.writeFileSync(path.join(dir, 'metrics.json'), JSON.stringify(job.metrics, null, 2));
    fs.writeFileSync(path.join(dir, 'training_logs.txt'), job.logs.join('\n'));
    // Save job config to job.json to avoid overwriting HF model config.json
    fs.writeFileSync(path.join(dir, 'job.json'), JSON.stringify({ job_id: job.job_id, status: job.status, config: job.config, progress: job.progress, error: job.error, user_id: job.user_id || job.owner || null }, null, 2));
    // Also keep trainer_config.json if Python runner saved it; do not overwrite HF config.json
    if (job.status === 'failed' && job.error) {
      fs.writeFileSync(path.join(dir, 'error.json'), JSON.stringify({ error: job.error, logs_tail: job.logs.slice(-20).join('\n') }, null, 2));
    } else if (job.status === 'finished') {
      const marker = path.join(dir, 'README.txt');
      if (!fs.existsSync(marker)) {
        fs.writeFileSync(marker, `Training job ${job.job_id}\nModel: ${job.config.model_id}\nDataset: ${job.config.dataset_id}\nStatus: ${job.status}\nMetrics: ${job.metrics.length}\n`);
      }
    }
    _log(job, `[TRAIN] Artifacts saved to ${dir} (metrics=${job.metrics.length})`);
  } catch (e) {
    _log(job, `[TRAIN] Failed to save artifacts: ${e.message}`);
  }
}

// SSE helpers
function attachSSE(job_id, res) {
  const job = jobs.get(job_id);
  if (!job) return false;
  job.sseClients.add(res);
  res.write(`event: status\ndata: ${JSON.stringify({ status: job.status, job_id })}\n\n`);
  res.write(`event: progress\ndata: ${JSON.stringify(job.progress)}\n\n`);
  for (const m of job.metrics.slice(-20)) {
    res.write(`event: metrics\ndata: ${JSON.stringify(m)}\n\n`);
  }
  // Replay recent logs + terminal state to late attachers, but ONLY for
  // already-terminal jobs. Without this, a job that fails before the UI's
  // EventSource connects (fast failures: no event_id, instant Space errors)
  // is observed as status=failed with an empty log box and no reason — a
  // silent failure. Terminal jobs never receive further live events, so this
  // replay cannot duplicate anything; running jobs stream live as before.
  // Event shapes are unchanged.
  const terminal = job.end_time && (job.status === 'finished' || job.status === 'failed');
  try {
    if (terminal) {
      for (const line of job.logs.slice(-20)) {
        res.write(`event: log\ndata: ${JSON.stringify({ line })}\n\n`);
      }
      res.write(`event: done\ndata: ${JSON.stringify({ status: job.status, error: job.error || null })}\n\n`);
    }
  } catch (_) {}
  res.on('close', () => { job.sseClients.delete(res); });
  return true;
}

function clearAllJobs() {
  for (const job of jobs.values()) {
    if (job._pythonProc) {
      try { job._pythonProc.kill('SIGKILL'); } catch (_) {}
    }
    if (job._remote) {
      try { if (job._remote.controller) job._remote.controller.abort(); } catch (_) {}
      try { if (job._remote.downloadController) job._remote.downloadController.abort(); } catch (_) {}
      if (job._remote.timeout) { clearTimeout(job._remote.timeout); job._remote.timeout = null; }
    }
    job.sseClients.clear();
  }
  jobs.clear();
}

module.exports = {
  validateTrainingRequest,
  createJob,
  getJob,
  listJobs,
  getJobsByUser,
  getActiveJobCountForUser,
  isOwner,
  startJob,
  stopJob,
  attachSSE,
  clearAllJobs,
  getArtifactMetadata,
  _isValidJobId,
  _safeArtifactDir,
  _broadcast,
  _pushMetric, // exported for tests that push real Trainer-like metrics directly
  _updateProgress,
  _setStatus,
  _log,
  _applyTrainerMetric,
  _applyRemoteTrainEvent,
  _extractZipBuffer,
  _zeroGpuTrainConfig,
  _classifyZeroGpuErrorFrame,
  _postZeroGpuFn,
  estimateTrainingSteps,
  refreshTotalStepsEstimate,
  ZeroGpuTrainError,
  jobs,
  ALLOWED_TASKS,
  MAX_EPOCHS,
  MAX_BATCH,
  MAX_SAMPLES,
  MAX_TIME_SEC,
  MAX_STEPS_LIMIT,
  normalizeTask,
  resolveTrainingProvider,
};

