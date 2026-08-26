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
const { spawn } = require('child_process');

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
  const model_id = String(body.model_id || body.modelId || '').trim();
  const dataset_id = String(body.dataset_id || body.datasetId || '').trim();
  const task_type = normalizeTask(body.task_type || body.task || body.taskType);

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

  if (!Number.isFinite(epochs) || !Number.isInteger(epochs) || epochs < 1 || epochs > MAX_EPOCHS) {
    throw Object.assign(new Error(`epochs must be integer 1..${MAX_EPOCHS}`), { code: 'bad_request', status: 400 });
  }
  if (!Number.isFinite(batch_size) || !Number.isInteger(batch_size) || batch_size < 1 || batch_size > MAX_BATCH) {
    throw Object.assign(new Error(`batch_size must be integer 1..${MAX_BATCH}`), { code: 'bad_request', status: 400 });
  }
  if (!Number.isFinite(learning_rate) || learning_rate < 1e-6 || learning_rate > 1e-2) {
    throw Object.assign(new Error('learning_rate must be between 1e-6 and 1e-2'), { code: 'bad_request', status: 400 });
  }
  if (max_steps !== null && max_steps !== undefined && String(max_steps).trim() !== '') {
    max_steps = Number(max_steps);
    if (!Number.isFinite(max_steps) || !Number.isInteger(max_steps) || max_steps < 1 || max_steps > MAX_STEPS_LIMIT) {
      throw Object.assign(new Error(`max_steps must be integer 1..${MAX_STEPS_LIMIT} or empty`), { code: 'bad_request', status: 400 });
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

  return {
    model_id,
    dataset_id,
    task_type,
    epochs,
    batch_size,
    learning_rate,
    max_steps,
    validation_split,
    provider: (body.provider || process.env.TRAINING_PROVIDER || 'local').toLowerCase(),
    max_samples: MAX_SAMPLES,
    max_time_sec: MAX_TIME_SEC,
    training_method,
    lora_r,
    lora_alpha,
    lora_dropout,
    target_modules,
  };
}

// ── Job lifecycle ──
function createJob(config) {
  const job_id = _genId();
  const job = {
    job_id,
    status: 'queued',
    config,
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

function resolveTrainingProvider(env) {
  const p = String(env.TRAINING_PROVIDER || env.GPU_PROVIDER || 'local').toLowerCase().trim();
  if (['local', 'zerogpu', 'modal'].includes(p)) return p;
  return 'local';
}

// ── Real Trainer via Python ──
function _startPythonTraining(job) {
  const cfg = job.config;
  const pythonBin = process.env.PYTHON_BIN || 'python3';
  const runnerPath = path.join(__dirname, 'training_runner.py');

  if (!fs.existsSync(runnerPath)) {
    const msg = `training_runner.py not found at ${runnerPath}`;
    _log(job, `[TRAIN] training_error: ${msg}`);
    _setStatus(job, 'failed');
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
    proc = spawn(pythonBin, args, { cwd: __dirname, env: process.env });
  } catch (e) {
    const msg = `Failed to spawn Python (${pythonBin}): ${e.message}`;
    _log(job, `[TRAIN] training_error: ${msg}`);
    _setStatus(job, 'failed');
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
          if (typeof m.step === 'number' && (typeof m.train_loss === 'number' || typeof m.eval_loss === 'number')) {
            const elapsed = Math.round((_now() - job.start_time) / 1000);
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
            _updateProgress(job, {
              current_step: metric.step,
              current_epoch: metric.epoch,
              train_loss: metric.train_loss !== undefined ? metric.train_loss : job.progress.train_loss,
              eval_loss: metric.eval_loss !== undefined ? metric.eval_loss : job.progress.eval_loss,
              learning_rate: metric.learning_rate,
              elapsed_time: elapsed,
              eta: Math.max(0, Math.round((total - metric.step) * 1.2)),
              percent: Math.min(100, Math.round((metric.step / total) * 100)),
              gpu_status: 'training',
            });
            _pushMetric(job, metric);
            // status: if eval_loss present, it's evaluation step
            if (m.eval_loss !== undefined) {
              _setStatus(job, 'evaluating');
              setTimeout(() => { if (job.status === 'evaluating' && !job._aborted) _setStatus(job, 'training'); }, 600);
            } else {
              if (job.status !== 'training') _setStatus(job, 'training');
            }
            continue;
          }
        } catch (_) {
          // not a metric JSON, treat as log
        }
      }

      // status markers from runner
      if (line.startsWith('[TRAIN]')) {
        _log(job, line);
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
      // already marked failed via training_error in stdout — ensure artifacts saved
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
        job.error = 'training_error';
        job.end_time = _now();
        _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
      }
      if (!job.artifacts_dir) _saveArtifacts(job);
    } else {
      // edge: code 0 but status already failed, ensure artifacts
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
      job.error = 'training_error';
      job.end_time = _now();
      _broadcast(job, 'done', { status: 'failed', error: job.error, message: msg });
      _saveArtifacts(job);
    }
  });
}

function startJob(job) {
  const provider = resolveTrainingProvider(process.env);
  job.config.provider = provider;
  // gpu_status reflects requested provider but real execution is via Python Trainer
  job.progress.gpu_status = provider === 'zerogpu' ? 'zerogpu' : provider === 'modal' ? 'modal' : 'local';
  _log(job, `[TRAIN] provider=${provider} starting real training (Trainer)`);
  _startPythonTraining(job);
  return job;
}

function stopJob(job_id) {
  const job = jobs.get(job_id);
  if (!job) return null;
  if (['finished', 'failed'].includes(job.status)) return job;
  job._aborted = true;

  // Terminate Python process and its children if possible
  if (job._pythonProc) {
    const proc = job._pythonProc;
    job._pythonProc = null;
    try {
      // try graceful SIGTERM first, then SIGKILL after 3s
      proc.kill('SIGTERM');
      _log(job, `[TRAIN] SIGTERM sent to ${proc.pid}`);
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
  }

  _setStatus(job, 'failed');
  job.error = 'stopped_by_user';
  job.end_time = _now();
  _updateProgress(job, { gpu_status: 'idle', eta: 0 });
  _broadcast(job, 'done', { status: 'failed', error: job.error, message: 'stopped by user' });
  _log(job, `[TRAIN] Stopped by user`);
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
    fs.writeFileSync(path.join(dir, 'job.json'), JSON.stringify({ job_id: job.job_id, status: job.status, config: job.config, progress: job.progress, error: job.error }, null, 2));
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
  res.on('close', () => { job.sseClients.delete(res); });
  return true;
}

function clearAllJobs() {
  for (const job of jobs.values()) {
    if (job._pythonProc) {
      try { job._pythonProc.kill('SIGKILL'); } catch (_) {}
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

