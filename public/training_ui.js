'use strict';
/**
 * training_ui.js — Frontend helpers for progressive-disclosure Training Cell
 * Shared between dashboard.html (browser) and Node tests.
 * No backend logic duplicated — backend remains authoritative.
 * All validation mirrors training_backend.js but stays friendly.
 */

const TASK_TYPES = ['text-generation','text-classification','image-classification','token-classification'];
const TASK_DISPLAY = {
  'text-generation': 'Text generation',
  'text-classification': 'Text classification',
  'image-classification': 'Image classification',
  'token-classification': 'Token classification',
};
const TASK_EXPLANATIONS = {
  'text-generation': 'Teach a model to generate or continue text.',
  'text-classification': 'Teach a model to assign categories or labels to text.',
  'image-classification': 'Teach a model to recognize categories in images.',
  'token-classification': 'Teach a model to label individual words or tokens.',
};
const TRAINING_METHOD_DISPLAY = {
  'auto': 'Recommended',
  'full': 'Full fine-tuning',
  'lora': 'LoRA',
};
const TRAINING_METHOD_EXPLANATIONS = {
  'auto': 'Claro.AI automatically chooses the best training method for the model.',
  'full': 'Updates the model\'s existing parameters. Best for smaller models.',
  'lora': 'Trains a small set of additional parameters. Usually much more efficient for large models.',
};
const EDUCATIONAL_HINTS = {
  epochs: 'How many times the model sees the training dataset.',
  batch_size: 'How many examples the model processes at once. Larger batches usually need more memory.',
  learning_rate: 'How strongly the model changes its parameters during training.',
  validation_split: 'The percentage of examples kept aside to check whether the model generalizes to data it wasn\'t trained on.',
  max_steps: 'Stop after this many training steps. Leave blank for Auto (based on epochs and dataset size).',
  lora: 'LoRA lets large models learn by updating a much smaller set of parameters.',
};

// Limits — must match training_backend defaults (env overrides not visible in frontend, use defaults)
const LIMITS = {
  epochs: { min: 1, max: 5 },
  batch_size: { min: 1, max: 32 },
  learning_rate: { min: 1e-6, max: 1e-2 },
  validation_split: { min: 0, max: 50 },
  max_steps: { min: 1, max: 10000 },
  lora_r: { min: 1, max: 64 },
  lora_alpha: { min: 1, max: 128 },
  lora_dropout: { min: 0, max: 0.5 },
};


// ── Curated Model + Dataset catalog ─────────────────────────────────────
// Stable preset IDs are the trust anchor: the backend resolves preset IDs
// server-side and ignores any frontend-supplied model/dataset IDs or limits.
// The UI keeps raw model_id/dataset_id in sync for display/validation, but
// /api/train/start must never trust them when a preset is present.
const CUSTOM_PRESET = 'custom';

const MODEL_PRESETS = {
  'distilbert-base': {
    id: 'distilbert-base',
    name: 'DistilBERT (base)',
    description: 'Compact 66M-parameter text model. Fast to fine-tune; great for sentiment and topic classification.',
    taskTypes: ['text-classification'],
    modelId: 'distilbert-base-uncased',
    datasets: ['imdb', 'ag-news'],
    trainingMethods: ['full', 'lora'],
    maxEpochs: 3,
    maxBatchSize: 16,
    maxSamples: 5000,
    maxSteps: 2000,
    resourceClass: 'small',
  },
  'mobilenet-v2': {
    id: 'mobilenet-v2',
    name: 'MobileNetV2',
    description: 'Lightweight vision model built for efficiency. Good default for small image classification.',
    taskTypes: ['image-classification'],
    modelId: 'google/mobilenet_v2_1.0_224',
    datasets: ['beans'],
    trainingMethods: ['full'],
    maxEpochs: 5,
    maxBatchSize: 16,
    maxSamples: 2000,
    maxSteps: 2000,
    resourceClass: 'small',
  },
  'resnet-18': {
    id: 'resnet-18',
    name: 'ResNet-18',
    description: 'Classic 18-layer residual network. Slightly heavier than MobileNetV2, strong on small image datasets.',
    taskTypes: ['image-classification'],
    modelId: 'microsoft/resnet-18',
    datasets: ['beans'],
    trainingMethods: ['full'],
    maxEpochs: 5,
    maxBatchSize: 16,
    maxSamples: 2000,
    maxSteps: 2000,
    resourceClass: 'small',
  },
};

const DATASET_PRESETS = {
  'imdb': {
    id: 'imdb',
    name: 'IMDb Movie Reviews',
    description: '25,000 movie reviews labeled positive/negative. Binary sentiment classification.',
    taskTypes: ['text-classification'],
    datasetId: 'stanfordnlp/imdb',
  },
  'ag-news': {
    id: 'ag-news',
    name: 'AG News',
    description: '120,000+ news headlines in 4 topic classes: World, Sports, Business, Sci/Tech.',
    taskTypes: ['text-classification'],
    datasetId: 'fancyzhx/ag_news',
  },
  'beans': {
    id: 'beans',
    name: 'Beans (leaf images)',
    description: '1,296 bean leaf photos in 3 classes: healthy, angular leaf spot, bean rust.',
    taskTypes: ['image-classification'],
    // Canonical repository. Resolve directly — no speculative owner fallbacks.
    datasetId: 'AI-Lab-Makerere/beans',
  },
};

// Known raw IDs migrate to presets (lower-cased, trimmed before lookup).
const MODEL_ID_ALIASES = {
  'distilbert-base-uncased': 'distilbert-base',
  'distilbert/distilbert-base-uncased': 'distilbert-base',
  'google/mobilenet_v2_1.0_224': 'mobilenet-v2',
  'mobilenet_v2_1.0_224': 'mobilenet-v2',
  'microsoft/resnet-18': 'resnet-18',
  'resnet-18': 'resnet-18',
};
const DATASET_ID_ALIASES = {
  'stanfordnlp/imdb': 'imdb',
  'imdb': 'imdb',
  'fancyzhx/ag_news': 'ag-news',
  'ag_news': 'ag-news',
  'ag-news': 'ag-news',
  'beans': 'beans',
};

function normPresetId(v) {
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  return s === '' ? null : s;
}
function getModelPreset(id) {
  const k = normPresetId(id);
  return (k && k !== CUSTOM_PRESET && MODEL_PRESETS[k]) ? MODEL_PRESETS[k] : null;
}
function getDatasetPreset(id) {
  const k = normPresetId(id);
  return (k && k !== CUSTOM_PRESET && DATASET_PRESETS[k]) ? DATASET_PRESETS[k] : null;
}
function modelsForTaskType(taskType) {
  const t = String(taskType || '').trim().toLowerCase();
  return Object.values(MODEL_PRESETS).filter((m) => m.taskTypes.includes(t));
}
function datasetsForModelPreset(modelPresetId) {
  const mp = getModelPreset(modelPresetId);
  if (!mp) return Object.values(DATASET_PRESETS);
  return mp.datasets.map((d) => DATASET_PRESETS[d]).filter(Boolean);
}
function inferModelPresetId(modelId) {
  const k = String(modelId || '').trim().toLowerCase();
  return MODEL_ID_ALIASES[k] || CUSTOM_PRESET;
}
function inferDatasetPresetId(datasetId) {
  const k = String(datasetId || '').trim().toLowerCase();
  return DATASET_ID_ALIASES[k] || CUSTOM_PRESET;
}
// Pure compatibility check shared by frontend validation and the backend.
// Either side may be 'custom'/missing (legacy raw mode) — only real preset
// IDs are verified. Returns { ok, error }.
function checkPresetCompatibility({ modelPreset, datasetPreset, taskType } = {}) {
  const mpId = normPresetId(modelPreset);
  const dpId = normPresetId(datasetPreset);
  const task = String(taskType || '').trim().toLowerCase();
  let mp = null;
  let dp = null;
  if (mpId && mpId !== CUSTOM_PRESET) {
    mp = getModelPreset(mpId);
    if (!mp) return { ok: false, error: `Unknown model preset: ${mpId}` };
  }
  if (dpId && dpId !== CUSTOM_PRESET) {
    dp = getDatasetPreset(dpId);
    if (!dp) return { ok: false, error: `Unknown dataset preset: ${dpId}` };
  }
  if (mp && task && !mp.taskTypes.includes(task)) {
    return { ok: false, error: `Model preset '${mp.id}' (${mp.name}) does not support task '${task}'` };
  }
  if (dp && task && !dp.taskTypes.includes(task)) {
    return { ok: false, error: `Dataset preset '${dp.id}' (${dp.name}) does not support task '${task}'` };
  }
  if (mp && dp && !mp.datasets.includes(dp.id)) {
    return { ok: false, error: `Dataset preset '${dp.id}' (${dp.name}) is not compatible with model preset '${mp.id}' (${mp.name})` };
  }
  return { ok: true, error: null };
}
// Migrate legacy notebook state (raw model_id/dataset_id) to preset IDs.
// Never destroys information: unknown values become 'custom' with raw IDs
// kept; an incompatible migrated pair is repaired toward the model preset's
// first dataset so the cell stays in a valid, startable state.
function migrateLegacyTrainingToPresets(t) {
  if (!t || typeof t !== 'object') return t;
  if (!normPresetId(t.model_preset)) {
    t.model_preset = inferModelPresetId(t.model_id);
  }
  if (!normPresetId(t.dataset_preset)) {
    t.dataset_preset = inferDatasetPresetId(t.dataset_id);
  }
  let mp = getModelPreset(t.model_preset);
  let dp = getDatasetPreset(t.dataset_preset);
  const task = String(t.task_type || '').trim().toLowerCase();
  // Task wins over presets: a preset that cannot do the task becomes custom.
  if (mp && task && !mp.taskTypes.includes(task)) {
    t.model_preset = CUSTOM_PRESET;
    mp = null;
  }
  if (dp && task && !dp.taskTypes.includes(task)) {
    t.dataset_preset = CUSTOM_PRESET;
    dp = null;
  }
  // Sync canonical underlying IDs for real presets.
  if (mp && String(t.model_id || '').trim().toLowerCase() !== mp.modelId.toLowerCase()) {
    t.model_id = mp.modelId;
  }
  if (mp && dp && !mp.datasets.includes(dp.id)) {
    const first = getDatasetPreset(mp.datasets[0]);
    if (first) {
      t.dataset_preset = first.id;
      t.dataset_id = first.datasetId;
      dp = first;
    }
  } else if (dp && String(t.dataset_id || '').trim().toLowerCase() !== dp.datasetId.toLowerCase()) {
    t.dataset_id = dp.datasetId;
  }
  return t;
}

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

function getEffectiveTrainingMethod(training) {
  if (!training) return 'full';
  const m = String(training.training_method || 'auto').toLowerCase();
  if (m === 'auto') return isLargeModelFrontend(training.model_id) ? 'lora' : 'full';
  if (m === 'lora' || m === 'full') return m;
  return 'full';
}

function getTaskDisplayName(task) {
  return TASK_DISPLAY[task] || task;
}
function getTaskExplanation(task) {
  return TASK_EXPLANATIONS[task] || '';
}
function getTrainingMethodDisplay(method) {
  return TRAINING_METHOD_DISPLAY[method] || method;
}
function getTrainingMethodExplanation(method) {
  return TRAINING_METHOD_EXPLANATIONS[method] || '';
}

// --- Friendly frontend format validation (mirrors backend regex but with helpful messages) ---
function validateModelIdFrontend(modelId) {
  const raw = String(modelId || '').trim();
  if (!raw) return 'Model is required. Use something like distilbert-base-uncased or google-bert/bert-base-uncased';
  // backend allows single segment OR owner/name with alphanum . _ -
  const pattern = /^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)?$/;
  if (!pattern.test(raw)) {
    return "Model name doesn't look valid. Use something like:\ndistilbert-base-uncased\nor\ngoogle-bert/bert-base-uncased";
  }
  return null;
}
function validateDatasetIdFrontend(datasetId) {
  const raw = String(datasetId || '').trim();
  if (!raw) return 'Dataset is required. Use something like stanfordnlp/imdb';
  const pattern = /^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)?$/;
  if (!pattern.test(raw)) {
    return "Dataset name doesn't look valid. Use something like:\nstanfordnlp/imdb\nor\ncifar10";
  }
  return null;
}

function _isInt(v) { return Number.isFinite(v) && Number.isInteger(v); }

function validateTrainingConfigFrontend(t) {
  const errors = {};
  // model / dataset
  const mErr = validateModelIdFrontend(t.model_id);
  if (mErr) errors.model_id = mErr;
  const dErr = validateDatasetIdFrontend(t.dataset_id);
  if (dErr) errors.dataset_id = dErr;
  if (!TASK_TYPES.includes(t.task_type)) errors.task_type = `Choose a task: ${TASK_TYPES.join(', ')}`;

  // epochs
  const epochs = Number(t.epochs);
  if (t.epochs === '' || t.epochs === null || t.epochs === undefined) errors.epochs = 'Epochs is required.';
  else if (!_isInt(epochs) || epochs < LIMITS.epochs.min || epochs > LIMITS.epochs.max) errors.epochs = `Use an integer between ${LIMITS.epochs.min} and ${LIMITS.epochs.max}.`;

  // batch_size
  const bs = Number(t.batch_size);
  if (t.batch_size === '' || t.batch_size === null || t.batch_size === undefined) errors.batch_size = 'Batch size is required.';
  else if (!_isInt(bs) || bs < LIMITS.batch_size.min || bs > LIMITS.batch_size.max) errors.batch_size = `Use an integer between ${LIMITS.batch_size.min} and ${LIMITS.batch_size.max}.`;

  // learning_rate
  const lr = Number(t.learning_rate);
  if (t.learning_rate === '' || t.learning_rate === null || t.learning_rate === undefined || !Number.isFinite(lr)) errors.learning_rate = 'Learning rate is required.';
  else if (lr < LIMITS.learning_rate.min || lr > LIMITS.learning_rate.max) errors.learning_rate = `Use a value between ${LIMITS.learning_rate.min} and ${LIMITS.learning_rate.max}.`;

  // validation_split
  const vs = Number(t.validation_split);
  if (t.validation_split === '' || t.validation_split === null || t.validation_split === undefined || !Number.isFinite(vs)) errors.validation_split = 'Validation split is required.';
  else if (vs < LIMITS.validation_split.min || vs > LIMITS.validation_split.max) errors.validation_split = `Use a value between ${LIMITS.validation_split.min} and ${LIMITS.validation_split.max}.`;

  // training_method
  const method = String(t.training_method || 'auto').toLowerCase();
  if (!['auto','full','lora'].includes(method)) errors.training_method = 'Choose Recommended, Full fine-tuning, or LoRA.';

  // LoRA params only when effective method is lora
  const eff = getEffectiveTrainingMethod(t);
  if (eff === 'lora') {
    const r = Number(t.lora_r);
    if (t.lora_r === '' || t.lora_r === null || t.lora_r === undefined || !_isInt(r) || r < LIMITS.lora_r.min || r > LIMITS.lora_r.max) errors.lora_r = `Use an integer between ${LIMITS.lora_r.min} and ${LIMITS.lora_r.max}.`;
    const a = Number(t.lora_alpha);
    if (t.lora_alpha === '' || t.lora_alpha === null || t.lora_alpha === undefined || !_isInt(a) || a < LIMITS.lora_alpha.min || a > LIMITS.lora_alpha.max) errors.lora_alpha = `Use an integer between ${LIMITS.lora_alpha.min} and ${LIMITS.lora_alpha.max}.`;
    const d = Number(t.lora_dropout);
    if (t.lora_dropout === '' || t.lora_dropout === null || t.lora_dropout === undefined || !Number.isFinite(d) || d < LIMITS.lora_dropout.min || d > LIMITS.lora_dropout.max) errors.lora_dropout = `Use a value between ${LIMITS.lora_dropout.min} and ${LIMITS.lora_dropout.max}.`;
    const tm = String(t.target_modules || 'auto').trim();
    if (tm.toLowerCase() !== 'auto' && tm !== '') {
      const mods = tm.split(',').map(s=>s.trim()).filter(Boolean);
      if (mods.length===0) errors.target_modules = 'Use "auto" or comma-separated module names.';
      else for (const mod of mods) {
        if (!/^[A-Za-z0-9_\.]+$/.test(mod)) { errors.target_modules = `Invalid target_modules entry: ${mod}`; break; }
      }
    }
  }

  // preset compatibility (the selectors normally prevent invalid combos;
  // this is defense-in-depth — the backend re-checks authoritatively)
  if (normPresetId(t.model_preset) || normPresetId(t.dataset_preset)) {
    const chk = checkPresetCompatibility({ modelPreset: t.model_preset, datasetPreset: t.dataset_preset, taskType: t.task_type });
    if (!chk.ok) errors.dataset_id = chk.error;
  }

  // max_steps optional
  if (t.max_steps !== '' && t.max_steps !== null && t.max_steps !== undefined) {
    const ms = Number(t.max_steps);
    if (String(t.max_steps).trim() !== '' && (!_isInt(ms) || ms < LIMITS.max_steps.min || ms > LIMITS.max_steps.max)) errors.max_steps = `Use an integer between ${LIMITS.max_steps.min} and ${LIMITS.max_steps.max}, or leave empty.`;
  }

  const valid = Object.keys(errors).length === 0;
  return { valid, errors, effectiveMethod: eff };
}

function getTrainingSummary(t) {
  const ep = t.epochs ?? 2;
  const bs = t.batch_size ?? 8;
  const vs = t.validation_split ?? 10;
  return `${ep} epochs \u00b7 batch ${bs} \u00b7 ${vs}% validation`;
}

function getCompatibilityInfo(t) {
  const task = String(t.task_type||'').toLowerCase();
  const dataset = String(t.dataset_id||'').toLowerCase();
  const model = String(t.model_id||'').toLowerCase();

  // heuristic indicators
  const textDatasetIndicators = ['imdb','glue','squad','wiki','sentiment','news','tweets','reviews','ag_news','yelp','amazon','stanfordnlp','tweet','text','civil_comments'];
  const imageDatasetIndicators = ['cifar','imagenet','mnist','coco','fashion','flowers','food101','celeba','voc','cityscapes','imagenette','stanford-cars','beans','cats_vs_dogs'];
  const textModelIndicators = ['bert','roberta','distilbert','albert','deberta','electra','gpt2','gpt-','llama','mistral','gemma','qwen','phi','falcon','bloom','t5','bart'];
  const visionModelIndicators = ['vit','resnet','efficientnet','beit','deit','swin','convnext','regnet','poolformer','convnext','eva','sam'];

  const hasTextDataset = textDatasetIndicators.some(k => dataset.includes(k));
  const hasImageDataset = imageDatasetIndicators.some(k => dataset.includes(k));
  const hasTextModel = textModelIndicators.some(k => model.includes(k));
  const hasVisionModel = visionModelIndicators.some(k => model.includes(k));

  const isTextTask = ['text-generation','text-classification','token-classification'].includes(task);
  const isImageTask = task === 'image-classification';

  // Known incompatible
  if (isImageTask && hasTextDataset) {
    return { status: 'error', code: 'dataset_task_mismatch', message: '\u274C These settings don\'t work together. Image classification needs an image dataset.', fix: 'Choose an image dataset like cifar10 or switch task to Text classification.' , fixAction: 'switch_to_text'};
  }
  if (isTextTask && hasImageDataset) {
    return { status: 'error', code: 'dataset_task_mismatch', message: '\u274C These settings don\'t work together. Text tasks need a text dataset.', fix: 'Choose a text dataset like stanfordnlp/imdb or switch task to Image classification.', fixAction: 'switch_to_image'};
  }
  if (isImageTask && hasTextModel && !hasVisionModel) {
    return { status: 'error', code: 'model_task_mismatch', message: '\u274C This model looks like a text model, but the task is Image classification.', fix: 'Try a vision model like google/vit-base-patch16-224 or switch task.', fixAction: 'suggest_vision_model'};
  }
  if (isTextTask && hasVisionModel && !hasTextModel) {
    return { status: 'error', code: 'model_task_mismatch', message: '\u274C This model looks like a vision model, but the task is a text task.', fix: 'Try a text model like distilbert-base-uncased or switch task to Image classification.', fixAction: 'suggest_text_model'};
  }

  // If we have no strong signal, say compatibility will be checked
  if (!hasTextDataset && !hasImageDataset && !hasTextModel && !hasVisionModel) {
    return { status: 'unknown', message: 'Compatibility will be checked when training starts.' };
  }
  return { status: 'ok', message: '\u2713 Configuration looks good' };
}

function translateBackendError(errMsg, code) {
  const low = String(errMsg||'').toLowerCase();
  if (code === 'invalid_model_id' || low.includes('invalid model')) return "Claro.AI couldn't load this model. Check the model name or try another model.";
  if (code === 'invalid_dataset_id' || low.includes('invalid dataset')) return "Claro.AI couldn't load this dataset. Check the dataset name or try another dataset.";
  if (low.includes('missing dependency') || low.includes('missing prompt')) return "Training environment is missing a required Python package.";
  if (low.includes('stopped_by_user') || low.includes('stopped by user')) return "Training was stopped.";
  if (low.includes('training_error')) return "Training failed. Open Training Logs for technical details.";
  if (low.includes('bad_request')) return "Something in the configuration looks off. Check the highlighted fields.";
  return errMsg || "Training failed. Open Training Logs for technical details.";
}

// Pure step-count math shared by the UI preview and tests.
// Returns a positive integer, or null when the inputs are insufficient
// (the caller must then display "Estimate unavailable" — never invent one).
// trainRows: effective training samples AFTER the max_samples cap and the
// validation split have been applied.
function estimateStepsFromSamples({ trainRows, batchSize, epochs, maxSteps } = {}) {
  const ms = (maxSteps === '' || maxSteps == null) ? null : Number(maxSteps);
  if (ms !== null && Number.isFinite(ms) && ms > 0) return Math.floor(ms);
  const tr = Number(trainRows);
  const bs = Number(batchSize);
  const ep = Number(epochs);
  if (!Number.isFinite(tr) || tr <= 0) return null;
  if (!Number.isFinite(bs) || bs < 1) return null;
  if (!Number.isFinite(ep) || ep < 1) return null;
  return Math.max(1, Math.ceil(tr / Math.floor(bs)) * Math.floor(ep));
}

// Human-readable classification labels (display only — never mutates model output).
// Priority: 1) meaningful config id2label values (normalized casing),
// 2) generic LABEL_X mapped ONLY when model/task metadata confirms a known
// classifier (currently: binary IMDb sentiment). Generic labels from any other
// model pass through unchanged. Returns { display, raw }; raw is always the
// untouched model label for debugging/compatibility.
function resolveDisplayLabel({ label, labelId, taskType, modelId, datasetId, numLabels } = {}) {
  const raw = label == null ? '' : String(label);
  const generic = /^LABEL_\d+$/i.test(raw.trim());
  if (generic) {
    const id = Number(labelId);
    const imdb = /imdb/i.test(String(datasetId || '')) || /imdb/i.test(String(modelId || ''));
    const binary = numLabels == null || Number(numLabels) === 2;
    if (String(taskType).toLowerCase() === 'text-classification' && imdb && binary && (id === 0 || id === 1)) {
      return { display: id === 0 ? 'Negative' : 'Positive', raw };
    }
    return { display: raw, raw };
  }
  // Meaningful label: clean up uniform casing only (NEGATIVE -> Negative);
  // mixed-case labels (Sci/Tech) are already human-readable — leave them.
  if (/^[A-Z][A-Z0-9_]*$/.test(raw) || /^[a-z][a-z0-9_]*$/.test(raw)) {
    const lowered = raw.toLowerCase();
    return { display: lowered.charAt(0).toUpperCase() + lowered.slice(1), raw };
  }
  return { display: raw, raw };
}

function getPreviewData(t) {
  const eff = getEffectiveTrainingMethod(t);
  const effDisplay = eff === 'lora' ? 'LoRA' : eff === 'full' ? 'Full fine-tuning' : 'Recommended';
  // Estimates — do not fake. If max_steps set, use that; else say Estimate unavailable for steps/time.
  let estimatedSteps;
  let estimatedTime = 'Estimate unavailable';
  let resourceUsage = 'Estimate unavailable';
  if (t.max_steps !== '' && t.max_steps != null && String(t.max_steps).trim() !== '' && Number.isFinite(Number(t.max_steps))) {
    estimatedSteps = String(Number(t.max_steps));
    estimatedTime = 'Estimate unavailable';
    resourceUsage = eff === 'lora' ? 'LoRA — efficient (updates <20% params)' : 'Full fine-tuning — updates all parameters';
  } else {
    estimatedSteps = 'Estimate unavailable';
    // we could note fallback is epochs*100 but clarify it's not exact
    // Use backend's fallback note: show as "Depends on dataset size (backend will use epochs \u00d7 ~100 as progress total)"
    resourceUsage = eff === 'lora' ? 'LoRA — efficient (updates <20% params)' : 'Full fine-tuning';
  }
  const maxStepsSet = t.max_steps !== '' && t.max_steps != null && String(t.max_steps).trim() !== '' && Number.isFinite(Number(t.max_steps));
  const pvModelPreset = getModelPreset(t.model_preset);
  const pvDatasetPreset = getDatasetPreset(t.dataset_preset);
  const pvModelName = pvModelPreset ? pvModelPreset.name : t.model_id;
  const pvDatasetName = pvDatasetPreset ? pvDatasetPreset.name : t.dataset_id;
  const pvModelId = pvModelPreset ? pvModelPreset.modelId : String(t.model_id || '');
  const pvDatasetId = pvDatasetPreset ? pvDatasetPreset.datasetId : String(t.dataset_id || '');
  const pvAdvanced = `Model ID ${pvModelId || '(none)'} · Dataset ID ${pvDatasetId || '(none)'} · presets ${normPresetId(t.model_preset) || 'custom'} / ${normPresetId(t.dataset_preset) || 'custom'}`;
  return {
    task: getTaskDisplayName(t.task_type),
    model: pvModelName,
    dataset: pvDatasetName,
    advancedDetails: pvAdvanced,
    maxSteps: maxStepsSet ? String(Number(t.max_steps)) : 'Auto',
    method: eff === 'lora' ? 'LoRA' : eff === 'full' ? 'Full fine-tuning' : TRAINING_METHOD_DISPLAY[t.training_method] || effDisplay,
    effectiveMethod: eff,
    epochs: t.epochs,
    batch_size: t.batch_size,
    validation: `${t.validation_split}%`,
    estimatedSteps,
    estimatedTime,
    resourceUsage,
  };
}

// Friendly inline message helpers
function getFieldHelp(key) {
  return EDUCATIONAL_HINTS[key] || '';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    TASK_TYPES, TASK_DISPLAY, TASK_EXPLANATIONS, TRAINING_METHOD_DISPLAY, TRAINING_METHOD_EXPLANATIONS, EDUCATIONAL_HINTS, LIMITS,
    isLargeModelFrontend, getEffectiveTrainingMethod, getTaskDisplayName, getTaskExplanation, getTrainingMethodDisplay, getTrainingMethodExplanation,
    validateModelIdFrontend, validateDatasetIdFrontend, validateTrainingConfigFrontend, getTrainingSummary, getCompatibilityInfo, translateBackendError, getPreviewData, getFieldHelp, estimateStepsFromSamples, resolveDisplayLabel,
    CUSTOM_PRESET, MODEL_PRESETS, DATASET_PRESETS, normPresetId, getModelPreset, getDatasetPreset, modelsForTaskType, datasetsForModelPreset, inferModelPresetId, inferDatasetPresetId, checkPresetCompatibility, migrateLegacyTrainingToPresets,
  };
}
if (typeof window !== 'undefined') {
  window.TrainingUI = {
    TASK_TYPES, TASK_DISPLAY, TASK_EXPLANATIONS, TRAINING_METHOD_DISPLAY, TRAINING_METHOD_EXPLANATIONS, EDUCATIONAL_HINTS, LIMITS,
    isLargeModelFrontend, getEffectiveTrainingMethod, getTaskDisplayName, getTaskExplanation, getTrainingMethodDisplay, getTrainingMethodExplanation,
    validateModelIdFrontend, validateDatasetIdFrontend, validateTrainingConfigFrontend, getTrainingSummary, getCompatibilityInfo, translateBackendError, getPreviewData, getFieldHelp, estimateStepsFromSamples, resolveDisplayLabel,
    CUSTOM_PRESET, MODEL_PRESETS, DATASET_PRESETS, normPresetId, getModelPreset, getDatasetPreset, modelsForTaskType, datasetsForModelPreset, inferModelPresetId, inferDatasetPresetId, checkPresetCompatibility, migrateLegacyTrainingToPresets,
  };
}
