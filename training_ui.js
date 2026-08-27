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
  lora: 'LoRA lets large models learn by updating a much smaller set of parameters.',
};

// Limits — must match training_backend defaults (env overrides not visible in frontend, use defaults)
const LIMITS = {
  epochs: { min: 1, max: 5 },
  batch_size: { min: 1, max: 32 },
  learning_rate: { min: 1e-6, max: 1e-2 },
  validation_split: { min: 0, max: 50 },
  lora_r: { min: 1, max: 64 },
  lora_alpha: { min: 1, max: 128 },
  lora_dropout: { min: 0, max: 0.5 },
};

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

  // max_steps optional
  if (t.max_steps !== '' && t.max_steps !== null && t.max_steps !== undefined) {
    const ms = Number(t.max_steps);
    if (String(t.max_steps).trim() !== '' && (!_isInt(ms) || ms < 1 || ms > 10000)) errors.max_steps = 'Use an integer between 1 and 10000, or leave empty.';
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
  return {
    task: getTaskDisplayName(t.task_type),
    model: t.model_id,
    dataset: t.dataset_id,
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
    validateModelIdFrontend, validateDatasetIdFrontend, validateTrainingConfigFrontend, getTrainingSummary, getCompatibilityInfo, translateBackendError, getPreviewData, getFieldHelp,
  };
}
if (typeof window !== 'undefined') {
  window.TrainingUI = {
    TASK_TYPES, TASK_DISPLAY, TASK_EXPLANATIONS, TRAINING_METHOD_DISPLAY, TRAINING_METHOD_EXPLANATIONS, EDUCATIONAL_HINTS, LIMITS,
    isLargeModelFrontend, getEffectiveTrainingMethod, getTaskDisplayName, getTaskExplanation, getTrainingMethodDisplay, getTrainingMethodExplanation,
    validateModelIdFrontend, validateDatasetIdFrontend, validateTrainingConfigFrontend, getTrainingSummary, getCompatibilityInfo, translateBackendError, getPreviewData, getFieldHelp,
  };
}
