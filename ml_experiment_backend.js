'use strict';

/**
 * Classical-ML experiment backend for Claro.AI (local tabular workflows).
 *
 * Separate from training_backend.js (Hugging Face fine-tuning) on purpose:
 * different engine (local sklearn via ml/experiment_runner.py), different
 * store (ml_experiments/), different lifecycle. Single-user, no auth —
 * same architecture as the rest of the app.
 *
 * CPU only. Structured configs only — never arbitrary code. All child
 * processes run the repo's own ml/* scripts with validated JSON args.
 */

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const childProcess = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(childProcess.execFile);

// ── Resource limits (env overrides) ──
const ML_MAX_DATASET_BYTES = parseInt(process.env.ML_MAX_DATASET_BYTES || String(10 * 1024 * 1024), 10);
const ML_MAX_DATASET_ROWS = parseInt(process.env.ML_MAX_DATASET_ROWS || '50000', 10);
const ML_MAX_FEATURES = parseInt(process.env.ML_MAX_FEATURES || '200', 10);
const ML_MAX_TRAIN_TIME_SEC = parseInt(process.env.ML_MAX_TRAIN_TIME_SEC || '600', 10);
const ML_MAX_CV_FOLDS = parseInt(process.env.ML_MAX_CV_FOLDS || '10', 10);
const ML_MAX_SEARCH_TRIALS = parseInt(process.env.ML_MAX_SEARCH_TRIALS || '30', 10);
const ML_MAX_PREDICT_ROWS = parseInt(process.env.ML_MAX_PREDICT_ROWS || '5000', 10);
const ML_MAX_ARTIFACT_BYTES = parseInt(process.env.ML_MAX_ARTIFACT_BYTES || String(200 * 1024 * 1024), 10);

const CLASSIFICATION_MODELS = ['LogisticRegression', 'DecisionTreeClassifier',
  'RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier',
  'SVC', 'KNeighborsClassifier', 'GaussianNB'];
const REGRESSION_MODELS = ['LinearRegression', 'Ridge', 'Lasso',
  'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
  'HistGradientBoostingRegressor', 'SVR', 'KNeighborsRegressor'];
const TASKS = ['classification', 'regression'];
const SPLIT_STRATEGIES = ['train_test', 'train_val_test', 'official', 'time'];
const CV_METHODS = ['auto', 'kfold', 'stratified', 'group', 'time', 'none'];
const SEARCH_METHODS = ['none', 'grid', 'random'];

// ── Job / experiment store (in-memory; durable copy in experiment.json) ──
const experiments = new Map(); // exp_id -> record

function _now() { return Date.now(); }
function _genWsId() { return 'mlw_' + crypto.randomBytes(6).toString('hex'); }
function _genExpId() { return 'exp_' + crypto.randomBytes(6).toString('hex'); }

function _err(message, code, status) {
  return Object.assign(new Error(message), { code, status: status || 400 });
}

function _isValidId(v, prefix) {
  return typeof v === 'string' && new RegExp(`^${prefix}_[a-f0-9]{12}$`).test(v);
}

// Derived server-side from ids only — never from client paths.
function _safeMlDir(kind, id) {
  const base = kind === 'ws' ? 'ml_workspaces' : 'ml_experiments';
  const prefix = kind === 'ws' ? 'mlw' : 'exp';
  if (!_isValidId(id, prefix)) throw _err(`Invalid id: ${id}`, 'bad_request', 400);
  const dir = path.resolve(path.join(__dirname, base, id));
  const root = path.resolve(path.join(__dirname, base));
  if (dir !== root && !dir.startsWith(root + path.sep)) {
    throw _err('Invalid artifact path', 'bad_request', 400);
  }
  return dir;
}

function _pythonBin() { return process.env.PYTHON_BIN || 'python3'; }

async function _runPython(args, { timeoutMs = 60000, maxBuffer = 8 * 1024 * 1024 } = {}) {
  const { stdout, stderr } = await execFileAsync(_pythonBin(), args,
    { cwd: __dirname, timeout: timeoutMs, maxBuffer });
  return { stdout: String(stdout || ''), stderr: String(stderr || '') };
}

function _lastJsonLine(stdout) {
  const lines = String(stdout || '').trim().split('\n');
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.startsWith('{') && line.endsWith('}')) {
      try { return JSON.parse(line); } catch (_) { /* keep scanning */ }
    }
  }
  return null;
}

// ── Validation (structural; Python re-validates semantics) ──
function _nonEmptyString(v, name) {
  const s = String(v == null ? '' : v).trim();
  if (!s) throw _err(`${name} is required.`, 'bad_request', 400);
  return s;
}

function _optInt(v, name, lo, hi, def) {
  if (v === '' || v == null) return def;
  const n = Number(v);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < lo || n > hi) {
    throw _err(`${name} must be an integer ${lo}..${hi}.`, 'bad_request', 400);
  }
  return n;
}

function _optFloat(v, name, lo, hi, def) {
  if (v === '' || v == null) return def;
  const n = Number(v);
  if (!Number.isFinite(n) || n < lo || n > hi) {
    throw _err(`${name} must be a number ${lo}..${hi}.`, 'bad_request', 400);
  }
  return n;
}

function validateDatasetUpload(body) {
  if (!body || typeof body !== 'object') throw _err('Request body must be an object.', 'bad_request', 400);
  const filename = _nonEmptyString(body.filename, 'filename').slice(0, 200);
  if (!/\.(csv|tsv|tab|json)$/i.test(filename)) {
    throw _err('Only .csv, .tsv and .json uploads are supported.', 'bad_request', 400);
  }
  const content = body.content == null ? '' : String(body.content);
  const nbytes = Buffer.byteLength(content, 'utf8');
  if (nbytes === 0) throw _err('Uploaded file is empty.', 'bad_request', 400);
  if (nbytes > ML_MAX_DATASET_BYTES) {
    throw _err(`Dataset too large: ${nbytes} bytes (limit ${ML_MAX_DATASET_BYTES}).`, 'bad_request', 400);
  }
  return { filename, content };
}

function validateHfDataset(body) {
  if (!body || typeof body !== 'object') throw _err('Request body must be an object.', 'bad_request', 400);
  const dataset_id = _nonEmptyString(body.dataset_id || body.datasetId, 'dataset_id');
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(dataset_id)) {
    throw _err(`Invalid dataset_id: ${dataset_id}. Use owner/name.`, 'bad_request', 400);
  }
  const split = body.split == null || body.split === '' ? null : String(body.split).slice(0, 64);
  const config = body.config == null || body.config === '' ? null : String(body.config).slice(0, 200);
  return { dataset_id, split, config };
}

// ── Hugging Face Hub dataset discovery (metadata only, never contents) ──
// Fixed upstream bases only — the client sends structured params, never URLs.
const HF_HUB_API = 'https://huggingface.co/api/datasets';
const HF_SPLITS_API = 'https://datasets-server.huggingface.co/splits';
const HF_SEARCH_TTL_MS = 60000;
const HF_SEARCH_CACHE_MAX = 100;
const _hfSearchCache = new Map(); // key -> { at, data }
const HF_SORTS = ['downloads', 'likes', 'lastModified'];
const HF_TASKS = ['text-classification', 'token-classification', 'question-answering',
  'summarization', 'translation', 'text-generation', 'image-classification',
  'object-detection', 'tabular-classification', 'tabular-regression',
  'tabular-to-tabular', 'audio-classification', 'reinforcement-learning'];

function _hfAuthHeaders() {
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || '';
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function _hfCacheGet(key) {
  const hit = _hfSearchCache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.at > HF_SEARCH_TTL_MS) {
    _hfSearchCache.delete(key);
    return null;
  }
  return hit.data;
}

function _hfCacheSet(key, data) {
  if (_hfSearchCache.size >= HF_SEARCH_CACHE_MAX) {
    const oldest = _hfSearchCache.keys().next().value;
    _hfSearchCache.delete(oldest);
  }
  _hfSearchCache.set(key, { at: Date.now(), data });
}

function _hfFriendlyError(status, context) {
  if (status === 401 || status === 403) {
    return _err('That dataset is gated and requires access.' +
      (process.env.HF_TOKEN ? '' : ' Set HF_TOKEN on the server to access gated datasets.'),
      'gated_dataset', status);
  }
  if (status === 404) {
    return _err(`Dataset could not be found${context ? ` (${context})` : ''}.`, 'not_found', 404);
  }
  if (status === 429) {
    return _err('Hugging Face is rate-limiting requests. Try again shortly.', 'rate_limited', 429);
  }
  return _err('Hugging Face is temporarily unavailable. Try again.', 'hf_unavailable', 502);
}

function validateHfSearch(query) {
  query = query || {};
  const q = query.q == null ? '' : String(query.q).slice(0, 100);
  const task = query.task == null || query.task === '' ? null : String(query.task).slice(0, 64);
  if (task && !HF_TASKS.includes(task)) throw _err(`Unknown task filter: ${task}.`, 'bad_request', 400);
  const language = query.language == null || query.language === '' ? null : String(query.language).slice(0, 10);
  if (language && !/^[a-z]{2,3}(-[A-Za-z]{2,4})?$/.test(language)) {
    throw _err(`Invalid language filter: ${language}.`, 'bad_request', 400);
  }
  const sort = query.sort == null || query.sort === '' ? 'downloads' : String(query.sort);
  if (!HF_SORTS.includes(sort)) throw _err(`Unknown sort: ${sort}.`, 'bad_request', 400);
  let limit = query.limit == null || query.limit === '' ? 20 : Number(query.limit);
  if (!Number.isFinite(limit) || !Number.isInteger(limit) || limit < 1) {
    throw _err('limit must be an integer >= 1.', 'bad_request', 400);
  }
  limit = Math.min(limit, 50);
  const cursor = query.cursor == null || query.cursor === '' ? null : String(query.cursor).slice(0, 500);
  return { q, task, language, sort, limit, cursor };
}

function _pickTags(tags, prefix) {
  const out = [];
  for (const t of tags || []) {
    if (typeof t === 'string' && t.startsWith(prefix)) out.push(t.slice(prefix.length));
  }
  return out.slice(0, 12);
}

function _mapHubDataset(d) {
  const id = String(d.id || '');
  const card = (d.cardData && typeof d.cardData === 'object') ? d.cardData : {};
  return {
    id,
    name: id.includes('/') ? id.split('/').slice(1).join('/') : id,
    author: id.includes('/') ? id.split('/')[0] : '',
    description: String(card.description || '').slice(0, 500),
    downloads: Number(d.downloads) || 0,
    likes: Number(d.likes) || 0,
    last_modified: d.lastModified || d.last_modified || null,
    tasks: _pickTags(d.tags, 'task_categories:'),
    languages: _pickTags(d.tags, 'language:'),
  };
}

function _nextCursorFromLink(link) {
  if (!link) return null;
  const m = String(link).match(/<([^>]*[?&]cursor=([^>&]+))>;\s*rel="next"/);
  return m ? decodeURIComponent(m[2]) : null;
}

async function searchHfDatasets(params) {
  const key = JSON.stringify(params);
  const cached = _hfCacheGet(key);
  if (cached) return cached;
  const qs = new URLSearchParams({
    sort: params.sort, direction: '-1', limit: String(params.limit),
  });
  if (params.q) qs.set('search', params.q);
  // NOTE: Hub `filter` accepts a single comma-joined value; task+language combine here.
  const filters = [];
  if (params.task) filters.push(`task_categories:${params.task}`);
  if (params.language) filters.push(`language:${params.language}`);
  if (filters.length) qs.set('filter', filters.join(','));
  if (params.cursor) qs.set('cursor', params.cursor);
  let res;
  try {
    res = await fetch(`${HF_HUB_API}?${qs.toString()}`, { headers: _hfAuthHeaders() });
  } catch (e) {
    throw _err('Hugging Face is temporarily unavailable. Try again.', 'hf_unavailable', 502);
  }
  if (!res.ok) throw _hfFriendlyError(res.status);
  let list;
  try {
    list = await res.json();
  } catch (_) {
    throw _err('Hugging Face is temporarily unavailable. Try again.', 'hf_unavailable', 502);
  }
  if (!Array.isArray(list)) throw _err('Hugging Face is temporarily unavailable. Try again.', 'hf_unavailable', 502);
  const data = {
    datasets: list.map(_mapHubDataset).filter((d) => d.id.includes('/')),
    next_cursor: _nextCursorFromLink(res.headers.get('link')),
  };
  _hfCacheSet(key, data);
  return data;
}

async function getHfDatasetInfo(dataset_id) {
  const id = _nonEmptyString(dataset_id, 'dataset_id');
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(id)) {
    throw _err(`Invalid dataset_id: ${id}. Use owner/name.`, 'bad_request', 400);
  }
  let meta;
  try {
    const res = await fetch(`${HF_HUB_API}/${id}`, { headers: _hfAuthHeaders() });
    if (!res.ok) throw _hfFriendlyError(res.status, id);
    meta = await res.json();
  } catch (e) {
    if (e && e.code) throw e;
    throw _err('Hugging Face is temporarily unavailable. Try again.', 'hf_unavailable', 502);
  }
  const mapped = _mapHubDataset(meta || {});
  mapped.id = id;
  if (!mapped.author && id.includes('/')) mapped.author = id.split('/')[0];
  let configs = [];
  let splits = {};
  try {
    const res = await fetch(`${HF_SPLITS_API}?dataset=${encodeURIComponent(id)}`, { headers: _hfAuthHeaders() });
    if (res.ok) {
      const body = await res.json();
      const rows = Array.isArray(body && body.splits) ? body.splits : [];
      const byConfig = {};
      for (const r of rows) {
        const c = String((r && r.config) || 'default');
        const s = String((r && r.split) || '');
        if (!s) continue;
        if (!byConfig[c]) byConfig[c] = [];
        if (!byConfig[c].includes(s)) byConfig[c].push(s);
      }
      configs = Object.keys(byConfig);
      splits = byConfig;
    }
  } catch (_) {
    // Split metadata is best-effort; the loader still tries the split.
  }
  return { ...mapped, configs, splits };
}

function validateView(view) {
  view = view || {};
  const out = {};
  if (view.drop_columns !== undefined) {
    if (!Array.isArray(view.drop_columns) || view.drop_columns.some((c) => typeof c !== 'string')) {
      throw _err('drop_columns must be a list of column names.', 'bad_request', 400);
    }
    out.drop_columns = view.drop_columns.map((c) => c.slice(0, 200));
  }
  if (view.filters !== undefined) {
    if (!Array.isArray(view.filters)) throw _err('filters must be a list.', 'bad_request', 400);
    out.filters = view.filters.map((f) => {
      if (!f || typeof f.column !== 'string' || typeof f.op !== 'string') {
        throw _err('Each filter needs {column, op, value}.', 'bad_request', 400);
      }
      if (!['==', '!=', '>', '>=', '<', '<=', 'contains', 'not_null', 'is_null'].includes(f.op)) {
        throw _err(`Unknown filter op: ${f.op}.`, 'bad_request', 400);
      }
      return { column: f.column.slice(0, 200), op: f.op, value: f.value == null ? null : String(f.value).slice(0, 500) };
    });
  }
  if (view.sample !== undefined && view.sample !== null) {
    const s = view.sample;
    out.sample = {
      n: s.n == null || s.n === '' ? null : _optInt(s.n, 'sample.n', 1, ML_MAX_DATASET_ROWS, null),
      frac: s.frac == null || s.frac === '' ? null : _optFloat(s.frac, 'sample.frac', 0, 1, null),
      seed: _optInt(s.seed, 'sample.seed', 0, 2 ** 31 - 1, 42),
    };
    if (out.sample.n == null && out.sample.frac == null) {
      throw _err('sample needs n or frac.', 'bad_request', 400);
    }
  }
  return out;
}

function validateSplitConfig(cfg) {
  cfg = cfg || {};
  const strategy = String(cfg.strategy || 'train_val_test');
  if (!SPLIT_STRATEGIES.includes(strategy)) throw _err(`Unknown split strategy: ${strategy}.`, 'bad_request', 400);
  if (strategy === 'official') {
    return { strategy, split_column: _nonEmptyString(cfg.split_column, 'split_column').slice(0, 200),
             split_values: cfg.split_values || { train: 'train', validation: 'validation', test: 'test' } };
  }
  if (strategy === 'time') {
    return { strategy, time_column: _nonEmptyString(cfg.time_column, 'time_column').slice(0, 200),
             train_frac: _optFloat(cfg.train_frac, 'train_frac', 0.01, 0.99, 0.7),
             val_frac: _optFloat(cfg.val_frac, 'val_frac', 0, 0.99, 0.15),
             test_frac: _optFloat(cfg.test_frac, 'test_frac', 0.01, 0.99, 0.15) };
  }
  const out = {
    strategy,
    train_frac: _optFloat(cfg.train_frac, 'train_frac', 0.01, 0.99, strategy === 'train_test' ? 0.8 : 0.7),
    val_frac: _optFloat(cfg.val_frac, 'val_frac', 0, 0.99, strategy === 'train_test' ? 0 : 0.15),
    test_frac: _optFloat(cfg.test_frac, 'test_frac', 0.01, 0.99, strategy === 'train_test' ? 0.2 : 0.15),
    seed: _optInt(cfg.seed, 'seed', 0, 2 ** 31 - 1, 42),
    shuffle: cfg.shuffle === undefined ? true : !!cfg.shuffle,
    stratify: !!cfg.stratify,
    group_column: cfg.group_column ? String(cfg.group_column).slice(0, 200) : null,
  };
  const total = out.train_frac + out.val_frac + out.test_frac;
  if (Math.abs(total - 1) > 1e-6) {
    throw _err(`Fractions must sum to 1 (got ${total}).`, 'bad_request', 400);
  }
  return out;
}

function validatePreprocessingConfig(cfg) {
  cfg = cfg || {};
  const num = cfg.numeric || {};
  const cat = cfg.categorical || {};
  const out = {
    numeric_columns: Array.isArray(cfg.numeric_columns) ? cfg.numeric_columns.map(String) : [],
    categorical_columns: Array.isArray(cfg.categorical_columns) ? cfg.categorical_columns.map(String) : [],
    numeric: {
      impute: ['median', 'mean', 'most_frequent', 'constant'].includes(num.impute) ? num.impute : 'median',
      impute_value: num.impute_value === undefined ? 0 : num.impute_value,
      scaler: ['none', 'standard', 'minmax'].includes(num.scaler) ? num.scaler : 'standard',
      log_columns: Array.isArray(num.log_columns) ? num.log_columns.map(String) : [],
    },
    categorical: {
      impute: ['most_frequent', 'constant'].includes(cat.impute) ? cat.impute : 'most_frequent',
      impute_value: cat.impute_value === undefined ? 'missing' : String(cat.impute_value),
      encoder: ['onehot', 'ordinal'].includes(cat.encoder) ? cat.encoder : 'onehot',
    },
  };
  if (out.numeric_columns.length + out.categorical_columns.length > ML_MAX_FEATURES) {
    throw _err(`Too many feature columns (limit ${ML_MAX_FEATURES}).`, 'bad_request', 400);
  }
  return out;
}

function validateModelConfig(cfg, task) {
  cfg = cfg || {};
  const family = _nonEmptyString(cfg.family, 'model family');
  const pool = task === 'regression' ? REGRESSION_MODELS : CLASSIFICATION_MODELS;
  if (!pool.includes(family)) {
    throw _err(`Model ${family} is not available for ${task}.`, 'bad_request', 400);
  }
  const params = cfg.params && typeof cfg.params === 'object' ? cfg.params : {};
  for (const [k, v] of Object.entries(params)) {
    if (typeof v !== 'number' && typeof v !== 'string' && v !== null) {
      throw _err(`Parameter ${k} must be a number, string or null.`, 'bad_request', 400);
    }
  }
  return { family, params };
}

function validateCvConfig(cfg) {
  if (!cfg) return null;
  const method = String(cfg.method || 'auto');
  if (!CV_METHODS.includes(method)) throw _err(`Unknown CV method: ${method}.`, 'bad_request', 400);
  if (method === 'none') return null;
  return {
    method,
    folds: _optInt(cfg.folds, 'cv.folds', 2, ML_MAX_CV_FOLDS, 5),
    scoring: cfg.scoring ? String(cfg.scoring).slice(0, 64) : null,
    seed: _optInt(cfg.seed, 'cv.seed', 0, 2 ** 31 - 1, 42),
  };
}

function validateSearchConfig(cfg) {
  if (!cfg) return null;
  const method = String(cfg.method || 'none');
  if (!SEARCH_METHODS.includes(method)) throw _err(`Unknown search method: ${method}.`, 'bad_request', 400);
  if (method === 'none') return null;
  const params = cfg.params && typeof cfg.params === 'object' ? cfg.params : null;
  if (!params || !Object.keys(params).length) throw _err('Search needs at least one parameter.', 'bad_request', 400);
  if (Object.keys(params).length > 8) throw _err('Too many search parameters (limit 8).', 'bad_request', 400);
  return {
    method,
    params,
    trials: _optInt(cfg.trials, 'search.trials', 1, ML_MAX_SEARCH_TRIALS, 10),
    scoring: cfg.scoring ? String(cfg.scoring).slice(0, 64) : null,
    cv_folds: _optInt(cfg.cv_folds, 'search.cv_folds', 2, Math.min(5, ML_MAX_CV_FOLDS), 3),
    seed: _optInt(cfg.seed, 'search.seed', 0, 2 ** 31 - 1, 42),
  };
}

function validateExperimentConfig(body) {
  if (!body || typeof body !== 'object') throw _err('Request body must be an object.', 'bad_request', 400);
  const workspace_id = _nonEmptyString(body.workspace_id || body.workspaceId, 'workspace_id');
  if (!_isValidId(workspace_id, 'mlw')) throw _err(`Invalid workspace_id: ${workspace_id}.`, 'bad_request', 400);
  const task = String(body.task || 'classification');
  if (!TASKS.includes(task)) throw _err(`task must be classification or regression.`, 'bad_request', 400);
  const target_column = _nonEmptyString(body.target_column || body.targetColumn, 'target_column').slice(0, 200);
  const feature_columns = body.feature_columns || body.featureColumns || null;
  if (feature_columns !== null) {
    if (!Array.isArray(feature_columns) || !feature_columns.length) {
      throw _err('feature_columns must be a non-empty list or null (all columns).', 'bad_request', 400);
    }
    if (feature_columns.length > ML_MAX_FEATURES) throw _err(`Too many features (limit ${ML_MAX_FEATURES}).`, 'bad_request', 400);
  }
  return {
    workspace_id,
    task,
    target_column,
    feature_columns: feature_columns ? feature_columns.map(String) : null,
    dataset_view: validateView(body.dataset_view || body.datasetView),
    split: validateSplitConfig(body.split),
    preprocessing: validatePreprocessingConfig(body.preprocessing),
    model: validateModelConfig(body.model, task),
    seed: _optInt(body.seed, 'seed', 0, 2 ** 31 - 1, 42),
    cv: validateCvConfig(body.cv),
    search: validateSearchConfig(body.search),
    name: body.name ? String(body.name).slice(0, 120) : '',
    notes: body.notes ? String(body.notes).slice(0, 2000) : '',
  };
}

function validatePredictRequest(body) {
  if (!body || typeof body !== 'object') throw _err('Request body must be an object.', 'bad_request', 400);
  const experiment_id = _nonEmptyString(body.experiment_id || body.experimentId, 'experiment_id');
  if (!_isValidId(experiment_id, 'exp')) throw _err(`Invalid experiment_id.`, 'bad_request', 400);
  const out = { experiment_id };
  if (body.rows !== undefined) {
    if (!Array.isArray(body.rows) || !body.rows.length) throw _err('rows must be a non-empty list.', 'bad_request', 400);
    if (body.rows.length > ML_MAX_PREDICT_ROWS) throw _err(`Too many rows (limit ${ML_MAX_PREDICT_ROWS}).`, 'bad_request', 400);
    out.rows = body.rows;
  } else if (body.csv !== undefined) {
    const csv = String(body.csv);
    if (!csv.trim()) throw _err('csv is empty.', 'bad_request', 400);
    if (Buffer.byteLength(csv, 'utf8') > ML_MAX_DATASET_BYTES) throw _err('Prediction CSV too large.', 'bad_request', 400);
    out.csv = csv;
  } else if (body.workspace_id && body.row_filter !== undefined) {
    if (!_isValidId(String(body.workspace_id), 'mlw')) throw _err('Invalid workspace_id.', 'bad_request', 400);
    out.workspace_id = String(body.workspace_id);
    out.row_filter = validateView({ filters: body.row_filter }).filters || [];
    out.max_rows = Math.min(_optInt(body.max_rows, 'max_rows', 1, ML_MAX_PREDICT_ROWS, 500), ML_MAX_PREDICT_ROWS);
  } else {
    throw _err('Provide rows, csv, or workspace_id + row_filter.', 'bad_request', 400);
  }
  return out;
}

// ── Dataset workspaces ──
function _wsPaths(ws_id) {
  const root = _safeMlDir('ws', ws_id);
  return { root, data: path.join(root, 'data.csv'), profile: path.join(root, 'profile.json'), manifest: path.join(root, 'manifest.json') };
}

function _readJson(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }

async function ingestUpload({ filename, content }) {
  const ws_id = _genWsId();
  const p = _wsPaths(ws_id);
  fs.mkdirSync(p.root, { recursive: true });
  // Write raw text to a temp file for the profiler (avoids argv limits).
  const tmp = path.join(os.tmpdir(), `mlup_${ws_id}.txt`);
  fs.writeFileSync(tmp, content, 'utf8');
  try {
    const { stdout } = await _runPython(
      ['ml/experiment_runner.py', 'dataset-ingest',
        '--ws', ws_id, '--workspace-base', path.join(__dirname),
        '--filename', filename, '--input', tmp,
        '--max-bytes', String(ML_MAX_DATASET_BYTES),
        '--max-rows', String(ML_MAX_DATASET_ROWS)],
      { timeoutMs: 120000 });
    const payload = _lastJsonLine(stdout);
    if (!payload || payload.type !== 'result') {
      throw _err(`Dataset ingestion failed: ${(payload && payload.error) || 'no result'}`, 'ingest_error', 502);
    }
    return { workspace_id: ws_id, manifest: payload.manifest, profile: payload.profile };
  } finally {
    try { fs.unlinkSync(tmp); } catch (_) {}
  }
}

async function ingestHf({ dataset_id, split, config }) {
  const ws_id = _genWsId();
  const args = ['ml/experiment_runner.py', 'dataset-ingest-hf',
      '--ws', ws_id, '--workspace-base', path.join(__dirname),
      '--dataset-id', dataset_id, '--split', split || '',
      '--max-rows', String(ML_MAX_DATASET_ROWS)];
  if (config) args.push('--config', config);
  const { stdout } = await _runPython(args,
    { timeoutMs: 300000, maxBuffer: 32 * 1024 * 1024 });
  const payload = _lastJsonLine(stdout);
  if (!payload || payload.type !== 'result') {
    throw _err(`Hugging Face dataset load failed: ${(payload && payload.error) || 'no result'}`, 'ingest_error', 502);
  }
  return { workspace_id: ws_id, manifest: payload.manifest, profile: payload.profile };
}

function getWorkspace(ws_id) {
  const p = _wsPaths(ws_id);
  if (!fs.existsSync(p.manifest)) throw _err(`Unknown dataset workspace: ${ws_id}.`, 'not_found', 404);
  return { workspace_id: ws_id, manifest: _readJson(p.manifest), profile: _readJson(p.profile) };
}

async function splitPreview(workspace_id, split, view, target, task) {
  const { stdout } = await _runPython(
    ['ml/experiment_runner.py', 'split-preview',
      '--workspace', path.join(__dirname), '--ws', workspace_id,
      '--split-json', JSON.stringify(split),
      '--view-json', JSON.stringify(view || {}),
      '--target', target || '', '--task', task || 'classification'],
    { timeoutMs: 120000 });
  const payload = _lastJsonLine(stdout);
  if (!payload || payload.type !== 'result') {
    throw _err(`Split preview failed: ${(payload && payload.error) || 'no result'}`, 'split_error', 502);
  }
  return payload.split;
}

// ── Experiments ──
function _expDir(exp_id) { return _safeMlDir('exp', exp_id); }

function _readRecord(exp_id) {
  const dir = _expDir(exp_id);
  const fp = path.join(dir, 'experiment.json');
  if (!fs.existsSync(fp)) throw _err(`Unknown experiment: ${exp_id}.`, 'not_found', 404);
  return _readJson(fp);
}

function _writeRecord(exp_id, record) {
  const dir = _expDir(exp_id);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'experiment.json'), JSON.stringify(record, null, 2));
}

function _publicRecord(rec) {
  return {
    experiment_id: rec.experiment_id,
    name: rec.name || '',
    status: rec.status,
    task: rec.task,
    created_at: rec.created_at,
    updated_at: rec.updated_at || rec.created_at,
    target_column: rec.target_column,
    feature_columns: rec.feature_columns,
    model_family: rec.model_family,
    metrics: rec.metrics || null,
    runtime_sec: rec.runtime_sec,
    error: rec.error || null,
    parent_id: rec.parent_id || null,
  };
}

function createExperimentRecord(config, extra) {
  const exp_id = _genExpId();
  const now = new Date().toISOString();
  const rec = {
    experiment_id: exp_id,
    name: config.name || '',
    notes: config.notes || '',
    status: 'queued',
    task: config.task,
    created_at: now,
    updated_at: now,
    config,
    metrics: null,
    runtime_sec: null,
    error: null,
    ...(extra || {}),
  };
  experiments.set(exp_id, rec);
  _writeRecord(exp_id, rec);
  return rec;
}

function _stripForDisk(rec) {
  const { _proc, ...rest } = rec;
  return rest;
}

function _updateRecord(exp_id, patch) {
  const rec = experiments.get(exp_id) || _readRecord(exp_id);
  Object.assign(rec, patch, { updated_at: new Date().toISOString() });
  experiments.set(exp_id, rec);
  _writeRecord(exp_id, _stripForDisk(rec));
  return rec;
}

function runExperiment(config, opts) {
  opts = opts || {};
  const rec = createExperimentRecord(config, {
    parent_id: opts.parent_id || null,
    rerun_of: opts.rerun_of || null,
  });
  const dir = _expDir(rec.experiment_id);
  const cfgPath = path.join(dir, 'config.json');
  const fullConfig = {
    ...config,
    experiment_id: rec.experiment_id,
    workspace_base: path.join(__dirname),
    limits: {
      max_trials: ML_MAX_SEARCH_TRIALS,
      max_time_sec: ML_MAX_TRAIN_TIME_SEC,
    },
  };
  fs.writeFileSync(cfgPath, JSON.stringify(fullConfig, null, 2));
  _updateRecord(rec.experiment_id, { status: 'running' });

  const timeoutMs = (ML_MAX_TRAIN_TIME_SEC + 60) * 1000;
  const proc = childProcess.spawn(_pythonBin(),
    ['ml/experiment_runner.py', 'run', '--workdir', dir, '--config', cfgPath],
    { cwd: __dirname, env: process.env });
  rec._proc = proc;
  let lineBuf = '';
  let fullOut = '';
  let lastProgress = '';

  const timer = setTimeout(() => {
    try { proc.kill('SIGKILL'); } catch (_) {}
    if (!experiments.get(rec.experiment_id) || experiments.get(rec.experiment_id).status === 'running') {
      _updateRecord(rec.experiment_id, {
        status: 'failed', error: `Training exceeded the ${ML_MAX_TRAIN_TIME_SEC}s limit.`,
      });
    }
  }, timeoutMs);
  if (timer.unref) timer.unref();

  proc.stdout.on('data', (chunk) => {
    const text = chunk.toString('utf8');
    fullOut += text;
    if (fullOut.length > 2 * 1024 * 1024) fullOut = fullOut.slice(-2 * 1024 * 1024);
    lineBuf += text;
    if (lineBuf.length > 512 * 1024) lineBuf = lineBuf.slice(-512 * 1024);
    const lines = lineBuf.split('\n');
    lineBuf = lines.pop();
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith('{')) continue;
      try {
        const msg = JSON.parse(t);
        if (msg && msg.type === 'progress' && msg.message) lastProgress = String(msg.message);
      } catch (_) {}
    }
    const cur = experiments.get(rec.experiment_id);
    if (cur && cur.status === 'running' && lastProgress) {
      cur.last_progress = lastProgress;
    }
  });
  let stderrBuf = '';
  proc.stderr.on('data', (chunk) => {
    stderrBuf += chunk.toString('utf8');
    if (stderrBuf.length > 64 * 1024) stderrBuf = stderrBuf.slice(-64 * 1024);
  });
  proc.on('close', (code) => {
    clearTimeout(timer);
    const cur = experiments.get(rec.experiment_id);
    if (!cur || cur.status !== 'running') return;
    const result = _lastJsonLine(fullOut + '\n' + lineBuf);
    if (result && result.type === 'result' && result.status === 'finished' && result.record) {
      const r = result.record;
      _updateRecord(rec.experiment_id, {
        status: 'finished',
        metrics: r.metrics || null,
        runtime_sec: r.runtime_sec,
        target_column: r.target_column,
        feature_columns: r.feature_columns,
        model_family: r.model_family,
        search: r.search || null,
        children: r.children || [],
        error: null,
      });
    } else {
      const detail = (result && result.error) || stderrBuf.slice(-2000).trim() || `exit code ${code}`;
      _updateRecord(rec.experiment_id, { status: 'failed', error: String(detail).slice(0, 2000) });
    }
  });
  proc.on('error', (err) => {
    clearTimeout(timer);
    _updateRecord(rec.experiment_id, { status: 'failed', error: `Failed to start Python: ${err.message}` });
  });
  return rec;
}

function getExperiment(exp_id) {
  if (!_isValidId(exp_id, 'exp')) throw _err(`Invalid experiment_id.`, 'bad_request', 400);
  if (experiments.has(exp_id)) {
    const rec = experiments.get(exp_id);
    if (rec._proc) { const { _proc, ...rest } = rec; return { ...rest }; }
    return { ...rec };
  }
  return _readRecord(exp_id);
}

function listExperiments() {
  const base = path.resolve(path.join(__dirname, 'ml_experiments'));
  let ids = [...experiments.keys()];
  try {
    if (fs.existsSync(base)) {
      for (const name of fs.readdirSync(base)) {
        if (_isValidId(name, 'exp') && !experiments.has(name)) ids.push(name);
      }
    }
  } catch (_) {}
  const out = [];
  for (const id of ids) {
    try { out.push(_publicRecord(getExperiment(id))); } catch (_) {}
  }
  out.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  return out;
}

function renameExperiment(exp_id, name) {
  const rec = getExperiment(exp_id);
  _updateRecord(exp_id, { name: String(name || '').slice(0, 120) });
  return getExperiment(exp_id);
}

function deleteExperiment(exp_id) {
  getExperiment(exp_id);
  const cur = experiments.get(exp_id);
  if (cur && cur._proc && cur.status === 'running') {
    try { cur._proc.kill('SIGKILL'); } catch (_) {}
  }
  experiments.delete(exp_id);
  const dir = _expDir(exp_id);
  try { fs.rmSync(dir, { recursive: true, force: true }); } catch (_) {}
  return { experiment_id: exp_id, deleted: true };
}

function rerunExperiment(exp_id) {
  const rec = getExperiment(exp_id);
  if (!rec.config) throw _err('Experiment has no stored config to rerun.', 'bad_request', 400);
  return runExperiment(rec.config, { rerun_of: exp_id });
}

function compareExperiments(ids) {
  if (!Array.isArray(ids) || ids.length < 2) {
    throw _err('Select at least two experiments to compare.', 'bad_request', 400);
  }
  if (ids.length > 8) throw _err('Compare at most 8 experiments.', 'bad_request', 400);
  return ids.map((id) => {
    const rec = getExperiment(String(id));
    if (!rec.config) throw _err(`Experiment ${id} has no stored config.`, 'bad_request', 400);
    return {
      experiment_id: rec.experiment_id,
      name: rec.name || '',
      status: rec.status,
      task: rec.task,
      created_at: rec.created_at,
      runtime_sec: rec.runtime_sec,
      dataset: rec.config.workspace_id,
      target_column: rec.config.target_column,
      feature_columns: rec.config.feature_columns,
      split: rec.config.split,
      preprocessing: rec.config.preprocessing,
      model_family: rec.config.model.family,
      model_params: rec.config.model.params,
      seed: rec.config.seed,
      cv: rec.config.cv,
      metrics: rec.metrics || null,
      error: rec.error || null,
    };
  });
}

async function predictWithExperiment(exp_id, input) {
  const rec = getExperiment(exp_id);
  if (rec.status !== 'finished') {
    throw _err(`Experiment is not finished (status=${rec.status}).`, 'bad_request', 400);
  }
  const dir = _expDir(exp_id);
  const modelPath = path.join(dir, 'model.joblib');
  if (!fs.existsSync(modelPath)) throw _err('Model artifact missing for experiment.', 'missing_artifact', 404);
  const args = ['ml/experiment_runner.py', 'predict', '--model', modelPath];
  let tmpCsv = null;
  if (input.rows) {
    args.push('--rows-json', JSON.stringify(input.rows));
  } else if (input.csv) {
    tmpCsv = path.join(os.tmpdir(), `mlpred_${exp_id}.csv`);
    fs.writeFileSync(tmpCsv, input.csv, 'utf8');
    args.push('--csv', tmpCsv);
  } else if (input.workspace_id) {
    const frame = await _workspaceRowsForPredict(input.workspace_id, input.row_filter, input.max_rows);
    args.push('--rows-json', JSON.stringify(frame));
  }
  const outPath = path.join(dir, 'predictions.csv');
  args.push('--out', outPath);
  try {
    const { stdout } = await _runPython(args, { timeoutMs: 120000 });
    const payload = _lastJsonLine(stdout);
    if (!payload || payload.type !== 'result') {
      throw _err(`Prediction failed: ${(payload && payload.error) || 'no result'}`, 'predict_error', 502);
    }
    return { experiment_id: exp_id, rows: payload.rows, count: payload.count, output: 'predictions.csv' };
  } finally {
    if (tmpCsv) { try { fs.unlinkSync(tmpCsv); } catch (_) {} }
  }
}

async function _workspaceRowsForPredict(workspace_id, filters, max_rows) {
  const p = _wsPaths(workspace_id);
  if (!fs.existsSync(p.data)) throw _err(`Unknown dataset workspace.`, 'not_found', 404);
  const { stdout } = await _runPython(
    ['ml/experiment_runner.py', 'predict-rows',
      '--workspace', path.join(__dirname), '--ws', workspace_id,
      '--filters-json', JSON.stringify(filters || []),
      '--max-rows', String(max_rows || 500)],
    { timeoutMs: 60000 });
  const payload = _lastJsonLine(stdout);
  if (!payload || payload.type !== 'result') throw _err('Could not read dataset rows.', 'predict_error', 502);
  return payload.rows;
}

function getModelCatalog() {
  return {
    classification: CLASSIFICATION_MODELS.map((f) => ({ family: f })),
    regression: REGRESSION_MODELS.map((f) => ({ family: f })),
  };
}

module.exports = {
  experiments,
  validateDatasetUpload,
  validateHfDataset,
  searchHfDatasets,
  getHfDatasetInfo,
  validateHfSearch,
  validateView,
  validateSplitConfig,
  validatePreprocessingConfig,
  validateModelConfig,
  validateCvConfig,
  validateSearchConfig,
  validateExperimentConfig,
  validatePredictRequest,
  ingestUpload,
  ingestHf,
  getWorkspace,
  splitPreview,
  createExperimentRecord,
  runExperiment,
  getExperiment,
  listExperiments,
  renameExperiment,
  deleteExperiment,
  rerunExperiment,
  compareExperiments,
  predictWithExperiment,
  getModelCatalog,
  ML_MAX_DATASET_BYTES,
  ML_MAX_DATASET_ROWS,
  ML_MAX_FEATURES,
  ML_MAX_TRAIN_TIME_SEC,
  ML_MAX_CV_FOLDS,
  ML_MAX_SEARCH_TRIALS,
  ML_MAX_PREDICT_ROWS,
  ML_MAX_ARTIFACT_BYTES,
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
};
