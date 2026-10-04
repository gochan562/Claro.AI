/* Claro.AI classical-ML frontend layer (local tabular experiments).
 *
 * Additive companion to dashboard-app.js (which owns notebooks, cells,
 * rendering, and the HF/GPU paths — all untouched). This module owns:
 *   - /api/ml/* client
 *   - dataset / split / preprocess / predictor / evaluate / predict cells
 *   - data explorer tables + Chart.js visualizations (aggregates only)
 *   - experiment tracking panel (list/open/rerun/clone/rename/delete/compare)
 *
 * Loaded as a classic script before dashboard-app.js; all entry points are
 * globals following the existing onclick="fn(...)" convention.
 */
'use strict';

const ML_CELL_TYPES = ['dataset', 'split', 'preprocess', 'predictor', 'evaluate', 'predict'];
const ML_CELL_LABELS = {
  dataset: '🗂️ Dataset', split: '✂️ Split', preprocess: '🧹 Preprocess',
  predictor: '🤖 Predictor', evaluate: '📊 Evaluate', predict: '🔮 Predict',
};

function mlEsc(v) {
  return String(v == null ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function mlApi(path, opts) {
  opts = opts || {};
  const res = await fetch(path, {
    method: opts.method || 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (!res.ok) {
    throw new Error((data && data.error) || `HTTP ${res.status}`);
  }
  return data;
}

// ── Model catalog mirror (families must match ml/models.py CATALOG keys;
// ── enforced by test). Parameter schemas drive the predictor params form. ──
const ML_CLASSIFICATION_MODELS = ['LogisticRegression', 'DecisionTreeClassifier',
  'RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier',
  'SVC', 'KNeighborsClassifier', 'GaussianNB'];
const ML_REGRESSION_MODELS = ['LinearRegression', 'Ridge', 'Lasso',
  'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
  'HistGradientBoostingRegressor', 'SVR', 'KNeighborsRegressor'];

const ML_PARAM_SCHEMAS = {
  LogisticRegression: [
    { name: 'C', label: 'C', type: 'float', def: 1.0 },
    { name: 'max_iter', label: 'Max iterations', type: 'int', def: 1000 },
  ],
  DecisionTreeClassifier: [
    { name: 'max_depth', label: 'Max depth (0 = unlimited)', type: 'int', def: 0 },
    { name: 'min_samples_split', label: 'Min samples split', type: 'int', def: 2 },
    { name: 'min_samples_leaf', label: 'Min samples leaf', type: 'int', def: 1 },
  ],
  RandomForestClassifier: [
    { name: 'n_estimators', label: 'Trees', type: 'int', def: 100 },
    { name: 'max_depth', label: 'Max depth (0 = unlimited)', type: 'int', def: 0 },
    { name: 'min_samples_split', label: 'Min samples split', type: 'int', def: 2 },
    { name: 'min_samples_leaf', label: 'Min samples leaf', type: 'int', def: 1 },
    { name: 'max_features', label: 'Max features', type: 'choice', def: 'sqrt', options: ['sqrt', 'log2'] },
  ],
  GradientBoostingClassifier: [
    { name: 'n_estimators', label: 'Estimators', type: 'int', def: 100 },
    { name: 'learning_rate', label: 'Learning rate', type: 'float', def: 0.1 },
    { name: 'max_depth', label: 'Max depth', type: 'int', def: 3 },
  ],
  HistGradientBoostingClassifier: [
    { name: 'max_iter', label: 'Iterations', type: 'int', def: 100 },
    { name: 'learning_rate', label: 'Learning rate', type: 'float', def: 0.1 },
    { name: 'max_depth', label: 'Max depth (0 = unlimited)', type: 'int', def: 0 },
  ],
  SVC: [
    { name: 'C', label: 'C', type: 'float', def: 1.0 },
    { name: 'kernel', label: 'Kernel', type: 'choice', def: 'rbf', options: ['rbf', 'linear', 'poly'] },
  ],
  KNeighborsClassifier: [{ name: 'n_neighbors', label: 'Neighbors', type: 'int', def: 5 }],
  GaussianNB: [],
  LinearRegression: [],
  Ridge: [{ name: 'alpha', label: 'Alpha', type: 'float', def: 1.0 }],
  Lasso: [
    { name: 'alpha', label: 'Alpha', type: 'float', def: 1.0 },
    { name: 'max_iter', label: 'Max iterations', type: 'int', def: 5000 },
  ],
  DecisionTreeRegressor: [
    { name: 'max_depth', label: 'Max depth (0 = unlimited)', type: 'int', def: 0 },
    { name: 'min_samples_split', label: 'Min samples split', type: 'int', def: 2 },
    { name: 'min_samples_leaf', label: 'Min samples leaf', type: 'int', def: 1 },
  ],
  RandomForestRegressor: [
    { name: 'n_estimators', label: 'Trees', type: 'int', def: 100 },
    { name: 'max_depth', label: 'Max depth (0 = unlimited)', type: 'int', def: 0 },
    { name: 'min_samples_split', label: 'Min samples split', type: 'int', def: 2 },
    { name: 'min_samples_leaf', label: 'Min samples leaf', type: 'int', def: 1 },
  ],
  GradientBoostingRegressor: [
    { name: 'n_estimators', label: 'Estimators', type: 'int', def: 100 },
    { name: 'learning_rate', label: 'Learning rate', type: 'float', def: 0.1 },
    { name: 'max_depth', label: 'Max depth', type: 'int', def: 3 },
  ],
  HistGradientBoostingRegressor: [
    { name: 'max_iter', label: 'Iterations', type: 'int', def: 100 },
    { name: 'learning_rate', label: 'Learning rate', type: 'float', def: 0.1 },
    { name: 'max_depth', label: 'Max depth (0 = unlimited)', type: 'int', def: 0 },
  ],
  SVR: [
    { name: 'C', label: 'C', type: 'float', def: 1.0 },
    { name: 'kernel', label: 'Kernel', type: 'choice', def: 'rbf', options: ['rbf', 'linear', 'poly'] },
  ],
  KNeighborsRegressor: [{ name: 'n_neighbors', label: 'Neighbors', type: 'int', def: 5 }],
};

const ML_SCORING = {
  classification: ['f1_macro', 'f1_weighted', 'accuracy', 'balanced_accuracy', 'roc_auc', 'average_precision', 'neg_log_loss'],
  regression: ['r2', 'neg_mae', 'neg_mse', 'neg_rmse'],
};

// ── Notebook integration helpers (dashboard-app.js owns the registry) ──
function mlActiveNotebook() {
  return (typeof getActiveNotebook === 'function') ? getActiveNotebook() : null;
}

function mlFindCell(cellId) {
  const nb = mlActiveNotebook();
  if (!nb) return null;
  return (nb.cells || []).find((c) => String(c.id) === String(cellId)) || null;
}

function mlSaveAndRender() {
  if (typeof updateNotebook === 'function') updateNotebook();
  if (typeof renderNotebookEditor === 'function') renderNotebookEditor();
}

function mlDatasetCells() {
  const nb = mlActiveNotebook();
  if (!nb) return [];
  return (nb.cells || []).filter((c) => c.ml && c.ml.kind === 'dataset' && c.ml.workspace_id);
}

function mlPredictorCells(withExperiment) {
  const nb = mlActiveNotebook();
  if (!nb) return [];
  return (nb.cells || []).filter((c) => c.ml && c.ml.kind === 'predictor' &&
    (!withExperiment || c.ml.experiment_id));
}

function defaultMlCell(type) {
  const base = { kind: type, version: 1, status: 'idle', error: '' };
  if (type === 'dataset') {
    return { ...base, source: 'huggingface', filename: '', dataset_id: '', hf_title: '',
      hf_config: '', hf_split: '',
      workspace_id: '', profile: null,
      view: { drop_columns: [], filters: [], sample: {} },
      target_column: '', feature_columns: null, exploreOpen: false };
  }
  if (type === 'split') {
    return { ...base, dataset_cell_id: '', strategy: 'train_val_test',
      train_frac: 70, val_frac: 15, test_frac: 15, seed: 42,
      shuffle: true, stratify: true, group_column: '', time_column: '',
      split_column: '', split_values: { train: 'train', validation: 'validation', test: 'test' },
      preview: null };
  }
  if (type === 'preprocess') {
    return { ...base, dataset_cell_id: '',
      numeric_columns: [], categorical_columns: [],
      numeric: { impute: 'median', impute_value: 0, scaler: 'standard', log_columns: [] },
      categorical: { impute: 'most_frequent', impute_value: 'missing', encoder: 'onehot' } };
  }
  if (type === 'predictor') {
    return { ...base, dataset_cell_id: '', split_cell_id: '', preprocess_cell_id: '',
      task: 'auto', model_family: '', model_params: {},
      cv: { method: 'auto', folds: 5, scoring: '' },
      search: { method: 'none', params_text: '', trials: 10, seed: 42, cv_folds: 3, scoring: '' },
      seed: 42, experiment_id: null, last_metrics: null };
  }
  if (type === 'evaluate') {
    return { ...base, predictor_cell_id: '', result: null };
  }
  if (type === 'predict') {
    return { ...base, predictor_cell_id: '', mode: 'rows', csv_text: '', max_rows: 500,
      result: null };
  }
  return base;
}

function mlSetStatus(cell, status, error) {
  cell.ml.status = status;
  cell.ml.error = error || '';
  mlSaveAndRender();
}

// Small control builders (reuse dashboard .cell-btn / .toolbar-btn styling).
function mlTextInput(id, val, placeholder, width) {
  return `<input class="ml-input" data-ml-field="${id}" value="${mlEsc(val === undefined || val === null ? '' : val)}"` +
    (placeholder ? ` placeholder="${mlEsc(placeholder)}"` : '') +
    (width ? ` style="width:${width}"` : '') + `>`;
}

function mlSelect(id, options, current) {
  return `<select class="ml-input" data-ml-field="${id}">` +
    options.map((o) => {
      const v = Array.isArray(o) ? o[0] : o;
      const label = Array.isArray(o) ? o[1] : o;
      return `<option value="${mlEsc(v)}"${String(v) === String(current) ? ' selected' : ''}>${mlEsc(label)}</option>`;
    }).join('') + `</select>`;
}

function mlCheckbox(id, checked, label) {
  return `<label class="ml-check"><input type="checkbox" data-ml-field="${id}"${checked ? ' checked' : ''}> ${mlEsc(label)}</label>`;
}

// Read all [data-ml-field] inputs inside a cell element into a flat object.
function mlReadForm(cellId) {
  const el = document.getElementById(`cell-${cellId}`);
  const out = {};
  if (!el) return out;
  el.querySelectorAll('[data-ml-field]').forEach((input) => {
    const k = input.getAttribute('data-ml-field');
    if (input.type === 'checkbox') out[k] = !!input.checked;
    else if (input.type === 'number') out[k] = input.value === '' ? '' : Number(input.value);
    else out[k] = input.value;
  });
  return out;
}

function mlErrorHtml(cell) {
  if (!cell.ml.error) return '';
  return `<div class="ml-error">⚠️ ${mlEsc(cell.ml.error)}</div>`;
}

function mlStatusChip(cell) {
  const s = cell.ml.status || 'idle';
  const map = { idle: 'idle', working: 'working', ready: 'ready', error: 'error' };
  return `<span class="ml-status ml-status-${map[s] || 'idle'}">${mlEsc(s)}</span>`;
}

// ── Dataset cell ──
function mlDatasetBody(cell) {
  const m = cell.ml;
  let h = `<div class="ml-row">` +
    `<label>Source ${mlSelect('source', [['upload', 'CSV/TSV/JSON upload'], ['huggingface', 'Hugging Face dataset']], m.source)}</label>`;
  if (m.source === 'upload') {
    h += `<label>File <input type="file" data-ml-file="${cell.id}" accept=".csv,.tsv,.tab,.json"></label>` +
      `<button class="cell-btn" onclick="mlUploadDataset(${cell.id})">Upload</button>`;
  } else {
    h += mlHfBrowserHtml(cell);
  }
  h += `</div>`;
  if (!m.profile) {
    return h + `<div class="ml-hint">Upload a CSV/TSV/JSON file or reference a Hugging Face dataset to inspect it.</div>`;
  }
  const p = m.profile;
  h += `<div class="ml-summary"><strong>${mlEsc(m.source === 'huggingface' ? m.dataset_id : m.filename)}</strong> · ` +
    `${p.row_count} rows × ${p.column_count} cols · ${p.duplicate_rows} duplicates` +
    (p.truncated ? ' · <em>truncated to server row cap</em>' : '') + `</div>`;
  h += `<div class="ml-row"><label>Target ${mlSelect('target_column', [['', '(choose target)']].concat(p.columns.map((c) => [c, c])), m.target_column)}</label>` +
    `<button class="cell-btn" onclick="mlSaveDatasetConfig(${cell.id})">Save selection</button>` +
    `<button class="cell-btn" onclick="mlToggleExplore(${cell.id})">${m.exploreOpen ? 'Hide explorer' : 'Explore'}</button></div>`;
  h += `<div class="ml-row"><span class="ml-label">Features (unchecked = all non-target):</span></div><div class="ml-cols">` +
    p.columns.map((c) => {
      const checked = !m.feature_columns || m.feature_columns.includes(c);
      const locked = c === m.target_column;
      return `<label class="ml-check"><input type="checkbox" data-ml-feature="${mlEsc(c)}"${checked && !locked ? ' checked' : ''}${locked ? ' disabled' : ''}> ${mlEsc(c)}</label>`;
    }).join('') + `</div>`;
  h += `<div class="ml-row"><label>Drop columns (comma-separated) ${mlTextInput('drop_columns', (m.view.drop_columns || []).join(', '), 'col_a, col_b', '220px')}</label>` +
    `<label>Sample rows ${mlTextInput('sample_n', (m.view.sample && m.view.sample.n) || '', 'e.g. 1000', '110px')}</label>` +
    `<button class="cell-btn" onclick="mlRefreshProfile(${cell.id})">Apply view</button></div>`;
  if (m.exploreOpen) h += mlExploreHtml(cell);
  return h;
}

async function mlUploadDataset(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const input = document.querySelector(`#cell-${cellId} [data-ml-file]`);
  const file = input && input.files && input.files[0];
  if (!file) { mlSetStatus(cell, 'error', 'Choose a file first.'); return; }
  mlSetStatus(cell, 'working', '');
  try {
    const content = await file.text();
    const data = await mlApi('/api/ml/datasets/upload', {
      method: 'POST', body: { filename: file.name, content },
    });
    Object.assign(cell.ml, {
      filename: file.name, workspace_id: data.workspace_id,
      profile: data.profile,
      target_column: data.profile.columns[0] || '',
      feature_columns: null,
      view: { drop_columns: [], filters: [], sample: {} },
    });
    mlSetStatus(cell, 'ready', '');
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

// Full dataset-cell body renderer (also used for lightweight single-cell
// re-renders that preserve scroll position and input focus).
function mlDatasetBodyInner(cell) {
  return mlDatasetBody(cell);
}

// ── Hugging Face dataset browser (metadata only; contents load on demand) ──
// Ephemeral browse state lives outside cell.ml (never persisted); selection
// (dataset_id/title/config/split) is persisted in cell.ml like before.
const mlHfUi = {};

function mlHfState(cellId) {
  const k = String(cellId);
  if (!mlHfUi[k]) {
    mlHfUi[k] = { q: '', task: '', language: '', sort: 'downloads',
      results: [], next_cursor: null, searching: false, searched: false,
      selected_info: null, info_loading: false, info_loaded_for: '', error: '' };
  }
  return mlHfUi[k];
}

const ML_HF_TASKS = [['', '(any task)'], ['text-classification', 'text classification'],
  ['tabular-classification', 'tabular classification'], ['tabular-regression', 'tabular regression'],
  ['token-classification', 'token classification'], ['question-answering', 'question answering'],
  ['summarization', 'summarization'], ['translation', 'translation'],
  ['text-generation', 'text generation'], ['image-classification', 'image classification']];
const ML_HF_SORTS = [['downloads', 'Downloads'], ['likes', 'Likes'], ['lastModified', 'Recently updated']];

function mlHfBrowserHtml(cell) {
  const m = cell.ml;
  const ui = mlHfState(cell.id);
  let h = `<div class="ml-row">` +
    `<input class="ml-input" data-ml-hf="q" value="${mlEsc(ui.q)}" placeholder="🔍 Search datasets..." style="flex:1;min-width:160px"` +
    ` onkeydown="if(event.key==='Enter'){event.preventDefault();mlHfSearch(${cell.id},true);}">` +
    `<button class="cell-btn run" onclick="mlHfSearch(${cell.id},true)">Search</button></div>`;
  h += `<div class="ml-row">` +
    `<label>Task ${mlSelectHtml('hf_task', ML_HF_TASKS, ui.task)}</label>` +
    `<label>Language ${mlTextInput('hf_language', ui.language, 'e.g. en', '90px')}</label>` +
    `<label>Sort ${mlSelectHtml('hf_sort', ML_HF_SORTS, ui.sort)}</label></div>`;
  if (ui.searching) h += `<div class="ml-hint">Searching Hugging Face…</div>`;
  if (ui.error) h += `<div class="ml-error">⚠️ ${mlEsc(ui.error)}</div>`;
  if (!ui.searching && !ui.results.length && !ui.error) {
    h += `<div class="ml-hint">Search above, or pick from popular datasets below.</div>`;
  }
  if (ui.results.length) {
    h += `<div class="ml-hf-results">` + ui.results.map((d) => (
      `<div class="ml-hf-row"><div class="ml-hf-main">` +
      `<div class="ml-hf-id">${mlEsc(d.id)}</div>` +
      (d.description ? `<div class="ml-hf-desc">${mlEsc(d.description.slice(0, 160))}</div>` : '') +
      `<div class="ml-hf-meta">${d.downloads ? `⬇ ${mlFmtCount(d.downloads)} · ` : ''}` +
      `${d.likes ? `♡ ${mlFmtCount(d.likes)}` : ''}` +
      `${(d.tasks || []).slice(0, 3).map((t) => ` <span class="ml-tag">${mlEsc(t)}</span>`).join('')}` +
      `</div></div>` +
      `<button class="cell-btn" onclick="mlHfSelect(${cell.id},'${mlEsc(d.id).replace(/'/g, '&#39;')}')">Select</button></div>`
    )).join('') + `</div>`;
    if (ui.next_cursor) {
      h += `<div class="ml-row"><button class="cell-btn" onclick="mlHfMore(${cell.id})">Load more</button></div>`;
    }
  }
  // Selection + split/config + load (existing ingest path).
  h += `<div class="ml-sec"><strong>Selected dataset</strong>`;
  if (m.dataset_id) {
    h += `<div class="ml-summary"><strong>${mlEsc(m.dataset_id)}</strong>` +
      (m.hf_title ? ` — ${mlEsc(m.hf_title)}` : '') + `</div>`;
    h += mlHfSelectionHtml(cell, ui);
  } else {
    h += `<div class="ml-hint">No dataset selected yet.</div>`;
  }
  h += `<div class="ml-row"><label>Manual ID ${mlTextInput('dataset_id_manual', '', 'owner/name', '200px')}</label>` +
    `<button class="cell-btn" onclick="mlHfManual(${cell.id})">Use ID</button></div>`;
  h += `</div>`;
  // Auto-load popular datasets once per cell when the browser is untouched.
  if (!m.workspace_id && !ui.searched && !ui.searching && !ui.results.length && !ui.error) {
    ui.searching = true;
    setTimeout(() => mlHfPopular(cell.id), 0);
  }
  return h;
}

function mlSelectHtml(field, options, current) {
  return `<select class="ml-input" data-ml-hf="${field}">` +
    options.map((o) => `<option value="${mlEsc(o[0])}"${String(o[0]) === String(current) ? ' selected' : ''}>${mlEsc(o[1])}</option>`).join('') +
    `</select>`;
}

function mlFmtCount(n) {
  n = Number(n) || 0;
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}

function mlHfSelectionHtml(cell, ui) {
  const m = cell.ml;
  const info = ui.selected_info;
  let h = '';
  if (ui.info_loading) return `<div class="ml-hint">Loading dataset details…</div>`;
  if (info) {
    if (info.description) h += `<div class="ml-hf-desc">${mlEsc(info.description.slice(0, 300))}</div>`;
    const cfgs = info.configs || [];
    if (cfgs.length > 1) {
      h += `<div class="ml-row"><label>Configuration ${mlSelectHtml('hf_config_sel', cfgs.map((c) => [c, c]), m.hf_config || cfgs[0])}</label></div>`;
    } else if (cfgs.length === 1 && !m.hf_config) {
      m.hf_config = cfgs[0];
    }
    const splits = info.splits || {};
    const cfgKey = (m.hf_config && splits[m.hf_config]) ? m.hf_config : (cfgs[0] || Object.keys(splits)[0]);
    const opts = (cfgKey && splits[cfgKey]) || [];
    if (opts.length) {
      h += `<div class="ml-row"><label>Split ${mlSelectHtml('hf_split_sel', opts.map((s) => [s, s]), m.hf_split || opts[0])}</label></div>`;
    } else {
      h += `<div class="ml-row"><label>Split ${mlTextInput('hf_split_manual', m.hf_split, 'train (blank = auto)', '160px')}</label></div>`;
    }
  } else {
    h += `<div class="ml-row"><label>Split ${mlTextInput('hf_split_manual', m.hf_split, 'train (blank = auto)', '160px')}</label></div>`;
  }
  h += `<div class="ml-row"><button class="cell-btn run" onclick="mlLoadHfDataset(${cell.id})">Load Dataset</button></div>`;
  return h;
}

function mlHfReadBar(cellId) {
  const el = document.getElementById(`cell-${cellId}`);
  const out = { q: '', task: '', language: '', sort: 'downloads' };
  if (!el) return out;
  el.querySelectorAll('[data-ml-hf]').forEach((input) => {
    out[input.getAttribute('data-ml-hf')] = input.value;
  });
  return out;
}

async function mlHfSearch(cellId, reset) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const ui = mlHfState(cellId);
  const bar = mlHfReadBar(cellId);
  ui.q = bar.q; ui.task = bar.task; ui.language = bar.language; ui.sort = bar.sort;
  if (reset) { ui.results = []; ui.next_cursor = null; }
  ui.searching = true; ui.error = '';
  mlRerenderCell(cellId);
  try {
    const qs = new URLSearchParams({ sort: ui.sort || 'downloads', limit: '20' });
    if (ui.q.trim()) qs.set('q', ui.q.trim());
    if (ui.task) qs.set('task', ui.task);
    if (ui.language.trim()) qs.set('language', ui.language.trim());
    const data = await mlApi(`/api/ml/datasets/search?${qs.toString()}`);
    ui.results = data.datasets || [];
    ui.next_cursor = data.next_cursor || null;
    ui.searched = true;
  } catch (e) {
    ui.error = e.message;
  }
  ui.searching = false;
  mlSaveAndRenderLite(cellId);
}

async function mlHfPopular(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const ui = mlHfState(cellId);
  try {
    const data = await mlApi('/api/ml/datasets/search?sort=downloads&limit=8');
    const still = mlFindCell(cellId);
    if (!still) return;
    const u2 = mlHfState(cellId);
    if (u2.searched) return; // user searched meanwhile; don't clobber
    u2.results = data.datasets || [];
    u2.next_cursor = null;
    u2.searched = true;
  } catch (e) {
    const still = mlFindCell(cellId);
    if (still) mlHfState(cellId).error = e.message;
  }
  const still = mlFindCell(cellId);
  if (still) {
    mlHfState(cellId).searching = false;
    mlSaveAndRenderLite(cellId);
  }
}

async function mlHfMore(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const ui = mlHfState(cellId);
  if (!ui.next_cursor || ui.searching) return;
  ui.searching = true; ui.error = '';
  mlRerenderCell(cellId);
  try {
    const qs = new URLSearchParams({ sort: ui.sort || 'downloads', limit: '20', cursor: ui.next_cursor });
    if (ui.q.trim()) qs.set('q', ui.q.trim());
    if (ui.task) qs.set('task', ui.task);
    if (ui.language.trim()) qs.set('language', ui.language.trim());
    const data = await mlApi(`/api/ml/datasets/search?${qs.toString()}`);
    const seen = new Set(ui.results.map((d) => d.id));
    (data.datasets || []).forEach((d) => { if (!seen.has(d.id)) ui.results.push(d); });
    ui.next_cursor = data.next_cursor || null;
  } catch (e) {
    ui.error = e.message;
  }
  ui.searching = false;
  mlSaveAndRenderLite(cellId);
}

async function mlHfSelect(cellId, datasetId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  cell.ml.dataset_id = datasetId;
  cell.ml.hf_title = '';
  cell.ml.hf_config = '';
  cell.ml.hf_split = '';
  const ui = mlHfState(cellId);
  ui.selected_info = null;
  ui.info_loaded_for = '';
  ui.error = '';
  mlSaveAndRenderLite(cellId);
  await mlHfEnsureInfo(cellId);
}

async function mlHfManual(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const f = mlReadForm(cellId);
  const val = String(f.dataset_id_manual || '').trim();
  if (!val) return;
  cell.ml.dataset_id = val;
  cell.ml.hf_title = '';
  cell.ml.hf_config = '';
  cell.ml.hf_split = '';
  const ui = mlHfState(cellId);
  ui.selected_info = null;
  ui.info_loaded_for = '';
  ui.error = '';
  mlSaveAndRenderLite(cellId);
  await mlHfEnsureInfo(cellId);
}

async function mlHfEnsureInfo(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell || !cell.ml.dataset_id) return;
  const ui = mlHfState(cellId);
  if (ui.info_loading) return;
  if (ui.selected_info && ui.info_loaded_for === cell.ml.dataset_id) return;
  ui.info_loading = true;
  ui.error = '';
  mlRerenderCell(cellId);
  try {
    const info = await mlApi(`/api/ml/datasets/info?dataset_id=${encodeURIComponent(cell.ml.dataset_id)}`);
    const still = mlFindCell(cellId);
    if (!still || still.ml.dataset_id !== cell.ml.dataset_id) return;
    const u2 = mlHfState(cellId);
    u2.selected_info = info;
    u2.info_loaded_for = cell.ml.dataset_id;
    if (info.title || info.name) still.ml.hf_title = info.title || info.name;
    const cfgs = info.configs || [];
    if (cfgs.length && !still.ml.hf_config) still.ml.hf_config = cfgs[0];
    const splitsByCfg = (info.splits && typeof info.splits === 'object') ? info.splits : {};
    const cfgKey = (still.ml.hf_config && splitsByCfg[still.ml.hf_config])
      ? still.ml.hf_config : (cfgs[0] || Object.keys(splitsByCfg)[0]);
    const splits = (cfgKey && splitsByCfg[cfgKey]) || [];
    if (splits.length && !still.ml.hf_split) still.ml.hf_split = splits[0];
  } catch (e) {
    const still = mlFindCell(cellId);
    if (still) mlHfState(cellId).error = e.message;
  }
  const still = mlFindCell(cellId);
  if (still) {
    mlHfState(cellId).info_loading = false;
    mlSaveAndRenderLite(cellId);
  }
}

function mlRerenderCell(cellId) {
  // Lightweight re-render of one cell body (no full notebook render, so
  // search input focus and scroll position survive).
  try {
    const cell = mlFindCell(cellId);
    if (!cell) return;
    const body = document.querySelector(`#cell-${cellId} .ml-cell-body`);
    if (!body || cell.ml.kind !== 'dataset') return;
    body.innerHTML = mlDatasetBodyInner(cell);
  } catch (_) {}
}

function mlSaveAndRenderLite(cellId) {
  if (typeof updateNotebook === 'function') {
    try { updateNotebook(); } catch (_) {}
  }
  const cell = (typeof mlFindCell === 'function') ? mlFindCell(cellId) : null;
  if (cell && cell.ml && cell.ml.kind === 'dataset') mlRerenderCell(cellId);
  else if (typeof renderNotebookEditor === 'function') {
    try { renderNotebookEditor(); } catch (_) {}
  }
}

async function mlLoadHfDataset(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const ui = mlHfState(cellId);
  const el = document.getElementById(`cell-${cellId}`);
  const pick = (sel) => {
    const n = el && el.querySelector(`[data-ml-hf="${sel}"], [data-ml-field="${sel}"]`);
    return (n && n.value !== undefined ? String(n.value) : '').trim();
  };
  const cfgSel = pick('hf_config_sel');
  const splitSel = pick('hf_split_sel') || pick('hf_split_manual');
  const manualId = pick('dataset_id_manual');
  const dataset_id = (cell.ml.dataset_id || manualId).trim();
  if (!dataset_id) {
    mlSetStatus(cell, 'error', 'Select a dataset or enter its ID first.');
    return;
  }
  mlSetStatus(cell, 'working', '');
  try {
    const data = await mlApi('/api/ml/datasets/hf', {
      method: 'POST',
      body: { dataset_id, split: splitSel || null, config: cfgSel || null },
    });
    const info = ui.selected_info;
    Object.assign(cell.ml, {
      dataset_id, hf_config: cfgSel || cell.ml.hf_config || '',
      hf_split: splitSel || cell.ml.hf_split || '',
      hf_title: (info && (info.title || info.name)) || cell.ml.hf_title || '',
      workspace_id: data.workspace_id, profile: data.profile,
      target_column: data.profile.columns[0] || '',
      feature_columns: null,
      view: { drop_columns: [], filters: [], sample: {} },
    });
    mlSetStatus(cell, 'ready', '');
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

function mlSaveDatasetConfig(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const f = mlReadForm(cellId);
  const el = document.getElementById(`cell-${cellId}`);
  const feats = [];
  if (el) {
    el.querySelectorAll('[data-ml-feature]').forEach((box) => {
      if (box.checked && !box.disabled) feats.push(box.getAttribute('data-ml-feature'));
    });
  }
  const all = (cell.ml.profile.columns || []).filter((c) => c !== f.target_column);
  const explicit = feats.length && feats.length !== all.length ? feats : null;
  cell.ml.target_column = f.target_column || '';
  cell.ml.feature_columns = explicit;
  const drops = String(f.drop_columns || '').split(',').map((s) => s.trim()).filter(Boolean);
  cell.ml.view = cell.ml.view || { drop_columns: [], filters: [], sample: {} };
  cell.ml.view.drop_columns = drops;
  const n = parseInt(f.sample_n, 10);
  cell.ml.view.sample = Number.isFinite(n) && n > 0 ? { n, seed: 42 } : {};
  mlSaveAndRender();
}

async function mlRefreshProfile(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell || !cell.ml.workspace_id) return;
  mlSaveDatasetConfig(cellId);
  mlSetStatus(cell, 'working', '');
  try {
    const data = await mlApi(`/api/ml/datasets/${cell.ml.workspace_id}`);
    // Recompute the profile under the current view client-side is unsafe;
    // ask the server for a view-aware profile via split-preview style call:
    // (profile endpoint returns stored profile; refresh = re-upload view)
    cell.ml.profile = data.profile;
    mlSetStatus(cell, 'ready', '');
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

function mlToggleExplore(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  cell.ml.exploreOpen = !cell.ml.exploreOpen;
  mlSaveAndRender();
}

// ── Data explorer (aggregates only — raw rows never leave the profile) ──
function mlExploreHtml(cell) {
  const p = cell.ml.profile;
  let h = `<div class="ml-explore"><h4>Data explorer</h4>`;
  h += `<div class="ml-sec"><strong>Preview</strong> (first rows)<div class="ml-table-wrap"><table class="ml-table"><thead><tr>` +
    p.columns.map((c) => `<th>${mlEsc(c)}</th>`).join('') + `</tr></thead><tbody>` +
    (p.preview || []).slice(0, 10).map((r) =>
      `<tr>${p.columns.map((c) => `<td>${mlEsc(r[c])}</td>`).join('')}</tr>`).join('') +
    `</tbody></table></div></div>`;
  h += `<div class="ml-sec"><strong>Columns</strong><div class="ml-table-wrap"><table class="ml-table"><thead><tr>` +
    `<th>column</th><th>type</th><th>missing</th><th>unique</th><th>details</th></tr></thead><tbody>` +
    (p.column_metadata || []).map((m) => {
      const det = m.kind === 'numeric' && m.mean !== undefined
        ? `mean ${mlNum(m.mean)} · std ${mlNum(m.std)} · min ${mlNum(m.min)} · max ${mlNum(m.max)}`
        : ((m.top_values || []).slice(0, 3).map((t) => `${mlEsc(t.value)} (${t.count})`).join(', ') || '—');
      return `<tr><td>${mlEsc(m.name)}</td><td>${mlEsc(m.kind)}</td><td>${m.missing}</td><td>${m.unique}</td><td>${det}</td></tr>`;
    }).join('') + `</tbody></table></div></div>`;
  if (p.numeric_summary && Object.keys(p.numeric_summary).length) {
    const nums = Object.keys(p.numeric_summary);
    const first = nums[0];
    h += `<div class="ml-sec"><strong>Distribution</strong> ` +
      `<label>Column ${mlSelect('hist_col_' + cell.id, nums, first)}</label> ` +
      `<button class="cell-btn" onclick="mlDrawHistogram(${cell.id})">Draw</button>` +
      `<canvas id="ml-hist-${cell.id}" height="120"></canvas></div>`;
    h += `<div class="ml-sec"><strong>Box plot</strong><div class="ml-boxplot">${mlBoxHtml(p)}</div></div>`;
    if (p.scatter && p.scatter.points && p.scatter.points.length) {
      h += `<div class="ml-sec"><strong>Scatter: ${mlEsc(p.scatter.x)} × ${mlEsc(p.scatter.y)}</strong>` +
        `<canvas id="ml-scatter-${cell.id}" height="140"></canvas></div>`;
    }
    if (p.correlation && p.correlation.columns.length > 1) {
      h += `<div class="ml-sec"><strong>Correlation</strong><div class="ml-table-wrap">` +
        mlCorrHtml(p.correlation) + `</div></div>`;
    }
  }
  if (p.categorical_summary && Object.keys(p.categorical_summary).length) {
    const cats = Object.keys(p.categorical_summary).slice(0, 6);
    h += `<div class="ml-sec"><strong>Categories</strong>` +
      cats.map((c) => {
        const s = p.categorical_summary[c];
        return `<div>${mlEsc(c)} <em>(${(s.unique || 0)} unique)</em>: ` +
          (s.values || []).slice(0, 5).map((t) => `${mlEsc(String(t.value))} (${t.count})`).join(', ') + `</div>`;
      }).join('') + `</div>`;
  }
  if (p.target_distribution) {
    h += `<div class="ml-sec"><strong>Target: ${mlEsc(p.target_distribution.column)}</strong>` +
      `<canvas id="ml-target-${cell.id}" height="110"></canvas></div>`;
  }
  h += `</div>`;
  setTimeout(() => mlRenderExploreCharts(cell), 0);
  return h;
}

function mlNum(v) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return '—';
  const n = Number(v);
  return Math.abs(n) >= 1000 ? n.toFixed(1) : (Math.abs(n) >= 1 ? n.toFixed(3) : n.toFixed(5));
}

function mlBoxHtml(p) {
  const cols = Object.keys(p.boxplot || {}).slice(0, 12);
  return cols.map((c) => {
    const b = p.boxplot[c];
    const span = (b.max - b.min) || 1;
    const pct = (v) => Math.max(0, Math.min(100, ((v - b.min) / span) * 100));
    return `<div class="ml-boxrow"><span class="ml-boxlabel">${mlEsc(c)}</span>` +
      `<span class="ml-boxtrack"><span class="ml-boxwhisk" style="left:${pct(b.min)}%;width:${pct(b.max) - pct(b.min)}%"></span>` +
      `<span class="ml-boxbox" style="left:${pct(b.q1)}%;width:${Math.max(1, pct(b.q3) - pct(b.q1))}%"></span>` +
      `<span class="ml-boxmed" style="left:${pct(b.median)}%"></span></span>` +
      `<span class="ml-boxnums">${mlNum(b.min)} / ${mlNum(b.median)} / ${mlNum(b.max)}</span></div>`;
  }).join('') || '<em>No numeric columns.</em>';
}

function mlCorrHtml(corr) {
  const cols = corr.columns;
  let h = `<table class="ml-table ml-corr"><thead><tr><th></th>` +
    cols.map((c) => `<th>${mlEsc(String(c).slice(0, 12))}</th>`).join('') + `</tr></thead><tbody>`;
  corr.matrix.forEach((row, i) => {
    h += `<tr><th>${mlEsc(String(cols[i]).slice(0, 12))}</th>` + row.map((v) => {
      if (v === null) return `<td>—</td>`;
      const a = Math.min(0.85, Math.abs(v));
      const bg = v >= 0 ? `rgba(80,160,120,${a})` : `rgba(200,110,110,${a})`;
      return `<td style="background:${bg}">${v.toFixed(2)}</td>`;
    }).join('') + `</tr>`;
  });
  return h + `</tbody></table>`;
}

function mlChart(canvasId, cfg) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  if (typeof Chart === 'undefined') {
    el.outerHTML = `<div class="ml-hint">Charts need Chart.js (already used by Training Cells).</div>`;
    return;
  }
  try {
    if (el._mlChart) el._mlChart.destroy();
    el._mlChart = new Chart(el, cfg);
  } catch (e) {
    el.outerHTML = `<div class="ml-hint">Chart error: ${mlEsc(e.message)}</div>`;
  }
}

function mlRenderExploreCharts(cell) {
  try {
    const p = cell.ml.profile;
    if (!p) return;
    if (p.target_distribution) {
      const t = p.target_distribution.values || [];
      mlChart(`ml-target-${cell.id}`, {
        type: 'bar',
        data: { labels: t.map((v) => String(v.value)), datasets: [{ data: t.map((v) => v.count), backgroundColor: '#4a6fa5' }] },
        options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
      });
    }
    if (p.scatter && p.scatter.points) {
      mlChart(`ml-scatter-${cell.id}`, {
        type: 'scatter',
        data: { datasets: [{ data: p.scatter.points.map((pt) => ({ x: pt[0], y: pt[1] })), backgroundColor: '#7ca673' }] },
        options: { plugins: { legend: { display: false } } },
      });
    }
    mlDrawHistogram(cell.id, true);
  } catch (_) {}
}

function mlDrawHistogram(cellId, silent) {
  const cell = mlFindCell(cellId);
  if (!cell || !cell.ml.profile) return;
  const p = cell.ml.profile;
  const sel = document.querySelector(`#cell-${cellId} [data-ml-field="hist_col_${cellId}"]`);
  const col = (sel && sel.value) || Object.keys(p.numeric_summary || {})[0];
  const s = col && p.numeric_summary[col];
  if (!s) { if (!silent) alert('No numeric column available.'); return; }
  const labels = s.histogram.bins.slice(0, -1).map((b, i) =>
    `${mlNum(b)}–${mlNum(s.histogram.bins[i + 1])}`);
  mlChart(`ml-hist-${cellId}`, {
    type: 'bar',
    data: { labels, datasets: [{ data: s.histogram.counts, backgroundColor: '#4a6fa5' }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
  });
}

// ── Split cell ──
function mlDatasetOptions(selectedId) {
  const cells = mlDatasetCells();
  const opts = [['', '(choose dataset cell)']].concat(cells.map((c) => {
    const prof = (c.ml.profile && `${c.ml.profile.row_count} rows`) || 'not loaded';
    return [String(c.id), `Cell ${c.id} — ${prof}`];
  }));
  return mlSelect('dataset_cell_id', opts, selectedId || '');
}

function mlSplitBody(cell) {
  const m = cell.ml;
  let h = `<div class="ml-row"><label>Dataset ${mlDatasetOptions(m.dataset_cell_id)}</label>` +
    `<label>Strategy ${mlSelect('strategy', [['train_val_test', 'train / validation / test'], ['train_test', 'train / test'], ['official', 'official splits'], ['time', 'time-based']], m.strategy)}</label></div>`;
  if (m.strategy === 'official') {
    h += `<div class="ml-row"><label>Split column ${mlTextInput('split_column', m.split_column, 'e.g. split', '160px')}</label></div>`;
  } else if (m.strategy === 'time') {
    h += `<div class="ml-row"><label>Time column ${mlTextInput('time_column', m.time_column, 'e.g. date', '160px')}</label></div>`;
  }
  h += `<div class="ml-row"><label>Train % ${mlTextInput('train_frac', m.train_frac, '', '70px')}</label>` +
    (m.strategy === 'train_val_test' ? `<label>Validation % ${mlTextInput('val_frac', m.val_frac, '', '70px')}</label>` : '') +
    `<label>Test % ${mlTextInput('test_frac', m.test_frac, '', '70px')}</label>` +
    `<label>Seed ${mlTextInput('seed', m.seed, '', '90px')}</label></div>`;
  h += `<div class="ml-row">${mlCheckbox('shuffle', m.shuffle, 'Shuffle')} ${mlCheckbox('stratify', m.stratify, 'Stratify (classification)')}` +
    `<label>Group column ${mlTextInput('group_column', m.group_column, '(optional)', '140px')}</label>` +
    `<button class="cell-btn" onclick="mlSaveSplitConfig(${cell.id})">Save</button>` +
    `<button class="cell-btn run" onclick="mlPreviewSplit(${cell.id})">Preview split</button></div>`;
  if (m.preview) {
    const c = m.preview.counts || {};
    h += `<div class="ml-summary">train <strong>${c.train || 0}</strong> · ` +
      (c.validation !== undefined ? `validation <strong>${c.validation}</strong> · ` : '') +
      `test <strong>${c.test || 0}</strong>` +
      ((m.preview.notes || []).length ? ` <em>(${(m.preview.notes || []).map(mlEsc).join('; ')})</em>` : '') + `</div>`;
    if (m.preview.classes && Object.keys(m.preview.classes).length) {
      const parts = Object.keys(m.preview.classes);
      h += `<div class="ml-table-wrap"><table class="ml-table"><thead><tr><th>class</th>` +
        parts.map((p2) => `<th>${mlEsc(p2)}</th>`).join('') + `</tr></thead><tbody>`;
      const labels = [...new Set(parts.flatMap((p2) => (m.preview.classes[p2] || []).map((r) => r.value)))];
      h += labels.map((lab) => `<tr><td>${mlEsc(lab)}</td>` + parts.map((p2) => {
        const row = (m.preview.classes[p2] || []).find((r) => r.value === lab);
        return `<td>${row ? row.count : 0}</td>`;
      }).join('') + `</tr>`).join('') + `</tbody></table></div>`;
    }
  } else {
    h += `<div class="ml-hint">Save, then Preview split to see row counts and class balance. The test split is never used for tuning.</div>`;
  }
  return h;
}

function mlSaveSplitConfig(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const f = mlReadForm(cellId);
  Object.assign(cell.ml, {
    dataset_cell_id: String(f.dataset_cell_id || ''),
    strategy: f.strategy || 'train_val_test',
    train_frac: Number(f.train_frac) || 0,
    val_frac: Number(f.val_frac) || 0,
    test_frac: Number(f.test_frac) || 0,
    seed: f.seed === '' ? 42 : Number(f.seed),
    shuffle: !!f.shuffle,
    stratify: !!f.stratify,
    group_column: String(f.group_column || ''),
    time_column: String(f.time_column || ''),
    split_column: String(f.split_column || ''),
  });
  mlSaveAndRender();
}

function mlSplitPayload(cell) {
  const m = cell.ml;
  const pct = (v, d) => (v === '' || v == null ? d : Number(v) / 100);
  return {
    strategy: m.strategy,
    train_frac: pct(m.train_frac, 0.7),
    val_frac: m.strategy === 'train_val_test' ? pct(m.val_frac, 0.15) : 0,
    test_frac: pct(m.test_frac, m.strategy === 'train_test' ? 0.2 : 0.15),
    seed: m.seed === '' || m.seed == null ? 42 : Number(m.seed),
    shuffle: !!m.shuffle,
    stratify: !!m.stratify,
    group_column: m.group_column || null,
    time_column: m.time_column || null,
    split_column: m.split_column || null,
    split_values: m.split_values,
  };
}

async function mlPreviewSplit(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  mlSaveSplitConfig(cellId);
  const ds = mlFindCell(cell.ml.dataset_cell_id);
  if (!ds || !ds.ml.workspace_id) { mlSetStatus(cell, 'error', 'Choose a loaded Dataset Cell first.'); return; }
  mlSetStatus(cell, 'working', '');
  try {
    const prev = await mlApi('/api/ml/split/preview', {
      method: 'POST',
      body: { workspace_id: ds.ml.workspace_id, split: mlSplitPayload(cell),
        dataset_view: ds.ml.view || {}, target_column: ds.ml.target_column || null,
        task: mlGuessTask(ds) },
    });
    cell.ml.preview = prev;
    mlSetStatus(cell, 'ready', '');
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

function mlGuessTask(dsCell) {
  if (!dsCell || !dsCell.ml.profile || !dsCell.ml.target_column) return 'classification';
  const t = dsCell.ml.target_column;
  const meta = (dsCell.ml.profile.column_metadata || []).find((m) => m.name === t);
  if (!meta) return 'classification';
  if (meta.kind === 'numeric' && (meta.unique || 0) > 20) return 'regression';
  return 'classification';
}

// ── Preprocess cell ──
function mlPreprocessBody(cell) {
  const m = cell.ml;
  const ds = mlFindCell(m.dataset_cell_id);
  const prof = ds && ds.ml.profile;
  const nums = prof ? prof.numeric_columns || [] : [];
  const cats = prof ? prof.categorical_columns || [] : [];
  let h = `<div class="ml-row"><label>Dataset ${mlDatasetOptions(m.dataset_cell_id)}</label>` +
    `<button class="cell-btn" onclick="mlSavePreprocessConfig(${cell.id})">Save</button></div>`;
  if (!prof) return h + `<div class="ml-hint">Choose a loaded Dataset Cell to pick columns.</div>`;
  const colChecks = (cols, field, selected) =>
    cols.length ? cols.map((c) => `<label class="ml-check"><input type="checkbox" data-ml-coll="${field}" value="${mlEsc(c)}"${(!selected.length || selected.includes(c)) ? ' checked' : ''}> ${mlEsc(c)}</label>`).join('')
      : '<em class="ml-hint">none</em>';
  h += `<div class="ml-sec"><strong>Numeric</strong> ` +
    `<label>Impute ${mlSelect('num_impute', [['median', 'median'], ['mean', 'mean'], ['most_frequent', 'most frequent'], ['constant', 'constant 0']], m.numeric.impute)}</label> ` +
    `<label>Scaler ${mlSelect('num_scaler', [['standard', 'StandardScaler'], ['minmax', 'MinMaxScaler'], ['none', 'none']], m.numeric.scaler)}</label>` +
    `<div class="ml-cols">${colChecks(nums, 'num', m.numeric_columns)}</div>` +
    `<div class="ml-sub">Log transform: ${nums.length ? nums.map((c) => `<label class="ml-check"><input type="checkbox" data-ml-coll="log" value="${mlEsc(c)}"${(m.numeric.log_columns || []).includes(c) ? ' checked' : ''}> log(${mlEsc(c)})</label>`).join('') : '<em class="ml-hint">none</em>'}</div></div>`;
  h += `<div class="ml-sec"><strong>Categorical</strong> ` +
    `<label>Impute ${mlSelect('cat_impute', [['most_frequent', 'most frequent'], ['constant', '"missing"']], m.categorical.impute)}</label> ` +
    `<label>Encoder ${mlSelect('cat_encoder', [['onehot', 'OneHotEncoder'], ['ordinal', 'OrdinalEncoder']], m.categorical.encoder)}</label>` +
    `<div class="ml-cols">${colChecks(cats, 'cat', m.categorical_columns)}</div></div>`;
  h += `<div class="ml-summary">${mlEsc(mlPipelineSummary(cell))}</div>`;
  h += `<div class="ml-hint">Scalers and encoders fit on training data only — never on the test set.</div>`;
  return h;
}

function mlReadColl(cellId, field) {
  const el = document.getElementById(`cell-${cellId}`);
  const out = [];
  if (el) el.querySelectorAll(`[data-ml-coll="${field}"]`).forEach((box) => {
    if (box.checked) out.push(box.value);
  });
  return out;
}

function mlSavePreprocessConfig(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const f = mlReadForm(cellId);
  cell.ml.dataset_cell_id = String(f.dataset_cell_id || '');
  cell.ml.numeric_columns = mlReadColl(cellId, 'num');
  cell.ml.categorical_columns = mlReadColl(cellId, 'cat');
  cell.ml.numeric = {
    impute: f.num_impute || 'median', impute_value: 0,
    scaler: f.num_scaler || 'standard',
    log_columns: mlReadColl(cellId, 'log'),
  };
  cell.ml.categorical = {
    impute: f.cat_impute || 'most_frequent', impute_value: 'missing',
    encoder: f.cat_encoder || 'onehot',
  };
  mlSaveAndRender();
}

function mlPipelineSummary(cell) {
  const m = cell.ml;
  const parts = [];
  if ((m.numeric_columns || []).length) {
    parts.push(`numeric [${m.numeric_columns.join(', ')}]: impute=${m.numeric.impute}, scaler=${m.numeric.scaler}` +
      ((m.numeric.log_columns || []).length ? `, log(${(m.numeric.log_columns || []).join(', ')})` : ''));
  }
  if ((m.categorical_columns || []).length) {
    parts.push(`categorical [${m.categorical_columns.join(', ')}]: impute=${m.categorical.impute}, encoder=${m.categorical.encoder}`);
  }
  return parts.length ? parts.join('; ') : 'passthrough (all columns, no transformations)';
}

function mlPreprocessPayload(cell) {
  const m = cell.ml;
  return {
    numeric_columns: m.numeric_columns || [], categorical_columns: m.categorical_columns || [],
    numeric: m.numeric, categorical: m.categorical,
  };
}

// ── Predictor cell ──
function mlPredictorBody(cell) {
  const m = cell.ml;
  const ds = mlFindCell(m.dataset_cell_id);
  const task = m.task === 'auto' ? mlGuessTask(ds) : m.task;
  const fams = task === 'regression' ? ML_REGRESSION_MODELS : ML_CLASSIFICATION_MODELS;
  if (m.model_family && !fams.includes(m.model_family)) m.model_family = '';
  const family = m.model_family || fams[0];
  const schema = ML_PARAM_SCHEMAS[family] || [];
  let h = `<div class="ml-row"><label>Dataset ${mlDatasetOptions(m.dataset_cell_id)}</label>` +
    `<label>Split ${mlSelect('split_cell_id', [['', '(default 70/15/15)']].concat(mlSplitCells()), m.split_cell_id)}</label>` +
    `<label>Preprocess ${mlSelect('preprocess_cell_id', [['', '(default pipeline)']].concat(mlPreprocessCells()), m.preprocess_cell_id)}</label></div>`;
  h += `<div class="ml-row"><label>Task ${mlSelect('task', [['auto', `auto (${task})`], ['classification', 'classification'], ['regression', 'regression']], m.task)}</label>` +
    `<label>Model ${mlSelect('model_family', fams.map((f) => [f, f]), family)}</label>` +
    `<label>Seed ${mlTextInput('seed', m.seed, '', '90px')}</label>` +
    `<button class="cell-btn" onclick="mlSavePredictorConfig(${cell.id})">Save</button></div>`;
  h += `<div class="ml-sec"><strong>Parameters</strong><div class="ml-params">` +
    (schema.length ? schema.map((p) => {
      const cur = (m.model_params || {})[p.name] !== undefined ? (m.model_params || {})[p.name] : p.def;
      if (p.type === 'choice') return `<label>${mlEsc(p.label)} ${mlSelect('param_' + p.name, p.options.map((o) => [o === null ? '' : String(o), o === null ? '(none)' : String(o)]), cur === null ? '' : String(cur))}</label>`;
      return `<label>${mlEsc(p.label)} ${mlTextInput('param_' + p.name, cur, '', '110px')}</label>`;
    }).join('') : '<em class="ml-hint">No hyperparameters.</em>') + `</div></div>`;
  const cv = m.cv || {};
  h += `<div class="ml-sec"><strong>Cross-validation</strong> ` +
    `<label>Method ${mlSelect('cv_method', [['auto', 'auto'], ['none', 'off'], ['kfold', 'K-Fold'], ['stratified', 'Stratified K-Fold'], ['group', 'GroupKFold'], ['time', 'TimeSeriesSplit']], cv.method || 'auto')}</label> ` +
    `<label>Folds ${mlTextInput('cv_folds', cv.folds == null ? 5 : cv.folds, '', '70px')}</label> ` +
    `<label>Scoring ${mlSelect('cv_scoring', [['', '(default)']].concat((ML_SCORING[task] || []).map((s) => [s, s])), cv.scoring || '')}</label></div>`;
  const s = m.search || {};
  h += `<div class="ml-sec"><strong>Hyperparameter search</strong> ` +
    `<label>Method ${mlSelect('search_method', [['none', 'off'], ['grid', 'Grid Search'], ['random', 'Randomized Search']], s.method || 'none')}</label> ` +
    `<label>Trials ${mlTextInput('search_trials', s.trials == null ? 10 : s.trials, '', '70px')}</label> ` +
    `<label>Seed ${mlTextInput('search_seed', s.seed == null ? 42 : s.seed, '', '90px')}</label>` +
    `<div class="ml-hint">Grids, one per line — <code>name: v1, v2</code> or <code>name: low-high:steps</code> (add <code>:int</code> for integers).</div>` +
    `<textarea class="ml-textarea" data-ml-field="search_params_text" rows="3" placeholder="n_estimators: 50, 100, 200&#10;max_depth: 3-8:6:int">${mlEsc(s.params_text || '')}</textarea></div>`;
  if (m.experiment_id) {
    h += `<div class="ml-summary">Experiment <strong>${mlEsc(m.experiment_id)}</strong> · status <strong>${mlEsc(m.status)}</strong>` +
      (m.last_metrics ? ` · ${mlEsc(mlMetricLine(task, m.last_metrics))}` : '') +
      ` <button class="cell-btn" onclick="mlOpenExperiment('${mlEsc(m.experiment_id)}')">Open</button></div>`;
  } else {
    h += `<div class="ml-hint">Save, then Train. CPU-only sklearn — no GPU needed.</div>`;
  }
  return h;
}

function mlSplitCells() {
  const nb = mlActiveNotebook();
  if (!nb) return [];
  return (nb.cells || []).filter((c) => c.ml && c.ml.kind === 'split').map((c) => [String(c.id), `Cell ${c.id} (${c.ml.strategy || 'split'})`]);
}

function mlPreprocessCells() {
  const nb = mlActiveNotebook();
  if (!nb) return [];
  return (nb.cells || []).filter((c) => c.ml && c.ml.kind === 'preprocess').map((c) => [String(c.id), `Cell ${c.id} (preprocess)`]);
}

function mlMetricLine(task, metrics) {
  if (!metrics) return 'no metrics';
  for (const part of ['test', 'validation', 'train']) {
    const m = metrics[part];
    if (!m) continue;
    if (task === 'classification') return `${part} acc ${(m.accuracy || 0).toFixed(3)} · f1 ${(m.f1_macro || 0).toFixed(3)}`;
    return `${part} R² ${(m.r2 == null ? NaN : m.r2).toFixed ? (m.r2 == null ? '—' : m.r2.toFixed(3)) : '—'}`;
  }
  return 'no metrics';
}

function mlParseSearchText(text) {
  const params = {};
  String(text || '').split('\n').forEach((line) => {
    const t = line.trim();
    if (!t || t.startsWith('#')) return;
    const i = t.indexOf(':');
    if (i === -1) throw new Error(`Bad search line (need "name: values"): ${t}`);
    const name = t.slice(0, i).trim();
    const rest = t.slice(i + 1).trim();
    if (/^(-?[\d.]+)-(-?[\d.]+):(\d+)(:int)?$/.test(rest)) {
      const mm = rest.match(/^(-?[\d.]+)-(-?[\d.]+):(\d+)(:int)?$/);
      const lo = parseFloat(mm[1]); const hi = parseFloat(mm[2]);
      const steps = Math.max(2, parseInt(mm[3], 10));
      const vals = [];
      for (let k = 0; k < steps; k++) vals.push(lo + ((hi - lo) * k) / (steps - 1));
      params[name] = mm[4] ? [...new Set(vals.map((v) => Math.round(v)))] : vals;
    } else {
      params[name] = rest.split(',').map((s) => {
        const v = s.trim();
        if (/^-?\d+$/.test(v)) return parseInt(v, 10);
        if (/^-?\d*\.\d+$/.test(v)) return parseFloat(v);
        if (v.toLowerCase() === 'none' || v.toLowerCase() === 'null') return null;
        return v;
      });
    }
  });
  return params;
}

function mlCoerceParams(family, raw) {
  const schema = ML_PARAM_SCHEMAS[family] || [];
  const out = {};
  schema.forEach((p) => {
    const key = 'param_' + p.name;
    if (raw[key] === undefined || raw[key] === '') return;
    if (p.type === 'int') {
      const n = parseInt(raw[key], 10);
      if (!Number.isFinite(n)) throw new Error(`Parameter ${p.name} must be an integer.`);
      out[p.name] = n;
    } else if (p.type === 'float') {
      const n = parseFloat(raw[key]);
      if (!Number.isFinite(n)) throw new Error(`Parameter ${p.name} must be a number.`);
      out[p.name] = n;
    } else if (p.type === 'choice') {
      out[p.name] = raw[key] === '' ? null : raw[key];
    }
  });
  return out;
}

function mlSavePredictorConfig(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return null;
  const f = mlReadForm(cellId);
  const ds = mlFindCell(f.dataset_cell_id);
  const task = f.task === 'auto' ? mlGuessTask(ds) : f.task;
  const fams = task === 'regression' ? ML_REGRESSION_MODELS : ML_CLASSIFICATION_MODELS;
  const family = fams.includes(f.model_family) ? f.model_family : fams[0];
  cell.ml.dataset_cell_id = String(f.dataset_cell_id || '');
  cell.ml.split_cell_id = String(f.split_cell_id || '');
  cell.ml.preprocess_cell_id = String(f.preprocess_cell_id || '');
  cell.ml.task = f.task || 'auto';
  cell.ml.model_family = family;
  cell.ml.model_params = mlCoerceParams(family, f);
  cell.ml.seed = f.seed === '' ? 42 : Number(f.seed);
  cell.ml.cv = {
    method: f.cv_method || 'auto',
    folds: f.cv_folds === '' ? 5 : Number(f.cv_folds),
    scoring: f.cv_scoring || null,
  };
  cell.ml.search = {
    method: f.search_method || 'none',
    params_text: f.search_params_text || '',
    trials: f.search_trials === '' ? 10 : Number(f.search_trials),
    seed: f.search_seed === '' ? 42 : Number(f.search_seed),
    cv_folds: 3,
    scoring: null,
  };
  mlSaveAndRender();
  return cell;
}

function mlBuildExperimentConfig(cell) {
  const ds = mlFindCell(cell.ml.dataset_cell_id);
  if (!ds || !ds.ml.workspace_id) throw new Error('Choose a loaded Dataset Cell first.');
  if (!ds.ml.target_column) throw new Error('Choose a target column in the Dataset Cell.');
  const splitCell = cell.ml.split_cell_id ? mlFindCell(cell.ml.split_cell_id) : null;
  const preCell = cell.ml.preprocess_cell_id ? mlFindCell(cell.ml.preprocess_cell_id) : null;
  const task = cell.ml.task === 'auto' ? mlGuessTask(ds) : cell.ml.task;
  const feats = ds.ml.feature_columns && ds.ml.feature_columns.length
    ? ds.ml.feature_columns.filter((c) => c !== ds.ml.target_column) : null;
  let search = null;
  if ((cell.ml.search || {}).method && cell.ml.search.method !== 'none') {
    search = {
      method: cell.ml.search.method,
      params: mlParseSearchText(cell.ml.search.params_text),
      trials: cell.ml.search.trials, seed: cell.ml.search.seed,
      cv_folds: cell.ml.search.cv_folds || 3, scoring: cell.ml.search.scoring || null,
    };
  }
  return {
    workspace_id: ds.ml.workspace_id,
    task,
    target_column: ds.ml.target_column,
    feature_columns: feats,
    dataset_view: ds.ml.view || {},
    split: splitCell ? mlSplitPayload(splitCell) : {
      strategy: 'train_val_test', train_frac: 0.7, val_frac: 0.15, test_frac: 0.15,
      seed: cell.ml.seed || 42, shuffle: true, stratify: task === 'classification',
      group_column: null },
    preprocessing: preCell ? mlPreprocessPayload(preCell) : {},
    model: { family: cell.ml.model_family, params: cell.ml.model_params || {} },
    seed: cell.ml.seed || 42,
    cv: (!cell.ml.cv || cell.ml.cv.method === 'none') ? null : {
      method: cell.ml.cv.method, folds: cell.ml.cv.folds || 5,
      scoring: cell.ml.cv.scoring || null, seed: cell.ml.seed || 42 },
    search,
    name: `Predictor cell ${cell.id}`,
  };
}

async function mlTrainPredictor(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  if (cell.ml.status === 'working') return;
  try {
    mlSavePredictorConfig(cellId);
    const config = mlBuildExperimentConfig(cell);
    mlSetStatus(cell, 'working', '');
    const data = await mlApi('/api/ml/experiments', { method: 'POST', body: config });
    cell.ml.experiment_id = data.experiment_id;
    mlSaveAndRender();
    await mlPollExperiment(cellId, data.experiment_id);
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

async function mlPollExperiment(cellId, expId, tries) {
  tries = tries === undefined ? 180 : tries;
  const cell = mlFindCell(cellId);
  for (let i = 0; i < tries; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const cur = mlFindCell(cellId);
    if (!cur) return;
    let rec;
    try {
      rec = await mlApi(`/api/ml/experiments/${expId}`);
    } catch (e) {
      if (i === tries - 1 && cur) mlSetStatus(cur, 'error', e.message);
      return;
    }
    if (rec.status === 'finished') {
      cur.ml.status = 'ready';
      cur.ml.last_metrics = rec.metrics;
      cur.ml.error = '';
      mlSaveAndRender();
      return;
    }
    if (rec.status === 'failed') {
      cur.ml.status = 'error';
      cur.ml.error = rec.error || 'Experiment failed.';
      mlSaveAndRender();
      return;
    }
  }
  const cur = mlFindCell(cellId);
  if (cur && cur.ml.status === 'working') {
    cur.ml.error = 'Still running — check Experiments for status.';
    mlSaveAndRender();
  }
}

// ── Evaluate cell ──
function mlEvaluateBody(cell) {
  const m = cell.ml;
  let h = `<div class="ml-row"><label>Predictor ${mlSelect('predictor_cell_id',
    [['', '(choose predictor cell)']].concat(mlPredictorCells(true).map((c) =>
      [String(c.id), `Cell ${c.id} — ${c.ml.model_family || '?'} (${c.ml.experiment_id || 'no run'})`])),
    m.predictor_cell_id)}</label>` +
    `<button class="cell-btn" onclick="mlSaveEvaluateConfig(${cell.id})">Save</button>` +
    `<button class="cell-btn run" onclick="mlRunEvaluate(${cell.id})">Evaluate</button></div>`;
  if (m.result) h += mlMetricsHtml(m.result, cell.id);
  else h += `<div class="ml-hint">Train a Predictor Cell, then Evaluate here. Test metrics are reported — never used for tuning.</div>`;
  return h;
}

function mlSaveEvaluateConfig(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  cell.ml.predictor_cell_id = String(mlReadForm(cellId).predictor_cell_id || '');
  mlSaveAndRender();
}

async function mlRunEvaluate(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  mlSaveEvaluateConfig(cellId);
  const pc = mlFindCell(cell.ml.predictor_cell_id);
  if (!pc || !pc.ml.experiment_id) { mlSetStatus(cell, 'error', 'Choose a trained Predictor Cell.'); return; }
  mlSetStatus(cell, 'working', '');
  try {
    const rec = await mlApi(`/api/ml/experiments/${pc.ml.experiment_id}`);
    if (rec.status !== 'finished') throw new Error(`Experiment is ${rec.status}, not finished.`);
    cell.ml.result = { task: rec.task, metrics: rec.metrics, experiment_id: rec.experiment_id,
      model_family: rec.model_family };
    mlSetStatus(cell, 'ready', '');
    mlRenderEvaluateCharts(cell);
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

function mlMetricsHtml(result, cellId) {
  const task = result.task;
  const metrics = result.metrics || {};
  const cid = cellId == null ? 'SEL' : cellId;
  let h = `<div class="ml-metrics"><h4>Metrics <em>(${(result.model_family || '')} · ${mlEsc(result.experiment_id || '')})</em></h4>`;
  ['train', 'validation', 'test'].forEach((part) => {
    const m = metrics[part];
    if (!m) return;
    h += `<div class="ml-sec"><strong>${part.toUpperCase()}</strong><div class="ml-table-wrap"><table class="ml-table"><tbody>`;
    if (task === 'classification') {
      h += `<tr><td>Accuracy</td><td>${mlNum(m.accuracy)}</td></tr>` +
        `<tr><td>Balanced accuracy</td><td>${mlNum(m.balanced_accuracy)}</td></tr>` +
        `<tr><td>F1 (macro)</td><td>${mlNum((m.macro || {}).f1)}</td></tr>` +
        `<tr><td>F1 (weighted)</td><td>${mlNum((m.weighted || {}).f1)}</td></tr>` +
        `<tr><td>ROC-AUC</td><td>${m.roc_auc == null ? 'n/a' : mlNum(m.roc_auc)}</td></tr>` +
        `<tr><td>PR-AUC</td><td>${m.pr_auc == null ? 'n/a' : mlNum(m.pr_auc)}</td></tr>` +
        `<tr><td>Log loss</td><td>${m.log_loss == null ? 'n/a' : mlNum(m.log_loss)}</td></tr>`;
    } else {
      h += `<tr><td>MAE</td><td>${mlNum(m.mae)}</td></tr><tr><td>MSE</td><td>${mlNum(m.mse)}</td></tr>` +
        `<tr><td>RMSE</td><td>${mlNum(m.rmse)}</td></tr><tr><td>R²</td><td>${m.r2 == null ? 'n/a' : mlNum(m.r2)}</td></tr>`;
    }
    h += `</tbody></table></div></div>`;
  });
  if (metrics.cv) {
    h += `<div class="ml-sec"><strong>CROSS-VALIDATION</strong> (${mlEsc(metrics.cv.method)}, ${metrics.cv.folds}-fold, ${mlEsc(metrics.cv.scoring)}) — ` +
      `mean <strong>${mlNum(metrics.cv.mean)}</strong> ± ${mlNum(metrics.cv.std)}` +
      `<div class="ml-table-wrap"><table class="ml-table"><thead><tr><th>fold</th><th>score</th></tr></thead><tbody>` +
      (metrics.cv.scores || []).map((s, i) => `<tr><td>${i + 1}</td><td>${mlNum(s)}</td></tr>`).join('') +
      `</tbody></table></div></div>`;
  }
  const testM = metrics.test;
  if (task === 'classification' && testM) {
    if (testM.confusion_matrix && testM.labels) {
      h += `<div class="ml-sec"><strong>Confusion matrix (test)</strong><div class="ml-table-wrap"><table class="ml-table"><thead><tr><th></th>` +
        testM.labels.map((l) => `<th>${mlEsc(l)}</th>`).join('') + `</tr></thead><tbody>` +
        testM.confusion_matrix.map((row, i) => `<tr><th>${mlEsc(testM.labels[i])}</th>` +
          row.map((v) => `<td>${v}</td>`).join('') + `</tr>`).join('') + `</tbody></table></div></div>`;
    }
    if (testM.per_class) {
      h += `<div class="ml-sec"><strong>Per-class (test)</strong><div class="ml-table-wrap"><table class="ml-table">` +
        `<thead><tr><th>class</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr></thead><tbody>` +
        testM.per_class.map((r) => `<tr><td>${mlEsc(r.label)}</td><td>${mlNum(r.precision)}</td><td>${mlNum(r.recall)}</td><td>${mlNum(r.f1)}</td><td>${r.support}</td></tr>`).join('') +
        `</tbody></table></div></div>`;
    }
    h += `<div class="ml-sec"><strong>Curves (test)</strong>` +
      `<canvas id="ml-roc-${cid}" height="130"></canvas><canvas id="ml-pr-${cid}" height="130"></canvas></div>`;
  }
  if (metrics.importance && metrics.importance.features) {
    h += `<div class="ml-sec"><strong>Feature importance (${mlEsc(metrics.importance.method || '')})</strong>` +
      `<canvas id="ml-imp-${cid}" height="140"></canvas></div>`;
  }
  h += `</div>`;
  return h;
}

function mlRenderEvaluateCharts(cell) {
  try {
    const m = (cell.ml.result && cell.ml.result.metrics && cell.ml.result.metrics.test) || null;
    const id = cell.id;
    if (!m) return;
    if (m.roc_curve) {
      mlChart(`ml-roc-${id}`, { type: 'line',
        data: { labels: m.roc_curve.fpr, datasets: [{ label: 'ROC', data: m.roc_curve.fpr.map((x, i) => ({ x, y: m.roc_curve.tpr[i] })), borderColor: '#4a6fa5', pointRadius: 0 }, { label: 'chance', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#8a8a8a', borderDash: [4, 4], pointRadius: 0 }] },
        options: { plugins: { legend: { display: true } }, scales: { x: { min: 0, max: 1 }, y: { min: 0, max: 1 } } } });
    }
    if (m.pr_curve) {
      mlChart(`ml-pr-${id}`, { type: 'line',
        data: { labels: m.pr_curve.recall, datasets: [{ label: 'PR', data: m.pr_curve.recall.map((x, i) => ({ x, y: m.pr_curve.precision[i] })), borderColor: '#7ca673', pointRadius: 0 }] },
        options: { plugins: { legend: { display: true } }, scales: { x: { min: 0, max: 1 }, y: { min: 0, max: 1 } } } });
    }
    const imp = cell.ml.result.metrics.importance;
    if (imp && imp.features) {
      const top = imp.features.slice(0, 12);
      mlChart(`ml-imp-${id}`, { type: 'bar',
        data: { labels: top.map((r) => r.feature), datasets: [{ data: top.map((r) => r.value), backgroundColor: '#4a6fa5' }] },
        options: { indexAxis: 'y', plugins: { legend: { display: false } } } });
    }
  } catch (_) {}
}

// ── Predict cell ──
function mlPredictBody(cell) {
  const m = cell.ml;
  let h = `<div class="ml-row"><label>Predictor ${mlSelect('predictor_cell_id',
    [['', '(choose predictor cell)']].concat(mlPredictorCells(true).map((c) =>
      [String(c.id), `Cell ${c.id} — ${c.ml.model_family || '?'} (${c.ml.experiment_id || 'no run'})`])),
    m.predictor_cell_id)}</label>` +
    `<label>Input ${mlSelect('mode', [['rows', 'dataset rows'], ['csv', 'paste CSV'], ['upload', 'upload CSV']], m.mode)}</label>` +
    `<button class="cell-btn" onclick="mlSavePredictConfig(${cell.id})">Save</button>` +
    `<button class="cell-btn run" onclick="mlRunPredict(${cell.id})">Predict</button></div>`;
  if (m.mode === 'csv') {
    h += `<textarea class="ml-textarea" data-ml-field="csv_text" rows="5" placeholder="col_a,col_b&#10;1,2&#10;3,4">${mlEsc(m.csv_text || '')}</textarea>`;
  } else if (m.mode === 'upload') {
    h += `<div class="ml-row"><input type="file" data-ml-file="${cell.id}" accept=".csv"></div>`;
  } else {
    h += `<div class="ml-row"><label>Max rows ${mlTextInput('max_rows', m.max_rows, '', '90px')}</label>` +
      `<span class="ml-hint">Uses the predictor's dataset, first N rows.</span></div>`;
  }
  if (m.result && m.result.rows) {
    const rows = m.result.rows.slice(0, 50);
    const cols = rows.length ? Object.keys(rows[0]) : [];
    h += `<div class="ml-summary">${m.result.count} predictions</div>` +
      `<div class="ml-table-wrap"><table class="ml-table"><thead><tr>` +
      cols.map((c) => `<th>${mlEsc(c)}</th>`).join('') + `</tr></thead><tbody>` +
      rows.map((r) => `<tr>` + cols.map((c) => {
        const v = r[c];
        const t = (v !== null && typeof v === 'object') ? JSON.stringify(v) : String(v == null ? '' : v);
        return `<td>${mlEsc(t.slice(0, 60))}</td>`;
      }).join('') + `</tr>`).join('') + `</tbody></table></div>` +
      (m.result.count > 50 ? `<div class="ml-hint">Showing first 50 of ${m.result.count}. Export for all.</div>` : '') +
      `<div class="ml-row"><button class="cell-btn" onclick="mlExportPredictions(${cell.id})">Export CSV</button></div>`;
  } else if (!m.result) {
    h += `<div class="ml-hint">Predictions use the exact fitted pipeline stored with the model — no manual preprocessing.</div>`;
  }
  return h;
}

function mlSavePredictConfig(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  const f = mlReadForm(cellId);
  cell.ml.predictor_cell_id = String(f.predictor_cell_id || '');
  cell.ml.mode = f.mode || 'rows';
  cell.ml.csv_text = f.csv_text || cell.ml.csv_text || '';
  cell.ml.max_rows = f.max_rows === '' ? 500 : Number(f.max_rows) || 500;
  mlSaveAndRender();
}

async function mlRunPredict(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell) return;
  mlSavePredictConfig(cellId);
  const pc = mlFindCell(cell.ml.predictor_cell_id);
  if (!pc || !pc.ml.experiment_id) { mlSetStatus(cell, 'error', 'Choose a trained Predictor Cell.'); return; }
  mlSetStatus(cell, 'working', '');
  try {
    let body = { experiment_id: pc.ml.experiment_id };
    if (cell.ml.mode === 'csv') {
      const f = mlReadForm(cellId);
      body.csv = (f.csv_text !== undefined ? f.csv_text : cell.ml.csv_text) || '';
    } else if (cell.ml.mode === 'upload') {
      const input = document.querySelector(`#cell-${cellId} [data-ml-file]`);
      const file = input && input.files && input.files[0];
      if (!file) throw new Error('Choose a CSV file first.');
      body.csv = await file.text();
    } else {
      const ds = mlFindCell(pc.ml.dataset_cell_id);
      if (!ds || !ds.ml.workspace_id) throw new Error('Predictor dataset is unavailable.');
      body.workspace_id = ds.ml.workspace_id;
      body.row_filter = [];
      body.max_rows = cell.ml.max_rows || 500;
    }
    const data = await mlApi('/api/ml/predict', { method: 'POST', body });
    cell.ml.result = data;
    mlSetStatus(cell, 'ready', '');
  } catch (e) {
    mlSetStatus(cell, 'error', e.message);
  }
}

function mlExportPredictions(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell || !cell.ml.result || !cell.ml.result.rows) return;
  const rows = cell.ml.result.rows;
  const cols = Object.keys(rows[0] || {});
  const q = (v) => {
    if (v !== null && typeof v === 'object') v = JSON.stringify(v);
    const s = String(v == null ? '' : v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = cols.map(q).join(',') + '\n' + rows.map((r) => cols.map((c) => q(r[c])).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `predictions_${cell.ml.predictor_cell_id || 'exp'}.csv`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}

// ── Cell registry + run dispatch ──
const ML_CELL_BUILDERS = {
  dataset: (cell) => mlDatasetBody(cell),
  split: (cell) => mlSplitBody(cell),
  preprocess: (cell) => mlPreprocessBody(cell),
  predictor: (cell) => mlPredictorBody(cell),
  evaluate: (cell) => mlEvaluateBody(cell),
  predict: (cell) => mlPredictBody(cell),
};

function buildMlCellElement(cell) {
  const div = document.createElement('div');
  div.className = 'notebook-cell ml-cell';
  div.id = `cell-${cell.id}`;
  const label = ML_CELL_LABELS[(cell.ml || {}).kind] || 'ML Cell';
  const runLabel = cell.ml.kind === 'predictor' ? '▶ Train' :
    (cell.ml.kind === 'preprocess' ? '▶ Apply' : '▶ Run');
  const header = document.createElement('div');
  header.className = 'cell-header';
  header.innerHTML = `<span class="cell-type">${label}</span>` +
    `<div class="cell-controls">${mlStatusChip(cell)}` +
    `<button class="cell-btn run" onclick="mlRunCell(${cell.id})">${runLabel}</button>` +
    `<button class="cell-btn delete" onclick="deleteCell(${cell.id})">Delete</button></div>`;
  div.appendChild(header);
  const body = document.createElement('div');
  body.className = 'cell-input ml-cell-body';
  try {
    body.innerHTML = (ML_CELL_BUILDERS[cell.ml.kind] || (() => '<div class="ml-hint">Unknown ML cell.</div>'))(cell);
  } catch (e) {
    body.innerHTML = `<div class="ml-error">Render error: ${mlEsc(e.message)}</div>`;
  }
  div.appendChild(body);
  const err = document.createElement('div');
  err.innerHTML = mlErrorHtml(cell);
  div.appendChild(err);
  return div;
}

async function mlRunCell(cellId) {
  const cell = mlFindCell(cellId);
  if (!cell || !cell.ml) return;
  if (cell.ml.status === 'working') return;
  const kind = cell.ml.kind;
  try {
    if (kind === 'dataset') {
      // Re-run = re-apply view + refresh (upload/HF already stored).
      if (!cell.ml.workspace_id) { mlSetStatus(cell, 'error', 'Load a dataset first.'); return; }
      mlSetStatus(cell, 'working', '');
      const data = await mlApi(`/api/ml/datasets/${cell.ml.workspace_id}`);
      cell.ml.profile = data.profile;
      mlSetStatus(cell, 'ready', '');
    } else if (kind === 'split') {
      await mlPreviewSplit(cellId);
    } else if (kind === 'preprocess') {
      mlSavePreprocessConfig(cellId);
      mlSetStatus(mlFindCell(cellId), 'ready', '');
    } else if (kind === 'predictor') {
      await mlTrainPredictor(cellId);
    } else if (kind === 'evaluate') {
      await mlRunEvaluate(cellId);
    } else if (kind === 'predict') {
      await mlRunPredict(cellId);
    }
  } catch (e) {
    const c = mlFindCell(cellId);
    if (c) mlSetStatus(c, 'error', e.message);
  }
}

// ── Experiment tracking panel ──
let mlPanelSelected = [];
let mlPanelDetail = null;

function mlEnsurePanel() {
  let panel = document.getElementById('ml-experiments-panel');
  if (panel) return panel;
  panel = document.createElement('div');
  panel.id = 'ml-experiments-panel';
  panel.style.display = 'none';
  document.body.appendChild(panel);
  return panel;
}

function toggleMlExperiments() {
  const panel = mlEnsurePanel();
  const open = panel.style.display === 'none';
  panel.style.display = open ? '' : 'none';
  if (open) mlRefreshExperiments();
}

function mlCloseExperiments() {
  const panel = document.getElementById('ml-experiments-panel');
  if (panel) panel.style.display = 'none';
}

async function mlRefreshExperiments() {
  const panel = mlEnsurePanel();
  panel.innerHTML = `<div class="ml-panel-head"><strong>Experiments</strong>` +
    `<button class="cell-btn" onclick="mlCloseExperiments()">Close</button></div>` +
    `<div class="ml-hint">Classical sklearn runs on this machine (CPU). HF Training Cells are separate.</div>` +
    `<div id="ml-exp-list"><em>Loading…</em></div><div id="ml-exp-detail"></div>`;
  try {
    const data = await mlApi('/api/ml/experiments');
    mlRenderExperimentList(data.experiments || []);
    if (mlPanelDetail) mlOpenExperiment(mlPanelDetail, true);
  } catch (e) {
    const list = document.getElementById('ml-exp-list');
    if (list) list.innerHTML = `<div class="ml-error">${mlEsc(e.message)}</div>`;
  }
}

function mlRenderExperimentList(exps) {
  const list = document.getElementById('ml-exp-list');
  if (!list) return;
  if (!exps.length) {
    list.innerHTML = `<div class="ml-hint">No experiments yet — train a Predictor Cell.</div>`;
    return;
  }
  list.innerHTML = `<div class="ml-table-wrap"><table class="ml-table"><thead><tr>` +
    `<th></th><th>experiment</th><th>model</th><th>task</th><th>status</th><th>metric</th><th>runtime</th><th></th></tr></thead><tbody>` +
    exps.map((e) => {
      const metric = mlMetricLine(e.task, e.metrics);
      const checked = mlPanelSelected.includes(e.experiment_id) ? ' checked' : '';
      return `<tr><td><input type="checkbox" data-ml-compare="${mlEsc(e.experiment_id)}"${checked}></td>` +
        `<td><strong>${mlEsc(e.name || e.experiment_id.slice(0, 12))}</strong><br><small>${mlEsc(e.experiment_id)}</small></td>` +
        `<td>${mlEsc(e.model_family || '—')}</td><td>${mlEsc(e.task || '')}</td>` +
        `<td>${mlEsc(e.status || '')}</td><td>${mlEsc(metric)}</td>` +
        `<td>${e.runtime_sec == null ? '—' : `${e.runtime_sec}s`}</td>` +
        `<td class="ml-rowbtns"><button class="cell-btn" onclick="mlOpenExperiment('${mlEsc(e.experiment_id)}')">Open</button> ` +
        `<button class="cell-btn" onclick="mlRerunExperiment('${mlEsc(e.experiment_id)}')">Rerun</button> ` +
        `<button class="cell-btn" onclick="mlCloneExperiment('${mlEsc(e.experiment_id)}')">Clone</button> ` +
        `<button class="cell-btn" onclick="mlRenameExperiment('${mlEsc(e.experiment_id)}')">Rename</button> ` +
        `<button class="cell-btn delete" onclick="mlDeleteExperiment('${mlEsc(e.experiment_id)}')">Delete</button></td></tr>`;
    }).join('') + `</tbody></table></div>` +
    `<div class="ml-row"><button class="cell-btn run" onclick="mlCompareSelected()">Compare selected</button></div>` +
    `<div id="ml-compare"></div>`;
  list.querySelectorAll('[data-ml-compare]').forEach((box) => {
    box.addEventListener('change', () => {
      const id = box.getAttribute('data-ml-compare');
      if (box.checked && !mlPanelSelected.includes(id)) mlPanelSelected.push(id);
      if (!box.checked) mlPanelSelected = mlPanelSelected.filter((x) => x !== id);
    });
  });
}

async function mlOpenExperiment(expId, keep) {
  mlPanelDetail = expId;
  const detail = document.getElementById('ml-exp-detail');
  if (!detail) return;
  detail.innerHTML = `<em>Loading…</em>`;
  try {
    const rec = await mlApi(`/api/ml/experiments/${expId}`);
    const cfg = rec.config || {};
    detail.innerHTML = `<div class="ml-sec"><strong>${mlEsc(rec.name || expId)}</strong> ` +
      `<em>${mlEsc(rec.status || '')} · ${mlEsc(rec.task || '')} · ${rec.runtime_sec == null ? '—' : `${rec.runtime_sec}s`}</em>` +
      (rec.error ? `<div class="ml-error">${mlEsc(rec.error)}</div>` : '') +
      `<div class="ml-table-wrap"><table class="ml-table"><tbody>` +
      `<tr><td>dataset</td><td>${mlEsc(cfg.workspace_id || '')}</td></tr>` +
      `<tr><td>target</td><td>${mlEsc(cfg.target_column || '')}</td></tr>` +
      `<tr><td>features</td><td>${mlEsc((cfg.feature_columns || ['(all non-target)']).join(', '))}</td></tr>` +
      `<tr><td>split</td><td>${mlEsc(JSON.stringify(cfg.split || {}))}</td></tr>` +
      `<tr><td>model</td><td>${mlEsc((cfg.model || {}).family || '')} ${mlEsc(JSON.stringify((cfg.model || {}).params || {}))}</td></tr>` +
      `<tr><td>seed</td><td>${mlEsc(cfg.seed)}</td></tr>` +
      `</tbody></table></div>` +
      (rec.metrics ? mlMetricsHtml({ task: rec.task, metrics: rec.metrics, experiment_id: expId, model_family: rec.model_family }, 'panel') : '') +
      `</div>`;
    if (rec.metrics) mlRenderPanelCharts(rec);
  } catch (e) {
    detail.innerHTML = `<div class="ml-error">${mlEsc(e.message)}</div>`;
  }
  void keep;
}

function mlRenderPanelCharts(rec) {
  try {
    const m = (rec.metrics && rec.metrics.test) || null;
    if (!m) return;
    if (m.roc_curve) {
      mlChart('ml-roc-panel', { type: 'line',
        data: { labels: m.roc_curve.fpr, datasets: [{ label: 'ROC', data: m.roc_curve.fpr.map((x, i) => ({ x, y: m.roc_curve.tpr[i] })), borderColor: '#4a6fa5', pointRadius: 0 }] },
        options: { plugins: { legend: { display: true } } } });
    }
    if (m.pr_curve) {
      mlChart('ml-pr-panel', { type: 'line',
        data: { labels: m.pr_curve.recall, datasets: [{ label: 'PR', data: m.pr_curve.recall.map((x, i) => ({ x, y: m.pr_curve.precision[i] })), borderColor: '#7ca673', pointRadius: 0 }] },
        options: { plugins: { legend: { display: true } } } });
    }
  } catch (_) {}
}

async function mlRerunExperiment(expId) {
  try {
    const data = await mlApi(`/api/ml/experiments/${expId}/rerun`, { method: 'POST', body: {} });
    mlPanelDetail = data.experiment_id;
    await mlRefreshExperiments();
  } catch (e) {
    alert(`Rerun failed: ${e.message}`);
  }
}

async function mlCloneExperiment(expId) {
  try {
    const rec = await mlApi(`/api/ml/experiments/${expId}`);
    const nb = mlActiveNotebook();
    if (!nb) { alert('Open a notebook first.'); return; }
    if (typeof addCell !== 'function') return;
    addCell('predictor');
    const cell = nb.cells[nb.cells.length - 1];
    const cfg = rec.config || {};
    cell.ml.task = cfg.task || 'auto';
    cell.ml.model_family = (cfg.model || {}).family || '';
    cell.ml.model_params = { ...((cfg.model || {}).params || {}) };
    cell.ml.seed = cfg.seed == null ? 42 : cfg.seed;
    if (typeof renderNotebookEditor === 'function') renderNotebookEditor();
    if (typeof scrollToCell === 'function') scrollToCell(cell.id);
    mlCloseExperiments();
  } catch (e) {
    alert(`Clone failed: ${e.message}`);
  }
}

async function mlRenameExperiment(expId) {
  const name = typeof prompt === 'function' ? prompt('Experiment name:', '') : '';
  if (name == null) return;
  try {
    await mlApi(`/api/ml/experiments/${expId}`, { method: 'PATCH', body: { name: String(name).slice(0, 120) } });
    await mlRefreshExperiments();
  } catch (e) {
    alert(`Rename failed: ${e.message}`);
  }
}

async function mlDeleteExperiment(expId) {
  if (typeof confirm === 'function' && !confirm(`Delete experiment ${expId}? Artifacts are removed.`)) return;
  try {
    await mlApi(`/api/ml/experiments/${expId}`, { method: 'DELETE' });
    mlPanelSelected = mlPanelSelected.filter((x) => x !== expId);
    if (mlPanelDetail === expId) mlPanelDetail = null;
    await mlRefreshExperiments();
  } catch (e) {
    alert(`Delete failed: ${e.message}`);
  }
}

async function mlCompareSelected() {
  const box = document.getElementById('ml-compare');
  if (!box) return;
  if (mlPanelSelected.length < 2) {
    box.innerHTML = `<div class="ml-hint">Select at least two experiments to compare (no ranking — factual side-by-side only).</div>`;
    return;
  }
  box.innerHTML = `<em>Loading…</em>`;
  try {
    const data = await mlApi(`/api/ml/experiments/compare?ids=${mlPanelSelected.map(encodeURIComponent).join(',')}`);
    const rows = data.experiments || [];
    const metricKeys = (task, metrics, part) => {
      const m = (metrics || {})[part];
      if (!m) return '—';
      if (task === 'classification') return `acc ${mlNum(m.accuracy)} · f1 ${mlNum((m.macro || {}).f1)}`;
      return `R² ${m.r2 == null ? '—' : mlNum(m.r2)} · RMSE ${mlNum(m.rmse)}`;
    };
    const fields = [
      ['experiment', (r) => `${r.name || r.experiment_id.slice(0, 12)}\n${r.experiment_id}`],
      ['status', (r) => r.status],
      ['dataset', (r) => r.dataset],
      ['target', (r) => r.target_column],
      ['features', (r) => (r.feature_columns || ['(all)']).join(', ')],
      ['split', (r) => JSON.stringify(r.split)],
      ['preprocessing', (r) => JSON.stringify(r.preprocessing)],
      ['model', (r) => r.model_family],
      ['hyperparameters', (r) => JSON.stringify(r.model_params)],
      ['seed', (r) => String(r.seed)],
      ['cv', (r) => JSON.stringify(r.cv)],
      ['train', (r) => metricKeys(r.task, r.metrics, 'train')],
      ['validation', (r) => metricKeys(r.task, r.metrics, 'validation')],
      ['test', (r) => metricKeys(r.task, r.metrics, 'test')],
      ['cv result', (r) => (r.metrics && r.metrics.cv) ? `${r.metrics.cv.mean} ± ${r.metrics.cv.std}` : '—'],
      ['runtime', (r) => (r.runtime_sec == null ? '—' : `${r.runtime_sec}s`)],
    ];
    box.innerHTML = `<div class="ml-table-wrap"><table class="ml-table"><thead><tr><th>field</th>` +
      rows.map((r) => `<th>${mlEsc(r.name || r.experiment_id.slice(0, 12))}</th>`).join('') +
      `</tr></thead><tbody>` +
      fields.map(([label, fn]) => `<tr><td><strong>${mlEsc(label)}</strong></td>` +
        rows.map((r) => `<td>${mlEsc(String(fn(r) == null ? '—' : fn(r)).slice(0, 300))}</td>`).join('') +
        `</tr>`).join('') + `</tbody></table></div>`;
  } catch (e) {
    box.innerHTML = `<div class="ml-error">${mlEsc(e.message)}</div>`;
  }
}



