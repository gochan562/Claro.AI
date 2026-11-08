'use strict';

/**
 * Storage Manager backend for Claro.AI.
 *
 * Inspects and cleans DISPOSABLE caches used by ML experiments. The managed
 * directories form a strict allowlist resolved from the same environment
 * variables the Hugging Face libraries use (HF_HOME / HF_HUB_CACHE /
 * HF_DATASETS_CACHE) — cleanup requests carry a CATEGORY, never a path, so
 * path traversal is impossible by construction.
 *
 * Managed categories:
 *   - model_cache   : <hub>/models--*          (downloaded weights; re-downloadable)
 *   - dataset_cache : <hub>/datasets--* + <HF_DATASETS_CACHE>/*  (download +
 *                     processed Arrow cache; workspaces in ml_workspaces/
 *                     hold self-contained copies and are NEVER touched)
 *   - temp_files    : *.incomplete / *.lock download artifacts inside the HF
 *                     caches + __pycache__ under the repo root
 *
 * Never touched (not in the allowlist): ml_workspaces/ (uploaded + ingested
 * source datasets), ml_experiments/ (results, metrics, joblib artifacts),
 * training_outputs/ (training artifacts), training_backend uploaded files.
 *
 * Protection model:
 *   - Running training jobs (queued/loading/training/evaluating) protect
 *     their config.model_id and base_model_id hub directories.
 *   - In-flight HF dataset ingestions protect the whole dataset category for
 *     that round (the datasets library cache is shared with the running
 *     subprocess; if we cannot prove a file is unused, we SKIP it).
 *   - Symlinks are never followed: lstat everywhere, symlinks are unlinked
 *     as entries (HF snapshots are symlinked into blobs/) — nothing outside
 *     an approved directory can be reached.
 *   - Concurrency: one cleanup per category at a time (per-category mutex);
 *     a second concurrent request fails with 409 instead of racing.
 */

const fs = require('fs');
const os = require('os');
const path = require('path');

const TRAINING_ACTIVE_STATUSES = new Set(['queued', 'loading', 'training', 'evaluating']);
const DEFAULT_WARN_PERCENT = 80;
const DEFAULT_MIN_AVAILABLE_BYTES = 2 * 1024 * 1024 * 1024; // 2 GB practical floor

const CATEGORY_NAMES = {
  models: 'Model cache',
  datasets: 'Dataset cache',
  temp: 'Temporary files',
};

// ── Directory resolution (strict allowlist) ─────────────────────────────────

function resolveCacheDirs(env) {
  const e = env || process.env;
  const hfRoot = path.resolve(e.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface'));
  const hub = path.resolve(e.HF_HUB_CACHE || path.join(hfRoot, 'hub'));
  const datasets = path.resolve(e.HF_DATASETS_CACHE || path.join(hfRoot, 'datasets'));
  const xet = path.resolve(e.HF_XET_CACHE || path.join(hfRoot, 'xet'));
  return { hfRoot, hub, datasets, xet, repoRoot: path.resolve(__dirname) };
}

// ── Safe filesystem primitives ──────────────────────────────────────────────

function _lstatSafe(p) {
  try { return fs.lstatSync(p); } catch (_) { return null; }
}

// Recursive size/count. lstat only — symlinked entries are counted as the
// link itself (0 bytes) and never traversed, so nothing outside the managed
// root is measured.
function dirStats(p, depth = 0) {
  const st = _lstatSafe(p);
  if (!st) return { bytes: 0, files: 0 };
  if (st.isSymbolicLink()) return { bytes: 0, files: 0 };
  if (st.isFile()) return { bytes: st.size, files: 1 };
  if (!st.isDirectory() || depth > 24) return { bytes: 0, files: 0 };
  let bytes = 0, files = 0;
  let entries = [];
  try { entries = fs.readdirSync(p); } catch (_) { return { bytes: 0, files: 0 }; }
  for (const name of entries) {
    const sub = dirStats(path.join(p, name), depth + 1);
    bytes += sub.bytes;
    files += sub.files;
  }
  return { bytes, files };
}

// Recursive delete. lstat-based: symlinks are UNLINKED (never traversed),
// so a symlink inside a cache cannot cause deletion outside it.
function rmrfSafe(p, result, depth = 0) {
  const st = _lstatSafe(p);
  if (!st) { result.missing += 1; return true; }
  if (st.isSymbolicLink()) {
    try { fs.unlinkSync(p); return true; }
    catch (e) { result.errors.push({ path: p, error: e.message }); return false; }
  }
  if (st.isFile()) {
    try { fs.unlinkSync(p); result.bytes += st.size; return true; }
    catch (e) { result.errors.push({ path: p, error: e.message }); return false; }
  }
  if (!st.isDirectory() || depth > 24) return true;
  let entries = [];
  try { entries = fs.readdirSync(p); } catch (e) {
    result.errors.push({ path: p, error: e.message }); return false;
  }
  let ok = true;
  for (const name of entries) {
    ok = rmrfSafe(path.join(p, name), result, depth + 1) && ok;
  }
  if (ok) {
    try { fs.rmdirSync(p); } catch (e) { result.errors.push({ path: p, error: e.message }); ok = false; }
  }
  return ok;
}

// ── Active-resource tracking (lazy imports avoid require cycles) ────────────

function activeTrainingRepos() {
  try {
    const tb = require('./training_backend');
    const repos = new Set();
    for (const j of tb.listJobs()) {
      if (!TRAINING_ACTIVE_STATUSES.has(j.status)) continue;
      const mid = j.config && (j.config.model_id || j.config.baseModelId);
      const bid = j.config && j.config.base_model_id;
      if (mid) repos.add(String(mid));
      if (bid) repos.add(String(bid));
    }
    return repos;
  } catch (_) {
    return new Set(); // if tracking is unavailable, per-item checks below still run
  }
}

function hasActiveIngest() {
  try {
    const ml = require('./ml_experiment_backend');
    return typeof ml.hasActiveIngest === 'function' ? ml.hasActiveIngest() : false;
  } catch (_) {
    return false;
  }
}

// repo id ("org/name") → hub directory name ("models--org--name")
function hubDirName(repoId, prefix) {
  return `${prefix}--${String(repoId).split('/')[0]}--${String(repoId).split('/')[1]}`;
}

// ── Enumeration of cleanup items (name + absolute path + reason) ────────────

function _listChildren(dir, filter) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  let entries = [];
  try { entries = fs.readdirSync(dir); } catch (_) { return out; }
  for (const name of entries.sort()) {
    const full = path.join(dir, name);
    const st = _lstatSafe(full);
    if (!st) continue;
    if (filter(name, full, st)) out.push({ name, path: full });
  }
  return out;
}

function enumerateItems(category, dirs, deps) {
  if (category === 'models') {
    const active = deps.activeTrainingRepos();
    return _listChildren(dirs.hub, (name) => name.startsWith('models--')).map((it) => {
      const repoId = it.name.replace(/^models--/, '').replace(/--/g, '/');
      const isActive = active.has(repoId);
      return { ...it, category, protected: isActive,
               reason: isActive ? 'in use by a running training job' : null };
    });
  }
  if (category === 'datasets') {
    if (deps.hasActiveIngest()) {
      // The datasets library cache is shared with any in-flight ingestion;
      // when we cannot prove files are unused, nothing here is deleted.
      const items = [
        ..._listChildren(dirs.hub, (name) => name.startsWith('datasets--')),
        ..._listChildren(dirs.datasets, () => true),
      ];
      return items.map((it) => ({ ...it, category, protected: true,
        reason: 'a dataset download/processing job is running' }));
    }
    return [
      ..._listChildren(dirs.hub, (name) => name.startsWith('datasets--')),
      ..._listChildren(dirs.datasets, () => true),
    ].map((it) => ({ ...it, category, protected: false, reason: null }));
  }
  if (category === 'temp') {
    // Incomplete downloads / stale locks can sit at any depth inside the
    // caches (hub/models--*/blobs/<hash>.incomplete is the real layout), so
    // scan bounded-depth with lstat-only walking. The xet chunk store is
    // deliberately NOT blanket-deleted: it is shared with in-flight
    // downloads and we cannot prove individual chunks are unused.
    const isTempFile = (name) => name.endsWith('.incomplete') || name.endsWith('.lock');
    const items = [];
    const walk = (dir, depth) => {
      if (depth > 6) return;
      for (const it of _listChildren(dir, () => true)) {
        if (isTempFile(it.name)) { items.push({ ...it, category, protected: false, reason: null }); continue; }
        const st = _lstatSafe(it.path);
        if (st && st.isDirectory()) walk(it.path, depth + 1);
      }
    };
    for (const root of [dirs.hub, dirs.datasets]) walk(root, 0);
    for (const it of _listChildren(dirs.hfRoot, (name) => isTempFile(name))) {
      items.push({ ...it, category, protected: false, reason: null });
    }
    // Python bytecode caches under the repo root (disposable, regenerated)
    for (const it of _listChildren(dirs.repoRoot, (name) => name === '__pycache__')) {
      items.push({ ...it, category, protected: false, reason: null });
    }
    return items;
  }
  const err = new Error(`Unknown cleanup category: ${category}`);
  err.code = 'bad_request'; err.status = 400;
  throw err;
}

// ── Stats ───────────────────────────────────────────────────────────────────

function getStorageStats(dirs, deps) {
  const d = dirs || resolveCacheDirs();
  const dep = deps || { activeTrainingRepos, hasActiveIngest };
  const hubStat = dirStats(d.hub);
  const modelItems = enumerateItems('models', d, dep);
  const datasetItems = enumerateItems('datasets', d, dep);

  const modelCache = { bytes: 0, files: 0, entries: modelItems.length,
                       protected_entries: modelItems.filter((i) => i.protected).length };
  for (const it of modelItems) {
    const s = dirStats(it.path);
    modelCache.bytes += s.bytes; modelCache.files += s.files;
  }
  const datasetHub = modelItems.length ? { bytes: 0, files: 0 } : { bytes: 0, files: 0 };
  let datasetBytes = 0, datasetFiles = 0, datasetProtected = 0;
  for (const it of datasetItems) {
    const s = dirStats(it.path);
    if (it.protected) datasetProtected += 1;
    datasetBytes += s.bytes; datasetFiles += s.files;
  }
  let incompleteBytes = 0;
  const tempItems = enumerateItems('temp', d, dep);
  for (const it of tempItems) {
    const st = _lstatSafe(it.path);
    if (st && st.isFile()) incompleteBytes += st.size;
    else if (st && st.isDirectory()) incompleteBytes += dirStats(it.path).bytes;
  }
  let pycacheBytes = 0;
  for (const it of tempItems) {
    if (it.name === '__pycache__') pycacheBytes += dirStats(it.path).bytes;
  }
  incompleteBytes = Math.max(0, incompleteBytes - pycacheBytes); // avoid double-count
  // Informational only: the xet chunk cache is shared with in-flight
  // downloads, so it is NOT claimed as reclaimable.
  const xetBytes = dirStats(d.xet).bytes;

  let availableBytes = null, totalBytes = null, volumePercent = null;
  try {
    const vfs = fs.statfsSync(d.hfRoot);
    availableBytes = Number(vfs.bavail) * Number(vfs.bsize);
    totalBytes = Number(vfs.blocks) * Number(vfs.bsize);
    if (totalBytes > 0) {
      volumePercent = Math.round(((totalBytes - availableBytes) / totalBytes) * 1000) / 10;
    }
  } catch (_) { /* volume stats unavailable — reported as null, never invented */ }

  const managedBytes = modelCache.bytes + datasetBytes + incompleteBytes + pycacheBytes;
  const warn = (volumePercent != null && volumePercent >= DEFAULT_WARN_PERCENT) ||
               (availableBytes != null && availableBytes < DEFAULT_MIN_AVAILABLE_BYTES);

  return {
    categories: {
      // Keys match the cleanup API's canonical categories exactly
      // (models | datasets | temp) so the frontend can look them up 1:1.
      models: { label: 'Model cache', bytes: modelCache.bytes,
                files: modelCache.files, entries: modelCache.entries,
                protected_entries: modelCache.protected_entries,
                path: d.hub, glob: 'models--*' },
      datasets: { label: 'Dataset cache', bytes: datasetBytes,
                  files: datasetFiles, entries: datasetItems.length,
                  protected_entries: datasetProtected,
                  path: d.datasets, glob: 'hub/datasets--* + datasets/*' },
      temp: { label: 'Temporary files',
              bytes: incompleteBytes + pycacheBytes,
              files: 0, entries: tempItems.length, protected_entries: 0,
              path: d.hfRoot,
              glob: '*.incomplete + *.lock + __pycache__' },
    },
    informational: { xet_chunk_cache_bytes: xetBytes,
                     note: 'xet download chunks are shared with in-flight downloads and are not claimed as reclaimable' },
    totals: {
      managed_bytes: managedBytes,
      dataset_hub_bytes: datasetHub.bytes,
      available_bytes: availableBytes,
      total_bytes: totalBytes,   // device volume — hosting quota may be lower (Replit)
      volume_percent: volumePercent,
      warning: warn,
      warning_reason: warn
        ? (volumePercent != null && volumePercent >= DEFAULT_WARN_PERCENT
            ? `Volume usage at ${volumePercent}% (threshold ${DEFAULT_WARN_PERCENT}%)`
            : 'Less than 2 GB available on the volume')
        : null,
    },
    protected_roots: ['ml_workspaces', 'ml_experiments', 'training_outputs',
                      'uploaded source datasets'],
    active: { training_jobs: [...dep.activeTrainingRepos()].length,
              dataset_ingestions_in_flight: dep.hasActiveIngest() ? 1 : 0 },
    generated_at: new Date().toISOString(),
  };
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

const _busy = new Map(); // category -> in-flight promise (per-category mutex)

function cleanup(category, opts) {
  const options = opts || {};
  const dirs = options.dirs || resolveCacheDirs();
  const deps = options.deps || { activeTrainingRepos, hasActiveIngest };
  const dryRun = !!options.dryRun;
  if (!CATEGORY_NAMES[category]) {
    const err = new Error(`Unknown cleanup category: ${category}. Use one of: ${Object.keys(CATEGORY_NAMES).join(', ')}.`);
    err.code = 'bad_request'; err.status = 400;
    return Promise.reject(err);
  }
  if (_busy.has(category)) {
    const err = new Error(`A ${CATEGORY_NAMES[category]} cleanup is already running.`);
    err.code = 'cleanup_busy'; err.status = 409;
    return Promise.reject(err);
  }

  const run = Promise.resolve().then(() => {
    const items = enumerateItems(category, dirs, deps);
    const result = {
      category, dry_run: dryRun, category_label: CATEGORY_NAMES[category],
      removed: [], skipped: [], errors: [],
      freed_bytes: 0, removed_count: 0, protected_count: 0,
    };
    for (const item of items) {
      if (item.protected) {
        result.protected_count += 1;
        result.skipped.push({ name: item.name, reason: item.reason });
        continue;
      }
      const size = dirStats(item.path).bytes;
      if (size === 0 && !fs.existsSync(item.path)) {
        continue; // vanished between enumeration and cleanup — fine
      }
      if (dryRun) {
        result.freed_bytes += size;
        result.removed.push({ name: item.name, bytes: size, dry_run: true });
        result.removed_count += 1;
        continue;
      }
      const delResult = { bytes: 0, errors: [], missing: 0 };
      const ok = rmrfSafe(item.path, delResult);
      if (ok) {
        result.freed_bytes += size;
        result.removed.push({ name: item.name, bytes: size });
        result.removed_count += 1;
      } else {
        // Item-granular failure: other items still cleaned; nothing is left
        // half-deleted beyond this item's own partial state, which is
        // reported so the UI shows an error, not success.
        result.errors.push({ name: item.name, path: item.path, error: delResult.errors });
      }
    }
    console.log(`[STORAGE] cleanup category=${category} dry_run=${dryRun} ` +
      `removed=${result.removed_count} freed=${result.freed_bytes} ` +
      `protected=${result.protected_count} errors=${result.errors.length}`);
    return result;
  });

  _busy.set(category, run);
  const clear = () => _busy.delete(category);
  run.then(clear, clear);
  return run;
}

module.exports = {
  resolveCacheDirs,
  dirStats,
  rmrfSafe,
  getStorageStats,
  cleanup,
  enumerateItems,
  hubDirName,
  activeTrainingRepos,
  hasActiveIngest,
  CATEGORY_NAMES,
  DEFAULT_WARN_PERCENT,
  DEFAULT_MIN_AVAILABLE_BYTES,
};
