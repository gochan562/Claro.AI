"""Classical-ML experiment runner (CPU, sklearn).

Subcommands (all print one JSON document to stdout unless noted):
  dataset-profile  --workspace <dir> [--view-json ...] [--target ...]
  split-preview    --workspace <dir> --split-json ... [--target ...] [--task ...]
  run              --workdir <exp dir> --config <config.json>
                   prints {"type":"progress",...} lines, final {"type":"result",...}
  predict          --model <model.joblib> (--rows-json ... | --csv <path>) [--out ...]

No arbitrary code execution: every op is driven by validated config.
"""
import argparse
import copy
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from ml import dataset_manager as dm
from ml import evaluation as ev
from ml import models as mm
from ml import preprocessing as pp
from ml import serialization as se
from ml import split as sp


def _emit(obj):
    print(json.dumps(obj, default=str), flush=True)


def _progress(job, message, **kw):
    out = {'type': 'progress', 'message': message}
    out.update(kw)
    _emit(out)


def _fail(message):
    _emit({'type': 'error', 'error': message})
    sys.exit(1)


def cmd_dataset_ingest(args):
    import tempfile
    try:
        with open(args.input, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        from ml import dataset_manager as _dm
        df, info = _dm.load_frame_from_text(
            args.filename, content, int(args.max_bytes), int(args.max_rows))
        if df.shape[1] > 500:
            raise ValueError('Too many columns (%d).' % df.shape[1])
        base = args.workspace_base
        manifest = {
            'workspace_id': args.ws,
            'source_type': 'upload',
            'source_name': args.filename,
            'source_id_or_filename': args.filename,
            'format': info['format'],
            'bytes': info['bytes'],
            'truncated': info['truncated'],
            'target_column': None,
            'feature_columns': None,
            'row_count': int(len(df)),
            'column_metadata': _dm.column_metadata(df),
            'transformation_reference': None,
            'dataset_version_or_hash': None,
        }
        manifest = _dm.save_workspace(base, args.ws, df, manifest)
        manifest['dataset_version_or_hash'] = manifest.get('sha256')
        with open(os.path.join(base, 'ml_workspaces', args.ws, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        profile = _dm.profile_dataframe(df)
        with open(os.path.join(base, 'ml_workspaces', args.ws, 'profile.json'), 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        _emit({'type': 'result', 'manifest': manifest, 'profile': profile})
    except Exception as e:
        _fail('%s: %s' % (type(e).__name__, e))


def cmd_dataset_ingest_hf(args):
    try:
        from ml import dataset_manager as _dm
        cfg = args.config.strip() if isinstance(args.config, str) and args.config.strip() else None
        df, info = _dm.load_frame_from_hf(args.dataset_id, args.split or None,
                                          int(args.max_rows), config=cfg)
        if df.shape[1] > 500:
            raise ValueError('Too many columns (%d).' % df.shape[1])
        base = args.workspace_base
        manifest = {
            'workspace_id': args.ws,
            'source_type': 'huggingface',
            'source_name': args.dataset_id,
            'source_id_or_filename': args.dataset_id,
            'format': 'hf',
            'bytes': 0,
            'truncated': info['truncated'],
            'split': info['split'],
            'config': cfg,
            'target_column': None,
            'feature_columns': None,
            'row_count': int(len(df)),
            'column_metadata': _dm.column_metadata(df),
            'transformation_reference': None,
            'dataset_version_or_hash': None,
        }
        manifest = _dm.save_workspace(base, args.ws, df, manifest)
        manifest['dataset_version_or_hash'] = manifest.get('sha256')
        with open(os.path.join(base, 'ml_workspaces', args.ws, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        profile = _dm.profile_dataframe(df)
        with open(os.path.join(base, 'ml_workspaces', args.ws, 'profile.json'), 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        _emit({'type': 'result', 'manifest': manifest, 'profile': profile})
    except Exception as e:
        _fail('%s: %s' % (type(e).__name__, e))


def cmd_predict_rows(args):
    try:
        from ml import dataset_manager as _dm
        frame = _dm.load_workspace_frame(args.workspace, args.ws)
        filters = json.loads(args.filters_json or '[]')
        out = frame
        for f in filters:
            col, op, val = f.get('column'), f.get('op'), f.get('value')
            if col not in out.columns:
                continue
            s = out[col]
            if op == '==':
                out = out[s.astype(str) == str(val)]
            elif op == '!=':
                out = out[s.astype(str) != str(val)]
        out = out.head(int(args.max_rows)).astype(object).where(pd.notna(out.head(int(args.max_rows))), None)
        _emit({'type': 'result', 'rows': out.to_dict(orient='records')})
    except Exception as e:
        _fail('%s: %s' % (type(e).__name__, e))


def cmd_dataset_profile(args):
    try:
        frame = dm.load_workspace_frame(args.workspace, args.ws)
        view = json.loads(args.view_json or '{}')
        target = args.target or None
        frame2, applied = dm.apply_view(frame, view)
        profile = dm.profile_dataframe(frame2, target=target)
        profile['view_applied'] = applied
        _emit({'type': 'result', 'profile': profile})
    except Exception as e:
        _fail('%s: %s' % (type(e).__name__, e))


def cmd_split_preview(args):
    try:
        frame = dm.load_workspace_frame(args.workspace, args.ws)
        view = json.loads(args.view_json or '{}')
        frame2, _ = dm.apply_view(frame, view)
        cfg = sp.validate_split_config(json.loads(args.split_json or '{}'))
        parts, notes = sp.make_splits(frame2, cfg, target=args.target,
                                      task_type=args.task or 'classification')
        desc = sp.describe_split(frame2, parts, target=args.target)
        desc['notes'] = notes
        _emit({'type': 'result', 'split': desc})
    except Exception as e:
        _fail('%s: %s' % (type(e).__name__, e))


def _resolve_xy(frame, target, features):
    if target not in frame.columns:
        raise ValueError('Target column %r not in dataset.' % target)
    if features:
        missing = [c for c in features if c not in frame.columns]
        if missing:
            raise ValueError('Unknown feature columns: %s.' % (missing,))
        feats = list(features)
    else:
        feats = [c for c in frame.columns if c != target]
    if not feats:
        raise ValueError('No feature columns selected.')
    if target in feats:
        feats = [c for c in feats if c != target]
    return feats


def _check_task_compat(frame, target, task):
    s = frame[target]
    if task == 'regression':
        try:
            pd.to_numeric(s.dropna(), errors='raise')
        except Exception:
            raise ValueError('Regression needs a numeric target column.')
        return sorted([str(v) for v in pd.Series(s.dropna().unique()).tolist()])[:0] or None
    n_unique = int(s.nunique(dropna=True))
    if n_unique < 2:
        raise ValueError('Classification needs at least 2 target classes.')
    if n_unique > 200:
        raise ValueError('Target has %d unique values; is this really classification?' % n_unique)
    return sorted([str(v) for v in s.dropna().unique().tolist()])


def _frame_xy(frame, feats, target):
    X = frame[feats].copy()
    y = frame[target]
    return X, y


def _fit_pipeline(pre, estimator, X_train, y_train):
    if pre is None:
        estimator.fit(X_train, y_train)
        return estimator
    pipe = __import__('sklearn.pipeline', fromlist=['Pipeline']).Pipeline(
        [('preprocess', pre), ('estimator', estimator)])
    pipe.fit(X_train, y_train)
    return pipe


def _predict_all(pipe, X):
    y_pred = np.asarray(pipe.predict(X))
    y_proba = None
    try:
        if hasattr(pipe, 'predict_proba'):
            y_proba = np.asarray(pipe.predict_proba(X))
    except Exception:
        y_proba = None
    return y_pred, y_proba


def _evaluate_split(pipe, X, y, task, labels):
    y_pred, y_proba = _predict_all(pipe, X)
    if task == 'classification':
        return ev.classification_metrics(
            np.asarray(y).astype(str), np.asarray(y_pred).astype(str),
            y_proba, labels=labels)
    return ev.regression_metrics(np.asarray(y, dtype=float),
                                 np.asarray(y_pred, dtype=float))


SCORERS = {
    'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy',
    'f1_macro': 'f1_macro', 'f1_weighted': 'f1_weighted',
    'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
    'neg_log_loss': 'neg_log_loss',
    'r2': 'r2', 'neg_mae': 'neg_mean_absolute_error',
    'neg_mse': 'neg_mean_squared_error', 'neg_rmse': 'neg_root_mean_squared_error',
}
PROBA_SCORERS = {'roc_auc', 'average_precision', 'neg_log_loss'}


def _cv_splitter(method, folds, seed, y=None, groups=None, task='classification'):
    method = (method or 'kfold').lower()
    if method == 'stratified' or (method == 'auto' and task == 'classification'):
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if method == 'group':
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=folds)
    if method == 'time':
        from sklearn.model_selection import TimeSeriesSplit
        return TimeSeriesSplit(n_splits=folds)
    from sklearn.model_selection import KFold
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _run_cv(make_pipe, X, y, cv_cfg, task, seed, proba_ok):
    from sklearn.model_selection import cross_validate
    folds = int(cv_cfg.get('folds', 5))
    if folds < 2 or folds > 10:
        raise ValueError('CV folds must be 2..10.')
    scoring = cv_cfg.get('scoring') or ('f1_macro' if task == 'classification' else 'r2')
    if scoring not in SCORERS:
        raise ValueError('Unknown scoring metric: %r.' % scoring)
    if scoring in PROBA_SCORERS and not proba_ok:
        raise ValueError('Scoring %r needs probability output, which this model lacks.' % scoring)
    method = cv_cfg.get('method', 'auto')
    splitter = _cv_splitter(method, folds, seed, task=task)
    res = cross_validate(make_pipe(), X, y, cv=splitter, scoring=SCORERS[scoring],
                         n_jobs=1, error_score='raise')
    key = 'test_score'
    scores = [float(v) for v in res[key]]
    return {'method': method, 'folds': folds, 'scoring': scoring,
            'scores': scores, 'mean': float(np.mean(scores)),
            'std': float(np.std(scores))}


def _expand_search_space(raw):
    """Grid values stay lists; {low,high[,steps]} becomes linspace; {choice} stays list."""
    space = {}
    for name, spec in (raw or {}).items():
        if isinstance(spec, list):
            space[name] = spec
        elif isinstance(spec, dict) and 'choice' in spec:
            space[name] = list(spec['choice'])
        elif isinstance(spec, dict) and 'low' in spec and 'high' in spec:
            lo, hi = float(spec['low']), float(spec['high'])
            steps = int(spec.get('steps', 5))
            vals = np.linspace(lo, hi, max(2, steps)).tolist()
            if spec.get('integer'):
                vals = sorted(set(int(round(v)) for v in vals))
            space[name] = vals
        else:
            raise ValueError('Bad search space for %r: use a list, {choice:[...]} or {low,high}.' % name)
        if not space[name]:
            raise ValueError('Empty search space for %r.' % name)
    return space


def _run_search(make_pipe, base_params, X, y, search_cfg, seed, max_trials, proba_ok, task, emit):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    method = (search_cfg.get('method') or 'grid').lower()
    if method not in ('grid', 'random'):
        raise ValueError('Search method must be grid or random.')
    space = _expand_search_space(search_cfg.get('params') or {})
    if not space:
        raise ValueError('Search needs at least one parameter.')
    scoring = search_cfg.get('scoring') or ('f1_macro' if task == 'classification' else 'r2')
    if scoring not in SCORERS:
        raise ValueError('Unknown scoring metric: %r.' % scoring)
    if scoring in PROBA_SCORERS and not proba_ok:
        raise ValueError('Scoring %r needs probability output.' % scoring)
    cv_folds = int(search_cfg.get('cv_folds', 3))
    if cv_folds < 2 or cv_folds > 5:
        raise ValueError('Search CV folds must be 2..5.')
    trials = int(search_cfg.get('trials', 10))
    trials = max(1, min(trials, max_trials))
    prefixed = {'estimator__' + k: v for k, v in space.items()}
    if method == 'grid':
        total = 1
        for v in prefixed.values():
            total *= len(v)
        if total > max_trials:
            raise ValueError('Grid has %d combinations (limit %d); narrow it.' % (total, max_trials))
        search = GridSearchCV(make_pipe(), prefixed, scoring=SCORERS[scoring],
                              cv=cv_folds, n_jobs=1, refit=True)
    else:
        search = RandomizedSearchCV(make_pipe(), prefixed, n_iter=trials,
                                    scoring=SCORERS[scoring], cv=cv_folds, n_jobs=1,
                                    random_state=int(search_cfg.get('seed', seed)), refit=True)
    search.fit(X, y)
    rows = []
    for i, params in enumerate(search.cv_results_['params']):
        rows.append({
            'trial': i,
            'params': {k.replace('estimator__', '', 1): (v.item() if hasattr(v, 'item') else v)
                       for k, v in params.items()},
            'mean_score': float(search.cv_results_['mean_test_score'][i]),
            'std_score': float(search.cv_results_['std_test_score'][i]),
        })
    rows.sort(key=lambda r: r['mean_score'], reverse=True)
    best = {k.replace('estimator__', '', 1): (v.item() if hasattr(v, 'item') else v)
            for k, v in search.best_params_.items()}
    emit('search complete: best %s=%.4f' % (scoring, float(search.best_score_)))
    return {'method': method, 'scoring': scoring, 'trials': rows,
            'best_params': best, 'best_score': float(search.best_score_)}, search.best_estimator_


def _feature_importance(pipe, estimator, X_sample, feature_names, task):
    """coef_ / feature_importances_ / permutation fallback. Top 20."""
    names = [str(n) for n in feature_names]
    try:
        est = estimator if estimator is not None else pipe
        if hasattr(est, 'named_steps') and 'estimator' in est.named_steps:
            est = est.named_steps['estimator']
        if hasattr(est, 'coef_'):
            coef = np.asarray(est.coef_)
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            vals = coef.tolist()
            return {'method': 'coefficients', 'features': _top(names, vals)}
        if hasattr(est, 'feature_importances_'):
            vals = np.asarray(est.feature_importances_).tolist()
            return {'method': 'feature_importance', 'features': _top(names, vals)}
    except Exception:
        pass
    try:
        from sklearn.inspection import permutation_importance
        Xs = X_sample[:200] if len(X_sample) > 200 else X_sample
        ys = None
        r = permutation_importance(pipe, Xs, ys, n_repeats=3, random_state=0, n_jobs=1)
        return {'method': 'permutation', 'features': _top(names, r.importances_mean.tolist()),
                'note': 'unsupervised proxy (no labels passed); rerun with labels for exact values'}
    except Exception as e:
        return {'method': 'unavailable', 'reason': '%s: %s' % (type(e).__name__, e)}


def _top(names, vals, k=20):
    order = sorted(range(len(vals)), key=lambda i: -abs(float(vals[i])))[:k]
    return [{'feature': names[i] if i < len(names) else 'f%d' % i,
             'value': float(vals[i])} for i in order]


def cmd_run(args):
    t0 = time.time()
    with open(args.config) as f:
        cfg = json.load(f)
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    def emit(m, **kw):
        _progress(None, m, **kw)

    try:
        from ml import dataset_manager as _dm  # local alias, same module
        ws = cfg['workspace_id']
        base = cfg.get('workspace_base') or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
        frame = _dm.load_workspace_frame(base, ws)
        manifest = _dm.load_manifest(base, ws)
        view = cfg.get('dataset_view') or {}
        frame, view_applied = _dm.apply_view(frame, view)
        target = cfg['target_column']
        feats = _resolve_xy(frame, target, cfg.get('feature_columns'))
        task = cfg.get('task', 'classification')
        labels = _check_task_compat(frame, target, task)
        seed = int(cfg.get('seed', 42))

        emit('split: %s' % cfg['split']['strategy'])
        parts, notes = sp.make_splits(
            frame.reset_index(drop=True), cfg['split'], target=target, task_type=task)
        se.dump_joblib(os.path.join(workdir, 'split.joblib'),
                       {'config': cfg['split'], 'notes': notes,
                        'parts': {k: np.asarray(v) for k, v in parts.items()}})
        X = frame[feats].reset_index(drop=True)
        y = frame[target].reset_index(drop=True)
        idx = {k: np.asarray(v) for k, v in parts.items()}
        splits = {k: (X.iloc[v].copy(), y.iloc[v].copy()) for k, v in idx.items()}
        X_train, y_train = splits['train']
        if len(X_train) < 5:
            raise ValueError('Training split too small (%d rows).' % len(X_train))

        num_cols = [c for c in feats if _dm._kind_of(X_train[c]) == 'numeric']
        cat_cols = [c for c in feats if _dm._kind_of(X_train[c]) in ('categorical', 'boolean')]
        emit('preprocessing: fit on train only')
        pre, pre_spec = pp.build_preprocessor(cfg.get('preprocessing') or {}, num_cols, cat_cols)

        family = cfg['model']['family']
        emit('train: %s' % family)
        base_est, used_params = mm.build_estimator(family, (cfg['model'] or {}).get('params'), seed)

        def make_pipe(fresh_estimator=None):
            est, _ = mm.build_estimator(family, (cfg['model'] or {}).get('params'), seed)
            if pre is None:
                return est
            from sklearn.pipeline import Pipeline
            return Pipeline([('preprocess', copy.deepcopy(pre)), ('estimator', est)])

        proba_ok = mm.supports_proba(base_est)

        search_cfg = cfg.get('search')
        search_result, children = None, []
        if search_cfg:
            max_trials = int(cfg.get('limits', {}).get('max_trials', 30))
            search_result, best_pipe = _run_search(
                make_pipe, (cfg['model'] or {}).get('params'), X_train, y_train,
                search_cfg, seed, max_trials, proba_ok, task, emit)
            used_params = dict(used_params)
            used_params.update(search_result['best_params'])
            for row in search_result['trials']:
                children.append({'trial': row['trial'], 'params': row['params'],
                                 'mean_score': row['mean_score'], 'std_score': row['std_score']})
            pipe = best_pipe
            cdir = os.path.join(workdir, 'children')
            os.makedirs(cdir, exist_ok=True)
            se.write_json(os.path.join(cdir, 'search.json'), search_result)
        else:
            pipe = _fit_pipeline(pre, base_est, X_train, y_train)

        try:
            feature_names = pp.fitted_feature_names(
                pipe.named_steps['preprocess'] if hasattr(pipe, 'named_steps') else pre, feats)
        except Exception:
            feature_names = list(feats)

        metrics = {'task': task}
        for name in ('train', 'validation', 'test'):
            if name not in splits:
                continue
            Xs, ys = splits[name]
            emit('evaluate: %s (%d rows)' % (name, len(Xs)))
            yt = np.asarray(ys).astype(str) if task == 'classification' else ys
            metrics[name] = _evaluate_split(pipe, Xs, yt, task, labels)

        cv_result = None
        if cfg.get('cv'):
            emit('cross-validation')
            cv_result = _run_cv(make_pipe, X_train, y_train, cfg['cv'], task, seed, proba_ok)
            metrics['cv'] = cv_result

        emit('feature importance')
        try:
            importance = _feature_importance(pipe, None, X_train, feature_names, task)
        except Exception as e:
            importance = {'method': 'unavailable', 'reason': str(e)}
        metrics['importance'] = importance

        model_bundle = {
            'pipeline': pipe,
            'family': family,
            'params': used_params,
            'feature_columns': list(feats),
            'target_column': target,
            'task': task,
            'labels': labels,
            'preprocessing': pre_spec,
            'feature_names': feature_names,
            'sklearn_version': __import__('sklearn').__version__,
        }
        se.dump_joblib(os.path.join(workdir, 'model.joblib'), model_bundle)
        se.write_json(os.path.join(workdir, 'metrics.json'), metrics)
        se.write_json(os.path.join(workdir, 'config.json'), cfg)
        record = {
            'experiment_id': cfg.get('experiment_id'),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'dataset': {'workspace_id': ws,
                        'source': manifest.get('source'),
                        'sha256': manifest.get('sha256'),
                        'rows': len(frame), 'columns': list(map(str, frame.columns))},
            'target_column': target, 'feature_columns': list(feats),
            'split': {'config': cfg['split'], 'notes': notes,
                      'counts': {k: int(len(v)) for k, v in idx.items()}},
            'preprocessing': pre_spec,
            'preprocessing_summary': pp.summarize_preprocessor(pre_spec),
            'model_family': family, 'model_params': used_params,
            'seed': seed,
            'cv': cv_result, 'search': search_result, 'children': children,
            'metrics': _summarize_metrics(metrics),
            'runtime_sec': round(time.time() - t0, 2),
            'status': 'finished',
            'artifacts': ['config.json', 'metrics.json', 'model.joblib', 'split.joblib'],
        }
        se.write_json(os.path.join(workdir, 'experiment.json'), record)
        _emit({'type': 'result', 'status': 'finished', 'record': record})
    except Exception:
        _emit({'type': 'result', 'status': 'failed',
               'error': traceback.format_exc(limit=8)[-2000:]})
        sys.exit(1)


def _summarize_metrics(metrics):
    out = {'task': metrics.get('task')}
    for split in ('train', 'validation', 'test'):
        m = metrics.get(split)
        if not m:
            continue
        if m['task'] == 'classification':
            out[split] = {k: m.get(k) for k in
                          ('accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc', 'log_loss')}
            out[split]['f1_macro'] = (m.get('macro') or {}).get('f1')
        else:
            out[split] = {k: m.get(k) for k in ('mae', 'mse', 'rmse', 'r2')}
    if metrics.get('cv'):
        out['cv'] = {k: metrics['cv'].get(k) for k in ('method', 'folds', 'scoring', 'mean', 'std')}
    if metrics.get('importance'):
        out['importance'] = metrics.get('importance')
    return out


def cmd_predict(args):
    try:
        bundle = se.load_joblib(args.model)
        pipe = bundle['pipeline']
        feats = list(bundle['feature_columns'])
        if args.rows_json:
            rows = json.loads(args.rows_json)
            X = pd.DataFrame(rows)
        elif args.csv:
            if not os.path.exists(args.csv):
                raise ValueError('Input CSV not found.')
            X = pd.read_csv(args.csv)
        else:
            raise ValueError('Provide rows-json or csv input.')
        missing = [c for c in feats if c not in X.columns]
        if missing:
            raise ValueError('Missing feature columns: %s.' % (missing,))
        X = X[feats]
        if len(X) == 0:
            raise ValueError('No rows to predict.')
        if len(X) > 50000:
            raise ValueError('Too many prediction rows (%d).' % len(X))
        y_pred, y_proba = _predict_all(pipe, X)
        out_rows = []
        for i in range(len(X)):
            r = {c: (None if pd.isna(X[c].iloc[i]) else X[c].iloc[i]) for c in feats}
            r['prediction'] = y_pred[i].item() if hasattr(y_pred[i], 'item') else y_pred[i]
            if y_proba is not None:
                probs = y_proba[i]
                r['probabilities'] = [float(p) for p in probs]
                try:
                    r['max_probability'] = float(np.max(probs))
                except Exception:
                    pass
            out_rows.append(r)
        if args.out:
            pd.DataFrame(out_rows).to_csv(args.out, index=False)
        _emit({'type': 'result', 'rows': out_rows,
               'count': len(out_rows), 'output': args.out})
    except Exception as e:
        _fail('%s: %s' % (type(e).__name__, e))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ml_experiment_runner')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('dataset-profile')
    p.add_argument('--workspace', required=True)
    p.add_argument('--ws', required=True)
    p.add_argument('--view-json', default='{}')
    p.add_argument('--target', default='')
    p.set_defaults(fn=cmd_dataset_profile)
    p = sub.add_parser('dataset-ingest')
    p.add_argument('--ws', required=True)
    p.add_argument('--workspace-base', required=True)
    p.add_argument('--filename', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--max-bytes', default='10485760')
    p.add_argument('--max-rows', default='50000')
    p.set_defaults(fn=cmd_dataset_ingest)
    p = sub.add_parser('dataset-ingest-hf')
    p.add_argument('--ws', required=True)
    p.add_argument('--workspace-base', required=True)
    p.add_argument('--dataset-id', required=True)
    p.add_argument('--split', default='')
    p.add_argument('--config', default='')
    p.add_argument('--max-rows', default='50000')
    p.set_defaults(fn=cmd_dataset_ingest_hf)
    p = sub.add_parser('predict-rows')
    p.add_argument('--workspace', required=True)
    p.add_argument('--ws', required=True)
    p.add_argument('--filters-json', default='[]')
    p.add_argument('--max-rows', default='500')
    p.set_defaults(fn=cmd_predict_rows)
    p = sub.add_parser('split-preview')
    p.add_argument('--workspace', required=True)
    p.add_argument('--ws', required=True)
    p.add_argument('--split-json', default='{}')
    p.add_argument('--view-json', default='{}')
    p.add_argument('--target', default='')
    p.add_argument('--task', default='classification')
    p.set_defaults(fn=cmd_split_preview)
    p = sub.add_parser('run')
    p.add_argument('--workdir', required=True)
    p.add_argument('--config', required=True)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('--rows-json', default='')
    p.add_argument('--csv', default='')
    p.add_argument('--out', default='')
    p.set_defaults(fn=cmd_predict)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
