"""Dataset ingestion, profiling, and view transforms for tabular ML.

Supported sources: CSV/TSV/JSON text upload, Hugging Face dataset reference.
Datasets live under ml_workspaces/<ws_id>/{data.csv,profile.json,manifest.json}.
Raw frames are NEVER stored in notebook localStorage; only the manifest is.
"""
import hashlib
import io
import json
import os

import pandas as pd

MAX_PREVIEW_ROWS = 100


class DatasetError(ValueError):
    pass


def sniff_format(filename):
    name = (filename or '').lower()
    if name.endswith('.tsv') or name.endswith('.tab'):
        return 'tsv'
    if name.endswith('.json'):
        return 'json'
    return 'csv'


def load_frame_from_text(filename, content, max_bytes, max_rows):
    raw = content if isinstance(content, str) else str(content or '')
    nbytes = len(raw.encode('utf-8'))
    if nbytes == 0:
        raise DatasetError('Uploaded file is empty.')
    if nbytes > max_bytes:
        raise DatasetError(
            'Dataset too large: %d bytes (limit %d bytes).' % (nbytes, max_bytes))
    fmt = sniff_format(filename)
    try:
        if fmt == 'tsv':
            df = pd.read_csv(io.StringIO(raw), sep='\t')
        elif fmt == 'json':
            df = pd.read_json(io.StringIO(raw))
            if not isinstance(df, pd.DataFrame):
                raise DatasetError('JSON must decode to a table (list of objects).')
        else:
            df = pd.read_csv(io.StringIO(raw))
    except DatasetError:
        raise
    except Exception as e:
        raise DatasetError('Could not parse %s file: %s' % (fmt.upper(), e))
    if df.shape[1] == 0:
        raise DatasetError('Dataset has no columns.')
    truncated = False
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
        truncated = True
    if len(df) == 0:
        raise DatasetError('Dataset has no rows.')
    return df, {'format': fmt, 'bytes': nbytes, 'truncated': truncated}


def load_frame_from_hf(dataset_id, split=None, max_rows=50000):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise DatasetError('Hugging Face datasets library unavailable: %s' % e)
    if not dataset_id or not isinstance(dataset_id, str):
        raise DatasetError('dataset_id is required.')
    try:
        ds = load_dataset(dataset_id.strip(), split=split, streaming=False)
    except Exception:
        try:
            full = load_dataset(dataset_id.strip(), streaming=False)
        except Exception as e:
            raise DatasetError('Could not load Hugging Face dataset %r: %s' % (dataset_id, e))
        names = list(full.keys())
        pick = split or ('train' if 'train' in names else names[0])
        if pick not in names:
            raise DatasetError('Split %r not found in %s (have %s).' % (split, dataset_id, names))
        ds = full[pick]
    try:
        df = ds.to_pandas()
    except Exception as e:
        raise DatasetError('Could not convert dataset to table: %s' % e)
    # Flatten list/struct columns to JSON strings so the frame stays tabular.
    for c in list(df.columns):
        if len(df) and isinstance(df[c].iloc[0], (list, dict)):
            df[c] = df[c].map(lambda v: json.dumps(v, default=str))
    truncated = False
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
        truncated = True
    return df, {'split': split or 'auto', 'truncated': truncated}


def _kind_of(series):
    if pd.api.types.is_bool_dtype(series):
        return 'boolean'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    return 'categorical'


def column_metadata(df, max_categories=50):
    meta = []
    for name in df.columns:
        s = df[name]
        kind = _kind_of(s)
        m = {
            'name': str(name),
            'dtype': str(s.dtype),
            'kind': kind,
            'missing': int(s.isna().sum()),
            'unique': int(s.nunique(dropna=True)),
        }
        if kind == 'numeric':
            d = s.dropna()
            if len(d):
                m.update({
                    'min': float(d.min()), 'max': float(d.max()),
                    'mean': float(d.mean()), 'std': float(d.std()) if len(d) > 1 else 0.0,
                })
        else:
            top = s.value_counts(dropna=True).head(max_categories)
            m['top_values'] = [{'value': str(v), 'count': int(c)} for v, c in top.items()]
        meta.append(m)
    return meta


def profile_dataframe(df, target=None, hist_bins=20, max_corr_cols=30,
                      scatter_cap=500):
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    num_cols = [c for c in df.columns if _kind_of(df[c]) == 'numeric']
    cat_cols = [c for c in df.columns if _kind_of(df[c]) in ('categorical', 'boolean')]
    profile = {
        'row_count': n_rows,
        'column_count': n_cols,
        'columns': [str(c) for c in df.columns],
        'column_metadata': column_metadata(df),
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_columns': [str(c) for c in num_cols],
        'categorical_columns': [str(c) for c in cat_cols],
        'numeric_summary': {},
        'categorical_summary': {},
        'target_distribution': None,
        'correlation': None,
        'boxplot': {},
        'scatter': None,
        'preview': df.head(MAX_PREVIEW_ROWS).astype(str).to_dict(orient='records'),
    }
    for c in num_cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if not len(s):
            continue
        q = s.quantile([0, 0.25, 0.5, 0.75, 1.0])
        counts, edges = _histogram(s, hist_bins)
        profile['numeric_summary'][str(c)] = {
            'count': int(len(s)), 'mean': float(s.mean()),
            'std': float(s.std()) if len(s) > 1 else 0.0,
            'min': float(q.iloc[0]), 'q1': float(q.iloc[1]),
            'median': float(q.iloc[2]), 'q3': float(q.iloc[3]),
            'max': float(q.iloc[4]),
            'histogram': {'bins': edges, 'counts': counts},
        }
        profile['boxplot'][str(c)] = {
            'min': float(q.iloc[0]), 'q1': float(q.iloc[1]),
            'median': float(q.iloc[2]), 'q3': float(q.iloc[3]),
            'max': float(q.iloc[4]),
        }
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(20)
        profile['categorical_summary'][str(c)] = {
            'unique': int(df[c].nunique(dropna=True)),
            'missing': int(df[c].isna().sum()),
            'values': [{'value': (None if pd.isna(v) else str(v)), 'count': int(n)}
                        for v, n in vc.items()],
        }
    if target is not None and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        profile['target_distribution'] = {
            'column': str(target),
            'values': [{'value': (None if pd.isna(v) else str(v)), 'count': int(n)}
                       for v, n in vc.items()],
        }
    use = num_cols[:max_corr_cols]
    if len(use) >= 2:
        try:
            corr = df[use].apply(pd.to_numeric, errors='coerce').corr(numeric_only=True)
            profile['correlation'] = {
                'columns': [str(c) for c in corr.columns],
                'matrix': [[(None if pd.isna(v) else round(float(v), 3)) for v in row]
                           for row in corr.values.tolist()],
            }
        except Exception:
            profile['correlation'] = None
    if len(num_cols) >= 2:
        a, b = num_cols[0], num_cols[1]
        sub = df[[a, b]].dropna().head(scatter_cap)
        try:
            profile['scatter'] = {
                'x': str(a), 'y': str(b),
                'points': [[float(x), float(y)] for x, y in
                            zip(pd.to_numeric(sub[a], errors='coerce'),
                                pd.to_numeric(sub[b], errors='coerce'))],
            }
        except Exception:
            profile['scatter'] = None
    return profile


def _histogram(s, bins):
    try:
        counts, edges = __import__('numpy').histogram(s.values, bins=bins)
        return [int(c) for c in counts], [float(e) for e in edges]
    except Exception:
        return [], []


def apply_view(df, view):
    """Apply drop_columns / filters / sampling. Returns (frame, applied)."""
    view = view or {}
    applied = {'dropped': [], 'filters': [], 'sampled': None}
    out = df
    drop = [c for c in (view.get('drop_columns') or []) if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
        applied['dropped'] = [str(c) for c in drop]
    for f in (view.get('filters') or []):
        col, op, val = f.get('column'), f.get('op'), f.get('value')
        if col not in out.columns or op not in ('==', '!=', '>', '>=', '<', '<=',
                                                'contains', 'not_null', 'is_null'):
            continue
        s = out[col]
        try:
            if op == '==':
                mask = s.astype(str) == str(val)
            elif op == '!=':
                mask = s.astype(str) != str(val)
            elif op in ('>', '>=', '<', '<='):
                num, v = pd.to_numeric(s, errors='coerce'), float(val)
                mask = num > v if op == '>' else (num >= v if op == '>=' else
                      (num < v if op == '<' else num <= v))
                mask = mask.fillna(False)
            elif op == 'contains':
                mask = s.astype(str).str.contains(str(val), na=False)
            elif op == 'not_null':
                mask = s.notna()
            else:
                mask = s.isna()
        except (ValueError, TypeError):
            continue
        out = out[mask]
        applied['filters'].append({'column': str(col), 'op': op, 'value': val})
    sample = view.get('sample') or {}
    n, frac = sample.get('n'), sample.get('frac')
    try:
        if n and int(n) > 0 and int(n) < len(out):
            out = out.sample(n=int(n), random_state=int(sample.get('seed', 42)))
            applied['sampled'] = {'n': int(n)}
        elif frac and 0 < float(frac) < 1:
            out = out.sample(frac=float(frac), random_state=int(sample.get('seed', 42)))
            applied['sampled'] = {'frac': float(frac)}
    except (ValueError, TypeError):
        pass
    return out.copy(), applied


def dataset_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def workspace_paths(base_dir, ws_id):
    root = os.path.join(base_dir, 'ml_workspaces', ws_id)
    return {
        'root': root,
        'data': os.path.join(root, 'data.csv'),
        'profile': os.path.join(root, 'profile.json'),
        'manifest': os.path.join(root, 'manifest.json'),
    }


def save_workspace(base_dir, ws_id, df, manifest):
    p = workspace_paths(base_dir, ws_id)
    os.makedirs(p['root'], exist_ok=True)
    df.to_csv(p['data'], index=False)
    manifest = dict(manifest)
    manifest['sha256'] = dataset_sha256(p['data'])
    with open(p['manifest'], 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_workspace_frame(base_dir, ws_id):
    p = workspace_paths(base_dir, ws_id)
    if not os.path.exists(p['data']):
        raise DatasetError('Dataset workspace %r has no data file.' % ws_id)
    return pd.read_csv(p['data'])


def load_manifest(base_dir, ws_id):
    p = workspace_paths(base_dir, ws_id)
    if not os.path.exists(p['manifest']):
        raise DatasetError('Unknown dataset workspace: %s' % ws_id)
    with open(p['manifest']) as f:
        return json.load(f)
