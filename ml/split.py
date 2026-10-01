"""Train/validation/test splitting. Test set stays isolated by construction.

Strategies:
  train_test            -> train / test
  train_val_test        -> train / validation / test
  official              -> preserve caller-provided official splits
  time                  -> time-ordered cut (no shuffle)

Splitting is stratified for classification when requested. Indices are
stored (split.joblib) so reruns with the same seed reproduce exactly.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedShuffleSplit,
    train_test_split,
)


class SplitError(ValueError):
    pass


STRATEGIES = ('train_test', 'train_val_test', 'official', 'time')


def validate_split_config(cfg):
    cfg = cfg or {}
    strategy = cfg.get('strategy', 'train_val_test')
    if strategy not in STRATEGIES:
        raise SplitError('Unknown split strategy: %r.' % (strategy,))
    if strategy == 'official':
        col = (cfg.get('split_column') or '').strip()
        if not col:
            raise SplitError('Official splits need a split_column naming train/validation/test.')
        mapping = cfg.get('split_values') or {'train': 'train', 'validation': 'validation', 'test': 'test'}
        return {'strategy': 'official', 'split_column': col, 'split_values': mapping}
    if strategy == 'time':
        col = (cfg.get('time_column') or '').strip()
        if not col:
            raise SplitError('Time-based splitting needs a time_column.')
        frac = _fractions(cfg)
        return {'strategy': 'time', 'time_column': col,
                'train_frac': frac[0], 'val_frac': frac[1], 'test_frac': frac[2]}
    frac = _fractions(cfg)
    out = {
        'strategy': strategy,
        'train_frac': frac[0], 'val_frac': frac[1], 'test_frac': frac[2],
        'seed': int(cfg.get('seed', 42)),
        'shuffle': bool(cfg.get('shuffle', True)),
        'stratify': bool(cfg.get('stratify', False)),
        'group_column': (cfg.get('group_column') or '').strip() or None,
    }
    if out['test_frac'] <= 0:
        raise SplitError('Test fraction must be > 0.')
    if strategy == 'train_val_test' and out['val_frac'] <= 0:
        raise SplitError('Validation fraction must be > 0 for train/validation/test.')
    return out


def _fractions(cfg):
    if cfg.get('strategy', 'train_val_test') == 'train_test':
        train = float(cfg.get('train_frac', 0.8))
        test = cfg.get('test_frac', None)
        test = float(test) if test else round(1.0 - train, 6)
        return train, 0.0, test
    train = float(cfg.get('train_frac', 0.7))
    val = float(cfg.get('val_frac', 0.15))
    test = cfg.get('test_frac', None)
    test = float(test) if test else round(1.0 - train - val, 6)
    total = train + val + test
    if abs(total - 1.0) > 1e-6 or min(train, val, test) <= 0:
        raise SplitError('Fractions must be positive and sum to 1 (got %s).' % total)
    return train, val, test


def _check_stratify_possible(y, test_size, seed, min_per_class=2):
    if y is None:
        return None, 'no target for stratification'
    counts = pd.Series(np.asarray(y)).value_counts()
    if len(counts) < 2:
        return None, 'stratification needs at least 2 classes'
    n_test = max(1, int(round(len(y) * test_size)))
    if (counts < min_per_class).any() or (counts == 1).any() or n_test < len(counts):
        return None, 'rare classes too small to stratify safely'
    return y, None


def random_split(n, test_frac, seed, shuffle, y=None, stratify=False, groups=None):
    """Single train/test index split. Returns (train_idx, test_idx, notes)."""
    notes = []
    idx = np.arange(n)
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tr, te = next(gss.split(idx, groups=groups))
        notes.append('group split (no group spans train/test)')
        return np.sort(tr), np.sort(te), notes
    strat = None
    if stratify and shuffle:
        strat, why = _check_stratify_possible(y, test_frac, seed)
        if strat is None:
            notes.append('stratification skipped: %s' % why)
    if not shuffle:
        n_test = max(1, int(round(n * test_frac)))
        return idx[:n - n_test], idx[n - n_test:], notes + ['ordered split (shuffle off)']
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed,
                              stratify=strat)
    if strat is not None:
        notes.append('stratified')
    return np.sort(tr), np.sort(te), notes


def _official_indices(df, cfg):
    col = cfg['split_column']
    if col not in df.columns:
        raise SplitError('Split column %r not in dataset.' % col)
    m = cfg['split_values']
    s = df[col].astype(str)
    out = {}
    for part, val in (('train', m.get('train', 'train')),
                      ('validation', m.get('validation', 'validation')),
                      ('test', m.get('test', 'test'))):
        out[part] = np.sort(df.index[s == str(val)].to_numpy())
    if len(out['train']) == 0 or len(out['test']) == 0:
        raise SplitError('Official split %r produced an empty train or test set.' % col)
    return out, ['official splits preserved from %r' % col]


def _time_indices(df, cfg):
    col = cfg['time_column']
    if col not in df.columns:
        raise SplitError('Time column %r not in dataset.' % col)
    order = np.argsort(pd.to_datetime(df[col], errors='coerce').values, kind='stable')
    n = len(df)
    n_test = max(1, int(round(n * cfg['test_frac'])))
    n_val = max(1, int(round(n * cfg['val_frac']))) if cfg['val_frac'] else 0
    te = order[n - n_test:]
    va = order[n - n_test - n_val:n - n_test] if n_val else np.array([], dtype=int)
    tr = order[:n - n_test - n_val]
    return ({'train': np.sort(tr), 'validation': np.sort(va), 'test': np.sort(te)},
            ['time-ordered split on %r (no shuffle)' % col])


def make_splits(df, cfg, target=None, task_type='classification'):
    """Returns (parts, notes). parts maps train/validation?/test -> int index array."""
    cfg = validate_split_config(cfg)
    y = None
    if target and target in df.columns and task_type == 'classification':
        y = df[target].astype(str).fillna('__missing__').to_numpy()
    groups = None
    if cfg.get('group_column'):
        if cfg['group_column'] not in df.columns:
            raise SplitError('Group column %r not in dataset.' % cfg['group_column'])
        groups = df[cfg['group_column']].astype(str).fillna('__missing__').to_numpy()
    if cfg['strategy'] == 'official':
        return _official_indices(df, cfg)
    if cfg['strategy'] == 'time':
        return _time_indices(df, cfg)
    n = len(df)
    seed = cfg['seed']
    strat = bool(cfg.get('stratify')) and task_type == 'classification'
    if cfg['strategy'] == 'train_test':
        tr, te, notes = random_split(n, cfg['test_frac'], seed, cfg['shuffle'],
                                     y=y, stratify=strat, groups=groups)
        return {'train': tr, 'test': te}, notes
    tr_rest, te, notes = random_split(n, cfg['test_frac'], seed, cfg['shuffle'],
                                      y=y, stratify=strat, groups=groups)
    if len(tr_rest) < 2:
        raise SplitError('Not enough rows left for train/validation after test split.')
    rel_val = cfg['val_frac'] / (cfg['train_frac'] + cfg['val_frac'])
    y_rest = y[tr_rest] if y is not None else None
    g_rest = groups[tr_rest] if groups is not None else None
    # Second split reuses seed+1 so train/val differ from the test cut.
    tr_rel, va_rel, notes2 = random_split(len(tr_rest), rel_val, seed + 1, cfg['shuffle'],
                                          y=y_rest, stratify=strat, groups=g_rest)
    tr, va = tr_rest[np.sort(tr_rel)], tr_rest[np.sort(va_rel)]
    return {'train': np.sort(tr), 'validation': np.sort(va), 'test': np.sort(te)}, notes + notes2


def describe_split(df, parts, target=None):
    """Row counts + class distributions for UI display."""
    desc = {'counts': {k: int(len(v)) for k, v in parts.items()}, 'classes': {}}
    if target and target in df.columns:
        for k, idx in parts.items():
            vc = df.iloc[np.asarray(idx)][target].astype(str).value_counts()
            desc['classes'][k] = [{'value': str(v), 'count': int(n)} for v, n in vc.items()]
    return desc
