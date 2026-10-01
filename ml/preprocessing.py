"""Sklearn Pipeline / ColumnTransformer construction.

CRITICAL INVARIANT (enforced by experiment_runner, unit-tested here):
preprocessors are FIT ONLY on training data. This module only BUILDS the
(unfitted) transformer from a serializable config; fitting happens on the
train split inside a Pipeline. Never call fit on full/test data.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

import numpy as np


class PreprocessingError(ValueError):
    pass


NUM_IMPUTE = ('median', 'mean', 'most_frequent', 'constant')
CAT_IMPUTE = ('most_frequent', 'constant')
NUM_SCALER = ('none', 'standard', 'minmax')
CAT_ENCODER = ('onehot', 'ordinal')


def validate_preprocessing_config(cfg, numeric_cols, categorical_cols):
    cfg = cfg or {}
    num_cols = [c for c in numeric_cols]
    cat_cols = [c for c in categorical_cols]
    num = cfg.get('numeric') or {}
    cat = cfg.get('categorical') or {}
    n_imp = num.get('impute', 'median')
    if n_imp not in NUM_IMPUTE:
        raise PreprocessingError('Unknown numeric imputation: %r.' % (n_imp,))
    n_scaler = num.get('scaler', 'standard')
    if n_scaler not in NUM_SCALER:
        raise PreprocessingError('Unknown numeric scaler: %r.' % (n_scaler,))
    c_imp = cat.get('impute', 'most_frequent')
    if c_imp not in CAT_IMPUTE:
        raise PreprocessingError('Unknown categorical imputation: %r.' % (c_imp,))
    c_enc = cat.get('encoder', 'onehot')
    if c_enc not in CAT_ENCODER:
        raise PreprocessingError('Unknown categorical encoder: %r.' % (c_enc,))
    for key in ('numeric_columns', 'categorical_columns'):
        vals = cfg.get(key) or []
        if not isinstance(vals, list) or any(not isinstance(v, str) for v in vals):
            raise PreprocessingError('%s must be a list of column names.' % key)
    for c in (cfg.get('numeric_columns') or []):
        if c not in num_cols:
            raise PreprocessingError('Numeric column %r is not numeric.' % c)
    for c in (cfg.get('categorical_columns') or []):
        if c not in cat_cols:
            raise PreprocessingError('Categorical column %r is not categorical.' % c)
    log_cols = [c for c in (num.get('log_columns') or []) if c in num_cols]
    return {
        'numeric_columns': list(cfg.get('numeric_columns') or num_cols),
        'categorical_columns': list(cfg.get('categorical_columns') or cat_cols),
        'numeric': {
            'impute': n_imp,
            'impute_value': num.get('impute_value', 0),
            'scaler': n_scaler,
            'log_columns': log_cols,
        },
        'categorical': {'impute': c_imp, 'impute_value': cat.get('impute_value', 'missing'),
                        'encoder': c_enc},
    }


def _numeric_steps(spec):
    steps = [('impute', SimpleImputer(strategy=spec['impute'],
                                      fill_value=spec['impute_value']))]
    if spec['scaler'] == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif spec['scaler'] == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    return steps


def build_preprocessor(cfg, numeric_cols, categorical_cols):
    """Build an UNFITTED ColumnTransformer (or None when nothing to do)."""
    spec = validate_preprocessing_config(cfg, numeric_cols, categorical_cols)
    transformers = []
    num_cols = [c for c in spec['numeric_columns'] if c in numeric_cols]
    cat_cols = [c for c in spec['categorical_columns'] if c in categorical_cols]
    log_cols = [c for c in spec['numeric'].get('log_columns', []) if c in num_cols]
    num_main = [c for c in num_cols if c not in log_cols]
    if num_main:
        transformers.append(('num', Pipeline(_numeric_steps(spec['numeric'])), num_main))
    if log_cols:
        # log1p branch on raw values (impute first so NaNs do not poison
        # the log); kept separate so the main numeric scaler is untouched.
        log_steps = [('impute', SimpleImputer(strategy=spec['numeric']['impute'],
                                              fill_value=spec['numeric']['impute_value'])),
                     ('log', FunctionTransformer(np.log1p, validate=False))]
        if spec['numeric']['scaler'] == 'standard':
            log_steps.append(('scaler', StandardScaler()))
        elif spec['numeric']['scaler'] == 'minmax':
            log_steps.append(('scaler', MinMaxScaler()))
        transformers.append(('log', Pipeline(log_steps), log_cols))
    if cat_cols:
        if spec['categorical']['encoder'] == 'onehot':
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy=spec['categorical']['impute'],
                                     fill_value=spec['categorical']['impute_value'])),
            ('encode', enc),
        ]), cat_cols))
    if not transformers:
        return None, spec
    ct = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
    return ct, spec


def summarize_preprocessor(spec):
    """Human-readable pipeline summary for the UI."""
    lines = []
    if spec['numeric_columns']:
        n = spec['numeric']
        lines.append('numeric [%s]: impute=%s, scaler=%s%s' % (
            ', '.join(spec['numeric_columns']), n['impute'], n['scaler'],
            (', log(%s)' % ', '.join(n['log_columns'])) if n['log_columns'] else ''))
    if spec['categorical_columns']:
        c = spec['categorical']
        lines.append('categorical [%s]: impute=%s, encoder=%s' % (
            ', '.join(spec['categorical_columns']), c['impute'], c['encoder']))
    if not lines:
        return 'passthrough (no transformations)'
    return '; '.join(lines)


def fitted_feature_names(preprocessor, input_columns):
    """Output feature names after fitting (best effort, never raises)."""
    try:
        names = list(preprocessor.get_feature_names_out())
        return [str(n) for n in names]
    except Exception:
        pass
    try:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'remainder' or trans == 'drop':
                continue
            cols = list(cols)
            if name == 'cat':
                enc = trans.named_steps.get('encode')
                if hasattr(enc, 'get_feature_names_out'):
                    try:
                        names.extend(str(n) for n in enc.get_feature_names_out(cols))
                        continue
                    except Exception:
                        pass
            names.extend(str(c) for c in cols)
        return names or list(input_columns)
    except Exception:
        return list(input_columns)
