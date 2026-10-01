"""Curated classical-ML estimator catalog (scikit-learn).

Task-specific, small, and explicit. Each entry carries defaults plus a UI
parameter schema: {name, label, type, default, min/max/options}.
"""
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelError(ValueError):
    pass


def _num(name, label, default, lo=None, hi=None, step=None, integer=False):
    s = {'name': name, 'label': label, 'type': 'int' if integer else 'float',
         'default': default}
    if lo is not None:
        s['min'] = lo
    if hi is not None:
        s['max'] = hi
    if step is not None:
        s['step'] = step
    return s


def _choice(name, label, options, default):
    return {'name': name, 'label': label, 'type': 'choice',
            'options': list(options), 'default': default}


CATALOG = {
    'LogisticRegression': {
        'task': 'classification', 'label': 'Logistic Regression',
        'class': LogisticRegression,
        'defaults': {'C': 1.0, 'max_iter': 1000},
        'params': [
            _num('C', 'C (regularization strength)', 1.0, 0.001, 1000),
            _num('max_iter', 'Max iterations', 1000, 100, 10000, 100, True),
        ],
    },
    'DecisionTreeClassifier': {
        'task': 'classification', 'label': 'Decision Tree',
        'class': DecisionTreeClassifier,
        'defaults': {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        'params': [
            _num('max_depth', 'Max depth (0 = unlimited)', 0, 0, 50, 1, True),
            _num('min_samples_split', 'Min samples split', 2, 2, 100, 1, True),
            _num('min_samples_leaf', 'Min samples leaf', 1, 1, 100, 1, True),
        ],
    },
    'RandomForestClassifier': {
        'task': 'classification', 'label': 'Random Forest',
        'class': RandomForestClassifier,
        'defaults': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
                     'min_samples_leaf': 1, 'max_features': 'sqrt'},
        'params': [
            _num('n_estimators', 'Trees', 100, 10, 1000, 10, True),
            _num('max_depth', 'Max depth (0 = unlimited)', 0, 0, 50, 1, True),
            _num('min_samples_split', 'Min samples split', 2, 2, 100, 1, True),
            _num('min_samples_leaf', 'Min samples leaf', 1, 1, 100, 1, True),
            _choice('max_features', 'Max features', ['sqrt', 'log2', None], 'sqrt'),
        ],
    },
    'GradientBoostingClassifier': {
        'task': 'classification', 'label': 'Gradient Boosting',
        'class': GradientBoostingClassifier,
        'defaults': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        'params': [
            _num('n_estimators', 'Estimators', 100, 10, 1000, 10, True),
            _num('learning_rate', 'Learning rate', 0.1, 0.01, 1.0),
            _num('max_depth', 'Max depth', 3, 1, 10, 1, True),
        ],
    },
    'HistGradientBoostingClassifier': {
        'task': 'classification', 'label': 'Hist Gradient Boosting',
        'class': HistGradientBoostingClassifier,
        'defaults': {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': None},
        'params': [
            _num('max_iter', 'Iterations', 100, 10, 1000, 10, True),
            _num('learning_rate', 'Learning rate', 0.1, 0.01, 1.0),
            _num('max_depth', 'Max depth (0 = unlimited)', 0, 0, 30, 1, True),
        ],
    },
    'SVC': {
        'task': 'classification', 'label': 'SVC',
        'class': SVC,
        'defaults': {'C': 1.0, 'kernel': 'rbf', 'probability': True},
        'params': [
            _num('C', 'C (regularization strength)', 1.0, 0.001, 1000),
            _choice('kernel', 'Kernel', ['rbf', 'linear', 'poly'], 'rbf'),
        ],
    },
    'KNeighborsClassifier': {
        'task': 'classification', 'label': 'K-Neighbors',
        'class': KNeighborsClassifier,
        'defaults': {'n_neighbors': 5},
        'params': [_num('n_neighbors', 'Neighbors', 5, 1, 100, 1, True)],
    },
    'GaussianNB': {
        'task': 'classification', 'label': 'Naive Bayes',
        'class': GaussianNB,
        'defaults': {},
        'params': [],
    },
    'LinearRegression': {
        'task': 'regression', 'label': 'Linear Regression',
        'class': LinearRegression,
        'defaults': {},
        'params': [],
    },
    'Ridge': {
        'task': 'regression', 'label': 'Ridge',
        'class': Ridge,
        'defaults': {'alpha': 1.0},
        'params': [_num('alpha', 'Alpha', 1.0, 0.0001, 1000)],
    },
    'Lasso': {
        'task': 'regression', 'label': 'Lasso',
        'class': Lasso,
        'defaults': {'alpha': 1.0, 'max_iter': 5000},
        'params': [
            _num('alpha', 'Alpha', 1.0, 0.0001, 1000),
            _num('max_iter', 'Max iterations', 5000, 100, 20000, 100, True),
        ],
    },
    'DecisionTreeRegressor': {
        'task': 'regression', 'label': 'Decision Tree',
        'class': DecisionTreeRegressor,
        'defaults': {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        'params': [
            _num('max_depth', 'Max depth (0 = unlimited)', 0, 0, 50, 1, True),
            _num('min_samples_split', 'Min samples split', 2, 2, 100, 1, True),
            _num('min_samples_leaf', 'Min samples leaf', 1, 1, 100, 1, True),
        ],
    },
    'RandomForestRegressor': {
        'task': 'regression', 'label': 'Random Forest',
        'class': RandomForestRegressor,
        'defaults': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
                     'min_samples_leaf': 1},
        'params': [
            _num('n_estimators', 'Trees', 100, 10, 1000, 10, True),
            _num('max_depth', 'Max depth (0 = unlimited)', 0, 0, 50, 1, True),
            _num('min_samples_split', 'Min samples split', 2, 2, 100, 1, True),
            _num('min_samples_leaf', 'Min samples leaf', 1, 1, 100, 1, True),
        ],
    },
    'GradientBoostingRegressor': {
        'task': 'regression', 'label': 'Gradient Boosting',
        'class': GradientBoostingRegressor,
        'defaults': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        'params': [
            _num('n_estimators', 'Estimators', 100, 10, 1000, 10, True),
            _num('learning_rate', 'Learning rate', 0.1, 0.01, 1.0),
            _num('max_depth', 'Max depth', 3, 1, 10, 1, True),
        ],
    },
    'HistGradientBoostingRegressor': {
        'task': 'regression', 'label': 'Hist Gradient Boosting',
        'class': HistGradientBoostingRegressor,
        'defaults': {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': None},
        'params': [
            _num('max_iter', 'Iterations', 100, 10, 1000, 10, True),
            _num('learning_rate', 'Learning rate', 0.1, 0.01, 1.0),
            _num('max_depth', 'Max depth (0 = unlimited)', 0, 0, 30, 1, True),
        ],
    },
    'SVR': {
        'task': 'regression', 'label': 'SVR',
        'class': SVR,
        'defaults': {'C': 1.0, 'kernel': 'rbf'},
        'params': [
            _num('C', 'C (regularization strength)', 1.0, 0.001, 1000),
            _choice('kernel', 'Kernel', ['rbf', 'linear', 'poly'], 'rbf'),
        ],
    },
    'KNeighborsRegressor': {
        'task': 'regression', 'label': 'K-Neighbors',
        'class': KNeighborsRegressor,
        'defaults': {'n_neighbors': 5},
        'params': [_num('n_neighbors', 'Neighbors', 5, 1, 100, 1, True)],
    },
}

# UI sends max_depth 0 for unlimited (None is not JSON-friendly in forms).
_NONE_SENTINELS = {'max_depth': 0, 'max_features': None}


def models_for_task(task):
    return {k: v for k, v in CATALOG.items() if v['task'] == task}


def build_estimator(family, params, seed):
    if family not in CATALOG:
        raise ModelError('Unknown model family: %r.' % (family,))
    entry = CATALOG[family]
    merged = dict(entry['defaults'])
    for k, v in (params or {}).items():
        if k not in merged and k not in [p['name'] for p in entry['params']]:
            raise ModelError('Unknown parameter %r for %s.' % (k, family))
        merged[k] = v
    if 'max_depth' in merged and merged['max_depth'] == 0:
        merged['max_depth'] = None
    cls = entry['class']
    try:
        est = cls(**merged)
    except TypeError as e:
        raise ModelError('Invalid parameters for %s: %s' % (family, e))
    # Seed where supported (deterministic runs by default).
    try:
        if 'random_state' in cls().get_params():
            est.set_params(random_state=seed)
    except Exception:
        pass
    return est, merged


def supports_proba(estimator):
    return hasattr(estimator, 'predict_proba')
