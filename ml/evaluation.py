"""Classification/regression metrics with validity guards.

Never computes a metric that is mathematically invalid for the task/model:
ROC/PR-AUC and log-loss require probability output; ROC-AUC needs at least
2 classes present; multiclass ROC uses one-vs-rest macro averaging.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)


def _clean(values):
    if isinstance(values, np.ndarray):
        return values.tolist()
    try:
        return [float(v) for v in values]
    except Exception:
        return list(values)


def classification_metrics(y_true, y_pred, y_proba=None, labels=None, max_curve_points=200):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [str(l) for l in (labels if labels is not None
                               else sorted(set(str(v) for v in y_true) |
                                           set(str(v) for v in y_pred)))]
    out = {
        'task': 'classification',
        'labels': labels,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
    }
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    out['macro'] = {'precision': float(np.mean(prec)), 'recall': float(np.mean(rec)),
                    'f1': float(np.mean(f1))}
    out['weighted'] = {
        'precision': float(np.average(prec, weights=sup)) if sup.sum() else 0.0,
        'recall': float(np.average(rec, weights=sup)) if sup.sum() else 0.0,
        'f1': float(np.average(f1, weights=sup)) if sup.sum() else 0.0,
    }
    out['per_class'] = [{
        'label': labels[i], 'precision': float(prec[i]), 'recall': float(rec[i]),
        'f1': float(f1[i]), 'support': int(sup[i]),
    } for i in range(len(labels))]
    try:
        out['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    except Exception:
        out['confusion_matrix'] = None
    out['roc_auc'] = None
    out['pr_auc'] = None
    out['log_loss'] = None
    out['roc_curve'] = None
    out['pr_curve'] = None
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        n_classes = len(labels)
        try:
            if n_classes == 2 and proba.ndim == 2 and proba.shape[1] == 2:
                p1 = proba[:, 1]
                yb = (y_true.astype(str) == labels[1]).astype(int)
                if len(set(yb.tolist())) == 2:
                    out['roc_auc'] = float(roc_auc_score(yb, p1))
                    out['pr_auc'] = float(average_precision_score(yb, p1))
                    fpr, tpr, _ = roc_curve(yb, p1)
                    out['roc_curve'] = {'fpr': _down(fpr, max_curve_points),
                                        'tpr': _down(tpr, max_curve_points)}
                    pr, rc, _ = precision_recall_curve(yb, p1)
                    out['pr_curve'] = {'precision': _down(pr, max_curve_points),
                                       'recall': _down(rc, max_curve_points)}
            elif n_classes > 2 and proba.ndim == 2 and proba.shape[1] == n_classes:
                yt = np.array([labels.index(str(v)) for v in y_true])
                out['roc_auc'] = float(roc_auc_score(yt, proba, multi_class='ovr', average='macro'))
        except Exception:
            pass
        try:
            if proba.ndim == 2 and proba.shape[1] == len(labels):
                yt = np.array([labels.index(str(v)) for v in y_true])
                out['log_loss'] = float(log_loss(yt, proba, labels=list(range(len(labels)))))
        except Exception:
            pass
        try:
            out['proba_summary'] = {
                'mean_max_proba': float(np.max(proba, axis=1).mean()) if proba.ndim == 2 else None,
            }
        except Exception:
            out['proba_summary'] = None
    else:
        out['proba_summary'] = None
    return out


def _down(arr, n):
    vals = [float(v) for v in np.asarray(arr).tolist()]
    if len(vals) <= n:
        return vals
    idx = np.linspace(0, len(vals) - 1, n).astype(int)
    return [vals[i] for i in idx]


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    out = {
        'task': 'regression',
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
    }
    try:
        out['r2'] = float(r2_score(y_true, y_pred))
    except Exception:
        out['r2'] = None
    return out
