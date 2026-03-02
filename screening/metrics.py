from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict[str, float]:
    y_pred = y_prob.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    if num_classes == 2:
        try:
            out["macro_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            out["macro_auc"] = float("nan")
        out["micro_auc"] = out["macro_auc"]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["sensitivity"] = float(tp / max(tp + fn, 1))
        out["specificity"] = float(tn / max(tn + fp, 1))
    else:
        try:
            out["macro_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            out["macro_auc"] = float("nan")
        try:
            onehot = np.eye(num_classes, dtype=float)[y_true.astype(int)]
            out["micro_auc"] = float(roc_auc_score(onehot, y_prob, average="micro", multi_class="ovr"))
        except Exception:
            out["micro_auc"] = float("nan")
        cm = confusion_matrix(y_true, y_pred)
        sens = []
        spec = []
        for c in range(num_classes):
            tp = cm[c, c]
            fn = cm[c].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sens.append(tp / max(tp + fn, 1))
            spec.append(tn / max(tn + fp, 1))
        out["sensitivity"] = float(np.mean(sens))
        out["specificity"] = float(np.mean(spec))

    return out
