"""Random-Forest land-cover classifier over 8 spectral features.

Training protocol (per skill: spatial-block CV is mandatory):
- Labels: WorldCover-2021-derived cells with purity >= 0.70 for epoch-2021.
- Spatial-block CV: GroupKFold(5) on 5x5-cell blocks, class-balanced
  subsampling so each fold sees balanced classes, per-fold test set kept
  unbalanced (real prior). Metrics: overall accuracy, per-class recall,
  macro-F1, kappa. Holdout overall accuracy is reported honestly.
- Change map: classify epoch 2019 and 2026, then transition filtering.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lib.aois import AOI, CLASSES, EPOCH_BEFORE, EPOCH_AFTER, EPOCH_TRAIN
from lib.stac_io import DERIVED_DIR, FEATURE_NAMES, build_features

MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models"))

BLOCK = 5  # spatial CV block size in cells (5 x 5 = 450 m blocks)


def _load_npz(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def epoch_features(aoi: AOI, epoch_key: str) -> Optional[np.ndarray]:
    d = _load_npz(os.path.join(DERIVED_DIR, f"{aoi.id}_{epoch_key}.npz"))
    if d is None or "features" not in d:
        return None
    return d["features"]


def training_matrix(aois: List[AOI]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stack labelled cells across AOIs: X [n, 8], y, block_id [n]."""
    Xs, ys, bs = [], [], []
    for aoi in aois:
        feats = epoch_features(aoi, EPOCH_TRAIN)
        labs = _load_npz(os.path.join(DERIVED_DIR, f"{aoi.id}_labels.npz"))
        if feats is None or labs is None:
            return None
        labels = labs["labels"]
        rows, cols = labels.shape
        f = feats[:rows, :cols]
        mask = labels != -1
        if not mask.any():
            continue
        rr, cc = np.nonzero(mask)
        Xs.append(f[rr, cc])
        ys.append(labels[rr, cc])
        blocks = ((rr // BLOCK) * 1000 + (cc // BLOCK)).astype("int64")
        bs.append(blocks)
    if not Xs:
        return None
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(bs))


def _balance_classes(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                     per_class: int = 12000) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample each class down to per_class rows (keeps all if smaller)."""
    keep_idx = []
    for c in range(len(CLASSES)):
        idx = np.nonzero(y == c)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep_idx.append(idx)
    keep = np.concatenate(keep_idx)
    return X[keep], y[keep]


def train_eval(aois: List[AOI], seed: int = 42) -> Optional[dict]:
    """Spatial-block 5-fold CV; returns metrics + the full-data model."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                                  cohen_kappa_score)

    mat = training_matrix(aois)
    if mat is None:
        print("  [model] missing composites or labels - cannot train")
        return None
    X, y, blocks = mat
    print(f"  [model] training cells: {len(y)}")

    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=5)
    accs, f1s, kappas, per_class_recall = [], [], [], {c: [] for c in CLASSES}
    for tr_idx, te_idx in gkf.split(X, y, groups=blocks):
        X_tr, y_tr = _balance_classes(X[tr_idx], y[tr_idx], rng)
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X[te_idx])
        accs.append(accuracy_score(y[te_idx], pred))
        f1s.append(f1_score(y[te_idx], pred, average="macro"))
        kappas.append(cohen_kappa_score(y[te_idx], pred))
        rec = recall_score(y[te_idx], pred, average=None,
                           labels=list(range(len(CLASSES))),
                           zero_division=0)
        for i, c in enumerate(CLASSES):
            per_class_recall[c].append(float(rec[i]))

    metrics = {
        "n_train": int(len(y)),
        "cv_accuracy": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_macro_f1": float(np.mean(f1s)),
        "cv_kappa": float(np.mean(kappas)),
        "per_class_recall": {c: float(np.mean(v)) for c, v in
                             per_class_recall.items()},
    }

    # final model on all labelled data (balanced)
    X_all, y_all = _balance_classes(X, y, rng)
    rf_full = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf_full.fit(X_all, y_all)

    os.makedirs(MODELS_DIR, exist_ok=True)
    import joblib
    joblib.dump(rf_full, os.path.join(MODELS_DIR, "rf_landcover.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  [model] spatial-CV acc={metrics['cv_accuracy']:.3f}"
          f" macro-F1={metrics['cv_macro_f1']:.3f} kappa={metrics['cv_kappa']:.3f}")
    return metrics


def load_model() -> RandomForestClassifier:
    import joblib
    return joblib.load(os.path.join(MODELS_DIR, "rf_landcover.joblib"))


def predict_epoch(aoi: AOI, epoch_key: str,
                  model: RandomForestClassifier) -> Optional[np.ndarray]:
    """Classify every usable cell of an epoch; -1 where features are NaN."""
    feats = epoch_features(aoi, epoch_key)
    if feats is None:
        return None
    rows, cols, _ = feats.shape
    valid = np.isfinite(feats).all(axis=2)
    pred = np.full((rows, cols), -1, dtype="int16")
    if valid.any():
        flat = feats[valid].reshape(-1, 8)
        pred_v = model.predict(flat)
        pred[valid] = pred_v.astype("int16")
    np.savez_compressed(
        os.path.join(DERIVED_DIR, f"{aoi.id}_{epoch_key}_pred.npz"),
        pred=pred)
    return pred
