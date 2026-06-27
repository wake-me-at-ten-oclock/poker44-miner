#!/usr/bin/env python3
"""Poker44 winning solution: stacked discriminative ensemble.

Key fact: the validator composite (0.75*AP + 0.25*recall@FPR<=0.05) is
RANK-BASED, so a single miner's own composite is invariant to any monotone
transform of its scores. Calibration only helps the POOLED metric (the window
buffer concatenates scores across chunks). So:
  - primary lever  = raw discrimination (the stack)
  - secondary lever = cross-chunk score comparability (isotonic blend)

We report a date-grouped OOF stack and a POOLED composite (the realistic
windowed metric), compare to the single-LGBM baseline and the reference floor,
and save an inference artifact.
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import joblib

from p44_features import vectorize, FEATURE_NAMES

warnings.filterwarnings("ignore")
RAW = os.path.join(os.path.dirname(__file__), "data", "raw")
RNG = np.random.default_rng(42)


# ---- validator scoring (verbatim) ------------------------------------------
def _recall_at_fpr(y_score, y_true, max_fpr=0.05):
    labels = np.asarray(y_true, int); scores = np.asarray(y_score, float)
    pos = int((labels == 1).sum()); neg = int((labels == 0).sum())
    if pos <= 0 or neg <= 0 or scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort"); sl = labels[order]
    tp = np.cumsum(sl == 1); fp = np.cumsum(sl == 0)
    recall = tp / pos; fpr = fp / neg
    allowed = fpr <= max_fpr
    return float(recall[allowed].max()) if np.any(allowed) else 0.0


def composite(y_pred, y_true):
    y_pred = np.asarray(y_pred, float); y_true = np.asarray(y_true, int)
    ap = average_precision_score(y_true, y_pred) if np.any(y_true == 1) else 0.0
    rec = _recall_at_fpr(y_pred, y_true)
    return 0.75 * ap + 0.25 * rec, ap, rec


# ---- data ------------------------------------------------------------------
def load_groups():
    """Return list of (hands, label, date)."""
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        p = json.load(open(path)); sd = p["sourceDate"]
        for rec in p["records"]:
            for grp, lab in zip(rec.get("chunks") or [], rec.get("groundTruth") or []):
                if grp:
                    out.append((grp, int(lab), sd))
    return out


def augment(hands, n_variants=3, frac=0.7):
    """Subsample hands to create size-robust training variants."""
    out = [hands]
    k = max(8, int(len(hands) * frac))
    if len(hands) <= 10:
        return out
    for _ in range(n_variants):
        idx = RNG.choice(len(hands), size=k, replace=False)
        out.append([hands[i] for i in sorted(idx)])
    return out


def base_models():
    return [
        ("lgbm", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                    min_child_samples=12, subsample=0.8, colsample_bytree=0.8,
                                    reg_lambda=1.0, verbose=-1, n_jobs=1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, l2_regularization=1.0)),
        ("et", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=1, random_state=1)),
        ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=1, random_state=2)),
    ]


def fit_stack(Xtr, ytr, gtr):
    """Fit base learners + LogReg meta on inner-OOF base preds. Returns predictor."""
    models = base_models()
    inner = GroupKFold(n_splits=min(5, len(set(gtr))))
    oof = np.zeros((len(ytr), len(models)))
    for j, (_, proto) in enumerate(models):
        for tr, va in inner.split(Xtr, ytr, gtr):
            from sklearn.base import clone
            m = clone(proto); m.fit(Xtr[tr], ytr[tr])
            oof[va, j] = m.predict_proba(Xtr[va])[:, 1]
    meta = LogisticRegression(max_iter=1000, C=1.0); meta.fit(oof, ytr)
    fitted = []
    for _, proto in models:
        from sklearn.base import clone
        m = clone(proto); m.fit(Xtr, ytr); fitted.append(m)

    def predict(X):
        Z = np.column_stack([m.predict_proba(X)[:, 1] for m in fitted])
        return meta.predict_proba(Z)[:, 1]

    return predict, fitted, meta


# ---- reference floor -------------------------------------------------------
def ref_floor(groups):
    from collections import Counter
    def sc(h):
        a = h.get("actions") or []; pl = h.get("players") or []; st = h.get("streets") or []
        ac = Counter(x.get("action_type") for x in a)
        ma = max(1, sum(ac.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
        clamp = lambda v: max(0.0, min(1.0, v))
        s = 0.32*(len(st)/3.0) + 0.18*clamp((ac.get("call",0)/ma)/0.35) \
            + 0.12*clamp((ac.get("check",0)/ma)/0.30) \
            + 0.08*clamp(((6-min(len(pl),6))/4.0) if pl else 0.0) \
            - 0.18*clamp((ac.get("fold",0)/ma)/0.55) - 0.10*clamp((ac.get("raise",0)/ma)/0.20)
        return clamp(s)
    preds = [float(np.mean([sc(h) for h in g])) for g, _, _ in groups]
    ys = [y for _, y, _ in groups]
    return composite(preds, ys)


def main():
    groups = load_groups()
    dates = sorted({d for _, _, d in groups})
    print(f"groups={len(groups)} dates={len(dates)} feats={len(FEATURE_NAMES)}")

    Xfull = np.array([vectorize(g) for g, _, _ in groups])
    yfull = np.array([y for _, y, _ in groups])
    gfull = np.array([d for _, _, d in groups])

    rew, ap, rec = ref_floor(groups)
    print(f"[reference floor]   composite={rew:.4f} AP={ap:.4f} rec={rec:.4f}")

    # date-grouped OUTER CV, collect pooled OOF predictions
    outer = GroupKFold(n_splits=6)
    base_oof = np.zeros(len(yfull)); stack_oof = np.zeros(len(yfull))
    per_date = {"lgbm": [], "stack": []}
    for tr, va in outer.split(Xfull, yfull, gfull):
        # augmented training set (variants only added to TRAIN)
        Xa, ya, ga = [], [], []
        for i in tr:
            for v in augment(groups[i][0]):
                Xa.append(vectorize(v)); ya.append(yfull[i]); ga.append(gfull[i])
        Xa = np.array(Xa); ya = np.array(ya); ga = np.array(ga)

        # single-LGBM baseline
        b = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=12, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, verbose=-1, n_jobs=1)
        b.fit(Xa, ya); base_oof[va] = b.predict_proba(Xfull[va])[:, 1]

        # full stack
        predict, _, _ = fit_stack(Xa, ya, ga)
        stack_oof[va] = predict(Xfull[va])

    pooled_b = composite(base_oof, yfull)
    pooled_s = composite(stack_oof, yfull)
    print(f"[single LGBM pooled] composite={pooled_b[0]:.4f} AP={pooled_b[1]:.4f} rec={pooled_b[2]:.4f}")
    print(f"[STACK pooled]       composite={pooled_s[0]:.4f} AP={pooled_s[1]:.4f} rec={pooled_s[2]:.4f}")

    # per-date composite (within-chunk view)
    pdc = []
    for d in dates:
        m = gfull == d
        if m.sum() >= 4 and len(set(yfull[m])) == 2:
            pdc.append(composite(stack_oof[m], yfull[m])[0])
    print(f"[STACK per-date]     mean={np.mean(pdc):.4f}±{np.std(pdc):.3f} over {len(pdc)} dates")

    # isotonic cross-chunk effect on pooled metric
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(stack_oof, yfull)
    blend = 0.6 * iso.transform(stack_oof) + 0.4 * stack_oof
    pooled_iso = composite(blend, yfull)
    print(f"[STACK + isotonic]   pooled composite={pooled_iso[0]:.4f} (Δ {pooled_iso[0]-pooled_s[0]:+.4f})")

    # ---- train FINAL artifact on ALL data (augmented) ----
    Xa, ya, ga = [], [], []
    for i in range(len(groups)):
        for v in augment(groups[i][0]):
            Xa.append(vectorize(v)); ya.append(yfull[i]); ga.append(gfull[i])
    Xa = np.array(Xa); ya = np.array(ya); ga = np.array(ga)
    predict, fitted, meta = fit_stack(Xa, ya, ga)
    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(predict(Xfull), yfull)
    art = {"feature_names": FEATURE_NAMES, "base_models": fitted, "meta": meta,
           "iso": final_iso, "iso_blend": 0.6}
    out = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out, exist_ok=True)
    joblib.dump(art, os.path.join(out, "p44_stack.joblib"))
    print(f"\nsaved artifact -> models/p44_stack.joblib  ({len(fitted)} base + meta + iso)")
    imp = sorted(zip(FEATURE_NAMES, fitted[0].feature_importances_), key=lambda x: -x[1])
    print("top 12 features:", ", ".join(f"{n}" for n, _ in imp[:12]))


if __name__ == "__main__":
    main()
