#!/usr/bin/env python3
"""Train the FINAL deployable model (combined features + tuned LGBM) on all
augmented benchmark data, and save models/p44_final.joblib."""
from __future__ import annotations
import glob, json, os, warnings
import numpy as np
import lightgbm as lgb
import joblib
warnings.filterwarnings("ignore")
from p44_combined import vectorize, FEATURE_NAMES

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")
RNG = np.random.default_rng(42)
P = dict(n_estimators=700, learning_rate=0.02, num_leaves=47, min_child_samples=8,
         subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=1)


def load():
    gs = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        p = json.load(open(path))
        for rec in p["records"]:
            for g, lab in zip(rec.get("chunks") or [], rec.get("groundTruth") or []):
                if g: gs.append((g, int(lab)))
    return gs


def augment(hands, n=3, frac=0.7):
    out = [hands]
    if len(hands) <= 10: return out
    k = max(8, int(len(hands)*frac))
    for _ in range(n):
        idx = sorted(RNG.choice(len(hands), size=k, replace=False))
        out.append([hands[i] for i in idx])
    return out


def main():
    groups = load()
    X, y = [], []
    for hands, lab in groups:
        for v in augment(hands):
            X.append(vectorize(v)); y.append(lab)
    X = np.array(X); y = np.array(y)
    print(f"train rows (augmented)={len(y)}  feats={len(FEATURE_NAMES)}")
    m = lgb.LGBMClassifier(**P); m.fit(X, y)
    art = {"feature_module": "p44_combined", "feature_names": FEATURE_NAMES, "model": m}
    out = os.path.join(os.path.dirname(__file__), "models"); os.makedirs(out, exist_ok=True)
    joblib.dump(art, os.path.join(out, "p44_final.joblib"))
    print("saved models/p44_final.joblib")
    imp = sorted(zip(FEATURE_NAMES, m.feature_importances_), key=lambda x: -x[1])
    print("top 15:", ", ".join(n for n, _ in imp[:15]))


if __name__ == "__main__":
    main()
