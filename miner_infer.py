"""Poker44 stacked-model inference for the miner forward path.

Loads models/p44_stack.joblib and scores a chunk (list of masked hand dicts)
into a bot-risk probability in [0,1]. Cross-chunk isotonic blend is applied so
pooled (windowed) scores stay comparable across chunks.
"""
from __future__ import annotations

import os
import numpy as np
import joblib

from p44_combined import vectorize

_ART = os.path.join(os.path.dirname(__file__), "models", "p44_final.joblib")


class Poker44Stack:
    """Loads the final combined-feature model and scores chunks -> bot risk."""

    def __init__(self, path: str = _ART):
        a = joblib.load(path)
        self.feature_names = a["feature_names"]
        self.model = a["model"]

    def score_chunks(self, chunks) -> list[float]:
        if not chunks:
            return []
        X = np.array([vectorize(c) for c in chunks], dtype=np.float64)
        p = self.model.predict_proba(X)[:, 1]
        return [float(min(1.0, max(0.0, v))) for v in p]

    def score_chunk(self, chunk) -> float:
        return self.score_chunks([chunk])[0] if chunk else 0.5


if __name__ == "__main__":
    import glob, json
    raw = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "raw", "*.json")))
    s = Poker44Stack()
    p = json.load(open(raw[-1]))
    rec = p["records"][0]
    groups = rec["chunks"]; gt = rec["groundTruth"]
    scores = s.score_chunks(groups)
    print("labels:", gt)
    print("scores:", [round(x, 3) for x in scores])
