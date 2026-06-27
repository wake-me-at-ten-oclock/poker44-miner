"""Winning combined feature extractor: 1st-place rich aggregation (winner_features)
+ our mechanical-repetition / pot-geometry signals (p44_features)."""
from __future__ import annotations
import numpy as np
import p44_features as ours
import winner_features as win


def combined_features(chunk) -> dict:
    d = dict(win.chunk_features(chunk))
    for k, v in ours.chunk_features(chunk).items():
        d[f"o_{k}"] = v
    d.pop("hand_count", None)  # release artifact, not behavior
    return d


# stable order from a non-empty dummy hand
_DUMMY = [{"metadata": {"hero_seat": 1, "button_seat": 1, "max_seats": 6},
           "players": [{"seat": 1, "starting_stack": 5.0}],
           "streets": [{"street": "preflop"}],
           "actions": [{"street": "preflop", "actor_seat": 1, "action_type": "call",
                        "normalized_amount_bb": 1.0, "pot_before": 1.0, "pot_after": 2.0}]}]
FEATURE_NAMES = sorted(combined_features(_DUMMY).keys())


def vectorize(chunk) -> np.ndarray:
    f = combined_features(chunk)
    return np.array([float(f.get(k, 0.0)) for k in FEATURE_NAMES], dtype=np.float64)
