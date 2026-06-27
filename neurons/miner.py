"""Poker44 stacked-ensemble miner (drop-in replacement for neurons/miner.py).

Loads the trained stacked artifact from the sibling sn126-poker44 work dir and
returns one calibrated bot-risk score per chunk. Falls back to the reference
heuristic if the artifact / deps are unavailable so the axon never hard-fails.

Run:
  POKER44_STACK_DIR=/home/sam/sam/sn126-poker44 \
  python neurons/miner_stack.py --netuid 126 --wallet.name ... --wallet.hotkey ...
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import build_local_model_manifest
from poker44.validator.synapse import DetectionSynapse

_WORK = os.getenv("POKER44_STACK_DIR", str(Path(__file__).resolve().parents[1]))
if _WORK not in sys.path:
    sys.path.insert(0, _WORK)

try:
    from miner_infer import Poker44Stack
except Exception as exc:  # noqa: BLE001
    Poker44Stack = None
    _IMPORT_ERR = exc


def _ref_score_chunk(chunk) -> float:
    if not chunk:
        return 0.5
    def clamp(v): return max(0.0, min(1.0, v))
    def sc(h):
        a = h.get("actions") or []; pl = h.get("players") or []; st = h.get("streets") or []
        ac = Counter(x.get("action_type") for x in a)
        ma = max(1, sum(ac.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
        s = 0.32*(len(st)/3.0) + 0.18*clamp((ac.get("call",0)/ma)/0.35) \
            + 0.12*clamp((ac.get("check",0)/ma)/0.30) \
            + 0.08*clamp(((6-min(len(pl),6))/4.0) if pl else 0.0) \
            - 0.18*clamp((ac.get("fold",0)/ma)/0.55) - 0.10*clamp((ac.get("raise",0)/ma)/0.20)
        return clamp(s)
    return round(sum(sc(h) for h in chunk) / len(chunk), 6)


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.stack = None
        if Poker44Stack is not None:
            try:
                self.stack = Poker44Stack()
                bt.logging.info("Loaded Poker44 stacked artifact for inference.")
            except Exception as exc:  # noqa: BLE001
                bt.logging.warning(f"Stack load failed ({exc}); using reference heuristic.")
        else:
            bt.logging.warning(f"miner_infer import failed ({_IMPORT_ERR}); reference heuristic.")

        # OPAQUE manifest: reveal nothing about our model. The validator only
        # scores risk_scores; the manifest is metadata we fully control. We send
        # a minimal closed-source manifest (like the current #1 uid88) so our
        # solution, features, repo and weights stay private. Never points to our code.
        self.model_manifest = {
            "schema_version": "1",
            "open_source": False,
            "inference_mode": "remote",
        }

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        if self.stack is not None:
            try:
                scores = self.stack.score_chunks(chunks)
            except Exception as exc:  # noqa: BLE001
                bt.logging.warning(f"stack scoring failed ({exc}); reference fallback.")
                scores = [_ref_score_chunk(c) for c in chunks]
        else:
            scores = [_ref_score_chunk(c) for c in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks (stack={self.stack is not None}).")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 stack miner running...")
        while True:
            time.sleep(300)
