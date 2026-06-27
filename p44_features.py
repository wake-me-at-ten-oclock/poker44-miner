"""Poker44 chunk-level feature extraction (size-robust, leakage-free).

A chunk = list of masked hand dicts (the validator-visible payload). We emit one
fixed-order feature vector per chunk. All features are ratios / distributional
stats so they transfer across the 30-40 hand benchmark groups and the ~100 hand
live eval chunks. NO counts, ids, dates, or hashes are used as features.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

BB = 0.02  # payload_view fixes visible bb to 0.02
ATYPES = ["fold", "call", "check", "bet", "raise"]
_BUCKETS = ["z", "xs", "s", "m", "l", "xl"]


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in ps) / math.log(len(ps))


def _max_run_share(seq):
    if not seq:
        return 0.0
    longest = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1; longest = max(longest, cur)
        else:
            cur = 1
    return longest / len(seq)


def _bucket(v):
    if v <= 0: return "z"
    if v < 1: return "xs"
    if v < 3: return "s"
    if v < 8: return "m"
    if v < 20: return "l"
    return "xl"


def _stats(arr, prefix, out):
    a = np.asarray(arr, float)
    if a.size == 0:
        out[f"{prefix}_mean"] = 0.0; out[f"{prefix}_std"] = 0.0
        out[f"{prefix}_max"] = 0.0; out[f"{prefix}_iqr"] = 0.0
        out[f"{prefix}_cv"] = 0.0
        return
    m = float(a.mean()); s = float(a.std())
    out[f"{prefix}_mean"] = m
    out[f"{prefix}_std"] = s
    out[f"{prefix}_max"] = float(a.max())
    out[f"{prefix}_iqr"] = float(np.subtract(*np.percentile(a, [75, 25]))) if a.size > 1 else 0.0
    out[f"{prefix}_cv"] = s / m if m else 0.0   # coeff of variation: bots = low CV


def chunk_features(hands) -> dict:
    f: dict[str, float] = {}
    at = Counter(); street_share = Counter()
    amounts = []; pot_growth = []; pot_before = []; pot_after = []
    raise_to_n = 0; call_to_n = 0; allin_n = 0
    nonzero = 0; tot = 0; postflop = 0
    pot_monotonic = 0; pot_steps = 0
    actor_switches = 0; actor_obs = 0
    hero_actions = 0; button_actions = 0
    stacks = []; pcounts = []; hero_button_same = 0
    per_hand_actions = []; per_hand_ent = []; per_hand_aggr = []
    sig_chunk = Counter(); first_action = Counter()
    amt_bucket_sig = Counter()
    type_run_shares = []; actor_run_shares = []

    for h in hands:
        md = h.get("metadata") or {}
        players = h.get("players") or []
        streets = h.get("streets") or []
        actions = h.get("actions") or []
        pcounts.append(len(players))
        for p in players:
            stacks.append(float(p.get("starting_stack") or 0.0) / BB)
        hero = int(md.get("hero_seat") or 0); btn = int(md.get("button_seat") or 0)
        if hero and hero == btn:
            hero_button_same += 1

        types_seq = []; actor_seq = []; hand_at = Counter(); aggr = 0
        sig = []; bsig = []
        for a in actions:
            t = str(a.get("action_type") or "").lower().strip()
            at[t] += 1; hand_at[t] += 1; tot += 1; types_seq.append(t)
            if t in ("bet", "raise"): aggr += 1
            amt = float(a.get("normalized_amount_bb") or 0.0); amounts.append(amt)
            if amt > 0: nonzero += 1
            if a.get("raise_to"): raise_to_n += 1
            if a.get("call_to"): call_to_n += 1
            st = str(a.get("street") or "").lower(); street_share[st] += 1
            if st and st != "preflop": postflop += 1
            pb = float(a.get("pot_before") or 0.0) / BB
            pa = float(a.get("pot_after") or 0.0) / BB
            pot_before.append(pb); pot_after.append(pa)
            if pa >= pb: pot_monotonic += 1
            pot_steps += 1; pot_growth.append(max(0.0, pa - pb))
            actor = int(a.get("actor_seat") or 0); actor_seq.append(actor)
            if actor_seq[:-1]:
                actor_obs += 1
                if actor != actor_seq[-2]: actor_switches += 1
            if hero and actor == hero: hero_actions += 1
            if btn and actor == btn: button_actions += 1
            b = _bucket(amt)
            sig.append(f"{t}:{b}:{st[:2]}"); bsig.append(b)
        per_hand_actions.append(len(actions))
        per_hand_ent.append(_entropy(list(hand_at.values())))
        per_hand_aggr.append(aggr / max(1, len(actions)))
        if types_seq:
            first_action[types_seq[0]] += 1
            type_run_shares.append(_max_run_share(types_seq))
            actor_run_shares.append(_max_run_share(actor_seq))
        sig_chunk["|".join(sig)] += 1
        amt_bucket_sig["".join(bsig)] += 1

    nh = max(1, len(hands)); ntot = max(1, tot)
    for t in ATYPES:
        f[f"share_{t}"] = at.get(t, 0) / ntot
    f["aggression_share"] = (at.get("bet", 0) + at.get("raise", 0)) / ntot
    f["passive_share"] = (at.get("call", 0) + at.get("check", 0)) / ntot
    f["nonzero_amount_share"] = nonzero / ntot
    f["raise_to_share"] = raise_to_n / ntot
    f["call_to_share"] = call_to_n / ntot
    f["postflop_share"] = postflop / ntot
    for st in ("preflop", "flop", "turn", "river"):
        f[f"street_{st}_share"] = street_share.get(st, 0) / ntot
    f["hero_action_share"] = hero_actions / ntot
    f["button_action_share"] = button_actions / ntot
    f["actor_switch_rate"] = actor_switches / max(1, actor_obs)

    _stats(amounts, "amount", f)
    _stats(pot_growth, "potgrow", f)
    _stats(pot_before, "potbefore", f)
    _stats(pot_after, "potafter", f)
    _stats(stacks, "stack", f)
    _stats(per_hand_actions, "actions_per_hand", f)
    f["pot_monotonic_rate"] = pot_monotonic / max(1, pot_steps)

    f["action_entropy_mean"] = float(np.mean(per_hand_ent)) if per_hand_ent else 0.0
    f["action_entropy_std"] = float(np.std(per_hand_ent)) if per_hand_ent else 0.0
    f["hand_aggr_mean"] = float(np.mean(per_hand_aggr)) if per_hand_aggr else 0.0
    f["hand_aggr_std"] = float(np.std(per_hand_aggr)) if per_hand_aggr else 0.0
    f["type_run_share_mean"] = float(np.mean(type_run_shares)) if type_run_shares else 0.0
    f["actor_run_share_mean"] = float(np.mean(actor_run_shares)) if actor_run_shares else 0.0
    f["player_count_mean"] = float(np.mean(pcounts)) if pcounts else 0.0
    f["player_count_std"] = float(np.std(pcounts)) if pcounts else 0.0
    f["hero_button_same_rate"] = hero_button_same / nh
    f["street_count_entropy"] = _entropy(list(street_share.values()))
    f["first_action_entropy"] = _entropy(list(first_action.values()))

    # mechanical-repetition tells (bots reuse identical action signatures)
    f["sig_unique_share"] = len(sig_chunk) / nh
    f["sig_top_share"] = (max(sig_chunk.values()) / nh) if sig_chunk else 0.0
    f["amt_bucket_sig_unique_share"] = len(amt_bucket_sig) / nh
    f["amt_bucket_sig_top_share"] = (max(amt_bucket_sig.values()) / nh) if amt_bucket_sig else 0.0
    return f


# stable feature order
_SAMPLE = chunk_features([
    {"metadata": {}, "players": [], "streets": [], "actions": []}
])
FEATURE_NAMES = sorted(_SAMPLE.keys())


def vectorize(hands) -> np.ndarray:
    f = chunk_features(hands)
    return np.array([f.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64)
