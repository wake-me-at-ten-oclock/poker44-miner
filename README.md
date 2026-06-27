# Poker44 Miner — stacked behavioral bot detector (SN126)

A supervised miner for the Poker44 subnet (netuid 126). It scores each
`DetectionSynapse` chunk (a group of poker hands) with a probability that the
acting player is a bot, optimizing the validator metric
`0.75 * AveragePrecision + 0.25 * recall@FPR<=0.05`.

## Approach

Chunk-level behavioral feature engineering + a gradient-boosted model:

- `p44_features.py` — size-robust behavioral features (action-type shares,
  aggression, bet-size dispersion, pot geometry, action/actor entropy,
  mechanical-repetition signatures, stack distribution). No ids/counts/dates
  are used as features.
- `winner_features.py` — additional per-hand→chunk quantile aggregation
  (mean/std/min/max/q10/q50/q90) over the same masked payload. *Adapted from
  the public reference feature set (MIT); see NOTICE in code.*
- `p44_combined.py` — merges both feature families into one vector.
- `finalize_model.py` — trains a tuned LightGBM on the public benchmark
  (with light hand-subsampling augmentation) and saves `models/p44_final.joblib`.

The model is trained only on the **public** Poker44 benchmark
(`https://api.poker44.net/api/v1/benchmark`), never on validator-only data.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_benchmark.py        # download labeled benchmark chunks
python finalize_model.py         # -> models/p44_final.joblib
python train_solution.py         # date-grouped CV report (composite vs reference)
```

## Run the miner

```bash
python neurons/miner.py --netuid 126 \
  --wallet.name <cold> --wallet.hotkey <hot> \
  --subtensor.network finney --axon.port 8091
```

`miner_infer.py` loads `models/p44_final.joblib` and returns one calibrated
risk score per chunk; it falls back to a deterministic heuristic if the model
artifact is unavailable.

## License

MIT. Portions of `winner_features.py` adapt MIT-licensed reference code.
