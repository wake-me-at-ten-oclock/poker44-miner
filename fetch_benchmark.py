#!/usr/bin/env python3
"""Download the full Poker44 public benchmark (labeled chunks) to local cache.

Each release date -> one JSON file under data/raw/<sourceDate>.json holding the
raw API payload (all chunk records, each with inner groups + groundTruth).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

API = "https://api.poker44.net/api/v1/benchmark"
OUT = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(OUT, exist_ok=True)


def get(path: str, params: dict | None = None):
    url = path if path.startswith("http") else f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("data", data)


def list_release_dates(limit_total: int = 400) -> list[str]:
    dates: list[str] = []
    before = None
    while len(dates) < limit_total:
        d = get("/releases", {"limit": 50, "before": before})
        rels = d.get("releases", []) if isinstance(d, dict) else []
        if not rels:
            break
        for r in rels:
            sd = r.get("sourceDate")
            if sd and sd not in dates:
                dates.append(sd)
        before = rels[-1].get("sourceDate")
        if len(rels) < 50:
            break
    return dates


def fetch_date(source_date: str) -> dict:
    """Page through all chunks for a date (both splits)."""
    records: list[dict] = []
    cursor = None
    while True:
        d = get("/chunks", {"sourceDate": source_date, "limit": 24, "cursor": cursor})
        chunks = d.get("chunks", []) if isinstance(d, dict) else []
        records.extend(chunks)
        cursor = d.get("nextCursor") if isinstance(d, dict) else None
        if not cursor or not chunks:
            break
    return {"sourceDate": source_date, "records": records}


def main():
    dates = list_release_dates()
    print(f"found {len(dates)} release dates")
    total_groups = 0
    for i, sd in enumerate(dates, 1):
        path = os.path.join(OUT, f"{sd}.json")
        if os.path.exists(path):
            payload = json.load(open(path))
        else:
            try:
                payload = fetch_date(sd)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(dates)}] {sd}  ERROR {e}")
                continue
            json.dump(payload, open(path, "w"))
            time.sleep(0.3)
        ngroups = sum(len(r.get("chunks", [])) for r in payload["records"])
        total_groups += ngroups
        print(f"  [{i}/{len(dates)}] {sd}  records={len(payload['records'])} groups={ngroups}")
    print(f"DONE. cached {len(dates)} dates, ~{total_groups} labeled groups -> {OUT}")


if __name__ == "__main__":
    main()
