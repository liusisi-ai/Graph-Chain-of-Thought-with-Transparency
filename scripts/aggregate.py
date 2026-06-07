"""Read ``tune_runs.jsonl`` and print mean ± std accuracy for each run group.

A "group" is the combination of (tag, hparams) — i.e. the same hyper-param
configuration evaluated across multiple random seeds.

Usage::

    python scripts/aggregate.py
    python scripts/aggregate.py --log tune_runs.jsonl --metric val_acc
    python scripts/aggregate.py --top 5 --metric test_acc

Columns
-------
    tag           : free-form tag passed via --tag in main.py
    n             : number of seeds aggregated
    mean ± std    : mean and population stddev of the metric (val/test acc)
    seeds         : the seeds that contributed to the row
    elapsed_sec   : average wall-clock per run
    hparams_diff  : only the hyper-params that differ from the most-common run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise SystemExit(f"❌ log file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ skipping malformed line {ln}: {e}")
    return rows


def _group_key(row: Dict[str, Any]) -> Tuple:
    """Two runs belong to the same group iff (tag, hparams) match."""
    tag = row.get("tag", "")
    hp = row.get("hparams", {}) or {}
    return (tag, tuple(sorted(hp.items())))


def _common_hparams(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find hparams that are identical across every row (used as the baseline)."""
    if not rows:
        return {}
    common = dict(rows[0].get("hparams", {}) or {})
    for r in rows[1:]:
        hp = r.get("hparams", {}) or {}
        for k in list(common):
            if hp.get(k) != common[k]:
                del common[k]
    return common


def _hparams_diff(hp: Dict[str, Any], common: Dict[str, Any]) -> str:
    diff = {k: v for k, v in (hp or {}).items() if common.get(k) != v}
    if not diff:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(diff.items()))


def aggregate(log_path: str, metric: str, top_k: int) -> None:
    rows = _load_jsonl(log_path)
    if not rows:
        print(f"❌ no rows in {log_path}")
        return

    common = _common_hparams(rows)

    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    summary: List[Dict[str, Any]] = []
    for (tag, _), members in groups.items():
        accs = [m.get(metric) for m in members if isinstance(m.get(metric), (int, float))]
        if not accs:
            continue
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        elapsed = [m.get("elapsed_sec", 0) or 0 for m in members]
        seeds = [m.get("seed") for m in members]
        diff = _hparams_diff(members[0].get("hparams", {}) or {}, common)
        summary.append({
            "tag": tag,
            "n": len(accs),
            "mean": mean,
            "std": std,
            "raw": accs,
            "seeds": seeds,
            "elapsed": statistics.mean(elapsed) if elapsed else 0.0,
            "diff": diff or "(default)",
        })

    summary.sort(key=lambda x: x["mean"], reverse=True)

    if top_k and top_k < len(summary):
        summary = summary[:top_k]

    print(f"\n📊 {len(summary)} groups, sorted by {metric} (descending)")
    print(f"   common hparams across all rows: {common}\n")

    print(f"{'tag':<30s}  {'n':>3s}  {metric:>10s}  {'std':>8s}  "
          f"{'sec':>6s}  hparams_diff / raw_accs")
    print("─" * 110)
    for s in summary:
        raw_str = "[" + ", ".join(f"{a:.4f}" for a in s["raw"]) + "]"
        print(
            f"{s['tag']:<30s}  {s['n']:>3d}  "
            f"{s['mean']:>10.4f}  {s['std']:>8.4f}  "
            f"{s['elapsed']:>6.0f}  {s['diff']}  seeds={s['seeds']}  raw={raw_str}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="tune_runs.jsonl")
    ap.add_argument("--metric", type=str, default="val_acc",
                    choices=["val_acc", "test_acc"],
                    help="Which accuracy column to aggregate.")
    ap.add_argument("--top", type=int, default=0,
                    help="Show only the top-K configurations (0 = all).")
    args = ap.parse_args()
    aggregate(args.log, args.metric, args.top)


if __name__ == "__main__":
    main()
