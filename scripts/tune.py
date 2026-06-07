"""Grid-search hyper-parameter tuner.

Each combination is run as a fresh ``python main.py`` subprocess with the
chosen hyper-parameters injected via ``CFG_*`` environment variables.  The
result of every run is appended to ``tune_runs.jsonl`` by ``main.py``.

Quick start (cora)
------------------
::

    # 1) Sanity check: 1 baseline run on val_mask, single seed
    python scripts/tune.py --grid baseline --seeds 42 --eval-on val

    # 2) Sweep PCA-alignment epochs at 3 seeds, on val
    python scripts/tune.py --grid pca --seeds 42 1 7 --eval-on val

    # 3) Sweep CoT (#thoughts × #clusters) at 3 seeds, on val
    python scripts/tune.py --grid cot --seeds 42 1 7 --eval-on val

    # 4) Custom grid
    python scripts/tune.py --seeds 42 1 7 --eval-on val \
        --vary CFG_NUM_THOUGHTS=1,2,3,4 \
        --vary CFG_NUM_CLUSTERS=7,14

After running the grid, aggregate over seeds with::

    python scripts/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple


# ── Predefined grids (recommended exploration order) ────────────────

PREDEFINED_GRIDS: Dict[str, Dict[str, List[str]]] = {
    "baseline": {
        # Just runs main.py with the current config defaults.
    },
    "pca": {
        "CFG_PCA_EPOCHS": ["50", "100", "200"],
        "CFG_PCA_LR":     ["1e-4", "5e-4", "1e-3"],
        "CFG_NUM_PCA_ANCHORS": ["128", "256"],
    },
    "cot": {
        "CFG_NUM_THOUGHTS":      ["1", "2", "3", "4"],
        "CFG_NUM_CLUSTERS":      ["7", "14"],
        "CFG_NUM_REPR_TOKENS":   ["8", "16"],
    },
    "fast": {
        "CFG_NUM_THOUGHTS":  ["1", "2"],
        "CFG_NUM_CLUSTERS":  ["7", "14"],
    },
}


def _parse_vary(args_vary: List[str]) -> Dict[str, List[str]]:
    """Convert ``--vary KEY=v1,v2,v3`` arguments to a grid dict."""
    grid: Dict[str, List[str]] = {}
    for item in args_vary:
        if "=" not in item:
            raise SystemExit(f"--vary requires KEY=v1,v2,...  got: {item!r}")
        k, v = item.split("=", 1)
        grid[k.strip()] = [s.strip() for s in v.split(",") if s.strip()]
    return grid


def _expand(grid: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Cartesian product of the grid → list of env-var dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def _short_tag(env: Dict[str, str]) -> str:
    """Human-readable tag for the run log."""
    if not env:
        return "default"
    return ".".join(f"{k.replace('CFG_', '').lower()}={v}" for k, v in env.items())


def _run_one(env_overrides: Dict[str, str], seed: int, eval_on: str,
             tag: str, log_path: str, dry_run: bool) -> Tuple[bool, float]:
    cmd = [sys.executable, "main.py",
           "--seed", str(seed),
           "--eval-on", eval_on,
           "--tag", tag,
           "--log-path", log_path]
    env = os.environ.copy()
    env.update(env_overrides)

    cmd_preview = " ".join(f"{k}={v}" for k, v in env_overrides.items()) + " " + " ".join(cmd)
    print("\n" + "─" * 70)
    print(f"▶️  {cmd_preview}")
    print("─" * 70)

    if dry_run:
        return True, 0.0

    t = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t
    ok = proc.returncode == 0
    print(f"   exit={proc.returncode}  elapsed={elapsed:.1f}s")
    return ok, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default=None, choices=list(PREDEFINED_GRIDS),
                    help="Use a predefined grid (baseline/pca/cot/fast).")
    ap.add_argument("--vary", action="append", default=[],
                    help="Custom override: --vary CFG_NUM_THOUGHTS=1,2,3 (can repeat).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Random seeds to run for *each* combination.")
    ap.add_argument("--eval-on", type=str, default="val",
                    choices=["val", "test", "both"],
                    help="Evaluate on val (tuning) or test (final report).")
    ap.add_argument("--log-path", type=str, default="tune_runs.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned commands but do not execute.")
    args = ap.parse_args()

    grid = PREDEFINED_GRIDS.get(args.grid, {}) if args.grid else {}
    grid.update(_parse_vary(args.vary))
    combos = _expand(grid)

    print(f"📦 grid: {grid}")
    print(f"🌱 seeds: {args.seeds}")
    print(f"🧪 total runs: {len(combos)} combinations × {len(args.seeds)} seeds "
          f"= {len(combos) * len(args.seeds)}")
    print(f"📝 logging to {args.log_path}")

    n_ok, n_fail = 0, 0
    t0 = time.time()
    for ov in combos:
        tag = _short_tag(ov)
        for seed in args.seeds:
            ok, _ = _run_one(ov, seed, args.eval_on, tag, args.log_path, args.dry_run)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    print("\n" + "═" * 70)
    print(f"✅ done: {n_ok} ok, {n_fail} failed, total elapsed {time.time() - t0:.0f}s")
    print(f"   aggregate with:  python scripts/aggregate.py --log {args.log_path}")


if __name__ == "__main__":
    main()
