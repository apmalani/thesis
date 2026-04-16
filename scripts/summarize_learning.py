#!/usr/bin/env python3
"""Summarize a training_history.csv for signs of policy change (entropy, top-1, greedy score)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from redistricting.utils.paths import get_outputs_dir


def _latest_csv(state: str) -> Path | None:
    runs = get_outputs_dir(state, "runs")
    if not runs.is_dir():
        return None
    dirs = sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in dirs:
        p = d / "training_history.csv"
        if p.is_file():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize training_history.csv for learning signals")
    p.add_argument("--csv", type=str, default="", help="Path to training_history.csv")
    p.add_argument("--state", type=str, default="az", help="With --csv omitted, use latest run under this state")
    args = p.parse_args()
    path = Path(args.csv) if args.csv else _latest_csv(args.state)
    if path is None or not path.is_file():
        raise SystemExit("No CSV found. Pass --csv or ensure outputs/runs/<state>/*/training_history.csv exists.")

    df = pd.read_csv(path)
    n = len(df)
    print(f"File: {path}")
    print(f"Rows (episodes): {n}")
    if n == 0:
        raise SystemExit("Empty CSV.")

    def seg_mean(col: str, head: int = 5, tail: int = 5) -> tuple[float, float] | None:
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < head + tail:
            return float(s.iloc[: min(head, len(s))].mean()), float(s.iloc[-min(tail, len(s)) :].mean())
        return float(s.iloc[:head].mean()), float(s.iloc[-tail:].mean())

    print("\n--- Segment means (first 5 vs last 5 episodes) ---")
    for col, label in [
        ("episode_mean_entropy", "rollout mean entropy (policy at states)"),
        ("episode_mean_top1_prob", "rollout mean top-1 prob"),
        ("entropies", "PPO update batch mean entropy"),
        ("greedy_mean_total_score", "greedy eval mean total_score (NaN ok)"),
        ("best_legal_score", "best legal map score so far"),
    ]:
        m = seg_mean(col)
        if m is None:
            print(f"  {col}: (column missing)")
        else:
            a, b = m
            delta = b - a
            print(f"  {col}: first≈{a:.4f}  last≈{b:.4f}  Δ={delta:+.4f}  ({label})")

    print("\n--- How to read ---")
    print("  Stronger policy (less uniform): episode_mean_entropy down and/or top-1 up.")
    print("  Greedy / best_legal trending up suggests map quality signal (depends on reward).")
    print("  If only `entropies` exists (old runs), it is coarser but should move with the policy.")
    print(f"\nRaw tail:\n{df.tail(3).to_string()}")


if __name__ == "__main__":
    main()
