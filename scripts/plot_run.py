#!/usr/bin/env python3
"""Replot training_progress.png from a saved training_history.csv."""

import argparse
from pathlib import Path

import pandas as pd

from redistricting.utils.visualization import plot_learning_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning dashboard from training_history.csv")
    parser.add_argument("history_csv", type=str, help="Path to training_history.csv")
    parser.add_argument("--out", type=str, default="", help="Output PNG (default: <csv_dir>/training_progress_replot.png)")
    parser.add_argument("--ma-window", type=int, default=10)
    args = parser.parse_args()
    csv_path = Path(args.history_csv)
    df = pd.read_csv(csv_path)
    out_path = Path(args.out) if args.out else csv_path.parent / "training_progress_replot.png"
    plot_learning_dashboard(
        df.to_dict(orient="list"),
        save_path=str(out_path),
        show=False,
        ma_window=args.ma_window,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
