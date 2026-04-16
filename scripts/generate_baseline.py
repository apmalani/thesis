#!/usr/bin/env python3
"""Generate MCMC baseline ensemble and baseline statistics."""

import argparse
from functools import partial

import pandas as pd
from gerrychain import MarkovChain
from gerrychain.accept import always_accept
from gerrychain.constraints import Validator, within_percent_of_ideal_population
from gerrychain.proposals import recom

from redistricting.graph.construction import build_precinct_graph, validate_precinct_graph
from redistricting.graph.metrics import MCalc
from redistricting.utils.paths import get_data_dir


def run_chain(state: str, basepath: str, steps: int, thinning: int, pop_tol: float) -> pd.DataFrame:
    """Run a single ReCom chain and return sampled metric rows."""
    graph, partition = build_precinct_graph(state, basepath)
    validation = validate_precinct_graph(graph, partition, tolerance=pop_tol)
    if not validation["overall"]:
        raise RuntimeError("Initial partition invalid")
    ideal_pop = sum(partition["population"].values()) / len(partition)
    proposal = partial(
        recom,
        pop_col="P0010001",
        pop_target=ideal_pop,
        epsilon=pop_tol,
        node_repeats=2,
    )
    chain = MarkovChain(
        proposal=proposal,
        constraints=Validator([within_percent_of_ideal_population(partition, pop_tol)]),
        accept=always_accept,
        initial_state=partition,
        total_steps=steps,
    )
    mc = MCalc()
    rows = []
    for idx, part in enumerate(chain):
        if idx % thinning != 0:
            continue
        metrics = mc.calculate_metrics(part, include_geometry=True).iloc[0].to_dict()
        metrics["step"] = idx
        rows.append(metrics)
    return pd.DataFrame(rows)


def compute_baseline_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline stat table (mean/median/std/q25/q75/min/max/count)."""
    metrics = [
        "EfficiencyGap",
        "PartisanProp",
        "SeatsVotesDiff",
        "PolPopperAvg",
        "PolPopperMin",
        "MinOppAvg",
        "MinOppMin",
    ]
    baseline_stats = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = df[metric].dropna()
        baseline_stats[metric] = {
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "q25": values.quantile(0.25),
            "q75": values.quantile(0.75),
            "min": values.min(),
            "max": values.max(),
            "count": len(values),
        }
    return pd.DataFrame(baseline_stats).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline ensemble stats")
    parser.add_argument("--state", type=str, default="az")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--pop-tol", type=float, default=0.05)
    args = parser.parse_args()

    basepath = str(get_data_dir(None, "processed"))
    ensemble_df = run_chain(args.state, basepath, args.steps, args.thinning, args.pop_tol)
    state_dir = get_data_dir(args.state, "processed")
    ensemble_path = state_dir / "ensemble_metrics.csv"
    baseline_path = state_dir / "baseline_stats.csv"
    ensemble_df.to_csv(ensemble_path, index=False)
    baseline_df = compute_baseline_stats(ensemble_df)
    baseline_df.to_csv(baseline_path)
    print(f"Saved ensemble metrics to {ensemble_path}")
    print(f"Saved baseline stats to {baseline_path}")


if __name__ == "__main__":
    main()

