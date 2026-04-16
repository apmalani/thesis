"""Visualization utilities for training and map outputs."""

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from redistricting.utils.paths import get_data_dir, get_outputs_dir


def _moving_average(values: list, window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_progress(history: dict, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot reward and efficiency-gap history."""
    plot_learning_dashboard(history, save_path=save_path, show=show)


def plot_learning_dashboard(
    history: dict,
    save_path: Optional[str] = None,
    show: bool = True,
    ma_window: int = 10,
) -> None:
    """Plot training return MA, best legal score, greedy eval metrics, policy stats."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    n = len(history.get("episode_rewards", []))
    episodes = np.arange(n)
    w = min(ma_window, max(n, 1))

    rewards = history.get("episode_rewards", [])
    if rewards:
        axes[0].plot(episodes, rewards, alpha=0.35, label="Episode return", color="C0")
        ma_r = _moving_average(rewards, w)
        if ma_r.size > 0:
            x_ma = np.arange(w - 1, w - 1 + len(ma_r))
            axes[0].plot(x_ma, ma_r, label=f"Return MA (w={w})", color="C0", linewidth=2)
        greedy_ret = history.get("greedy_mean_return", [])
        if greedy_ret:
            valid = [(i, v) for i, v in enumerate(greedy_ret) if not np.isnan(v)]
            if valid:
                gx, gy = zip(*valid)
                axes[0].scatter(gx, gy, marker="x", s=40, color="C1", label="Greedy eval mean return", zorder=5)
                wg = min(5, max(1, len(gy)))
                gma = _moving_average(list(gy), wg)
                if gma.size > 1:
                    x_ma_g = np.array(gx)[wg - 1 : wg - 1 + len(gma)]
                    axes[0].plot(
                        x_ma_g,
                        gma,
                        color="C1",
                        linewidth=1.5,
                        linestyle="--",
                        label=f"Greedy return MA (w={wg})",
                    )
    axes[0].set_title("Episode return vs greedy eval return")
    axes[0].set_xlabel("Episode")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    best_scores = history.get("best_legal_score", [])
    if best_scores:
        axes[1].plot(episodes[: len(best_scores)], best_scores, label="Best legal total_score", color="C2")
    greedy_score = history.get("greedy_mean_total_score", [])
    if greedy_score:
        valid = [(i, v) for i, v in enumerate(greedy_score) if not np.isnan(v)]
        if valid:
            sx, sy = zip(*valid)
            axes[1].scatter(sx, sy, marker="o", s=30, facecolors="none", edgecolors="C3", label="Greedy eval mean score")
    eg_train = history.get("efficiency_gaps", [])
    if eg_train:
        ma_eg = _moving_average(eg_train, w)
        if ma_eg.size > 0:
            x_ma = np.arange(w - 1, w - 1 + len(ma_eg))
            axes[1].plot(x_ma, ma_eg, alpha=0.4, label="Train EG MA", color="gray")
    greedy_eg = history.get("greedy_mean_efficiency_gap", [])
    if greedy_eg:
        valid = [(i, v) for i, v in enumerate(greedy_eg) if not np.isnan(v)]
        if valid:
            ex, ey = zip(*valid)
            axes[1].scatter(ex, ey, marker="+", color="purple", label="Greedy eval mean EG")
    axes[1].set_title("Best legal score & efficiency gap (train MA vs greedy eval)")
    axes[1].set_xlabel("Episode")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    ent = history.get("episode_mean_entropy", [])
    if ent:
        ma_e = _moving_average(ent, w)
        if ma_e.size > 0:
            x_ma = np.arange(w - 1, w - 1 + len(ma_e))
            axes[2].plot(x_ma, ma_e, label=f"Entropy MA (w={w})", color="C4")
    top1 = history.get("episode_mean_top1_prob", [])
    if top1:
        ma_t = _moving_average(top1, w)
        if ma_t.size > 0:
            x_ma = np.arange(w - 1, w - 1 + len(ma_t))
            axes[2].plot(x_ma, ma_t, label="Top-1 prob MA", color="C5")
    axes[2].set_title("Policy concentration (entropy & top-1)")
    axes[2].set_xlabel("Episode")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_frontier(
    metrics_df: pd.DataFrame, save_path: str, x_col: str = "EfficiencyGap", y_col: str = "PolPopperAvg"
) -> None:
    """Plot Pareto-style scatter for two map quality metrics."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(metrics_df[x_col], metrics_df[y_col], alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Pareto Frontier (Observed)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boundary_heatmap(precinct_gdf: gpd.GeoDataFrame, frequency_col: str, save_path: str) -> None:
    """Plot boundary heatmap using a reassignment-frequency column."""
    fig, ax = plt.subplots(figsize=(10, 10))
    precinct_gdf.plot(column=frequency_col, ax=ax, legend=True, linewidth=0.1, edgecolor="black")
    ax.set_axis_off()
    ax.set_title("Ensemble Boundary Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_best_map(
    state: str,
    map_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    show: bool = True,
) -> Optional[Path]:
    """Render saved best-map assignment, optionally alongside enacted map."""
    data_dir = get_data_dir(state, "processed")
    candidates = [f"{state}_precincts.shp", "precincts_with_vap.shp", "precincts_with_districts.shp"]
    shapefile = next((data_dir / name for name in candidates if (data_dir / name).exists()), None)
    if shapefile is None:
        return None
    gdf = gpd.read_file(shapefile).reset_index(drop=True)
    gdf["graph_node_id"] = gdf.index.astype(str)

    if map_file is None:
        best_maps_dir = get_outputs_dir(state, "best_maps")
        map_files = list(best_maps_dir.glob("best_map_*.csv"))
        if not map_files:
            return None
        map_file = str(max(map_files, key=lambda p: p.stat().st_mtime))
    assignments = pd.read_csv(map_file)
    if "precinct_id" not in assignments.columns:
        return None
    gdf["graph_node_id"] = gdf["graph_node_id"].astype(str)
    assignments["precinct_id"] = assignments["precinct_id"].astype(str)
    gdf = gdf.merge(assignments, left_on="graph_node_id", right_on="precinct_id", how="left")
    if "district" not in gdf.columns or gdf["district"].isna().all():
        return None

    has_current_map = "CONG_DIST" in gdf.columns
    if has_current_map:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    gdf_best = gdf[gdf["district"].notna()].copy()
    gdf_best["district_num"] = pd.to_numeric(gdf_best["district"], errors="coerce")
    all_districts = sorted(
        set(gdf_best["district_num"].dropna().tolist() + (gdf["CONG_DIST"].dropna().tolist() if has_current_map else []))
    )

    cmap = plt.colormaps["tab20"] if len(all_districts) > 10 else plt.colormaps["tab10"]
    if has_current_map:
        gdf_current = gdf.copy()
        gdf_current["CONG_DIST_num"] = pd.to_numeric(gdf_current["CONG_DIST"], errors="coerce")
        gdf_current.plot(column="CONG_DIST_num", cmap=cmap, ax=ax1, edgecolor="black", linewidth=0.1)
        ax1.set_title(f"{state.upper()} Current Map")
        ax1.axis("off")

    target_ax = ax2 if ax2 is not None else ax1
    gdf_best.plot(column="district_num", cmap=cmap, ax=target_ax, edgecolor="black", linewidth=0.1)
    target_ax.set_title(f"{state.upper()} Best Map")
    target_ax.axis("off")

    colors = [cmap(i / max(len(all_districts) - 1, 1)) for i in range(len(all_districts))]
    handles = [Patch(facecolor=colors[i], edgecolor="black", label=f"District {int(d)}") for i, d in enumerate(all_districts)]
    target_ax.legend(handles=handles, title="District", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    output_dir = get_outputs_dir(state, "best_maps")
    output_file = output_dir / f"{Path(map_file).stem}_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return output_file

