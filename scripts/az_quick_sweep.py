#!/usr/bin/env python3
"""
Small AZ hyperparameter sweep: short runs, compare to random legal baseline on total_score.

Example:
  .venv/bin/python -u scripts/az_quick_sweep.py --skip-audit --episodes 15 --max-steps 70
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from redistricting.env.core import GerrymanderingEnv
from redistricting.rl.agent import PPOAgent, PPOHyperParams
from redistricting.rl.trainer import TrainingConfig, TrainingLoop
from redistricting.utils.data_audit import audit_data_directory, print_audit_report
from redistricting.utils.paths import get_data_dir, get_outputs_dir


@dataclass
class SweepArm:
    name: str
    reward_mode: str = "delta"
    score_reward_scale: float = 1.0
    delta_scale: float = 100.0
    exploration_coef: float = 0.0001
    ema_alpha: float = 0.1
    lr: float = 3e-4
    lr_policy: Optional[float] = None
    lr_value: Optional[float] = None
    entropy_coef: float = 0.004
    entropy_coef_start: Optional[float] = None
    huber: bool = False
    k_epochs: int = 3


def default_arms() -> List[SweepArm]:
    """A few diverse, short-run-friendly settings."""
    return [
        SweepArm(
            name="score_mild",
            reward_mode="score",
            score_reward_scale=0.35,
            lr=5e-4,
            lr_value=1.5e-4,
            entropy_coef=0.004,
            entropy_coef_start=0.02,
            huber=True,
        ),
        SweepArm(
            name="ema_delta_soft",
            reward_mode="ema_delta",
            delta_scale=45.0,
            exploration_coef=0.00005,
            lr=5e-4,
            lr_value=1.5e-4,
            entropy_coef=0.005,
            entropy_coef_start=0.018,
            huber=True,
        ),
        SweepArm(
            name="delta_scaled",
            reward_mode="delta",
            delta_scale=55.0,
            exploration_coef=0.00005,
            lr=4e-4,
            lr_value=1.2e-4,
            entropy_coef=0.003,
            entropy_coef_start=0.015,
            huber=True,
        ),
    ]


def random_baseline_end_scores(
    env_template: GerrymanderingEnv,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[float, float]:
    """Uniform random legal policy: mean/std of final-step total_score per episode."""
    rng = np.random.default_rng(seed)
    scores: List[float] = []
    for ep in range(n_episodes):
        t_ep = time.perf_counter()
        env_template.reset(seed=seed + ep)
        last_score = 0.0
        for _ in range(max_steps):
            mask = env_template.get_valid_action_mask()
            legal = np.flatnonzero(mask > 0.5)
            if legal.size == 0:
                break
            a = int(rng.choice(legal))
            _, _, term, trunc, info = env_template.step(a)
            last_score = float(info.get("total_score", 0.0))
            if term or trunc:
                break
        scores.append(last_score)
        print(
            f"[baseline] random ep {ep + 1}/{n_episodes} end_score={last_score:.4f} ({time.perf_counter() - t_ep:.1f}s)",
            flush=True,
        )
    return float(np.mean(scores)), float(np.std(scores))


def _last_finite(series: List[Any]) -> float:
    for x in reversed(series):
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if not np.isnan(v):
            return v
    return float("nan")


def summarize_history(h: Dict[str, List]) -> Dict[str, float]:
    ent = h.get("episode_mean_entropy", [])
    top = h.get("episode_mean_top1_prob", [])
    rew = h.get("episode_rewards", [])
    out: Dict[str, float] = {}
    out["greedy_score_last"] = _last_finite(h.get("greedy_mean_total_score", []))
    out["best_legal_last"] = float(h["best_legal_score"][-1]) if h.get("best_legal_score") else float("nan")
    out["entropy_first3"] = float(np.nanmean(ent[:3])) if len(ent) >= 3 else float(np.nanmean(ent)) if ent else float("nan")
    out["entropy_last3"] = float(np.nanmean(ent[-3:])) if len(ent) >= 3 else float(np.nanmean(ent)) if ent else float("nan")
    out["top1_first3"] = float(np.nanmean(top[:3])) if len(top) >= 3 else float(np.nanmean(top)) if top else float("nan")
    out["top1_last3"] = float(np.nanmean(top[-3:])) if len(top) >= 3 else float(np.nanmean(top)) if top else float("nan")
    out["return_first3"] = float(np.mean(rew[:3])) if len(rew) >= 3 else float(np.mean(rew)) if rew else float("nan")
    out["return_last3"] = float(np.mean(rew[-3:])) if len(rew) >= 3 else float(np.mean(rew)) if rew else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Quick AZ sweep vs random baseline (short runs)")
    p.add_argument("--state", type=str, default="az")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=70)
    p.add_argument("--max-actions", type=int, default=192)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--greedy-eval-episodes", type=int, default=2)
    p.add_argument(
        "--random-baseline-episodes",
        type=int,
        default=4,
        help="Random legal rollouts (each step is expensive on AZ; keep small)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = p.parse_args()

    basepath = str(get_data_dir(None, "processed"))
    if not args.skip_audit:
        result = audit_data_directory(args.state, basepath)
        print_audit_report(result)
        if result["status"] != "PASS":
            raise SystemExit(1)

    max_actions = None if args.max_actions == 0 else int(args.max_actions)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = get_outputs_dir(args.state, "runs") / f"sweep_quick_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] batch_dir={batch_dir}", flush=True)
    print(
        f"[sweep] episodes={args.episodes} max_steps={args.max_steps} max_actions={max_actions} "
        f"eval_every={args.eval_every}",
        flush=True,
    )

    # Reference env for random baseline (default reward)
    env_ref = GerrymanderingEnv(
        state=args.state,
        basepath=basepath,
        max_steps=args.max_steps,
        max_action_space_size=max_actions,
    )
    print("[baseline] random legal end-of-episode scores (first lines can take several minutes)...", flush=True)
    t0 = time.perf_counter()
    rand_mean, rand_std = random_baseline_end_scores(
        env_ref, args.random_baseline_episodes, args.max_steps, args.seed + 9000
    )
    t_rand = time.perf_counter() - t0
    print(
        f"[baseline] random legal: end-of-episode total_score mean={rand_mean:.4f} std={rand_std:.4f} "
        f"(n={args.random_baseline_episodes}, {t_rand:.1f}s)",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    arms = default_arms()

    for arm in arms:
        print(f"\n[sweep] === {arm.name} ===", flush=True)
        run_dir = batch_dir / arm.name
        run_dir.mkdir(parents=True, exist_ok=True)

        env = GerrymanderingEnv(
            state=args.state,
            basepath=basepath,
            max_steps=args.max_steps,
            max_action_space_size=max_actions,
            reward_mode=arm.reward_mode,
            score_reward_scale=arm.score_reward_scale,
            delta_scale_factor=arm.delta_scale,
            exploration_coef=arm.exploration_coef,
            ema_alpha=arm.ema_alpha,
        )
        _, feats = env.get_graph_observation()
        hyper = PPOHyperParams(
            lr=arm.lr,
            lr_policy=arm.lr_policy,
            lr_value=arm.lr_value,
            entropy_coef=arm.entropy_coef,
            entropy_coef_start=arm.entropy_coef_start,
            use_huber_value_loss=arm.huber,
            k_epochs=arm.k_epochs,
        )
        agent = PPOAgent(
            node_feature_dim=feats.shape[1],
            action_dim=env.action_space.n,
            hyperparams=hyper,
            device="cpu" if args.cpu else None,
        )
        trainer = TrainingLoop(
            env=env,
            agent=agent,
            config=TrainingConfig(
                num_episodes=args.episodes,
                seed=args.seed,
                eval_every_n_episodes=args.eval_every,
                greedy_eval_episodes=args.greedy_eval_episodes,
                save_frequency=10_000,
                early_stop_patience=10_000,
                verbose=args.verbose,
            ),
            run_dir=run_dir,
        )
        t1 = time.perf_counter()
        hist = trainer.train()
        wall = time.perf_counter() - t1
        summ = summarize_history(hist)
        row = {
            "name": arm.name,
            "reward_mode": arm.reward_mode,
            "score_reward_scale": arm.score_reward_scale,
            "delta_scale": arm.delta_scale,
            "lr": arm.lr,
            "entropy_coef": arm.entropy_coef,
            "entropy_coef_start": arm.entropy_coef_start or "",
            "huber": arm.huber,
            "wall_s": wall,
            "run_dir": str(run_dir),
            "random_score_mean": rand_mean,
            "random_score_std": rand_std,
            **summ,
        }
        rows.append(row)
        beat = summ["greedy_score_last"] > rand_mean + 0.25 * max(rand_std, 1e-6)
        sharper = summ["entropy_last3"] < summ["entropy_first3"] - 0.02
        print(
            f"[sweep] {arm.name}: greedy_last={summ['greedy_score_last']:.4f} best_legal={summ['best_legal_last']:.4f} "
            f"H first3/last3={summ['entropy_first3']:.3f}/{summ['entropy_last3']:.3f} "
            f"top1 first3/last3={summ['top1_first3']:.3f}/{summ['top1_last3']:.3f} ({wall:.0f}s) "
            f"beat_rand={beat} sharper={sharper}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    csv_path = batch_dir / "sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "random_baseline": {"mean": rand_mean, "std": rand_std, "n": args.random_baseline_episodes},
        "settings": vars(args),
        "arms": [asdict(a) for a in arms],
    }
    (batch_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[sweep] wrote {csv_path}", flush=True)
    print(df[["name", "greedy_score_last", "best_legal_last", "entropy_first3", "entropy_last3", "wall_s"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
