#!/usr/bin/env python3
"""Train GNN-PPO redistricting model."""

import argparse

from redistricting.env.core import GerrymanderingEnv
from redistricting.rl.agent import PPOAgent, PPOHyperParams
from redistricting.rl.trainer import TrainingConfig, TrainingLoop
from redistricting.utils.data_audit import audit_data_directory, print_audit_report
from redistricting.utils.paths import get_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train redistricting GNN-PPO model")
    parser.add_argument("--state", type=str, default="az")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-policy", type=float, default=None, help="Actor LR (default: --lr)")
    parser.add_argument("--lr-value", type=float, default=None, help="Critic LR (default: 0.5 * --lr)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=256, help="Cap legal moves; use 0 for no cap")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=("delta", "score", "ema_delta"),
        default="delta",
        help="delta=z-score delta; score=total_score*scale; ema_delta=delta vs EMA baseline",
    )
    parser.add_argument("--score-reward-scale", type=float, default=1.0)
    parser.add_argument("--delta-scale", type=float, default=100.0)
    parser.add_argument("--exploration-coef", type=float, default=0.0001)
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10, help="Greedy eval every N episodes; 0 disables")
    parser.add_argument("--greedy-eval-episodes", type=int, default=3)
    parser.add_argument("--accumulate-episodes", type=int, default=1, help="PPO update every K episodes")
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument(
        "--entropy-coef-start",
        type=float,
        default=None,
        help="If set, linear decay to --entropy-coef over training",
    )
    parser.add_argument("--huber-value-loss", action="store_true")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true", help="Print per-episode stats to stdout (line-buffer with python -u)")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available (default: use CUDA when torch sees a GPU)",
    )
    args = parser.parse_args()

    basepath = str(get_data_dir(None, "processed"))
    if not args.skip_audit:
        result = audit_data_directory(args.state, basepath)
        print_audit_report(result)
        if result["status"] != "PASS":
            raise SystemExit(1)

    max_actions = None if args.max_actions == 0 else args.max_actions
    env = GerrymanderingEnv(
        state=args.state,
        basepath=basepath,
        max_steps=args.max_steps,
        max_action_space_size=max_actions,
        reward_mode=args.reward_mode,
        score_reward_scale=args.score_reward_scale,
        delta_scale_factor=args.delta_scale,
        exploration_coef=args.exploration_coef,
        ema_alpha=args.ema_alpha,
    )
    _, node_features = env.get_graph_observation()
    hyper = PPOHyperParams(
        lr=args.lr,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        entropy_coef=args.entropy_coef,
        entropy_coef_start=args.entropy_coef_start,
        use_huber_value_loss=args.huber_value_loss,
        huber_delta=args.huber_delta,
    )
    dev = "cpu" if args.cpu else None
    agent = PPOAgent(
        node_feature_dim=node_features.shape[1],
        action_dim=env.action_space.n,
        hyperparams=hyper,
        device=dev,
    )
    print(f"[train] compute_device={agent.device}", flush=True)
    trainer = TrainingLoop(
        env=env,
        agent=agent,
        config=TrainingConfig(
            num_episodes=args.episodes,
            seed=args.seed,
            eval_every_n_episodes=args.eval_every,
            greedy_eval_episodes=args.greedy_eval_episodes,
            accumulate_episodes_before_update=args.accumulate_episodes,
            verbose=args.verbose,
        ),
    )
    history = trainer.train()
    print(f"Training complete: episodes={len(history['episode_rewards'])}, run_dir={trainer.run_dir}")


if __name__ == "__main__":
    main()
