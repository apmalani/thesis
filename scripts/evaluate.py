#!/usr/bin/env python3
"""Evaluate a saved GNN-PPO checkpoint."""

import argparse
from pathlib import Path

from redistricting.env.core import GerrymanderingEnv
from redistricting.rl.agent import PPOAgent
from redistricting.rl.trainer import TrainingConfig, TrainingLoop
from redistricting.utils.paths import get_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained redistricting model")
    parser.add_argument("--state", type=str, default="az")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=256, help="0 = no cap (all legal moves)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    basepath = str(get_data_dir(None, "processed"))
    max_actions = None if args.max_actions == 0 else args.max_actions
    env = GerrymanderingEnv(state=args.state, basepath=basepath, max_action_space_size=max_actions)
    _, node_features = env.get_graph_observation()
    agent = PPOAgent(
        node_feature_dim=node_features.shape[1],
        action_dim=env.action_space.n,
        device="cpu" if args.cpu else None,
    )
    agent.load_model(args.checkpoint)

    trainer = TrainingLoop(
        env=env,
        agent=agent,
        config=TrainingConfig(num_episodes=1, eval_episodes=args.episodes),
        run_dir=Path(args.checkpoint).parent,
    )
    summary = trainer.evaluate(num_episodes=args.episodes)
    print(summary)


if __name__ == "__main__":
    main()

