"""Training loop orchestration for PPO redistricting experiments."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys

import numpy as np
import pandas as pd
import torch

from redistricting.env.core import GerrymanderingEnv
from redistricting.rl.agent import PPOAgent
from redistricting.utils.logger import BestMapLogger
from redistricting.utils.paths import get_outputs_dir
from redistricting.utils.visualization import plot_learning_dashboard


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    num_episodes: int = 1000
    update_frequency: int = 1
    save_frequency: int = 100
    eval_episodes: int = 10
    early_stop_patience: int = 150
    early_stop_min_improvement: float = 1e-3
    seed: int = 42
    eval_every_n_episodes: int = 10
    greedy_eval_episodes: int = 3
    accumulate_episodes_before_update: int = 1
    verbose: bool = False


class TrainingLoop:
    """High-level orchestrator for environment interaction and PPO updates."""

    def __init__(
        self,
        env: GerrymanderingEnv,
        agent: PPOAgent,
        config: Optional[TrainingConfig] = None,
        run_dir: Optional[Path] = None,
    ):
        self.env = env
        self.agent = agent
        self.config = config or TrainingConfig()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir or (get_outputs_dir(env.state, "runs") / timestamp)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config.verbose:
            print(
                f"[train] run_dir={self.run_dir} episodes={self.config.num_episodes}",
                flush=True,
                file=sys.stdout,
            )
        self.best_map_logger = BestMapLogger(get_outputs_dir(env.state, "best_maps"))
        self.training_history: Dict[str, list] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "approx_kl": [],
            "clip_fraction": [],
            "efficiency_gaps": [],
            "best_legal_score": [],
            "episode_mean_entropy": [],
            "episode_mean_top1_prob": [],
            "episode_mean_mask_sparsity": [],
            "greedy_mean_total_score": [],
            "greedy_mean_efficiency_gap": [],
            "greedy_mean_max_pop_deviation": [],
            "greedy_mean_return": [],
        }
        self.eval_metrics_rows: List[Dict[str, float]] = []
        self.best_reward = -float("inf")
        self.patience_counter = 0
        self._set_seeds(self.config.seed)

    def _set_seeds(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _check_early_stopping(self, episode_reward: float) -> bool:
        if len(self.training_history["episode_rewards"]) < 25:
            return False
        if episode_reward > self.best_reward + self.config.early_stop_min_improvement:
            self.best_reward = episode_reward
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stop_patience

    def _append_loss_row(self, loss_info: Optional[Dict[str, float]]) -> None:
        if loss_info:
            self.training_history["policy_losses"].append(loss_info["policy_loss"])
            self.training_history["value_losses"].append(loss_info["value_loss"])
            self.training_history["entropies"].append(loss_info["entropy"])
            self.training_history["approx_kl"].append(loss_info["approx_kl"])
            self.training_history["clip_fraction"].append(loss_info["clip_fraction"])
        else:
            for key in ("policy_losses", "value_losses", "entropies", "approx_kl", "clip_fraction"):
                self.training_history[key].append(float("nan"))

    def _greedy_eval(self) -> Dict[str, float]:
        scores: List[float] = []
        egs: List[float] = []
        pops: List[float] = []
        returns: List[float] = []
        for ge in range(self.config.greedy_eval_episodes):
            _, _ = self.env.reset(seed=self.config.seed + 17_000 + ge)
            ep_ret = 0.0
            done = False
            last_info: Dict = {}
            while not done:
                graph, features = self.env.get_graph_observation()
                action_mask = self.env.get_valid_action_mask()
                action = self.agent.greedy_action(graph, features, action_mask)
                _, reward, terminated, truncated, last_info = self.env.step(action)
                ep_ret += float(reward)
                done = bool(terminated or truncated)
            scores.append(float(last_info.get("total_score", 0.0)))
            egs.append(float(last_info.get("efficiency_gap", 0.0)))
            pops.append(float(last_info.get("max_pop_deviation", 0.0)))
            returns.append(ep_ret)
        return {
            "mean_total_score": float(np.mean(scores)),
            "mean_efficiency_gap": float(np.mean(egs)),
            "mean_max_pop_deviation": float(np.mean(pops)),
            "mean_return": float(np.mean(returns)),
        }

    def _save_history_csv(self) -> Path:
        history_path = self.run_dir / "training_history.csv"
        pd.DataFrame(self.training_history).to_csv(history_path, index=False)
        return history_path

    def _save_eval_csv(self) -> None:
        if not self.eval_metrics_rows:
            return
        pd.DataFrame(self.eval_metrics_rows).to_csv(self.run_dir / "eval_metrics.csv", index=False)

    def train(self) -> Dict[str, list]:
        """Run training episodes and periodic PPO updates."""
        acc = max(1, int(self.config.accumulate_episodes_before_update))
        last_ep = self.config.num_episodes - 1
        for episode in range(self.config.num_episodes):
            _, _ = self.env.reset(seed=self.config.seed + episode)
            episode_reward = 0.0
            episode_length = 0
            done = False
            step_entropies: List[float] = []
            step_top1: List[float] = []
            step_sparsity: List[float] = []
            info: Dict = {}
            while not done:
                graph, features = self.env.get_graph_observation()
                action_mask = self.env.get_valid_action_mask()
                action, log_prob, value, _entropy = self.agent.get_action(graph, features, action_mask)
                diag = self.agent.policy_diagnostics(graph, features, action_mask)
                step_entropies.append(diag["entropy"])
                step_top1.append(diag["top1_prob"])
                step_sparsity.append(diag["mask_sparsity"])
                _, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)
                self.agent.store_transition(
                    graph,
                    features,
                    action,
                    reward,
                    log_prob,
                    value,
                    done,
                    action_mask=action_mask,
                )
                episode_reward += float(reward)
                episode_length += 1

                if self.best_map_logger.is_best_legal_map(
                    max_pop_deviation=info.get("max_pop_deviation", 0.0),
                    current_reward=info.get("total_score", float("-inf")),
                ):
                    self.best_map_logger.save_best_map(
                        partition=self.env.partition,
                        episode=episode + 1,
                        step=episode_length,
                        reward=float(info.get("total_score", 0.0)),
                        max_pop_deviation=float(info.get("max_pop_deviation", 0.0)),
                        metrics={"efficiency_gap": float(info.get("efficiency_gap", 0.0))},
                    )

            if acc == 1:
                should_flush = episode % self.config.update_frequency == 0
            else:
                should_flush = (episode + 1) % acc == 0 or episode == last_ep
            loss_info = None
            if should_flush and len(self.agent.memory["actions"]) > 0:
                self.agent.set_entropy_coef_for_episode(episode, self.config.num_episodes)
                loss_info = self.agent.update()
            self._append_loss_row(loss_info)

            self.training_history["episode_rewards"].append(episode_reward)
            self.training_history["episode_lengths"].append(episode_length)
            self.training_history["efficiency_gaps"].append(float(info.get("efficiency_gap", 0.0)))
            self.training_history["best_legal_score"].append(float(self.best_map_logger.get_best_score()))
            self.training_history["episode_mean_entropy"].append(
                float(np.mean(step_entropies)) if step_entropies else float("nan")
            )
            self.training_history["episode_mean_top1_prob"].append(
                float(np.mean(step_top1)) if step_top1 else float("nan")
            )
            self.training_history["episode_mean_mask_sparsity"].append(
                float(np.mean(step_sparsity)) if step_sparsity else float("nan")
            )

            greedy_stats = {
                "mean_total_score": float("nan"),
                "mean_efficiency_gap": float("nan"),
                "mean_max_pop_deviation": float("nan"),
                "mean_return": float("nan"),
            }
            ran_greedy_eval = False
            if self.config.eval_every_n_episodes > 0 and (
                (episode + 1) % self.config.eval_every_n_episodes == 0
                or episode == self.config.num_episodes - 1
            ):
                greedy_stats = self._greedy_eval()
                ran_greedy_eval = True
                row = {
                    "episode": float(episode + 1),
                    "mean_total_score": greedy_stats["mean_total_score"],
                    "mean_efficiency_gap": greedy_stats["mean_efficiency_gap"],
                    "mean_max_pop_deviation": greedy_stats["mean_max_pop_deviation"],
                    "mean_return": greedy_stats["mean_return"],
                }
                self.eval_metrics_rows.append(row)

            self.training_history["greedy_mean_total_score"].append(greedy_stats["mean_total_score"])
            self.training_history["greedy_mean_efficiency_gap"].append(greedy_stats["mean_efficiency_gap"])
            self.training_history["greedy_mean_max_pop_deviation"].append(
                greedy_stats["mean_max_pop_deviation"]
            )
            self.training_history["greedy_mean_return"].append(greedy_stats["mean_return"])

            if self.config.verbose:
                parts = [
                    f"[train] ep={episode + 1}/{self.config.num_episodes}",
                    f"len={episode_length}",
                    f"ret={episode_reward:.4f}",
                    f"H_mean={self.training_history['episode_mean_entropy'][-1]:.4f}",
                    f"top1_mean={self.training_history['episode_mean_top1_prob'][-1]:.4f}",
                    f"best_legal={self.training_history['best_legal_score'][-1]:.4f}",
                ]
                if loss_info is not None:
                    parts.append(f"ppo_H={loss_info['entropy']:.4f}")
                    parts.append(f"kl={loss_info['approx_kl']:.5f}")
                    parts.append(f"v_loss={loss_info['value_loss']:.4f}")
                if ran_greedy_eval:
                    parts.append(f"greedy_score={greedy_stats['mean_total_score']:.4f}")
                    parts.append(f"greedy_ret={greedy_stats['mean_return']:.4f}")
                print(" ".join(parts), flush=True, file=sys.stdout)

            if episode > 0 and episode % self.config.save_frequency == 0:
                self.agent.save_model(str(self.run_dir / f"checkpoint_ep{episode}.pth"))
                self._save_history_csv()

            if self._check_early_stopping(episode_reward):
                break

        self.agent.save_model(str(self.run_dir / "final_model.pth"))
        self._save_history_csv()
        self._save_eval_csv()
        plot_learning_dashboard(
            self.training_history,
            save_path=str(self.run_dir / "training_progress.png"),
            show=False,
        )
        return self.training_history

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Run deterministic evaluation episodes and return summary stats."""
        episodes = num_episodes or self.config.eval_episodes
        rewards = []
        for _ in range(episodes):
            _, _ = self.env.reset()
            done = False
            total = 0.0
            while not done:
                graph, features = self.env.get_graph_observation()
                action_mask = self.env.get_valid_action_mask()
                action, _, _, _ = self.agent.get_action(graph, features, action_mask)
                _, reward, terminated, truncated, _info = self.env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            rewards.append(total)
        return {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}
