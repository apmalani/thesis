"""PPO agent with GNN actor-critic models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Batch

from redistricting.models.gnn_encoder import GraphStateEncoder, networkx_to_pyg_data
from redistricting.utils.device import get_device, setup_kernel_optimizations


class GNNActor(nn.Module):
    """Actor network mapping graph states to action logits."""

    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        policy_hidden_dim: int = 64,
        encoder_type: str = "graphsage",
        aggregation: str = "mean",
    ):
        super().__init__()
        self.gnn_encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(gnn_embedding_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None) -> torch.Tensor:
        state_embedding = self.gnn_encoder(x, edge_index, batch)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        logits = self.policy_head(state_embedding)
        if batch is None:
            logits = logits.squeeze(0)
        return logits


class GNNCritic(nn.Module):
    """Critic network estimating scalar state value."""

    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        value_hidden_dim: int = 64,
        encoder_type: str = "graphsage",
        aggregation: str = "mean",
    ):
        super().__init__()
        self.gnn_encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation,
        )
        self.value_head = nn.Sequential(
            nn.Linear(gnn_embedding_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None) -> torch.Tensor:
        state_embedding = self.gnn_encoder(x, edge_index, batch)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        value = self.value_head(state_embedding).squeeze(-1)
        if batch is None:
            value = value.squeeze(0)
        return value


@dataclass
class PPOHyperParams:
    """PPO update hyperparameters."""

    lr: float = 3e-4
    lr_policy: Optional[float] = None
    lr_value: Optional[float] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    k_epochs: int = 4
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.001
    entropy_coef_start: Optional[float] = None
    value_loss_coef: float = 0.5
    use_huber_value_loss: bool = False
    huber_delta: float = 1.0


class PPOAgent:
    """PPO agent that stores graph transitions and performs policy updates."""

    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        hyperparams: Optional[PPOHyperParams] = None,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        encoder_type: str = "graphsage",
        aggregation: str = "mean",
        device: Optional[Union[str, torch.device]] = None,
    ):
        dev: Optional[torch.device] = None
        if device is not None:
            dev = torch.device(device) if isinstance(device, str) else device
        self.device = get_device(dev)
        if self.device.type == "cuda":
            setup_kernel_optimizations()
        self.hyperparams = hyperparams or PPOHyperParams()
        self.action_dim = action_dim
        self.policy = GNNActor(
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation,
        ).to(self.device)
        self.value = GNNCritic(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation,
        ).to(self.device)
        lr_p = self.hyperparams.lr_policy if self.hyperparams.lr_policy is not None else self.hyperparams.lr
        lr_v = self.hyperparams.lr_value if self.hyperparams.lr_value is not None else self.hyperparams.lr * 0.5
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_p)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_v)
        self._entropy_coef_effective = float(self.hyperparams.entropy_coef)
        self.memory: Dict[str, List] = {
            "graph_data": [],
            "action_masks": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

    def get_action(
        self, graph: nx.Graph, node_features: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float, float, float]:
        """Sample an action and return (action, log_prob, value, entropy)."""
        data = networkx_to_pyg_data(graph, node_features, self.device)
        with torch.no_grad():
            logits = self.policy(data.x, data.edge_index)
            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
                logits = logits + (1 - mask) * -1e9
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.value(data.x, data.edge_index)
        return int(action.item()), float(log_prob.item()), float(value.item()), float(entropy.item())

    def set_entropy_coef_for_episode(self, episode: int, total_episodes: int) -> None:
        """Linear schedule from entropy_coef_start to entropy_coef when start is set."""
        h = self.hyperparams
        if h.entropy_coef_start is None:
            self._entropy_coef_effective = float(h.entropy_coef)
            return
        denom = max(total_episodes - 1, 1)
        t = min(float(episode) / float(denom), 1.0)
        self._entropy_coef_effective = (1.0 - t) * float(h.entropy_coef_start) + t * float(h.entropy_coef)

    def greedy_action(
        self, graph: nx.Graph, node_features: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Argmax over legal actions (greedy evaluation)."""
        data = networkx_to_pyg_data(graph, node_features, self.device)
        with torch.no_grad():
            logits = self.policy(data.x, data.edge_index)
            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
                logits = logits + (1 - mask) * -1e9
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            return int(torch.argmax(logits, dim=-1).item())

    def policy_diagnostics(
        self, graph: nx.Graph, node_features: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Entropy, top-k mass, effective action count, and mask sparsity for masked distribution."""
        data = networkx_to_pyg_data(graph, node_features, self.device)
        with torch.no_grad():
            logits = self.policy(data.x, data.edge_index)
            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
                logits = logits + (1 - mask) * -1e9
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            dist = Categorical(probs)
            entropy = float(dist.entropy().item())
            sorted_p, _ = torch.sort(probs, descending=True)
            top1 = float(sorted_p[0].item()) if sorted_p.numel() > 0 else 0.0
            k5 = min(5, sorted_p.numel())
            top5 = float(sorted_p[:k5].sum().item())
            dim = float(probs.numel())
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
                n_legal = float(mask_t.sum().item())
            else:
                n_legal = dim
            mask_sparsity = n_legal / dim if dim > 0 else 0.0
            eff_actions = float(np.exp(entropy)) if entropy < 50 else float("inf")
        return {
            "entropy": entropy,
            "top1_prob": top1,
            "top5_prob": top5,
            "effective_actions": eff_actions,
            "n_legal_actions": n_legal,
            "mask_sparsity": mask_sparsity,
        }

    def store_transition(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Append a transition to in-memory rollout buffer."""
        self.memory["graph_data"].append((graph, node_features))
        if action_mask is None:
            self.memory["action_masks"].append(np.ones(self.action_dim, dtype=np.float32))
        else:
            self.memory["action_masks"].append(action_mask.astype(np.float32))
        self.memory["actions"].append(action)
        self.memory["rewards"].append(float(reward))
        self.memory["log_probs"].append(float(log_prob))
        self.memory["values"].append(float(value))
        self.memory["dones"].append(bool(done))

    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = np.array(self.memory["rewards"], dtype=np.float32)
        values = np.array(self.memory["values"], dtype=np.float32)
        dones = np.array(self.memory["dones"], dtype=np.bool_)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.hyperparams.gamma * next_value * mask - values[t]
            gae = delta + self.hyperparams.gamma * self.hyperparams.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )
        return returns_t, advantages_t

    def update(self) -> Optional[Dict[str, float]]:
        """Run PPO update on collected rollout buffer."""
        if len(self.memory["graph_data"]) == 0:
            return None
        graph_batch = [
            networkx_to_pyg_data(graph, node_features, self.device)
            for graph, node_features in self.memory["graph_data"]
        ]
        batch = Batch.from_data_list(graph_batch)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32, device=self.device)
        returns, advantages = self._compute_gae()
        action_masks = torch.tensor(np.array(self.memory["action_masks"]), dtype=torch.float32, device=self.device)

        policy_losses, value_losses, entropies, approx_kls, clip_fracs = [], [], [], [], []
        for _ in range(self.hyperparams.k_epochs):
            logits = self.policy(batch.x, batch.edge_index, batch.batch)
            logits = logits + (1 - action_masks) * -1e9
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            values = self.value(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.hyperparams.eps_clip, 1 + self.hyperparams.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self._entropy_coef_effective * entropy
            if self.hyperparams.use_huber_value_loss:
                value_loss = nn.functional.smooth_l1_loss(values, returns, beta=self.hyperparams.huber_delta)
            else:
                value_loss = nn.MSELoss()(values, returns)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyperparams.max_grad_norm)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            (value_loss * self.hyperparams.value_loss_coef).backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.hyperparams.max_grad_norm)
            self.value_optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean()
                clip_fraction = ((ratio > (1 + self.hyperparams.eps_clip)) | (ratio < (1 - self.hyperparams.eps_clip))).float().mean()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fracs.append(float(clip_fraction.item()))

        self.clear_memory()
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
        }

    def clear_memory(self) -> None:
        """Clear rollout buffer."""
        for key in self.memory:
            self.memory[key] = []

    def save_model(self, filepath: str) -> None:
        """Persist model and optimizer state."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """Load model and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

