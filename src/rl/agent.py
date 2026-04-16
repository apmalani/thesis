"""
Proximal Policy Optimization (PPO) Agent with Graph Neural Network Encoder.

This module implements a PPO agent that uses a GNN to encode precinct graphs
into state representations for redistricting optimization.
"""

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*torch_geometric.distributed.*")
warnings.filterwarnings("ignore", message=".*distributed.*")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_encoder import GraphStateEncoder, networkx_to_pyg_data
from torch_geometric.data import Data, Batch
import networkx as nx


class GNNActor(nn.Module):
    """
    Actor network that uses GNN encoder for state representation
    and outputs action logits over valid precinct reassignments.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        policy_hidden_dim: int = 64,
        encoder_type: str = 'graphsage',
        aggregation: str = 'mean',
        temperature: float = 0.5
    ):
        """
        Initialize GNN-based actor network.
        
        Args:
            node_feature_dim: Dimension of node features
            action_dim: Number of possible actions
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_embedding_dim: Output dimension of GNN encoder
            policy_hidden_dim: Hidden dimension for policy head
            encoder_type: Type of GNN encoder ('graphsage' or 'gcn')
            aggregation: How to aggregate node embeddings
        """
        super(GNNActor, self).__init__()
        
        # GNN encoder for state representation
        self.gnn_encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation
        )
        
        # Policy head: maps state embedding to action logits
        self.policy_head = nn.Sequential(
            nn.Linear(gnn_embedding_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, action_dim)
        )
        self.temperature = temperature
        
        # WEIGHT INITIALIZATION: Orthogonal init to give policy a little 'opinion' instead of perfect randomness
        # This ensures weights start with some structure, not too close to zero
        for module in self.policy_head:
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: encode graph and output action logits.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment (optional)
            
        Returns:
            Action logits [batch_size, action_dim] or [action_dim]
        """
        # Encode graph to state representation
        state_embedding = self.gnn_encoder(x, edge_index, batch)
        
        # If single graph, add batch dimension
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        
        # Get action logits
        action_logits = self.policy_head(state_embedding)
        
        # TEMPERATURE SCALING: Sharpen or soften the distribution
        action_logits = action_logits / max(self.temperature, 1e-6)
        
        # Remove batch dimension if single graph
        if batch is None:
            action_logits = action_logits.squeeze(0)
        
        return action_logits

    def set_temperature(self, temperature: float):
        self.temperature = temperature


class GNNCritic(nn.Module):
    """
    Critic network that uses GNN encoder to estimate state values.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        value_hidden_dim: int = 64,
        encoder_type: str = 'graphsage',
        aggregation: str = 'mean'
    ):
        """
        Initialize GNN-based critic network.
        
        Args:
            node_feature_dim: Dimension of node features
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_embedding_dim: Output dimension of GNN encoder
            value_hidden_dim: Hidden dimension for value head
            encoder_type: Type of GNN encoder ('graphsage' or 'gcn')
            aggregation: How to aggregate node embeddings
        """
        super(GNNCritic, self).__init__()
        
        # GNN encoder for state representation (shared with actor)
        self.gnn_encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation
        )
        
        # Value head: maps state embedding to scalar value
        self.value_head = nn.Sequential(
            nn.Linear(gnn_embedding_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: encode graph and output state value.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment (optional)
            
        Returns:
            State value [batch_size, 1] or scalar
        """
        # Encode graph to state representation
        state_embedding = self.gnn_encoder(x, edge_index, batch)
        
        # If single graph, add batch dimension
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        
        # Get state value
        value = self.value_head(state_embedding)
        
        # Remove batch dimension if single graph
        if batch is None:
            value = value.squeeze(0).squeeze(-1)
        else:
            value = value.squeeze(-1)
        
        return value


class PPOAgent:
    """
    Proximal Policy Optimization agent with GNN-based state encoding.
    
    Uses Graph Neural Networks to encode precinct graphs into state representations,
    enabling the agent to learn from the graph structure of redistricting problems.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        max_grad_norm: float = 0.5,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        encoder_type: str = 'graphsage',
        aggregation: str = 'mean'
    ):
        """
        Initialize PPO agent with GNN encoder.
        
        Args:
            node_feature_dim: Dimension of node (precinct) features
            action_dim: Number of possible actions (action space size, e.g., 1068 for AZ)
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of update epochs per batch
            max_grad_norm: Maximum gradient norm for clipping
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_embedding_dim: Output dimension of GNN encoder
            encoder_type: Type of GNN encoder ('graphsage' or 'gcn')
            aggregation: How to aggregate node embeddings
            
        Note:
            The action_dim must match the environment's action_space.n, which is
            fixed at initialization based on the initial number of valid precinct
            reassignments (e.g., 1068 for Arizona).
        """
        from utils.device_utils import get_device, get_device_name
        self.device = get_device()
        
        # Force GPU usage check and print confirmation
        device_name = get_device_name()
        if self.device.type == 'cuda':
            print(f"✅ SUCCESS: Training on {device_name} (GPU)")
        else:
            print(f"⚠️  WARNING: Training on {device_name} (CPU) - GPU not available")
        
        # Initialize actor and critic networks
        self.policy = GNNActor(
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation
        ).to(self.device)
        
        self.value = GNNCritic(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type,
            aggregation=aggregation
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_loss_coef = 0.1  # Scale value loss to prevent it from overwhelming policy updates (reduced from 0.5)
        
        # Entropy decay tracking
        self.best_score_seen = -float('inf')  # Track best score for entropy decay
        self.base_ent_coef = 0.0005  # Reduced from 0.01 - stop paying agent to be random
        self.min_ent_coef = 0.0001  # Reduced from 0.001 - minimum entropy coefficient
        self.entropy_coef = self.base_ent_coef
        
        # Memory for storing transitions
        self.memory = {
            'graph_data': [],  # Store graph data (x, edge_index)
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
    
    def get_action(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float, float, float]:
        """
        Get action from policy given graph state.
        
        Args:
            graph: NetworkX graph representing precinct adjacency
            node_features: Node feature matrix [num_nodes, node_feature_dim]
            action_mask: Binary mask for valid actions [action_dim]
            
        Returns:
            Tuple of (action, log_prob, value, entropy)
        """
        # Convert to PyTorch Geometric format
        data = networkx_to_pyg_data(graph, node_features, device=self.device)
        
        with torch.no_grad():
            # Get action logits from policy
            action_logits = self.policy(data.x, data.edge_index)
            
            # HARD ACTION MASKING: Set invalid action logits to -1e9 before Softmax
            # This ensures agent's 'curiosity' (entropy) only explores legal moves
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                # Set invalid actions (mask=0) to -1e9, keep valid actions (mask=1) unchanged
                masked_logits = action_logits + (1 - mask_tensor) * -1e9
                action_probs = torch.softmax(masked_logits, dim=-1)
                
                valid_actions = mask_tensor.sum().item()
                if valid_actions == 0:
                    return 0, 0.0, 0.0, 0.0
            else:
                action_probs = torch.softmax(action_logits, dim=-1)
            
            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()  # Calculate policy entropy
            
            # Get state value
            value = self.value(data.x, data.edge_index)
            
        return action.item(), log_prob.item(), value.item(), entropy.item()
    
    def store_transition(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """
        Store transition in memory.
        
        Args:
            graph: NetworkX graph
            node_features: Node feature matrix
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: State value estimate
            done: Whether episode is done
        """
        self.memory['graph_data'].append((graph, node_features))
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)
    
    def compute_returns(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute discounted returns and advantages.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (returns, advantages) as tensors
        """
        returns = []
        
        discounted_return = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_return = 0
            discounted_return = rewards[i] + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        
        advantages = returns - values_tensor
        
        # Normalize advantages (but check for zero std first)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        
        # If std is too small, advantages are all similar - this is a problem
        if adv_std < 1e-6:
            print(f"  ⚠️  WARNING: Advantage std is near zero ({adv_std.item():.6f})!")
            print(f"    This means all transitions have similar advantage - no learning signal!")
            # Don't normalize if std is too small - it would blow up
            # Instead, use raw advantages (they're already centered around value estimates)
        else:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return returns, advantages
    
    def update(self, best_score: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        Update policy and value networks using PPO algorithm.
        
        Args:
            best_score: Current best score seen (for entropy decay)
        
        Returns:
            Dictionary with loss information or None if no data
        """
        # Update best score for entropy decay
        if best_score is not None and best_score > self.best_score_seen:
            self.best_score_seen = best_score
        if len(self.memory['graph_data']) == 0:
            return None
        
        # Convert stored graphs to PyTorch Geometric format
        graph_batch = []
        for graph, node_features in self.memory['graph_data']:
            data = networkx_to_pyg_data(graph, node_features, device=self.device)
            graph_batch.append(data)
        
        # Batch graphs
        batch = Batch.from_data_list(graph_batch)
        
        # Get stored values
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        
        # REWARD SCALING: Keep rewards as-is (already scaled by 100 in environment)
        # REMOVED normalization division - we want the agent to see the full reward signal
        # Small score improvements (0.01) become visible when multiplied by 100
        scaled_rewards = rewards
        
        # Compute returns and advantages using scaled rewards
        returns, advantages = self.compute_returns(scaled_rewards, self.memory['values'], dones)
        
        # VERIFY ADVANTAGES: Check if advantages are meaningful
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        if abs(adv_mean) < 1e-6 and adv_std < 1e-6:
            print(f"  ⚠️  WARNING: Advantages are near zero! mean={adv_mean:.6f}, std={adv_std:.6f}")
            print(f"    This means rewards are not providing learning signal!")
            print(f"    Reward stats: mean={np.mean(scaled_rewards):.4f}, std={np.std(scaled_rewards):.4f}")
            print(f"    Value stats: mean={np.mean(self.memory['values']):.4f}, std={np.std(self.memory['values']):.4f}")
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        # PPO update loop with heartbeat logging
        for epoch in range(self.k_epochs):
            # Get new action logits and values
            action_logits = self.policy(batch.x, batch.edge_index, batch.batch)
            new_values = self.value(batch.x, batch.edge_index, batch.batch)
            
            # Compute new action probabilities
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            entropies.append(entropy.item())
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # ENTROPY COEFFICIENT: Use low fixed value to force policy convergence
            # Reduced significantly to stop 'paying' the agent to be random
            # No decay needed - use low value from start to force learning
            ent_coef = self.entropy_coef
            
            policy_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy
            
            # Check for NaN/Inf in policy loss
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                print(f"  ⚠️  Warning: NaN/Inf detected in policy loss at epoch {epoch+1}/{self.k_epochs}")
                break
            
            policy_losses.append(policy_loss.item())
            
            # Value loss (scaled to prevent overwhelming policy updates)
            value_loss = nn.MSELoss()(new_values, returns)
            scaled_value_loss = value_loss * self.value_loss_coef
            
            # Check for NaN/Inf in value loss
            if torch.isnan(scaled_value_loss) or torch.isinf(scaled_value_loss):
                print(f"  ⚠️  Warning: NaN/Inf detected in value loss at epoch {epoch+1}/{self.k_epochs}")
                break
            
            value_losses.append(value_loss.item())  # Store unscaled for logging
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value (using scaled loss)
            self.value_optimizer.zero_grad()
            scaled_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
        
        self.clear_memory()
        
        # Calculate average entropy across all epochs
        avg_entropy = np.mean(entropies) if entropies else 0.0
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': avg_entropy
        }
    
    def clear_memory(self):
        """Clear stored transitions."""
        for key in self.memory:
            self.memory[key] = []
    
    def save_model(self, filepath: str):
        """Save model checkpoints."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

    def set_entropy_coef(self, entropy_coef: float):
        self.entropy_coef = max(entropy_coef, 0.0)

    def reset_optimizers(self):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)

    def randomize_policy_head(self, std: float = 0.01):
        for module in self.policy.policy_head:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
