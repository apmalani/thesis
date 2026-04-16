"""
Training module for GNN-PPO redistricting agent.

Handles training loop, evaluation, and progress tracking.
"""

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*torch_geometric.distributed.*")

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import json
import shutil
from tqdm import tqdm

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import GerrymanderingEnv
from agent import PPOAgent
from utils.paths import get_project_root, get_models_dir, get_outputs_dir
from utils.device_utils import (
    get_device_name, check_gpu_utilization, setup_kernel_optimizations
)
from utils.logger import BestMapLogger

# Optional live plotting
try:
    from analysis.live_plotter import LiveTrainingPlotter
    LIVE_PLOTTING_AVAILABLE = True
except ImportError:
    LIVE_PLOTTING_AVAILABLE = False


class GerrymanderingTrainer:
    """
    Trainer for GNN-PPO redistricting agent.
    
    Manages training loop, curriculum learning, and evaluation.
    """
    
    def __init__(
        self,
        state: str,
        basepath: str,
        reward_weights: Optional[Dict] = None,
        pop_tol: float = 0.05,
        max_steps: int = 1000,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        max_grad_norm: float = 0.5,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_embedding_dim: int = 64,
        encoder_type: str = 'graphsage'
    ):
        """
        Initialize trainer.
        
        Args:
            state: State abbreviation
            basepath: Base path to processed data
            reward_weights: Reward component weights
            pop_tol: Population tolerance
            max_steps: Maximum steps per episode
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Update epochs per batch
            max_grad_norm: Max gradient norm
            gnn_hidden_dim: GNN hidden dimension
            gnn_num_layers: Number of GNN layers
            gnn_embedding_dim: GNN embedding dimension
            encoder_type: GNN encoder type ('graphsage' or 'gcn')
        """
        # Diagnostic check for GNN extension libraries
        self._check_gnn_extensions()
        
        # Set up kernel optimizations (cuDNN benchmark, ROCm settings)
        setup_kernel_optimizations()
        
        # Check for GPU availability (CUDA for NVIDIA, ROCm for AMD)
        from utils.device_utils import get_device, check_rocm_installed
        import torch
        import os
        
        # Check for ROCm (AMD GPU) even if torch.cuda.is_available() is False
        rocm_available = False
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            rocm_available = True
        elif os.path.exists('/opt/rocm') or check_rocm_installed():
            # ROCm might be installed but PyTorch not compiled with ROCm support
            rocm_available = True
        
        test_device = get_device()
        if test_device.type != 'cuda' and not rocm_available:
            print("\n" + "="*80)
            print("⚠️  WARNING: GPU NOT AVAILABLE - USING CPU")
            print("="*80)
            print("GNN training will be VERY SLOW on CPU.")
            print("\nFor AMD GPU users:")
            print("  - Install PyTorch with ROCm support:")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
            print("  - Verify ROCm installation: rocm-smi")
            print("="*80)
            print("Continuing with CPU fallback (training will be slow)...\n")
        elif rocm_available and test_device.type != 'cuda':
            # Check if PyTorch has ROCm compiled in
            has_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
            if has_hip:
                print("\n" + "="*80)
                print("⚠️  WARNING: ROCm SUPPORT DETECTED BUT GPU NOT ACCESSIBLE")
                print("="*80)
                print("PyTorch has ROCm support (HIP version: {})".format(torch.version.hip))
                print("but the GPU is not accessible to WSL2.")
                print("\nThis is a WSL2 GPU passthrough issue, not a PyTorch installation issue.")
                print("To enable GPU access:")
                print("1. Install AMD Windows Driver for WSL2 on Windows")
                print("2. Restart WSL2: wsl --shutdown (in PowerShell)")
                print("3. Install ROCm runtime in WSL2: sudo apt install rocm-smi rocm-dev")
                print("\nSee WSL2_AMD_GPU_SETUP.md for detailed instructions.")
                print("="*80)
                print("Continuing with CPU fallback (training will be slow)...\n")
            else:
                print("\n" + "="*80)
                print("⚠️  WARNING: ROCm DETECTED BUT NOT ACTIVE")
                print("="*80)
                print("AMD GPU detected but PyTorch may not be using it.")
                print("Please install PyTorch with ROCm support:")
                print("  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
                print("="*80)
                print("Continuing with CPU fallback (training will be slow)...\n")
        
        # Disable cuDNN if causing issues with ROCm (can be re-enabled if needed)
        if hasattr(torch.backends, 'cudnn'):
            # Only disable if we detect ROCm and cuDNN is causing issues
            try:
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    # ROCm detected - cuDNN may not be available
                    torch.backends.cudnn.enabled = False
            except:
                pass
        
        # FIXED EPISODE LENGTH: 250 steps for meaningful PPO updates
        # PPO needs more than 30 samples to perform a meaningful update
        FIXED_EPISODE_LENGTH = 250
        self.env = GerrymanderingEnv(state, basepath, pop_tol, reward_weights, max_steps=FIXED_EPISODE_LENGTH)
        
        # Pre-move graph data to GPU once per episode (optimization)
        # This avoids repeated CPU->GPU transfers
        self._episode_graph_data = None
        
        # Get node feature dimension from environment
        _, node_features = self.env.get_graph_observation()
        node_feature_dim = node_features.shape[1]
        
        # Initialize GNN-PPO agent
        self.agent = PPOAgent(
            node_feature_dim=node_feature_dim,
            action_dim=self.env.action_space.n,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            max_grad_norm=max_grad_norm,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_embedding_dim=gnn_embedding_dim,
            encoder_type=encoder_type
        )
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],  # Added for explainability
            'efficiency_gaps': [],
            'partisan_props': [],
            'polsby_poppers': []
        }
        
        self.verbose = False  # Verbose mode flag
        
        # Initialize best map logger
        outputs_dir = get_outputs_dir(state, subdir='best_maps')
        
        # Clean up best_maps folder at the start of each training run
        if outputs_dir.exists():
            # Remove all files in the directory
            for file in outputs_dir.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            # Note: cleaned up at start of training (verbose flag set later in train())
        
        self.best_map_logger = BestMapLogger(outputs_dir, min_improvement=0.0005)
    
    def _check_gnn_extensions(self):
        """
        Check for GNN extension library compatibility.
        
        Detects undefined symbol errors in torch-scatter, torch-sparse, torch-cluster
        and provides repair instructions if issues are found.
        """
        import warnings
        import sys
        
        extensions_to_check = [
            ('torch_scatter', 'torch-scatter'),
            ('torch_sparse', 'torch-sparse'),
            ('torch_cluster', 'torch-cluster')
        ]
        
        failed_extensions = []
        warning_messages = []
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            for module_name, package_name in extensions_to_check:
                try:
                    __import__(module_name)
                except ImportError:
                    failed_extensions.append(package_name)
                except Exception as e:
                    # Check if it's an undefined symbol error
                    error_str = str(e).lower()
                    if 'undefined symbol' in error_str or 'symbol' in error_str:
                        failed_extensions.append(package_name)
                        warning_messages.append(f"{package_name}: {e}")
            
            # Check for warnings about disabled extensions
            for warning in w:
                if 'disabling' in str(warning.message).lower() or \
                   'undefined symbol' in str(warning.message).lower():
                    for module_name, package_name in extensions_to_check:
                        if package_name in str(warning.message).lower():
                            if package_name not in failed_extensions:
                                failed_extensions.append(package_name)
                            warning_messages.append(str(warning.message))
        
        if failed_extensions:
            print("\n" + "="*80)
            print("❌ GNN EXTENSION LIBRARY COMPATIBILITY ERROR")
            print("="*80)
            print("\nThe following PyTorch Geometric extension libraries have issues:")
            for ext in failed_extensions:
                print(f"  - {ext}")
            
            if warning_messages:
                print("\nError details:")
                for msg in warning_messages:
                    print(f"  {msg}")
            
            print("\n" + "="*80)
            print("REPAIR INSTRUCTIONS")
            print("="*80)
            print("\nThe torch-scatter, torch-sparse, and torch-cluster libraries")
            print("must be reinstalled with wheels compatible with your PyTorch version.")
            print("\nStep 1: Uninstall existing extensions:")
            print("  pip uninstall torch-scatter torch-sparse torch-cluster -y")
            print("\nStep 2: Reinstall with compatible wheels:")
            print("  # For CPU:")
            print("  pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html")
            print("\n  # For CUDA (replace cu121 with your CUDA version):")
            print("  pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html")
            print("\nStep 3: Verify installation:")
            print("  python -c 'import torch_scatter; import torch_sparse; import torch_cluster; print(\"OK\")'")
            print("\nFor more information, see:")
            print("  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
            print("="*80 + "\n")
            
            sys.exit(1)
        
        # Early stopping parameters
        self.best_reward = float('-inf')
        self.patience = 150  # Patience counter limit (episodes without improvement before stopping)
        self.patience_counter = 0
        self.min_improvement = 0.001
        
        # Reward normalization
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
    
    def _interpolate_weights(self, weights1: Dict, weights2: Dict, alpha: float) -> Dict:
        """Smoothly interpolate between two weight dictionaries."""
        interpolated = {}
        for key in weights1:
            if key in weights2:
                interpolated[key] = weights1[key] + alpha * (weights2[key] - weights1[key])
            else:
                interpolated[key] = weights1[key]
        return interpolated
    
    def _check_early_stopping(self, episode_reward: float) -> bool:
        """Check if training should stop early."""
        if len(self.training_history['episode_rewards']) < 50:
            return False
        
        if episode_reward > self.best_reward + self.min_improvement:
            self.best_reward = episode_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward based on recent history."""
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 50:
            self.reward_history = self.reward_history[-50:]
        
        self.reward_mean = np.mean(self.reward_history)
        self.reward_std = np.std(self.reward_history) + 1e-8
        
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        return max(-3.0, min(3.0, normalized_reward))
    
    def _print_district_balance_report(self, step: int):
        """Print district balance report in verbose mode."""
        try:
            metrics_df = self.env.metrics_calc.calculate_metrics(
                self.env.partition, include_geometry=True
            )
            metrics = metrics_df.iloc[0]
            
            # Calculate population deviations
            total_pop = sum(
                self.env.graph.nodes[n]['P0010001'] 
                for n in self.env.graph.nodes()
            )
            ideal_pop = total_pop / self.env.n_districts
            
            pop_deviations = []
            for district_id in self.env.partition.parts.keys():
                district_pop = sum(
                    self.env.graph.nodes[n]['P0010001']
                    for n in self.env.partition.parts[district_id]
                )
                deviation = (district_pop - ideal_pop) / ideal_pop
                pop_deviations.append(deviation)
            
            max_dev = max(abs(d) for d in pop_deviations) * 100
            avg_dev = np.mean([abs(d) for d in pop_deviations]) * 100
            eg = metrics.get('EfficiencyGap', 0.0)
            
            print(f"\n{'='*60}")
            print(f"DISTRICT BALANCE REPORT (Step {step})")
            print(f"{'='*60}")
            print(f"Max Population Deviation: {max_dev:.2f}%")
            print(f"Avg Population Deviation: {avg_dev:.2f}%")
            print(f"Current Efficiency Gap: {eg:.4f}")
            print(f"Population Tolerance: {self.env.pop_tol*100:.1f}%")
            print(f"{'='*60}\n")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not generate district balance report: {e}")
    
    def train(
        self,
        num_episodes: int = 1000,
        update_frequency: int = 1,  # Update every episode for faster learning
        save_frequency: int = 100,
        save_path: Optional[str] = None,
        phase1_weights: Optional[dict] = None,
        phase2_weights: Optional[dict] = None,
        phase3_weights: Optional[dict] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Train the GNN-PPO agent.
        
        Args:
            num_episodes: Number of training episodes
            update_frequency: Update agent every N episodes
            save_frequency: Save checkpoints every N episodes
            save_path: Path to save models and logs
            phase1_weights: Reward weights for phase 1 (curriculum learning)
            phase2_weights: Reward weights for phase 2
            phase3_weights: Reward weights for phase 3
            verbose: Enable verbose mode with detailed logging
            
        Returns:
            Training history dictionary
        """
        self.verbose = verbose
        
        if save_path is None:
            # Use project-relative path
            save_path = str(get_models_dir(self.env.state))
        
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize live plotter if available
        live_plotter = None
        if LIVE_PLOTTING_AVAILABLE:
            try:
                live_plotter = LiveTrainingPlotter(
                    update_frequency=5,
                    save_path=save_path
                )
                print("📊 Live plotting enabled - plots will update every 5 episodes")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not initialize live plotter: {e}")
                live_plotter = None
        
        import time
        start_time = time.time()
        
        # Get graph info for verbose output
        graph, node_features = self.env.get_graph_observation()
        num_nodes = len(graph.nodes())
        num_edges = graph.number_of_edges()
        from utils.device_utils import get_device_name
        device_name = get_device_name()
        
        print(f"\n{'='*80}")
        print(f"GNN-PPO TRAINING INITIALIZATION")
        print(f"{'='*80}")
        print(f"State: {self.env.state}")
        # Force GPU usage check and print confirmation
        if self.agent.device.type == 'cuda':
            print(f"✅ SUCCESS: Training on {device_name} (GPU)")
        else:
            print(f"⚠️  WARNING: Training on {device_name} (CPU) - GPU not available")
        print(f"Graph Nodes (Precincts): {num_nodes}")
        print(f"Graph Edges: {num_edges}")
        print(f"Districts: {self.env.n_districts}")
        print(f"Node Feature Dim: {node_features.shape[1]}")
        print(f"Action Space Size: {self.env.action_space.n}")
        print(f"GNN Encoder: {self.agent.policy.gnn_encoder.encoder.__class__.__name__}")
        print(f"Verbose Mode: {'ON' if verbose else 'OFF'}")
        print(f"{'='*80}\n")
        
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Curriculum learning with weight interpolation
            # NOTE: max_steps is now fixed at 30 for all phases (reduced for faster debugging)
            if phase1_weights and phase2_weights and phase3_weights:
                if episode < 30:
                    self.env.reward_weights = phase1_weights
                    self.agent.eps_clip = 0.4
                    self.agent.lr = 3e-4
                    phase_name = "Fairness Focus"
                elif episode < 50:
                    alpha = (episode - 30) / 20
                    self.env.reward_weights = self._interpolate_weights(
                        phase1_weights, phase2_weights, alpha
                    )
                    self.agent.eps_clip = 0.4 - 0.1 * alpha
                    self.agent.lr = 3e-4
                    phase_name = "Early Transition"
                elif episode < 70:
                    alpha = (episode - 50) / 20
                    self.env.reward_weights = self._interpolate_weights(
                        phase2_weights, phase3_weights, alpha
                    )
                    self.agent.eps_clip = 0.3 - 0.1 * alpha
                    self.agent.lr = 3e-4
                    phase_name = "Late Transition"
                else:
                    self.env.reward_weights = phase3_weights
                    self.agent.eps_clip = 0.2
                    self.agent.lr = 3e-4
                    phase_name = "Compactness Focus"
                
                # Update optimizer learning rates
                for param_group in self.agent.policy_optimizer.param_groups:
                    param_group['lr'] = self.agent.lr
                for param_group in self.agent.value_optimizer.param_groups:
                    param_group['lr'] = self.agent.lr
                
                if episode % 5 == 0:
                    print(f"\n=== Episode {episode + 1}: {phase_name} Phase ===")
            
            episode_reward = 0
            episode_length = 0
            self.env.reset()
            
            # Pre-move graph to GPU once at episode start (optimization)
            graph, node_features = self.env.get_graph_observation()
            if self.agent.device.type == 'cuda':
                from models.gnn_encoder import networkx_to_pyg_data
                self._episode_graph_data = networkx_to_pyg_data(
                    graph, node_features, device=self.agent.device
                )
            
            if episode % 5 == 0:
                valid_actions_start = self.env.get_valid_action_mask().sum()
                print(f"\nEpisode {episode + 1}/{num_episodes}: "
                      f"Starting with {valid_actions_start:.0f} valid actions")
            
            # Reset patience and best reward at start of each episode
            self.patience_counter = 0
            self.best_reward = float('-inf')
            
            # Early stopping tracking: Reset patience when new best legal map is found
            recent_rewards = []  # Track for logging only
            recent_max_devs = []  # Track for logging only
            best_score_this_episode = -float('inf')  # Track best score in this episode
            global_best_score = self.best_map_logger.get_best_score()  # Track global best across all episodes
            consecutive_illegal_steps = 0  # Track for logging only (no forced termination)
            steps_since_improvement = 0  # Count steps since last meaningful improvement
            last_improvement_step = -1  # Track when last improvement occurred
            
            for step in range(self.env.max_steps):
                made_progress = False
                # Get graph observation (cached, only updates when partition changes)
                graph, node_features = self.env.get_graph_observation()
                action_mask = self.env.get_valid_action_mask()
                
                # Update GPU data if partition changed (cache invalidated)
                if self.agent.device.type == 'cuda' and self._episode_graph_data is None:
                    from models.gnn_encoder import networkx_to_pyg_data
                    self._episode_graph_data = networkx_to_pyg_data(
                        graph, node_features, device=self.agent.device
                    )
                
                # Get action from agent (now returns entropy)
                action, log_prob, value, entropy = self.agent.get_action(
                    graph, node_features, action_mask
                )
                
                # Step environment
                _, reward, done, info = self.env.step(action)
                
                # Get score (raw un-magnified quality) from environment
                # The environment stores this in _previous_score after calculating it
                # This is the EXACT same value used for saving best maps
                score = self.env._previous_score if self.env._previous_score is not None else 0.0
                
                # Track best score in this episode
                if score > best_score_this_episode:
                    best_score_this_episode = score
                
                # Track metrics for early stopping
                max_pop_dev = info.get('max_pop_deviation', 100.0)  # Default to high if missing
                recent_rewards.append(reward)
                recent_max_devs.append(max_pop_dev)
                
                # Track consecutive illegal steps (Max_Dev > 5.0%) for logging only
                # REMOVED: No forced termination - agent must learn to recover
                if max_pop_dev > 5.0:
                    consecutive_illegal_steps += 1
                else:
                    consecutive_illegal_steps = 0  # Reset if map becomes legal
                
                # Log illegal state but continue episode
                if consecutive_illegal_steps == 30 and (episode % 5 == 0 or self.verbose):
                    print(f"  ⚠️  Map has been illegal (Max Dev: {max_pop_dev:.2f}%) "
                          f"for {consecutive_illegal_steps} consecutive steps - continuing to allow recovery")
                
                # Check for best legal map using the raw score (not magnified reward)
                # This ensures the score used for saving matches the score printed in logs
                if self.best_map_logger.is_best_legal_map(max_pop_dev, score):
                    # Extract metrics for metadata
                    metrics_dict = {
                        'efficiency_gap': info.get('efficiency_gap', 0.0),
                        'max_pop_deviation': max_pop_dev,
                        'score': score  # Use raw score, not magnified reward
                    }
                    
                    # Try to get additional metrics if available
                    try:
                        metrics_df = self.env.metrics_calc.calculate_metrics(
                            self.env.partition, include_geometry=False
                        )
                        if not metrics_df.empty:
                            row = metrics_df.iloc[0]
                            metrics_dict.update({
                                'polsby_popper_avg': row.get('PolPopperAvg', 0.0),
                                'polsby_popper_min': row.get('PolPopperMin', 0.0),
                                'min_opp_avg': row.get('MinOppAvg', 0.0),
                                'min_opp_min': row.get('MinOppMin', 0.0),
                            })
                    except:
                        pass
                    
                    # Save best map using raw score
                    saved_path = self.best_map_logger.save_best_map(
                        partition=self.env.partition,
                        episode=episode,
                        step=step,
                        reward=score,  # Use raw score, not magnified reward
                        max_pop_deviation=max_pop_dev,
                        metrics=metrics_dict
                    )
                    
                    # Console print - score matches the value used for saving
                    print(f"  ⭐ New Best Legal Map Found! Score: {score:.4f} | "
                          f"Max Dev: {max_pop_dev:.2f}% | "
                          f"Saving to {saved_path}")
                    
                    # RESET EARLY STOPPING: New best legal map found, reset patience
                    steps_since_improvement = 0
                    last_improvement_step = step
                    global_best_score = score  # Update global best
                    made_progress = True  # Signal that progress was made
                
                # Track learning progress based on GLOBAL best score (not just episode score)
                # Progress = improvement in global best score (the score used for saving best maps)
                # If a new best legal map was found above, made_progress is already True
                if not made_progress:
                    # Check if score improved global best (with small threshold to avoid noise)
                    # Handle case where global_best_score might be -inf (first episode)
                    if global_best_score == -float('inf') or score > global_best_score + 0.01:
                        global_best_score = score
                        made_progress = True
                        steps_since_improvement = 0
                        last_improvement_step = step
                
                # Update steps since last improvement
                if not made_progress:
                    steps_since_improvement += 1
                
                # Early stopping: If no learning progress for 150 steps, end episode
                # This prevents wasting time when agent has stopped learning
                # Patience is based on GLOBAL best score, not episode score
                if steps_since_improvement >= 150:
                    done = True
                    print(f"  Early stopping: No learning progress for 150 steps "
                          f"(Last improvement at step {last_improvement_step}, "
                          f"Global Best Score: {global_best_score:.4f}, "
                          f"Episode Best Score: {best_score_this_episode:.4f})")
                    break
                
                # Keep only last 50 steps for logging
                if len(recent_rewards) > 50:
                    recent_rewards.pop(0)
                    recent_max_devs.pop(0)
                
                # Invalidate GPU cache if partition changed (step invalidates cache)
                if self.agent.device.type == 'cuda':
                    self._episode_graph_data = None
                
                # Store transition (reuse graph observation - it's cached and updated in step)
                self.agent.store_transition(
                    graph, node_features, action, reward, log_prob, value, done
                )
                
                episode_reward += reward
                episode_length += 1
                
                # Verbose mode: District balance report every 50 steps
                if self.verbose and step > 0 and step % 50 == 0:
                    self._print_district_balance_report(step)
                
                # Enhanced console logging: "Step X | Reward: [R] | Score: [S] | Entropy: [E] | Max Dev: [D]% | EG: [G]"
                # Score [S] is the EXACT same value used for saving best maps
                if self.verbose and ((step % 10 == 0 and step > 0) or step < 5):
                    max_dev = info.get('max_pop_deviation', 0.0)
                    eg = info.get('efficiency_gap', 0.0)
                    is_disconnected = info.get('is_disconnected', False)
                    is_illegal = info.get('is_illegal', False)
                    
                    # Build status markers
                    status_markers = []
                    if is_disconnected:
                        status_markers.append("[DISCONNECTED]")
                    if is_illegal:
                        status_markers.append("⚠️ ILLEGAL")
                    
                    status_str = " " + " ".join(status_markers) if status_markers else ""
                    
                    print(f"  Step {step} | Reward: {reward:.4f} | Score: {score:.4f}{status_str} | "
                          f"Entropy: {entropy:.4f} | Max Dev: {max_dev:.2f}% | EG: {eg:.4f}")
                
                # Episode ends only when max_steps reached (250 steps)
                # No early termination - agent must learn to recover from illegal states
                if done:
                    # Print episode end for all episodes
                    final_max_dev = info.get('max_pop_deviation', 0.0)
                    final_eg = info.get('efficiency_gap', 0.0)
                    distance = info.get('distance_from_baseline', 0)
                    print(f"  Episode {episode + 1} completed at step {step + 1}/{self.env.max_steps} "
                          f"(Max Dev: {final_max_dev:.2f}%, EG: {final_eg:.4f}, "
                          f"Distance from baseline: {distance} precincts)")
                    break
            
            # Update agent with error handling and GPU cache clearing
            if episode % update_frequency == 0:
                if episode % 5 == 0 or self.verbose or episode < 3:  # Always print for first 3 episodes
                    print(f"  Updating agent (Episode {episode + 1}, {len(self.agent.memory['actions'])} transitions)...")
                    
                # Clear GPU cache before update to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    # Get best score for entropy decay (from current episode)
                    # Use the best score seen in this episode
                    best_score_for_entropy = best_score_this_episode if best_score_this_episode > -float('inf') else None
                    
                    loss_info = self.agent.update(best_score=best_score_for_entropy)
                    if loss_info:
                        # Check for NaN/Inf in losses
                        if (np.isnan(loss_info.get('policy_loss', 0)) or 
                            np.isinf(loss_info.get('policy_loss', 0)) or
                            np.isnan(loss_info.get('value_loss', 0)) or 
                            np.isinf(loss_info.get('value_loss', 0))):
                            print(f"  ⚠️  Warning: NaN/Inf detected in losses, skipping update")
                        else:
                            self.training_history['policy_losses'].append(loss_info['policy_loss'])
                            self.training_history['value_losses'].append(loss_info['value_loss'])
                            if 'entropy' in loss_info:
                                self.training_history['entropies'].append(loss_info['entropy'])
                            
                            # Show post-update entropy statistics
                            if episode % 5 == 0 or self.verbose or episode < 3:
                                if 'entropy' in loss_info:
                                    print(f"    Post-update entropy: {loss_info['entropy']:.4f}")
                            
                            # Verbose mode: PPO explainability metrics
                            if self.verbose:
                                entropy = loss_info.get('entropy', 0.0)
                                print(f"\n  PPO EXPLAINABILITY METRICS:")
                                print(f"    Policy Loss: {loss_info['policy_loss']:.6f}")
                                print(f"    Value Loss: {loss_info['value_loss']:.6f}")
                                print(f"    Entropy: {entropy:.6f} "
                                      f"({'High exploration' if entropy > 1.0 else 'Low exploration'})")
                                print()
                            elif episode % 5 == 0:
                                print(f"  Policy Loss: {loss_info['policy_loss']:.4f}, "
                                      f"Value Loss: {loss_info['value_loss']:.4f}")
                except Exception as e:
                    print(f"  ❌ Error during agent update: {e}")
                    print(f"  Continuing training...")
                    # Clear GPU cache on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Normalize reward
            normalized_reward = self._normalize_reward(episode_reward)
            
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            
            # Update live plotter
            if live_plotter is not None:
                live_plotter.update(episode, self.training_history)
            
            if episode % 5 == 0:
                print(f"  Episode {episode + 1} completed: "
                      f"Reward={episode_reward:.2f} (norm={normalized_reward:.2f}), "
                      f"Steps={episode_length}")
            
            # Early stopping
            if self._check_early_stopping(episode_reward):
                print(f"\nEarly stopping triggered at episode {episode + 1}")
                break
            
            # Record metrics
            try:
                metrics_df = self.env.metrics_calc.calculate_metrics(
                    self.env.partition, include_geometry=True
                )
                metrics = metrics_df.iloc[0]
                self.training_history['efficiency_gaps'].append(metrics['EfficiencyGap'])
                self.training_history['partisan_props'].append(metrics['PartisanProp'])
                self.training_history['polsby_poppers'].append(metrics['PolPopperAvg'])
            except:
                self.training_history['efficiency_gaps'].append(0.0)
                self.training_history['partisan_props'].append(0.0)
                self.training_history['polsby_poppers'].append(0.0)
            
            # Save checkpoint
            if episode % save_frequency == 0 and episode > 0:
                self.save_training_history(f"{save_path}/training_history.csv")
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                print(f"Episode {episode}: Avg Reward (last 100): {avg_reward:.4f}")
        
        # Save final model
        self.agent.save_model(f"{save_path}/final_model.pth")
        
        # Close live plotter and save final plot
        if live_plotter is not None:
            live_plotter.save("training_progress_live.png")
            live_plotter.close()
            print("📊 Live plotting closed")
        
        print("Generating final plots...")
        self.plot_training_progress(save_path)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"Total episodes: {num_episodes}")
        if self.training_history['episode_rewards']:
            print(f"Average reward: {np.mean(self.training_history['episode_rewards']):.4f}")
            print(f"Best reward: {np.max(self.training_history['episode_rewards']):.4f}")
        print(f"Best legal maps saved: {self.best_map_logger.get_best_map_count()}")
        print(f"Best legal score: {self.best_map_logger.get_best_score():.4f}")
        print(f"Best maps saved to: {self.best_map_logger.output_dir}")
        print(f"{'='*80}\n")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average reward: {np.mean(self.training_history['episode_rewards']):.2f}")
        print(f"Best reward: {np.max(self.training_history['episode_rewards']):.2f}")
        
        return self.training_history
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Tuple[List, List]:
        """Evaluate trained agent."""
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_metrics = []
        
        for episode in range(num_episodes):
            print(f"  Evaluation Episode {episode + 1}/{num_episodes}...")
            episode_reward = 0
            self.env.reset()
            
            for step in range(self.env.max_steps):
                graph, node_features = self.env.get_graph_observation()
                action_mask = self.env.get_valid_action_mask()
                action, _, _ = self.agent.get_action(graph, node_features, action_mask)
                
                _, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            print(f"    Episode {episode + 1} completed: Reward={episode_reward:.2f}")
            
            try:
                metrics_df = self.env.metrics_calc.calculate_metrics(
                    self.env.partition, include_geometry=True
                )
                metrics = metrics_df.iloc[0].to_dict()
                eval_metrics.append(metrics)
            except:
                pass
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"Evaluation Results:")
        print(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        
        return eval_rewards, eval_metrics
    
    def save_training_history(self, filepath: str):
        """Save training history to CSV."""
        import pandas as pd
        
        max_length = max(len(self.training_history[key]) for key in self.training_history.keys())
        
        data = {}
        for key, values in self.training_history.items():
            if len(values) < max_length:
                data[key] = values + [None] * (max_length - len(values))
            else:
                data[key] = values
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def plot_training_progress(self, save_path: str):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(self.training_history['episode_rewards']))
        
        axes[0, 0].plot(episodes, self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(episodes, self.training_history['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        if self.training_history['efficiency_gaps']:
            axes[1, 0].plot(self.training_history['efficiency_gaps'])
            axes[1, 0].set_title('Efficiency Gap Over Time')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Efficiency Gap')
            axes[1, 0].grid(True)
        
        if self.training_history['polsby_poppers']:
            axes[1, 1].plot(self.training_history['polsby_poppers'])
            axes[1, 1].set_title('Polsby-Popper Average Over Time')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Polsby-Popper Average')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {save_path}/training_progress.png")


if __name__ == "__main__":
    from utils.paths import get_data_dir
    data_path = str(get_data_dir("az", "processed"))
    trainer = GerrymanderingTrainer("az", data_path)
    trainer.train(num_episodes=1000)
