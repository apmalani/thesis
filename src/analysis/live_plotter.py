"""
Live plotting utilities for real-time training visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
import os


class LiveTrainingPlotter:
    """
    Live plotter for real-time training progress visualization.
    
    Updates plots during training to show progress as it happens.
    """
    
    def __init__(self, update_frequency: int = 5, save_path: Optional[str] = None):
        """
        Initialize live plotter.
        
        Args:
            update_frequency: Update plots every N episodes
            save_path: Optional path to save plots
        """
        self.update_frequency = update_frequency
        self.save_path = save_path
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('GNN-PPO Training Progress (Live)', fontsize=16, fontweight='bold')
        
        # Initialize plot data
        self.episode_rewards = []
        self.efficiency_gaps = []
        self.policy_losses = []
        self.value_losses = []
        self.episodes = []
        
        # Plot handles for updating
        self.reward_line = None
        self.eg_line = None
        self.policy_loss_line = None
        self.value_loss_line = None
        
        self._initialize_plots()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def _initialize_plots(self):
        """Initialize empty plots with labels."""
        # Reward plot
        self.axes[0, 0].set_title('Episode Rewards', fontweight='bold')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Efficiency Gap plot
        self.axes[0, 1].set_title('Efficiency Gap', fontweight='bold')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Efficiency Gap')
        self.axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Policy Loss plot
        self.axes[1, 0].set_title('Policy Loss', fontweight='bold')
        self.axes[1, 0].set_xlabel('Update')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Value Loss plot
        self.axes[1, 1].set_title('Value Loss', fontweight='bold')
        self.axes[1, 1].set_xlabel('Update')
        self.axes[1, 1].set_ylabel('Loss')
        self.axes[1, 1].grid(True, alpha=0.3)
    
    def update(self, episode: int, history: Dict):
        """
        Update plots with latest training data.
        
        Args:
            episode: Current episode number
            history: Training history dictionary
        """
        # Update data
        if 'episode_rewards' in history and len(history['episode_rewards']) > 0:
            self.episode_rewards = history['episode_rewards']
            self.episodes = list(range(len(self.episode_rewards)))
        
        if 'efficiency_gaps' in history and len(history['efficiency_gaps']) > 0:
            self.efficiency_gaps = [
                eg for eg in history['efficiency_gaps'] 
                if eg is not None and not np.isnan(eg)
            ]
        
        if 'policy_losses' in history and len(history['policy_losses']) > 0:
            self.policy_losses = history['policy_losses']
        
        if 'value_losses' in history and len(history['value_losses']) > 0:
            self.value_losses = history['value_losses']
        
        # Update plots periodically
        if episode % self.update_frequency == 0:
            self._update_plots()
    
    def _update_plots(self):
        """Update all plot lines."""
        # Clear and redraw for better performance
        for ax in self.axes.flat:
            ax.clear()
        
        self._initialize_plots()
        
        # Plot 1: Episode Rewards
        if len(self.episode_rewards) > 0:
            self.axes[0, 0].plot(self.episodes, self.episode_rewards, 'b-', 
                                alpha=0.7, linewidth=1.5, label='Reward')
            
            # Moving average
            if len(self.episode_rewards) > 10:
                window = min(20, len(self.episode_rewards) // 5)
                moving_avg = np.convolve(
                    self.episode_rewards, np.ones(window)/window, mode='valid'
                )
                moving_episodes = self.episodes[window-1:]
                self.axes[0, 0].plot(moving_episodes, moving_avg, 'r-', 
                                   linewidth=2, label=f'MA({window})')
            
            self.axes[0, 0].legend()
            
            # Add current value text
            if len(self.episode_rewards) > 0:
                latest = self.episode_rewards[-1]
                self.axes[0, 0].text(0.02, 0.98, f'Latest: {latest:.2f}',
                                   transform=self.axes[0, 0].transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Efficiency Gap
        if len(self.efficiency_gaps) > 0:
            eg_episodes = list(range(len(self.efficiency_gaps)))
            self.axes[0, 1].plot(eg_episodes, self.efficiency_gaps, 'g-',
                               alpha=0.7, linewidth=1.5, label='EG')
            
            # Moving average
            if len(self.efficiency_gaps) > 10:
                window = min(20, len(self.efficiency_gaps) // 5)
                moving_avg = np.convolve(
                    self.efficiency_gaps, np.ones(window)/window, mode='valid'
                )
                moving_episodes = eg_episodes[window-1:]
                self.axes[0, 1].plot(moving_episodes, moving_avg, 'orange',
                                   linewidth=2, label=f'MA({window})')
            
            self.axes[0, 1].legend()
            
            # Add current value text
            if len(self.efficiency_gaps) > 0:
                latest = self.efficiency_gaps[-1]
                self.axes[0, 1].text(0.02, 0.98, f'Latest: {latest:.4f}',
                                   transform=self.axes[0, 1].transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Plot 3: Policy Loss
        if len(self.policy_losses) > 0:
            updates = list(range(len(self.policy_losses)))
            self.axes[1, 0].plot(updates, self.policy_losses, 'purple',
                               alpha=0.7, linewidth=1.5, label='Policy Loss')
            self.axes[1, 0].legend()
            
            if len(self.policy_losses) > 0:
                latest = self.policy_losses[-1]
                self.axes[1, 0].text(0.02, 0.98, f'Latest: {latest:.4f}',
                                    transform=self.axes[1, 0].transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
        
        # Plot 4: Value Loss
        if len(self.value_losses) > 0:
            updates = list(range(len(self.value_losses)))
            self.axes[1, 1].plot(updates, self.value_losses, 'brown',
                               alpha=0.7, linewidth=1.5, label='Value Loss')
            self.axes[1, 1].legend()
            
            if len(self.value_losses) > 0:
                latest = self.value_losses[-1]
                self.axes[1, 1].text(0.02, 0.98, f'Latest: {latest:.4f}',
                                    transform=self.axes[1, 1].transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='peachpuff', alpha=0.5))
        
        # Update and draw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to allow GUI to update
    
    def save(self, filename: str = "training_progress_live.png"):
        """Save current plot."""
        if self.save_path:
            filepath = os.path.join(self.save_path, filename)
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Live plot saved to {filepath}")
        else:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Live plot saved to {filename}")
    
    def close(self):
        """Close the plotter."""
        plt.ioff()
        plt.close(self.fig)

