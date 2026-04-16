import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_distributions(state, basepath):
    df = pd.read_csv(f"{basepath}/{state}/ensemble_metrics.csv")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
        
    for i, metric in enumerate(df.columns):
        if i >= len(axes):
            break
            
        ax = axes[i]
        values = df[metric].dropna()
        
        ax.hist(values, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax.axvline(values.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {values.median():.4f}')
        ax.axvline(values.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean: {values.mean():.4f}')
        
        ax.set_title(f'{metric} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i in range(len(df.columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{basepath}/{state}/metric_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_chain_comparison(state, basepath, metric):
    df = pd.read_csv(f"{basepath}/{state}/ensemble_metrics.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for chain_id in sorted(df['chain'].unique()):
        chain_data = df[df['chain'] == chain_id][metric].dropna()
        ax1.plot(chain_data.index, chain_data.values, alpha=0.7, label=f'Chain {chain_id}')
    
    ax1.set_title(f'{metric} by Chain (Time Series)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel(metric)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for chain_id in sorted(df['chain'].unique()):
        chain_data = df[df['chain'] == chain_id][metric].dropna()
        ax2.hist(chain_data, alpha=0.6, bins=30, label=f'Chain {chain_id}', density=True)
    
    ax2.set_title(f'{metric} Distribution by Chain', fontsize=14, fontweight='bold')
    ax2.set_xlabel(metric)
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{basepath}/{state}/{metric}_chain_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_progress(history: dict, save_path: Optional[str] = None, show: bool = True):
    """
    Plot training progress with multi-pane visualization.
    
    Creates plots for:
    - Reward vs Episode
    - Efficiency Gap vs Episode
    
    Args:
        history: Training history dictionary with keys:
            - 'episode_rewards': List of episode rewards
            - 'efficiency_gaps': List of efficiency gaps
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    episodes = range(len(history.get('episode_rewards', [])))
    
    # Plot 1: Reward vs Episode
    if 'episode_rewards' in history and len(history['episode_rewards']) > 0:
        rewards = history['episode_rewards']
        axes[0].plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1.5, label='Episode Reward')
        
        # Add moving average
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_episodes = range(window-1, len(rewards))
            axes[0].plot(moving_episodes, moving_avg, 'r-', linewidth=2, 
                        label=f'Moving Avg ({window} episodes)')
        
        axes[0].set_title('Training Progress: Reward vs Episode', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No reward data available', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Training Progress: Reward vs Episode', fontsize=14, fontweight='bold')
    
    # Plot 2: Efficiency Gap vs Episode
    if 'efficiency_gaps' in history and len(history['efficiency_gaps']) > 0:
        eg_values = history['efficiency_gaps']
        # Filter out None/NaN values
        valid_eg = [(i, v) for i, v in enumerate(eg_values) if v is not None and not np.isnan(v)]
        if valid_eg:
            valid_episodes, valid_values = zip(*valid_eg)
            axes[1].plot(valid_episodes, valid_values, 'g-', alpha=0.7, linewidth=1.5, 
                        label='Efficiency Gap')
            
            # Add horizontal line at 0 for reference
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add moving average
            if len(valid_values) > 10:
                window = min(50, len(valid_values) // 10)
                moving_avg = np.convolve(valid_values, np.ones(window)/window, mode='valid')
                moving_episodes = valid_episodes[window-1:]
                axes[1].plot(moving_episodes, moving_avg, 'orange', linewidth=2,
                            label=f'Moving Avg ({window} episodes)')
            
            axes[1].set_title('Training Progress: Efficiency Gap vs Episode', 
                            fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Efficiency Gap')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No efficiency gap data available',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Training Progress: Efficiency Gap vs Episode',
                            fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No efficiency gap data available',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Training Progress: Efficiency Gap vs Episode',
                        fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

