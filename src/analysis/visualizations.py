import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


