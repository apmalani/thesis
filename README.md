# GNN-PPO Redistricting Optimization Pipeline

A reinforcement learning system for computational redistricting using Graph Neural Networks (GNN) and Proximal Policy Optimization (PPO).

## Architecture

The pipeline implements an **Ensemble-Based Reinforcement Learning** algorithm with:
- **State Encoder**: GraphSAGE or GCN (via PyTorch Geometric) that processes precinct graphs
- **RL Agent**: PPO Actor-Critic with GNN-based state representation
- **Reward Function**: Z-score ensemble of redistricting metrics (Efficiency Gap, Partisan Proportionality, Compactness, VRA compliance)
- **Action Masking**: Ensures only legal moves that maintain district contiguity

### Data Flow

```
Precinct Graph → GNN Encoder → PPO Actor → Action Selection → Environment Step → Reward Function
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA, ROCm, or CPU support)
- PyTorch Geometric 2.4+

### GPU Support

The pipeline supports multiple GPU backends:

#### NVIDIA GPUs (CUDA)
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

#### AMD GPUs (ROCm)
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

#### CPU Only
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### Full Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd thesis
```

2. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Install PyTorch Geometric extensions:
```bash
# For CPU
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# For CUDA (replace version as needed)
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu121.html

# For ROCm (check PyG website for ROCm wheels)
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+rocm6.0.html
```

## Usage

### Training

Train the GNN-PPO agent:

```bash
python train.py --state az --episodes 1000 --verbose
```

Arguments:
- `--state`: State abbreviation (e.g., 'az' for Arizona)
- `--episodes`: Number of training episodes (default: 1000)
- `--verbose`: Enable detailed logging and progress reports

## Project Structure

```
thesis/
├── src/
│   ├── rl/              # Reinforcement learning components
│   │   ├── agent.py     # PPO agent with GNN encoder
│   │   ├── environment.py  # Redistricting environment
│   │   └── trainer.py   # Training loop and curriculum learning
│   ├── models/          # Neural network architectures
│   │   └── gnn_encoder.py  # GraphSAGE/GCN encoders
│   ├── graph/           # Graph construction utilities
│   ├── analysis/        # Metrics and visualization
│   └── utils/           # Path utilities and device detection
├── data/                # Shapefiles and processed data
├── models/              # Saved model checkpoints
├── train.py             # Training entry point
└── requirements.txt     # Python dependencies
```

## Key Features

### Device-Agnostic Design
- Automatic detection of CUDA, ROCm (AMD), or CPU
- Optimized with `torch.compile()` when available
- Seamless fallback to CPU if GPU unavailable

### Action Space Synchronization
- Fixed action dimension (e.g., 1068 for Arizona) ensures compatibility
- Action masking prevents invalid moves
- Cut-vertex detection maintains district contiguity

### Curriculum Learning
- Phase 1: Fairness focus (Efficiency Gap, Partisan Proportionality)
- Phase 2: Transition with increasing complexity
- Phase 3: Compactness optimization

### Live Visualization
- Real-time training progress plots
- District balance reports
- PPO explainability metrics (Policy Loss, Value Loss, Entropy)

## Performance Optimization

The pipeline includes several optimizations:

1. **torch.compile()**: Automatically compiles GNN encoders for faster execution on GPU
2. **Batch Processing**: Efficient batching of graph data for training
3. **Action Masking**: Reduces action space to only valid moves
4. **Early Stopping**: Prevents overfitting with patience-based stopping

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_ppo_redistricting,
  title = {GNN-PPO Redistricting Optimization Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/thesis}
}
```

## License

[Your License Here]

## Contact

[Your Contact Information]

