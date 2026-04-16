"""
Graph Neural Network Encoder for Redistricting State Representation.

This module implements a GraphSAGE-based encoder that processes precinct graphs
to extract meaningful state representations for the RL agent.

The encoder processes precinct graphs where:
- Nodes represent precincts with demographic/partisan features
- Edges represent adjacency relationships
- Output is a fixed-size state embedding for the PPO agent

The action space size is fixed at environment initialization (e.g., 1068 for AZ)
to ensure compatibility between the GNN encoder output and the agent's policy network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple
import numpy as np
import networkx as nx


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for learning node embeddings from precinct graphs.
    
    Uses GraphSAGE (Sample and Aggregate) to learn node representations that
    capture both local and neighborhood features.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize GraphSAGE encoder.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for graph convolutions
            num_layers: Number of GraphSAGE layers
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
        """
        super(GraphSAGEEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_feature_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE encoder.
        
        Args:
            x: Node feature matrix [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer without activation (for downstream tasks)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        
        return x


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder alternative to GraphSAGE.
    
    Uses GCN layers for node embedding learning.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize GCN encoder.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for graph convolutions
            num_layers: Number of GCN layers
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
        """
        super(GCNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN encoder.
        
        Args:
            x: Node feature matrix [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer without activation
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        
        return x


class GraphStateEncoder(nn.Module):
    """
    Complete graph state encoder that processes graph structure
    and outputs a global state representation for RL.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        embedding_dim: int = 64,
        aggregation: str = 'mean',
        encoder_type: str = 'graphsage',
        dropout: float = 0.1
    ):
        """
        Initialize graph state encoder.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for graph convolutions
            num_layers: Number of GNN layers
            embedding_dim: Node embedding dimension
            aggregation: How to aggregate node embeddings ('mean', 'max', 'sum', 'attention')
            encoder_type: Type of encoder ('graphsage' or 'gcn')
            dropout: Dropout rate
        """
        super(GraphStateEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.use_compile = False  # Will be set if compile is supported
        
        # Choose encoder type
        if encoder_type.lower() == 'graphsage':
            self.encoder = GraphSAGEEncoder(
                node_feature_dim, hidden_dim, num_layers, embedding_dim, dropout
            )
        elif encoder_type.lower() == 'gcn':
            self.encoder = GCNEncoder(
                node_feature_dim, hidden_dim, num_layers, embedding_dim, dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Try to compile the encoder for performance (if supported)
        # REMOVED torch.compile() temporarily - CPU training with compilation can swallow gradients
        # in older PyTorch versions. Using raw GNN model for debugging gradient flow.
        try:
            from utils.device import supports_compile
            if supports_compile():
                # Compile the encoder for faster execution on GPU
                # mode='reduce-overhead' is good for inference-heavy workloads
                self.encoder = torch.compile(self.encoder, mode='reduce-overhead')
                self.use_compile = True
        except Exception:
            # Compilation not supported or failed - use regular encoder
            self.use_compile = False
        
        # Attention mechanism for aggregation (optional)
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 1)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: encode graph and aggregate to global state.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for multiple graphs (optional)
            
        Returns:
            Global state representation [batch_size, embedding_dim] or [embedding_dim]
        """
        # Get node embeddings
        node_embeddings = self.encoder(x, edge_index)  # [num_nodes, embedding_dim]
        
        # Aggregate node embeddings to global state
        if self.aggregation == 'mean':
            if batch is not None:
                # For batched graphs
                from torch_geometric.nn import global_mean_pool
                state = global_mean_pool(node_embeddings, batch)
            else:
                state = node_embeddings.mean(dim=0, keepdim=True)
        elif self.aggregation == 'max':
            if batch is not None:
                from torch_geometric.nn import global_max_pool
                state = global_max_pool(node_embeddings, batch)
            else:
                state = node_embeddings.max(dim=0, keepdim=True)[0]
        elif self.aggregation == 'sum':
            if batch is not None:
                from torch_geometric.nn import global_add_pool
                state = global_add_pool(node_embeddings, batch)
            else:
                state = node_embeddings.sum(dim=0, keepdim=True)
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation
            attention_weights = self.attention(node_embeddings)  # [num_nodes, 1]
            attention_weights = F.softmax(attention_weights, dim=0)
            state = (node_embeddings * attention_weights).sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Remove batch dimension if single graph
        if batch is None:
            state = state.squeeze(0)
        
        return state


def networkx_to_pyg_data(
    graph: nx.Graph,
    node_features: np.ndarray,
    device: Optional[torch.device] = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        graph: NetworkX graph
        node_features: Node feature matrix [num_nodes, feature_dim]
        device: Device to place tensors on (if None, uses default device)
        
    Returns:
        PyTorch Geometric Data object
    """
    # Determine device if not provided
    if device is None:
        from utils.device import get_device
        device = get_device()
    
    # Get edge index
    edge_index = []
    for edge in graph.edges():
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])  # Undirected graph
    
    if len(edge_index) == 0:
        # Handle isolated nodes
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    
    # Convert node features and move to device
    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    
    data = Data(x=x, edge_index=edge_index)
    
    return data

