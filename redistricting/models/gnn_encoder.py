"""GNN encoders and graph conversion helpers."""

from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

from redistricting.utils.device import get_device


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for node embeddings."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(node_feature_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run GraphSAGE layers."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        return x


class GCNEncoder(nn.Module):
    """GCN encoder for node embeddings."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(node_feature_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run GCN layers."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        return x


class GraphStateEncoder(nn.Module):
    """Encode graph to a single state embedding."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        embedding_dim: int = 64,
        aggregation: str = "mean",
        encoder_type: str = "graphsage",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aggregation = aggregation
        if encoder_type.lower() == "graphsage":
            self.encoder = GraphSAGEEncoder(node_feature_dim, hidden_dim, num_layers, embedding_dim, dropout)
        elif encoder_type.lower() == "gcn":
            self.encoder = GCNEncoder(node_feature_dim, hidden_dim, num_layers, embedding_dim, dropout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 1),
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode node features and aggregate to state embedding."""
        node_embeddings = self.encoder(x, edge_index)
        if self.aggregation == "mean":
            if batch is not None:
                from torch_geometric.nn import global_mean_pool

                state = global_mean_pool(node_embeddings, batch)
            else:
                state = node_embeddings.mean(dim=0, keepdim=True)
        elif self.aggregation == "max":
            if batch is not None:
                from torch_geometric.nn import global_max_pool

                state = global_max_pool(node_embeddings, batch)
            else:
                state = node_embeddings.max(dim=0, keepdim=True)[0]
        elif self.aggregation == "sum":
            if batch is not None:
                from torch_geometric.nn import global_add_pool

                state = global_add_pool(node_embeddings, batch)
            else:
                state = node_embeddings.sum(dim=0, keepdim=True)
        elif self.aggregation == "attention":
            attention_weights = self.attention(node_embeddings)
            attention_weights = F.softmax(attention_weights, dim=0)
            state = (node_embeddings * attention_weights).sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        if batch is None:
            state = state.squeeze(0)
        return state


def networkx_to_pyg_data(
    graph: nx.Graph, node_features: np.ndarray, device: Optional[torch.device] = None
) -> Data:
    """Convert networkx graph + node features into PyG Data object."""
    if device is None:
        device = get_device()
    edge_index = []
    for edge in graph.edges():
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])
    if len(edge_index) == 0:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_index_tensor = (
            torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        )
    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    return Data(x=x, edge_index=edge_index_tensor)

