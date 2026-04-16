"""Model utility smoke tests."""

import numpy as np
import torch

from redistricting.models.gnn_encoder import GraphStateEncoder, networkx_to_pyg_data


def test_gnn_encoder_forward(tiny_graph):
    node_features = np.random.randn(len(tiny_graph.nodes()), 12).astype(np.float32)
    data = networkx_to_pyg_data(tiny_graph, node_features, device=torch.device("cpu"))
    encoder = GraphStateEncoder(node_feature_dim=12, embedding_dim=32)
    out = encoder(data.x, data.edge_index)
    assert out.shape[-1] == 32


def test_networkx_to_pyg(tiny_graph):
    node_features = np.random.randn(len(tiny_graph.nodes()), 8).astype(np.float32)
    data = networkx_to_pyg_data(tiny_graph, node_features, device=torch.device("cpu"))
    assert data.x.shape[0] == len(tiny_graph.nodes())
    assert data.edge_index.dtype == torch.long

