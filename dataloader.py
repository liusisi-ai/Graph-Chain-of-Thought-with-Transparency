import torch
import os
import numpy as np
import networkx as nx
from typing import Dict, Optional
from torch_geometric.data import Data


def _load_raw_data(dataset_name, filename):
    base_dir = f"dataset/{dataset_name}"
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = torch.load(path, weights_only=False)
    if not hasattr(data, 'edge_index') and hasattr(data, 'adj_t'):
        row, col, _ = data.adj_t.t().coo()
        data.edge_index = torch.stack([row, col], dim=0)

    return data, base_dir


def _load_simteg_features(base_dir, num_nodes):
    emb_files = ["simteg_sbert_x.pt", "simteg_roberta_x.pt", "simteg_e5_x.pt"]
    embs = []
    for f in emb_files:
        p = os.path.join(base_dir, f)
        if os.path.exists(p):
            embs.append(torch.load(p, map_location='cpu', weights_only=False))
        else:
            raise FileNotFoundError(f"Embedding missing: {p}")

    x = torch.cat(embs, dim=-1)
    if num_nodes is not None:
        assert x.shape[0] == num_nodes
    return x


def build_token_map(data) -> Optional[Dict[str, dict]]:
    """
    从 Data 对象构建 token_map: node_id -> {node_feature, node_id, node_degree, node_index}
    要求 data 包含 raw_texts, x, edge_index 属性
    """
    if not hasattr(data, 'raw_texts') or data.raw_texts is None:
        print("⚠️ Data 缺少 raw_texts，无法构建 token_map")
        return None

    features = data.x.cpu().numpy() if torch.is_tensor(data.x) else np.array(data.x)

    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)

    all_node_ids = data.raw_texts
    token_map: Dict[str, dict] = {}
    for i, node_id in enumerate(all_node_ids):
        token_map[node_id] = {
            'node_feature': features[i],
            'node_id': node_id,
            'node_degree': G.degree(i),
            'node_index': i,
        }

    print(f"✅ Token map built from Data. Total: {len(token_map)}")
    return token_map


def load_gnn_dataset(dataset_name="cora", task="nc"):
 
    if task == "nc":
        filename = "processed_data.pt"
    elif task == "lp":
        filename = "processed_data_link_notest.pt"
    else:
        raise ValueError(f"Unknown task: {task}")

    pyg_data, base_dir = _load_raw_data(dataset_name, filename)
    x_features = _load_simteg_features(base_dir, pyg_data.num_nodes)

    final_data = Data(
        x=x_features,
        edge_index=pyg_data.edge_index,
        y=getattr(pyg_data, 'y', None),
        num_nodes=x_features.shape[0]
    )

    for key in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_data, key):
            setattr(final_data, key, getattr(pyg_data, key))

    if hasattr(pyg_data, 'raw_texts'):
        final_data.raw_texts = pyg_data.raw_texts

    return final_data


def load_lp_data_with_test_split(dataset_name="cora"):
    print(">>> Computing LP Split (Diff Logic)...")
    full_data, _ = _load_raw_data(dataset_name, "processed_data.pt")
    train_data, base_dir = _load_raw_data(dataset_name, "processed_data_link_notest.pt")
    x_features = _load_simteg_features(base_dir, train_data.num_nodes)
    def edges_to_set(edge_index):
        s = set()
        if edge_index.shape[1] == 0: return s
        row, col = edge_index
        for i in range(edge_index.shape[1]):
            u, v = row[i].item(), col[i].item()
            if u > v: u, v = v, u
            s.add((u, v))
        return s

    full_set = edges_to_set(full_data.edge_index)
    train_set = edges_to_set(train_data.edge_index)
    test_set = full_set - train_set

    print(f"    Full Edges: {len(full_set)} | Train Edges: {len(train_set)}")
    print(f"    Diff (Test Edges): {len(test_set)}")

    if len(test_set) > 0:
        test_edges = torch.tensor(list(test_set), dtype=torch.long).t()
    else:
        test_edges = torch.empty((2, 0), dtype=torch.long)
    final_data = Data(
        x=x_features,
        edge_index=train_data.edge_index,
        y=getattr(train_data, 'y', None),
        num_nodes=x_features.shape[0]
    )

    final_data.test_pos_edge_index = test_edges

    if hasattr(full_data, 'raw_texts'):
        final_data.raw_texts = full_data.raw_texts
    elif hasattr(train_data, 'raw_texts'):
        final_data.raw_texts = train_data.raw_texts

    return final_data

data = load_lp_data_with_test_split()

print("-" * 30)
print("✅ Loading Done！")