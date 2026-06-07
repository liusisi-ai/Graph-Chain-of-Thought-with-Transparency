import torch
import os
import numpy as np
import networkx as nx
from typing import Dict, Optional
from torch_geometric.data import Data


_AUTODL_BASE = "/root/autodl-tmp"


def _resolve_data_path(dataset_name, filename):
    """Find the dataset file across the conventional locations.

    Search order:
      1. dataset/<name>/<filename>                     (canonical path)
      2. dataset/<name>/graph_data_<name>.pt           (single-file variant)
      3. <name>/<filename>                             (project-root folder, e.g. ``cora/processed_data.pt``)
      4. <name>/graph_data_<name>.pt
      5. ./graph_data_<name>.pt                        (project-root drop-in)
      6. /root/autodl-tmp/graph_data_<name>.pt         (AutoDL upload location)
      7. /root/autodl-tmp/<name>/<filename>
      8. /root/autodl-tmp/<name>/graph_data_<name>.pt
    Returns ``(actual_path, base_dir_for_aux_files)``.
    """
    primary_dir = f"dataset/{dataset_name}"
    flat_dir = dataset_name  # e.g. ./cora/
    autodl_dir = os.path.join(_AUTODL_BASE, dataset_name)
    single_file = f"graph_data_{dataset_name}.pt"
    alias_files = []
    # 'arxiv' / 'ogbn-arxiv' / 'ogbn_arxiv' 都映射到 OGBN-Arxiv 风格的
    # paper graph 文件 graph_data_paper.pt。
    if dataset_name in ("arxiv", "ogbn-arxiv", "ogbn_arxiv"):
        alias_files.append("graph_data_paper.pt")

    candidates = [
        (os.path.join(primary_dir, filename), primary_dir),
        (os.path.join(primary_dir, single_file), primary_dir),
        # Project-root <name>/ folder (e.g. cora/processed_data.pt)
        (os.path.join(flat_dir, filename), flat_dir),
        (os.path.join(flat_dir, single_file), flat_dir),
        (single_file, primary_dir),
        (os.path.join(_AUTODL_BASE, single_file), primary_dir),
        (os.path.join(autodl_dir, filename), autodl_dir),
        (os.path.join(autodl_dir, single_file), autodl_dir),
    ]
    for alias in alias_files:
        candidates.extend([
            # dataset/<name>/<alias>
            (os.path.join(primary_dir, alias), primary_dir),
            # <name>/<alias>  ← 用户的 ogbn-arxiv/graph_data_paper.pt 在这里命中
            (os.path.join(flat_dir, alias), flat_dir),
            # /root/autodl-tmp/<name>/<alias>
            (os.path.join(autodl_dir, alias), autodl_dir),
            # /root/autodl-tmp/<alias>
            (os.path.join(_AUTODL_BASE, alias), primary_dir),
            # ./<alias>
            (alias, primary_dir),
        ])
    for path, base in candidates:
        if os.path.exists(path):
            return path, base
    tried = "\n  - ".join(p for p, _ in candidates)
    raise FileNotFoundError(
        f"Could not locate dataset '{dataset_name}'. Tried:\n  - {tried}"
    )


def _describe_obj(obj, depth=0, max_depth=2):
    """Pretty short description of a tensor/list/dict for diagnostic printing."""
    if torch.is_tensor(obj):
        return f"Tensor{tuple(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape} dtype={obj.dtype}"
    if isinstance(obj, (list, tuple)):
        head = obj[:3]
        return f"{type(obj).__name__}(len={len(obj)})  e.g. {head!r}"[:160]
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"dict(keys={list(obj.keys())[:8]})"
        return "{ " + ", ".join(
            f"{k!r}: {_describe_obj(v, depth+1, max_depth)}" for k, v in list(obj.items())[:8]
        ) + (" ...}" if len(obj) > 8 else " }")
    return f"{type(obj).__name__}: {str(obj)[:80]}"


def _flatten_dict(raw):
    """If the dump is nested (e.g. {'data': {...}}), unwrap one level when it
    contains the actual graph payload."""
    if not isinstance(raw, dict):
        return raw
    if any(k in raw for k in ('x', 'edge_index', 'features', 'edges',
                              'feat', 'node_features')):
        return raw
    # Look one level deeper
    for v in raw.values():
        if isinstance(v, dict) and any(
            k in v for k in ('x', 'edge_index', 'features', 'edges')
        ):
            return v
        if isinstance(v, Data):
            return v
    return raw


def _heuristic_match(raw):
    """Try to recover x/edge_index/y from a dict whose keys we don't recognise,
    by inspecting tensor shapes."""
    inferred = {}
    n_nodes = None
    edge_idx = None

    # 1. find edge_index: any 2D LongTensor with one dim == 2
    for k, v in raw.items():
        if torch.is_tensor(v) and v.dtype in (torch.long, torch.int64, torch.int32):
            if v.dim() == 2 and (v.shape[0] == 2 or v.shape[1] == 2):
                ei = v if v.shape[0] == 2 else v.t().contiguous()
                edge_idx = ei.long()
                inferred['edge_index'] = edge_idx
                n_nodes = int(edge_idx.max().item()) + 1
                print(f"   ↳ heuristic: using key '{k}' as edge_index → {tuple(edge_idx.shape)}")
                break

    # 2. find x: 2D float tensor with N rows = n_nodes (or the largest 2D float tensor)
    best_x_key, best_x = None, None
    for k, v in raw.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.is_floating_point():
            if n_nodes is not None and v.shape[0] == n_nodes:
                best_x_key, best_x = k, v
                break
            if best_x is None or v.shape[0] > best_x.shape[0]:
                best_x_key, best_x = k, v
    if best_x is not None:
        inferred['x'] = best_x.float()
        if n_nodes is None:
            n_nodes = best_x.shape[0]
        print(f"   ↳ heuristic: using key '{best_x_key}' as x → {tuple(best_x.shape)}")

    # 3. find y: 1D int tensor whose length matches n_nodes
    if n_nodes is not None:
        for k, v in raw.items():
            if torch.is_tensor(v) and v.dim() == 1 and v.shape[0] == n_nodes:
                if v.dtype in (torch.long, torch.int32, torch.int16, torch.int8, torch.uint8):
                    inferred['y'] = v.long()
                    print(f"   ↳ heuristic: using key '{k}' as y → {tuple(v.shape)}")
                    break

    # 4. find raw_texts: list/tuple of strings whose length matches n_nodes
    if n_nodes is not None:
        for k, v in raw.items():
            if isinstance(v, (list, tuple)) and len(v) == n_nodes \
                    and v and isinstance(v[0], str):
                inferred['raw_texts'] = list(v)
                print(f"   ↳ heuristic: using key '{k}' as raw_texts (n={len(v)})")
                break

    return inferred, n_nodes


def _coerce_to_pyg_data(raw):
    """Best-effort conversion of a dict / namespace dump into ``torch_geometric.data.Data``.

    The single-file dumps used by the project (e.g. ``graph_data_paper.pt``)
    are sometimes stored as plain ``dict``s. We map common alias keys onto the
    canonical PyG attribute names so the downstream code can keep using
    ``data.x`` / ``data.edge_index`` / ``data.y`` etc.
    """
    if isinstance(raw, Data):
        return raw

    if not isinstance(raw, dict):
        # torch_geometric Batch / namespace-like — try to access attributes directly
        try:
            kwargs = {k: getattr(raw, k) for k in raw.keys()}  # type: ignore[attr-defined]
            raw = kwargs
        except Exception:
            return raw  # leave it; downstream will surface a clearer error

    raw = _flatten_dict(raw)
    if isinstance(raw, Data):
        return raw

    print(f"📋 Raw dump contents: {_describe_obj(raw)}")

    aliases = {
        'x': ['x', 'feat', 'features', 'node_features', 'node_feat',
              'node_embedding', 'node_emb', 'embedding', 'node_attr'],
        'edge_index': ['edge_index', 'edges', 'adj', 'edge_list', 'edge'],
        'y': ['y', 'label', 'labels', 'node_label', 'node_labels',
              'class', 'class_id'],
        'train_mask': ['train_mask', 'train_idx', 'train_index'],
        'val_mask': ['val_mask', 'valid_mask', 'val_idx', 'valid_idx',
                     'val_index', 'valid_index'],
        'test_mask': ['test_mask', 'test_idx', 'test_index'],
        'raw_texts': ['raw_texts', 'node_text', 'texts', 'titles',
                      'paper_title', 'title', 'text', 'node_id_text', 'name'],
        'num_nodes': ['num_nodes', 'n_nodes'],
    }

    canonical = {}
    for tgt, names in aliases.items():
        for n in names:
            if n in raw and raw[n] is not None:
                canonical[tgt] = raw[n]
                break

    # Heuristic fallback for the missing critical fields
    if 'x' not in canonical or 'edge_index' not in canonical:
        print("⚠️ Standard keys missing — falling back to shape-based heuristics")
        inferred, _ = _heuristic_match(raw)
        for k, v in inferred.items():
            canonical.setdefault(k, v)

    if 'edge_index' not in canonical or 'x' not in canonical:
        raise RuntimeError(
            "Could not recover 'x' and/or 'edge_index' from the dataset dump. "
            f"Available keys: {list(raw.keys())}.\n"
            "Add the key name to the `aliases` table in dataloader._coerce_to_pyg_data."
        )

    # Some dumps store edges as adjacency lists / coo dicts
    ei = canonical['edge_index']
    if isinstance(ei, dict) and 'row' in ei and 'col' in ei:
        ei = torch.stack([torch.as_tensor(ei['row']), torch.as_tensor(ei['col'])], dim=0)
    elif isinstance(ei, (list, tuple)) and len(ei) == 2:
        ei = torch.stack([torch.as_tensor(ei[0]), torch.as_tensor(ei[1])], dim=0)
    else:
        ei = torch.as_tensor(ei)
    if ei.dim() == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.t().contiguous()
    canonical['edge_index'] = ei.long()

    # Convert {train,val,test}_idx → boolean mask if needed
    n_nodes_hint = None
    if 'x' in canonical and torch.is_tensor(canonical['x']):
        n_nodes_hint = canonical['x'].shape[0]
    elif 'num_nodes' in canonical:
        n_nodes_hint = int(canonical['num_nodes'])

    for mkey in ('train_mask', 'val_mask', 'test_mask'):
        if mkey in canonical:
            v = canonical[mkey]
            if torch.is_tensor(v) and v.dtype != torch.bool and v.dim() == 1 and n_nodes_hint:
                idx = v.long()
                mask = torch.zeros(n_nodes_hint, dtype=torch.bool)
                mask[idx] = True
                canonical[mkey] = mask

    # raw_texts should be a list of strings
    if 'raw_texts' in canonical and not isinstance(canonical['raw_texts'], list):
        canonical['raw_texts'] = list(canonical['raw_texts'])

    # Cast tensors to the right dtype where needed
    if torch.is_tensor(canonical['x']) and canonical['x'].dtype != torch.float32:
        canonical['x'] = canonical['x'].float()
    if 'y' in canonical and torch.is_tensor(canonical['y']) and canonical['y'].dtype != torch.long:
        try:
            canonical['y'] = canonical['y'].long()
        except Exception:
            pass

    raw_text_list = canonical.pop('raw_texts', None)
    if 'num_nodes' not in canonical:
        canonical['num_nodes'] = canonical['x'].shape[0]
    data = Data(**canonical)
    if raw_text_list is not None:
        data.raw_texts = raw_text_list
    return data


def _load_raw_data(dataset_name, filename):
    path, base_dir = _resolve_data_path(dataset_name, filename)
    print(f"📁 Loading {dataset_name} from {path}")
    raw = torch.load(path, weights_only=False)
    data = _coerce_to_pyg_data(raw)

    if not hasattr(data, 'edge_index') and hasattr(data, 'adj_t'):
        row, col, _ = data.adj_t.t().coo()
        data.edge_index = torch.stack([row, col], dim=0)

    return data, base_dir


def _load_simteg_features(base_dir, num_nodes, fallback_x=None):
    """Load the three SimTEG embeddings if they exist; otherwise return ``fallback_x``.

    This lets datasets without SimTEG checkpoints (e.g. raw PubMed) still go
    through the same pipeline by reusing the ``x`` field of ``processed_data.pt``.
    """
    emb_files = ["simteg_sbert_x.pt", "simteg_roberta_x.pt", "simteg_e5_x.pt"]
    paths = [os.path.join(base_dir, f) for f in emb_files]

    if not all(os.path.exists(p) for p in paths):
        if fallback_x is None:
            raise FileNotFoundError(
                f"SimTEG embeddings missing in {base_dir} and no fallback_x provided"
            )
        print(f"ℹ️ SimTEG embeddings not found in {base_dir} — using processed_data.x as features")
        x = fallback_x if torch.is_tensor(fallback_x) else torch.tensor(fallback_x, dtype=torch.float32)
        if num_nodes is not None:
            assert x.shape[0] == num_nodes, f"x shape {x.shape} vs num_nodes {num_nodes}"
        return x

    embs = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
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


def _make_random_split(num_nodes: int, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Generate a random {train,val,test}_mask when the dataset has none."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


def _sample_subgraph_nodes(edge_index, num_nodes: int, max_nodes: int, seed: int = 42):
    """Sample a connected-ish subset using BFS expansion.

    Random node sampling on a sparse citation graph often removes most edges.
    Starting from a few random seeds and expanding through the graph preserves
    much more local structure, which is important for neighbor prompts.
    """
    if max_nodes <= 0 or max_nodes >= num_nodes:
        return torch.arange(num_nodes, dtype=torch.long)

    gen = torch.Generator().manual_seed(seed)
    row = edge_index[0].cpu().tolist()
    col = edge_index[1].cpu().tolist()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(row, col):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)

    selected = set()
    queue = []
    # Multiple seeds avoid getting stuck in one small component.
    num_seeds = min(16, max_nodes, num_nodes)
    seeds = torch.randperm(num_nodes, generator=gen)[:num_seeds].tolist()
    for s in seeds:
        selected.add(int(s))
        queue.append(int(s))

    qpos = 0
    while qpos < len(queue) and len(selected) < max_nodes:
        u = queue[qpos]
        qpos += 1
        neigh = adj[u]
        if len(neigh) > 1:
            order = torch.randperm(len(neigh), generator=gen).tolist()
            neigh = [neigh[i] for i in order]
        for v in neigh:
            if v not in selected:
                selected.add(v)
                queue.append(v)
                if len(selected) >= max_nodes:
                    break

    if len(selected) < max_nodes:
        # Fill isolated remainder if the graph is disconnected.
        for v in torch.randperm(num_nodes, generator=gen).tolist():
            selected.add(int(v))
            if len(selected) >= max_nodes:
                break

    return torch.tensor(sorted(selected), dtype=torch.long)


def _induced_subgraph(data: Data, max_nodes: int, seed: int = 42) -> Data:
    """Return an induced subgraph with node ids remapped to [0, n)."""
    if max_nodes <= 0 or data.num_nodes <= max_nodes:
        return data

    keep = _sample_subgraph_nodes(data.edge_index, data.num_nodes, max_nodes, seed)
    keep_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    keep_mask[keep] = True

    src, dst = data.edge_index.cpu()
    edge_mask = keep_mask[src] & keep_mask[dst]
    old_edges = data.edge_index[:, edge_mask]

    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    remap[keep] = torch.arange(keep.numel(), dtype=torch.long)
    new_edge_index = remap[old_edges.cpu()]

    sub = Data(
        x=data.x[keep],
        edge_index=new_edge_index.long(),
        y=(data.y[keep] if getattr(data, "y", None) is not None else None),
        num_nodes=keep.numel(),
    )

    for key in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, key) and getattr(data, key) is not None:
            setattr(sub, key, getattr(data, key)[keep])

    if hasattr(data, "raw_texts") and data.raw_texts is not None:
        sub.raw_texts = [data.raw_texts[int(i)] for i in keep.tolist()]

    sub.original_node_ids = keep
    print(
        f"✂️ Subgraph sampled: {data.num_nodes} → {sub.num_nodes} nodes, "
        f"{data.edge_index.shape[1]} → {sub.edge_index.shape[1]} edges"
    )
    return sub


def load_gnn_dataset(dataset_name="cora", task="nc", max_nodes: int = 0, seed: int = 42):

    if task == "nc":
        filename = "processed_data.pt"
    elif task == "lp":
        filename = "processed_data_link_notest.pt"
    else:
        raise ValueError(f"Unknown task: {task}")

    pyg_data, base_dir = _load_raw_data(dataset_name, filename)

    # Some single-file dumps (e.g. graph_data_paper.pt) may not expose num_nodes
    if not hasattr(pyg_data, "num_nodes") or pyg_data.num_nodes is None:
        if hasattr(pyg_data, "x") and pyg_data.x is not None:
            pyg_data.num_nodes = pyg_data.x.shape[0]
        elif hasattr(pyg_data, "edge_index") and pyg_data.edge_index is not None:
            pyg_data.num_nodes = int(pyg_data.edge_index.max().item()) + 1

    fallback_x = getattr(pyg_data, 'x', None)
    x_features = _load_simteg_features(base_dir, pyg_data.num_nodes, fallback_x=fallback_x)

    final_data = Data(
        x=x_features,
        edge_index=pyg_data.edge_index,
        y=getattr(pyg_data, 'y', None),
        num_nodes=x_features.shape[0]
    )

    has_any_mask = False
    for key in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_data, key) and getattr(pyg_data, key) is not None:
            setattr(final_data, key, getattr(pyg_data, key))
            has_any_mask = True

    if not has_any_mask and final_data.y is not None:
        print(f"ℹ️ No train/val/test masks in {dataset_name} → generating 60/20/20 random split")
        tr, va, te = _make_random_split(final_data.num_nodes)
        final_data.train_mask, final_data.val_mask, final_data.test_mask = tr, va, te

    # Pick the first available text-like field as raw_texts
    text_keys = ['raw_texts', 'node_text', 'titles', 'text', 'paper_title']
    for k in text_keys:
        if hasattr(pyg_data, k) and getattr(pyg_data, k) is not None:
            final_data.raw_texts = list(getattr(pyg_data, k))
            break

    if max_nodes and max_nodes > 0:
        final_data = _induced_subgraph(final_data, max_nodes=max_nodes, seed=seed)

    return final_data


def load_lp_data_with_test_split(dataset_name="cora"):
    print(">>> Computing LP Split (Diff Logic)...")
    full_data, _ = _load_raw_data(dataset_name, "processed_data.pt")
    train_data, base_dir = _load_raw_data(dataset_name, "processed_data_link_notest.pt")
    fallback_x = getattr(train_data, 'x', None)
    x_features = _load_simteg_features(base_dir, train_data.num_nodes, fallback_x=fallback_x)
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


if __name__ == "__main__":
    # quick sanity check
    data = load_gnn_dataset("pubmed", task="nc")
    print("-" * 30)
    print(f"✅ Loaded: x={tuple(data.x.shape)}, y={data.y.unique().tolist() if data.y is not None else None}, "
          f"edges={data.edge_index.shape[1]}, nodes={data.num_nodes}")