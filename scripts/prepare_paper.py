"""Convert ``graph_data_paper.pt`` (GraphTranslator-style paper graph) into the
format expected by ``dataloader.py``:

    dataset/paper/
        processed_data.pt     — PyG Data(x, edge_index, y, train/val/test masks, raw_texts)
        node_info.csv         — paper_id, title, abstract  (used by cluster_prompt.py)

Run from the project root::

    python scripts/prepare_paper.py
    python scripts/prepare_paper.py --src /root/autodl-tmp/graph_data_paper.pt
    python scripts/prepare_paper.py --src ... --tsv /root/autodl-tmp/arxiv.tsv
    python scripts/prepare_paper.py --src ... --inspect       # dump structure only

The script auto-detects whether the .pt holds a PyG ``Data`` object, a plain
``dict``, or a tuple, and extracts the standard fields.  When an
``arxiv.tsv`` (paper_id\\ttitle\\tabstract) is provided it is merged into
``node_info.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ── helpers to dig keys out of dict / Data ────────────────────────


def _get_first(obj, *keys):
    """Return the first attribute / dict-key that exists and is non-empty."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
    else:
        for k in keys:
            if hasattr(obj, k) and getattr(obj, k) is not None:
                return getattr(obj, k)
    return None


def _to_tensor(x, dtype=None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(dtype) if dtype else x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(dtype) if dtype else t
    try:
        t = torch.tensor(x)
        return t.to(dtype) if dtype else t
    except Exception:
        return None


def _to_str_list(obj) -> Optional[List[str]]:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [str(x) for x in obj.tolist()]
    if torch.is_tensor(obj):
        return [str(int(x)) for x in obj.tolist()]
    if isinstance(obj, dict):
        # node_id -> str  ; iterate by ascending int key if possible
        try:
            keys = sorted(obj.keys(), key=lambda k: int(k))
        except Exception:
            keys = list(obj.keys())
        return [str(obj[k]) for k in keys]
    return None


# ── inspect mode ──────────────────────────────────────────────────


def inspect(obj, indent: int = 0, prefix: str = ""):
    """Pretty-print the structure of a loaded .pt object."""
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}{prefix}dict ({len(obj)} keys):")
        for k, v in obj.items():
            inspect(v, indent + 1, f"[{k!r}] -> ")
    elif isinstance(obj, (list, tuple)):
        kind = type(obj).__name__
        print(f"{pad}{prefix}{kind} (len={len(obj)})")
        for i, v in enumerate(obj[:3]):
            inspect(v, indent + 1, f"[{i}] -> ")
        if len(obj) > 3:
            print(f"{pad}  ... ({len(obj) - 3} more)")
    elif torch.is_tensor(obj):
        print(f"{pad}{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{prefix}ndarray shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, Data):
        print(f"{pad}{prefix}PyG Data:")
        for k, v in obj:
            inspect(v, indent + 1, f".{k} -> ")
    else:
        s = repr(obj)
        if len(s) > 80:
            s = s[:80] + "..."
        print(f"{pad}{prefix}{type(obj).__name__}: {s}")


# ── tsv loader (paper_id \t title \t abstract) ────────────────────


def load_arxiv_tsv(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return out
    print(f"  loading text from {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            pid = row[0].strip()
            title = row[1].strip() if len(row) > 1 else ""
            abstract = row[2].strip() if len(row) > 2 else ""
            out[pid] = {"title": title, "abstract": abstract}
    print(f"  parsed {len(out)} papers from tsv")
    return out


# ── main extractor ────────────────────────────────────────────────


def extract_fields(obj) -> Dict[str, Any]:
    """Pull the canonical fields out of whatever structure ``obj`` is."""
    # If wrapped in tuple/list of (data, ..) take the first element
    if isinstance(obj, (list, tuple)) and len(obj) >= 1:
        if isinstance(obj[0], (Data, dict)):
            obj = obj[0]

    x = _get_first(obj, "x", "node_feat", "features", "feat", "node_features")
    edge_index = _get_first(obj, "edge_index", "edges", "edge")
    y = _get_first(obj, "y", "labels", "label", "node_label")
    train_idx = _get_first(obj, "train_idx", "train", "train_idx_list")
    val_idx = _get_first(obj, "val_idx", "valid_idx", "val", "valid")
    test_idx = _get_first(obj, "test_idx", "test")
    train_mask = _get_first(obj, "train_mask")
    val_mask = _get_first(obj, "val_mask")
    test_mask = _get_first(obj, "test_mask")
    raw_texts = _get_first(obj, "raw_texts", "raw_text", "node_id", "paper_id",
                           "node_ids", "paper_ids", "ids")
    titles = _get_first(obj, "title", "titles", "node_title")
    abstracts = _get_first(obj, "abstract", "abstracts", "node_abstract")
    text = _get_first(obj, "text", "node_text")

    return {
        "x": x, "edge_index": edge_index, "y": y,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "raw_texts": raw_texts,
        "titles": titles, "abstracts": abstracts, "text": text,
    }


def _idx_to_mask(idx, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    idx_t = _to_tensor(idx, dtype=torch.long)
    if idx_t is not None:
        idx_t = idx_t.flatten()
        mask[idx_t] = True
    return mask


def _make_random_split(num_nodes: int, seed: int = 42,
                       train_ratio: float = 0.6, val_ratio: float = 0.2):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


# ── main ──────────────────────────────────────────────────────────


def _resolve_src(arg_src: Optional[str]) -> str:
    candidates = [
        arg_src,
        "graph_data_paper.pt",
        "../graph_data_paper.pt",
        "/root/autodl-tmp/graph_data_paper.pt",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Cannot locate graph_data_paper.pt. Pass --src /path/to/graph_data_paper.pt"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="Path to graph_data_paper.pt")
    ap.add_argument("--tsv", default=None,
                    help="Optional arxiv.tsv with paper_id\\ttitle\\tabstract")
    ap.add_argument("--dst", default="dataset/paper",
                    help="Destination dataset directory")
    ap.add_argument("--inspect", action="store_true",
                    help="Just print the structure of the .pt and exit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = _resolve_src(args.src)
    print(f"📦 Loading {src} ...")
    obj = torch.load(src, map_location="cpu", weights_only=False)
    print(f"   top-level type: {type(obj).__name__}")

    if args.inspect:
        inspect(obj)
        return

    fields = extract_fields(obj)

    # ── x ──
    x = _to_tensor(fields["x"], dtype=torch.float32)
    if x is None:
        print("❌ No node feature matrix (x/node_feat/features) found.")
        print("Run with --inspect to see what's inside.")
        sys.exit(1)
    if x.dim() != 2:
        x = x.view(x.shape[0], -1)
    num_nodes = x.shape[0]
    print(f"   x: shape={tuple(x.shape)}  dim={x.shape[1]}")

    # ── edge_index ──
    ei = fields["edge_index"]
    edge_index = _to_tensor(ei, dtype=torch.long)
    if edge_index is None:
        print("❌ No edge_index found.")
        sys.exit(1)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    print(f"   edge_index: shape={tuple(edge_index.shape)}")

    # ── y ──
    y_raw = fields["y"]
    y = _to_tensor(y_raw, dtype=torch.long)
    if y is not None:
        y = y.flatten()
        n_class = int(y.max().item()) + 1 if y.numel() else 0
        print(f"   y: shape={tuple(y.shape)}  num_classes={n_class}")
    else:
        print("   y: none")

    # ── splits ──
    if fields["train_mask"] is not None:
        train_mask = _to_tensor(fields["train_mask"], dtype=torch.bool).flatten()
        val_mask   = _to_tensor(fields["val_mask"],   dtype=torch.bool).flatten() if fields["val_mask"]   is not None else torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = _to_tensor(fields["test_mask"],  dtype=torch.bool).flatten() if fields["test_mask"]  is not None else torch.zeros(num_nodes, dtype=torch.bool)
    elif fields["train_idx"] is not None:
        train_mask = _idx_to_mask(fields["train_idx"], num_nodes)
        val_mask   = _idx_to_mask(fields["val_idx"],   num_nodes) if fields["val_idx"]  is not None else torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = _idx_to_mask(fields["test_idx"],  num_nodes) if fields["test_idx"] is not None else torch.zeros(num_nodes, dtype=torch.bool)
    else:
        print("   no splits found → random 60/20/20")
        train_mask, val_mask, test_mask = _make_random_split(num_nodes, args.seed)
    print(f"   splits: train={train_mask.sum().item()} val={val_mask.sum().item()} test={test_mask.sum().item()}")

    # ── raw_texts (paper ids / strings) ──
    raw_texts = _to_str_list(fields["raw_texts"])
    if raw_texts is None or len(raw_texts) != num_nodes:
        raw_texts = [str(i) for i in range(num_nodes)]
    print(f"   raw_texts: {len(raw_texts)} entries (sample: {raw_texts[:2]})")

    # ── titles / abstracts ──
    titles_list   = _to_str_list(fields["titles"])
    abstracts_list = _to_str_list(fields["abstracts"])
    text_list     = _to_str_list(fields["text"])

    title_map: Dict[str, str] = {}
    abstract_map: Dict[str, str] = {}

    if titles_list and len(titles_list) == num_nodes:
        for i, t in enumerate(titles_list):
            title_map[raw_texts[i]] = t
    if abstracts_list and len(abstracts_list) == num_nodes:
        for i, a in enumerate(abstracts_list):
            abstract_map[raw_texts[i]] = a
    if text_list and len(text_list) == num_nodes and not title_map:
        # Fall back: split combined "title. abstract" by first '. '
        for i, t in enumerate(text_list):
            if '. ' in t:
                ti, ab = t.split('. ', 1)
            else:
                ti, ab = t, ""
            title_map[raw_texts[i]] = ti
            abstract_map[raw_texts[i]] = ab

    # Optional external arxiv.tsv merge
    if args.tsv:
        tsv_meta = load_arxiv_tsv(args.tsv)
        for pid, m in tsv_meta.items():
            title_map.setdefault(pid, m.get("title", ""))
            abstract_map.setdefault(pid, m.get("abstract", ""))

    # ── Persist ──
    print(f"\n💾 Writing artefacts to {args.dst} ...")
    os.makedirs(args.dst, exist_ok=True)

    info_rows = []
    for pid in raw_texts:
        info_rows.append({
            "paper_id": pid,
            "title":    title_map.get(pid, ""),
            "abstract": abstract_map.get(pid, ""),
        })
    info_df = pd.DataFrame(info_rows)
    info_path = os.path.join(args.dst, "node_info.csv")
    info_df.to_csv(info_path, index=False)
    matched = (info_df["title"].str.len() > 0).sum() + (info_df["abstract"].str.len() > 0).sum()
    print(f"   node_info.csv: {len(info_df)} rows  ({matched} have title/abstract)")

    final = Data(
        x=x,
        edge_index=edge_index,
        y=y if y is not None else torch.zeros(num_nodes, dtype=torch.long),
        num_nodes=num_nodes,
    )
    final.train_mask = train_mask
    final.val_mask   = val_mask
    final.test_mask  = test_mask
    final.raw_texts  = raw_texts

    out_pt = os.path.join(args.dst, "processed_data.pt")
    torch.save(final, out_pt)
    print(f"   processed_data.pt → {out_pt}")
    print(f"\n✅ Done. nodes={num_nodes}, edges={edge_index.shape[1]}, "
          f"feature_dim={x.shape[1]}, classes={int(y.max().item()) + 1 if y is not None and y.numel() else '?'}")


if __name__ == "__main__":
    main()
