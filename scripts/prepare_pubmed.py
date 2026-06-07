"""Convert ``PubMed_orig/`` (raw Pubmed-Diabetes dataset + pubmed.json) into the
format expected by ``dataloader.py``:

    dataset/pubmed/
        processed_data.pt    — PyG Data(x, edge_index, y, train/val/test masks, raw_texts)
        node_info.csv        — paper_id, title, abstract  (used by cluster_prompt.py)

Run from the project root::

    python scripts/prepare_pubmed.py
    python scripts/prepare_pubmed.py --src /root/autodl-tmp/PubMed_orig

The 500-dimensional TF-IDF features that ship with the Pubmed-Diabetes
dataset are used directly as ``data.x``.  ``dataloader._load_simteg_features``
gracefully falls back to ``data.x`` when no ``simteg_*.pt`` files are present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _resolve_src(arg_src: str | None) -> str:
    """Pick the first existing candidate path."""
    candidates = [
        arg_src,
        "PubMed_orig",
        os.path.join("..", "PubMed_orig"),
        "/root/autodl-tmp/PubMed_orig",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Cannot locate PubMed_orig.  Pass --src /path/to/PubMed_orig"
    )


# ── 1. Parse the .NODE.paper.tab feature file ─────────────────────


def parse_node_tab(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return (paper_ids, X[500-dim TF-IDF], y[0..2])."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Line 0: "NODE\tpaper"        (header)
    # Line 1: column descriptors   (use it to extract feature-word order)
    # Line 2+: nodes

    header = lines[1].split("\t")
    word_cols: List[str] = []
    for col in header:
        if col.startswith("numeric:w-"):
            wname = col.split(":")[1].split("=")[0]   # "w-xxx"
            word_cols.append(wname)
    word_idx: Dict[str, int] = {w: i for i, w in enumerate(word_cols)}
    n_dim = len(word_cols)
    print(f"  features: {n_dim}-dim TF-IDF")

    paper_ids: List[str] = []
    labels: List[int] = []
    X_rows: List[np.ndarray] = []

    field_re = re.compile(r"^(?P<key>[^=]+)=(?P<val>.*)$")
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        pid = parts[0].strip()
        x = np.zeros(n_dim, dtype=np.float32)
        label_val = -1
        for tok in parts[1:]:
            m = field_re.match(tok)
            if not m:
                continue
            k, v = m.group("key"), m.group("val")
            if k == "label":
                label_val = int(v) - 1   # 1/2/3 → 0/1/2
            elif k.startswith("w-"):
                j = word_idx.get(k)
                if j is not None:
                    try:
                        x[j] = float(v)
                    except ValueError:
                        pass
        paper_ids.append(pid)
        labels.append(label_val)
        X_rows.append(x)

    X = np.stack(X_rows, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    print(f"  parsed {len(paper_ids)} nodes, label dist: {np.bincount(y[y >= 0])}")
    return paper_ids, X, y


# ── 2. Parse the .DIRECTED.cites.tab edge file ────────────────────


def parse_cites_tab(path: str, id2idx: Dict[str, int]) -> torch.Tensor:
    edges: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Skip the two header lines: "DIRECTED\tcites" and "NO_FEATURES"
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        # Format: "<edge_id>\tpaper:<src>\t|\tpaper:<dst>"
        src = dst = None
        for p in parts:
            if p.startswith("paper:"):
                pid = p.split(":", 1)[1].strip()
                if src is None:
                    src = pid
                else:
                    dst = pid
                    break
        if src is None or dst is None:
            continue
        if src not in id2idx or dst not in id2idx:
            continue
        u, v = id2idx[src], id2idx[dst]
        edges.append((u, v))
        edges.append((v, u))   # undirected for GNN

    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"  parsed {edge_index.shape[1]} edges (undirected)")
    return edge_index


# ── 3. Parse pubmed.json for title + abstract ─────────────────────


def parse_pubmed_json(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    out: Dict[str, Dict[str, str]] = {}
    for r in records:
        pmid = str(r.get("PMID", "")).strip()
        if not pmid:
            continue
        title = str(r.get("TI", "")).strip()
        abstract = str(r.get("AB", "")).strip()
        out[pmid] = {"title": title, "abstract": abstract}
    print(f"  metadata for {len(out)} PMIDs")
    return out


# ── 4. Assemble + persist ─────────────────────────────────────────


def _make_random_split(num_nodes: int, seed: int = 42, train_ratio: float = 0.6,
                       val_ratio: float = 0.2):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="Path to PubMed_orig directory")
    ap.add_argument("--dst", default="dataset/pubmed",
                    help="Destination dataset directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = _resolve_src(args.src)
    print(f"📦 PubMed_orig source: {src}")
    print(f"📦 Output target     : {args.dst}")
    os.makedirs(args.dst, exist_ok=True)

    # ── Parse raw files ──
    print("\n[1/4] Parsing NODE.paper.tab ...")
    node_path = os.path.join(src, "data", "Pubmed-Diabetes.NODE.paper.tab")
    paper_ids, X, y = parse_node_tab(node_path)
    id2idx = {pid: i for i, pid in enumerate(paper_ids)}

    print("\n[2/4] Parsing DIRECTED.cites.tab ...")
    cites_path = os.path.join(src, "data", "Pubmed-Diabetes.DIRECTED.cites.tab")
    edge_index = parse_cites_tab(cites_path, id2idx)

    print("\n[3/4] Parsing pubmed.json ...")
    json_path = os.path.join(src, "pubmed.json")
    meta = parse_pubmed_json(json_path)

    # ── Build node_info.csv ──
    print("\n[4/4] Writing artefacts ...")
    info_rows = []
    matched = 0
    for pid in paper_ids:
        m = meta.get(pid, {"title": "", "abstract": ""})
        if m["title"] or m["abstract"]:
            matched += 1
        info_rows.append({
            "paper_id": pid,
            "title": m["title"],
            "abstract": m["abstract"],
        })
    pd.DataFrame(info_rows).to_csv(
        os.path.join(args.dst, "node_info.csv"), index=False
    )
    print(f"  node_info.csv: {len(info_rows)} rows  ({matched} with metadata)")

    # ── Build processed_data.pt ──
    train_mask, val_mask, test_mask = _make_random_split(len(paper_ids), args.seed)
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=len(paper_ids),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.raw_texts = paper_ids
    out_pt = os.path.join(args.dst, "processed_data.pt")
    torch.save(data, out_pt)
    print(f"  processed_data.pt → {out_pt}")
    print(f"  nodes: {data.num_nodes}, edges: {edge_index.shape[1]}, classes: {len(set(y.tolist()))}")
    print(f"  train/val/test = {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
    print("\n✅ Done. You can now run the project with DATASET_NAME='pubmed'.")


if __name__ == "__main__":
    main()
