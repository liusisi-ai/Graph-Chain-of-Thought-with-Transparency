"""Prepare local TAG datasets for this project.

The training pipeline reads ``dataset/<name>/processed_data.pt`` and optionally
``dataset/<name>/node_info.csv``.  The downloaded folders in this workspace are
raw sources:

* ``Pubmed/``: Planetoid PubMed pickle shards.
* ``Children/``, ``Computers/``, ``History/``, ``Photo/``, ``Sports/``: CSV
  files with text, labels and neighbor lists.

This script converts those raw formats into PyG ``Data`` objects with
``x``, ``edge_index``, ``y``, train/val/test masks and ``raw_texts``.

Examples
--------
    python scripts/prepare_tag_datasets.py --all
    python scripts/prepare_tag_datasets.py --datasets pubmed Children Photo
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ECOMMERCE_DATASETS = ("Children", "Computers", "History", "Photo", "Sports")
PUBMED_LABEL_NAMES = [
    "Diabetes Mellitus, Experimental",
    "Diabetes Mellitus Type 1",
    "Diabetes Mellitus Type 2",
]


def _canonical_dataset_name(name: str) -> str:
    lower = name.lower()
    if lower == "pubmed":
        return "pubmed"
    for candidate in ECOMMERCE_DATASETS:
        if lower == candidate.lower():
            return candidate
    raise ValueError(f"Unsupported dataset: {name}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_random_split(
    num_nodes: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _index_to_mask(indices: Iterable[int], size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    idx = torch.tensor(list(indices), dtype=torch.long)
    if idx.numel():
        mask[idx] = True
    return mask


def _pickle_load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _read_test_index(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def prepare_pubmed_planetoid(src_dir: str, dst_dir: str, seed: int) -> None:
    """Convert Planetoid PubMed shards into ``processed_data.pt``."""
    required = [
        "ind.pubmed.x",
        "ind.pubmed.tx",
        "ind.pubmed.allx",
        "ind.pubmed.y",
        "ind.pubmed.ty",
        "ind.pubmed.ally",
        "ind.pubmed.graph",
        "ind.pubmed.test.index",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(src_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing PubMed files in {src_dir}: {missing}")

    print(f"\n[pubmed] loading Planetoid shards from {src_dir}")
    x = _pickle_load(os.path.join(src_dir, "ind.pubmed.x"))
    tx = _pickle_load(os.path.join(src_dir, "ind.pubmed.tx"))
    allx = _pickle_load(os.path.join(src_dir, "ind.pubmed.allx"))
    y = _pickle_load(os.path.join(src_dir, "ind.pubmed.y"))
    ty = _pickle_load(os.path.join(src_dir, "ind.pubmed.ty"))
    ally = _pickle_load(os.path.join(src_dir, "ind.pubmed.ally"))
    graph = _pickle_load(os.path.join(src_dir, "ind.pubmed.graph"))
    test_idx_reorder = _read_test_index(os.path.join(src_dir, "ind.pubmed.test.index"))
    test_idx_range = sorted(test_idx_reorder)

    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - dependency exists in project env
        raise RuntimeError("scipy is required to prepare Planetoid PubMed") from exc

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    x_tensor = torch.from_numpy(features.toarray()).float()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    y_tensor = torch.from_numpy(labels.argmax(axis=1)).long()
    num_nodes = x_tensor.shape[0]

    edges = set()
    for src, neighbors in graph.items():
        src_i = int(src)
        if src_i >= num_nodes:
            continue
        for dst in neighbors:
            dst_i = int(dst)
            if dst_i >= num_nodes or dst_i == src_i:
                continue
            edges.add((src_i, dst_i))
            edges.add((dst_i, src_i))
    edge_index = (
        torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    train_mask = _index_to_mask(range(y.shape[0]), num_nodes)
    val_start = y.shape[0]
    val_stop = min(val_start + 500, num_nodes)
    val_mask = _index_to_mask(range(val_start, val_stop), num_nodes)
    test_mask = _index_to_mask(test_idx_range, num_nodes)
    if not train_mask.any() or not test_mask.any():
        train_mask, val_mask, test_mask = _make_random_split(num_nodes, seed, 0.6, 0.2)

    raw_texts = [str(i) for i in range(num_nodes)]
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, num_nodes=num_nodes)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.raw_texts = raw_texts
    data.label_names = PUBMED_LABEL_NAMES
    data.category_names = PUBMED_LABEL_NAMES

    _ensure_dir(dst_dir)
    torch.save(data, os.path.join(dst_dir, "processed_data.pt"))
    pd.DataFrame(
        {
            "paper_id": raw_texts,
            "title": [f"PubMed paper {i}" for i in range(num_nodes)],
            "abstract": ["" for _ in range(num_nodes)],
        }
    ).to_csv(os.path.join(dst_dir, "node_info.csv"), index=False)

    print(
        f"[pubmed] wrote {dst_dir}: nodes={num_nodes}, "
        f"edges={edge_index.shape[1]}, x_dim={x_tensor.shape[1]}, "
        f"splits={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}"
    )


def _parse_neighbors(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = text

    if isinstance(parsed, (list, tuple, set)):
        return [str(v) for v in parsed]
    if isinstance(parsed, (int, np.integer)):
        return [str(int(parsed))]
    if isinstance(parsed, str):
        stripped = parsed.strip().strip("[]")
        if not stripped:
            return []
        return [part.strip().strip("'\"") for part in stripped.split(",") if part.strip()]
    return []


def _label_names_from_csv(df: pd.DataFrame, labels: Sequence[int]) -> List[str]:
    max_label = max(labels) if labels else -1
    label_names = [f"class_{i}" for i in range(max_label + 1)]
    if "category" not in df.columns:
        return label_names

    for label, group in df.groupby("label"):
        try:
            label_id = int(label)
        except (TypeError, ValueError):
            continue
        categories = [
            str(v).strip()
            for v in group["category"].tolist()
            if str(v).strip() and str(v).lower() != "nan"
        ]
        if 0 <= label_id < len(label_names) and categories:
            label_names[label_id] = Counter(categories).most_common(1)[0][0]
    return label_names


def _split_title_abstract(text: str) -> Tuple[str, str]:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return "Unknown", ""

    title_match = re.search(r"(?:^|;\s*)Title:\s*(.+)$", raw, flags=re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip(" .;")
        abstract = raw[: title_match.start()].strip(" .;")
        abstract = re.sub(r"^Description:\s*", "", abstract, flags=re.IGNORECASE)
        return title[:220] or "Unknown", abstract

    sports_match = re.search(r"\btitle of .*? category is\s+(.+)$", raw, flags=re.IGNORECASE)
    if sports_match:
        title = sports_match.group(1).strip(" .;")
        return title[:220] or "Unknown", raw

    parts = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)
    title = parts[0].strip()
    if len(title) > 220:
        title = title[:220].rstrip() + "..."
    return title or "Unknown", raw


def _build_edges_from_csv(df: pd.DataFrame, node_ids: List[str]) -> torch.Tensor:
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edges = set()
    if "neighbour" not in df.columns:
        return torch.empty((2, 0), dtype=torch.long)

    for row_idx, value in enumerate(df["neighbour"].tolist()):
        for neighbor_id in _parse_neighbors(value):
            dst_idx = id_to_idx.get(str(neighbor_id))
            if dst_idx is None or dst_idx == row_idx:
                continue
            edges.add((row_idx, dst_idx))
            edges.add((dst_idx, row_idx))

    return (
        torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )


def prepare_ecommerce_csv(
    dataset_name: str,
    src_csv: str,
    dst_dir: str,
    seed: int,
    max_features: int,
    min_df: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Convert one e-commerce CSV into ``processed_data.pt``."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"\n[{dataset_name}] loading CSV from {src_csv}")
    df = pd.read_csv(src_csv, low_memory=False)
    required = {"text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{src_csv} is missing required columns: {missing}")

    if "node_id" in df.columns:
        node_ids = [str(v) for v in df["node_id"].tolist()]
    else:
        node_ids = [str(i) for i in range(len(df))]

    texts = df["text"].fillna("").astype(str).tolist()
    labels = [int(v) for v in df["label"].tolist()]
    label_names = _label_names_from_csv(df, labels)
    edge_index = _build_edges_from_csv(df, node_ids)

    print(f"[{dataset_name}] vectorizing {len(texts)} texts with TF-IDF({max_features})")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        dtype=np.float32,
        strip_accents="unicode",
        lowercase=True,
    )
    x_sparse = vectorizer.fit_transform(texts)
    x_tensor = torch.from_numpy(x_sparse.toarray()).float()
    y_tensor = torch.tensor(labels, dtype=torch.long)
    train_mask, val_mask, test_mask = _make_random_split(
        len(df), seed, train_ratio, val_ratio
    )

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        num_nodes=len(df),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.raw_texts = node_ids
    data.label_names = label_names
    data.category_names = label_names

    info_rows = []
    for node_id, text in zip(node_ids, texts):
        title, abstract = _split_title_abstract(text)
        info_rows.append({"paper_id": node_id, "title": title, "abstract": abstract})

    _ensure_dir(dst_dir)
    torch.save(data, os.path.join(dst_dir, "processed_data.pt"))
    pd.DataFrame(info_rows).to_csv(os.path.join(dst_dir, "node_info.csv"), index=False)
    pd.DataFrame({"label": range(len(label_names)), "category": label_names}).to_csv(
        os.path.join(dst_dir, "label_names.csv"), index=False
    )

    print(
        f"[{dataset_name}] wrote {dst_dir}: nodes={len(df)}, "
        f"edges={edge_index.shape[1]}, x_dim={x_tensor.shape[1]}, "
        f"classes={len(label_names)}, "
        f"splits={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}"
    )


def prepare_one(args: argparse.Namespace, dataset_name: str) -> None:
    canonical = _canonical_dataset_name(dataset_name)
    if canonical == "pubmed":
        src_dir = args.pubmed_src
        dst_dir = os.path.join(args.dst_root, "pubmed")
        prepare_pubmed_planetoid(src_dir, dst_dir, args.seed)
        return

    src_csv = os.path.join(args.raw_root, canonical, f"{canonical}.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(src_csv)
    dst_dir = os.path.join(args.dst_root, canonical)
    prepare_ecommerce_csv(
        dataset_name=canonical,
        src_csv=src_csv,
        dst_dir=dst_dir,
        seed=args.seed,
        max_features=args.max_features,
        min_df=args.min_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Datasets to prepare, e.g. pubmed Children Photo.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Prepare pubmed plus all local e-commerce datasets.",
    )
    ap.add_argument("--raw-root", default=".", help="Root containing raw CSV folders.")
    ap.add_argument("--pubmed-src", default="Pubmed", help="Planetoid PubMed folder.")
    ap.add_argument("--dst-root", default="dataset", help="Output root directory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=768)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["pubmed", *ECOMMERCE_DATASETS] if args.all else args.datasets
    if not datasets:
        raise SystemExit("Pass --all or --datasets ...")
    for dataset_name in datasets:
        prepare_one(args, dataset_name)


if __name__ == "__main__":
    main()
