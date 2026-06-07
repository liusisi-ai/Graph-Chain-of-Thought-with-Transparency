"""训练完成后的诊断脚本。

功能
----
1. 选一个节点（默认 train_mask 中第一个有标签的；可用 ``--node-idx`` 指定），
   打印它的原始文本（``data.raw_texts``）以及它在每一轮 thought 中 LLM
   生成的所有响应，再附上最终分类结果（如有）。

2. 按论文图中的公式计算每一轮 thought 生成的邻接矩阵 A 的 homophily：

       α_v = |{u ∈ N_v | ℓ(u) = ℓ(v)}| / |N_v|
       Homophily Ratio = (1/N) * Σ_v α_v

   其中 ℓ(v) 取 ``data.y`` 中的 ground-truth class label（不是 K-means 聚类标签）。
   N 为「至少有一个邻居且有有效标签」的节点数。
   原始 ``data.edge_index`` 也会一并计算作为 baseline 对照。

工件查找路径（与 ``cluster_prompt.py`` 中保存的位置一致）
    dataset/<name>/cluster/thought_<t>_edge_index.pt        # 第 t 轮后的 A
    dataset/<name>/cluster/thought_<t>_node_results.csv     # 第 t 轮逐节点 LLM 输出
    dataset/<name>/cluster/<name>_cluster_llm_results.csv   # 簇中心 LLM 输出
    dataset/<name>/cluster/cluster_labels.pt                # K-means 聚类 id
    dataset/<name>/cluster/<name>_classification_results.csv # 最终分类结果

CLI
----
    python -m scripts.analyze_thoughts
    python -m scripts.analyze_thoughts --node-idx 123
    python -m scripts.analyze_thoughts --dataset cora --output reports/cora_demo.txt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Allow ``python scripts/analyze_thoughts.py`` from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from config import (  # noqa: E402
    DATASET_NAME,
    MAX_SOURCE_NODES,
    SOURCE_DOMAIN,
    SUBGRAPH_SEED,
)
from dataloader import load_gnn_dataset  # noqa: E402


# ── Homophily ───────────────────────────────────────────────


def compute_homophily(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
) -> Dict[str, float]:
    """根据公式 α_v = |{u∈N_v|ℓ(u)=ℓ(v)}| / |N_v| 计算图的 homophily ratio。

    - 邻居集 N_v 视为无向图的去重集合（去掉自环）。
    - 仅统计「至少有 1 个邻居且自身标签有效」的节点。
    """
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    y_np = y.cpu().numpy().astype(np.int64)

    neighbors: Dict[int, set] = {}
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            neighbors.setdefault(u, set()).add(v)
            neighbors.setdefault(v, set()).add(u)

    alphas: List[float] = []
    isolated = 0
    for node in range(num_nodes):
        nbrs = neighbors.get(node)
        if not nbrs:
            isolated += 1
            continue
        my_label = int(y_np[node])
        if my_label < 0:  # 缺失标签 → 跳过
            continue
        same = sum(1 for n in nbrs if int(y_np[n]) == my_label)
        alphas.append(same / len(nbrs))

    homo = float(np.mean(alphas)) if alphas else float("nan")
    return {
        "homophily_ratio": homo,
        "num_nodes_counted": len(alphas),
        "num_isolated": isolated,
        "num_total_nodes": num_nodes,
        "num_directed_edges": int(edge_index.shape[1]),
        "num_undirected_edges": sum(len(s) for s in neighbors.values()) // 2,
    }


# ── Locate per-thought artefacts ────────────────────────────


def _list_thought_edge_files(cluster_dir: str) -> List[Tuple[int, str]]:
    """返回按轮次升序排列的 (t, path) 列表。"""
    pattern = os.path.join(cluster_dir, "thought_*_edge_index.pt")
    out: List[Tuple[int, str]] = []
    for p in glob.glob(pattern):
        m = re.search(r"thought_(\d+)_edge_index\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _list_thought_node_results(cluster_dir: str) -> Dict[int, str]:
    """返回 t -> CSV 路径。"""
    pattern = os.path.join(cluster_dir, "thought_*_node_results.csv")
    out: Dict[int, str] = {}
    for p in glob.glob(pattern):
        m = re.search(r"thought_(\d+)_node_results\.csv$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


# ── Pick a demo node ────────────────────────────────────────


def pick_demo_node(data, override: Optional[int] = None) -> int:
    """选一个用于展示的节点。

    优先级:
      1. ``override`` (CLI ``--node-idx``)
      2. ``train_mask`` 中第一个有标签的节点
      3. 节点 0
    """
    n = int(data.num_nodes)
    if override is not None:
        if not (0 <= override < n):
            raise ValueError(f"--node-idx={override} 越界 (0, {n - 1})")
        return int(override)

    if hasattr(data, "train_mask") and data.train_mask is not None:
        idx_list = torch.nonzero(data.train_mask, as_tuple=False).flatten().tolist()
        for idx in idx_list:
            if data.y is not None and int(data.y[idx].item()) >= 0:
                return int(idx)
    return 0


# ── Pretty-print one node's full trajectory ─────────────────


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:  # pragma: no cover
        print(f"⚠️ 读取 {path} 失败: {e}")
        return None


def collect_node_story(
    node_idx: int,
    data,
    cluster_dir: str,
    label_names: Optional[List[str]] = None,
) -> List[str]:
    """聚合一个节点在整个 pipeline 中的所有文本/输出，返回逐行字符串列表。"""
    lines: List[str] = []
    raw_texts = getattr(data, "raw_texts", None)
    node_id = raw_texts[node_idx] if raw_texts is not None else str(node_idx)

    true_label_str = "<unknown>"
    if data.y is not None:
        try:
            li = int(data.y[node_idx].item())
            true_label_str = (
                label_names[li] if (label_names and 0 <= li < len(label_names)) else str(li)
            )
        except Exception:
            pass

    cluster_id: Optional[int] = None
    cl_path = os.path.join(cluster_dir, "cluster_labels.pt")
    if os.path.exists(cl_path):
        try:
            cl = torch.load(cl_path, weights_only=False)
            if hasattr(cl, "numpy"):
                cl = cl.numpy()
            if 0 <= node_idx < len(cl):
                cluster_id = int(cl[node_idx])
        except Exception:
            pass

    lines.append("=" * 78)
    lines.append(f"🔎 Demo node trajectory  (node_idx = {node_idx})")
    lines.append("=" * 78)
    lines.append(f"  node_id      : {node_id}")
    lines.append(f"  true_label   : {true_label_str}")
    if cluster_id is not None:
        lines.append(f"  cluster_id   : {cluster_id}  (K-means 簇)")
    lines.append("")

    lines.append("── 1) 原始文本 (data.raw_texts) ──")
    if raw_texts is None:
        lines.append("  (该数据集未提供 raw_texts)")
    else:
        lines.append(f"  {raw_texts[node_idx]}")
    lines.append("")

    cluster_csv = None
    for fn in os.listdir(cluster_dir) if os.path.isdir(cluster_dir) else []:
        if fn.endswith("_cluster_llm_results.csv"):
            cluster_csv = os.path.join(cluster_dir, fn)
            break
    if cluster_csv and cluster_id is not None:
        df = _safe_read_csv(cluster_csv)
        if df is not None and "cluster_id" in df.columns:
            row = df[df["cluster_id"].astype(int) == int(cluster_id)]
            if len(row):
                lines.append(f"── 2) 所在簇 ({cluster_id}) 的中心节点 LLM 响应 ──")
                lines.append(f"  central_node_id : {row.iloc[0].get('central_node_id', '?')}")
                lines.append(f"  llm_response    : {row.iloc[0].get('llm_response', '')}")
                lines.append("")

    th_csvs = _list_thought_node_results(cluster_dir)
    if not th_csvs:
        lines.append("⚠️ 未找到 thought_*_node_results.csv，请先运行 stage2 thought-loop。")
    for t in sorted(th_csvs):
        df = _safe_read_csv(th_csvs[t])
        if df is None:
            continue
        sub = df[df["node_idx"].astype(int) == int(node_idx)] if "node_idx" in df.columns else df.iloc[0:0]
        lines.append(f"── 3.{t}) Thought {t} ──")
        if not len(sub):
            lines.append("  (该节点在本轮没有记录, 可能是孤立节点 / 没有候选邻居)")
            lines.append("")
            continue
        r = sub.iloc[0]
        lines.append(f"  cluster_id          : {r.get('cluster_id', '?')}")
        lines.append(f"  selected_neighbors  : {r.get('selected_neighbors', '?')}")
        lines.append(f"  llm_response        :")
        for ln in str(r.get("llm_response", "")).splitlines() or [""]:
            lines.append(f"    {ln}")
        lines.append("")

    cls_csv = None
    for fn in os.listdir(cluster_dir) if os.path.isdir(cluster_dir) else []:
        if fn.endswith("_classification_results.csv"):
            cls_csv = os.path.join(cluster_dir, fn)
            break
    if cls_csv:
        df = _safe_read_csv(cls_csv)
        if df is not None and "node_idx" in df.columns:
            row = df[df["node_idx"].astype(int) == int(node_idx)]
            if len(row):
                r = row.iloc[0]
                lines.append("── 4) 最终 LLM 分类结果 ──")
                lines.append(f"  predicted    : {r.get('predicted', '?')}")
                lines.append(f"  true_label   : {r.get('true_label', '?')}")
                lines.append(f"  llm_response : {r.get('llm_response', '')}")
                lines.append("")

    return lines


# ── Top-level analysis ──────────────────────────────────────


def analyze(
    dataset_name: str,
    node_idx_override: Optional[int] = None,
    output_path: Optional[str] = None,
    label_names: Optional[List[str]] = None,
) -> str:
    """主入口：返回一份格式化好的报告字符串，并按需写入磁盘。"""
    data = load_gnn_dataset(
        dataset_name,
        task="nc",
        max_nodes=MAX_SOURCE_NODES,
        seed=SUBGRAPH_SEED,
    )
    if data.y is None:
        raise RuntimeError(
            f"数据集 '{dataset_name}' 缺少 ground-truth 标签 y, "
            f"无法按公式计算 homophily ratio。"
        )

    cluster_dir = f"dataset/{dataset_name}/cluster"
    if not os.path.isdir(cluster_dir):
        raise RuntimeError(
            f"未找到簇/thought 工件目录: {cluster_dir}\n"
            f"请先运行完整 pipeline (python main.py) 让 stage2 thought-loop 落盘工件。"
        )

    blocks: List[str] = []
    blocks.append("=" * 78)
    blocks.append(f"📊 Homophily 分析  (dataset = {dataset_name})")
    blocks.append("=" * 78)
    blocks.append(
        "  α_v = |{u ∈ N(v) : ℓ(u)=ℓ(v)}| / |N(v)|;  "
        "Homophily = mean_v α_v"
    )
    blocks.append(f"  ℓ(·) 来自 data.y (共 {int(data.y.max().item()) + 1} 类), "
                  f"N(v) 取无向 1-hop 邻居 (去重去自环)")
    blocks.append("")

    rows: List[Dict] = []

    orig_metric = compute_homophily(data.edge_index, data.y, data.num_nodes)
    rows.append({"name": "Original A (input graph)", **orig_metric})

    for t, path in _list_thought_edge_files(cluster_dir):
        try:
            ei = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"⚠️ 加载 {path} 失败: {e}")
            continue
        m = compute_homophily(ei, data.y, data.num_nodes)
        rows.append({"name": f"Thought {t}  (after step-3 selection)", **m})

    header = (
        f"  {'name':<40s}  {'homophily':>10s}  "
        f"{'#nodes':>8s}  {'#iso':>6s}  {'#edges':>10s}"
    )
    blocks.append(header)
    blocks.append("  " + "-" * (len(header) - 2))
    for r in rows:
        blocks.append(
            f"  {r['name']:<40s}  {r['homophily_ratio']:>10.4f}  "
            f"{r['num_nodes_counted']:>8d}  {r['num_isolated']:>6d}  "
            f"{r['num_undirected_edges']:>10d}"
        )
    blocks.append("")

    target_node = pick_demo_node(data, node_idx_override)
    blocks.extend(collect_node_story(target_node, data, cluster_dir, label_names))

    report = "\n".join(blocks)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n📝 报告已写入: {output_path}")

    return report


_SENTINEL_DEFAULT_OUTPUT = "__AUTO__"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, default=SOURCE_DOMAIN or DATASET_NAME,
                    help="数据集名 (默认取 config.SOURCE_DOMAIN)")
    ap.add_argument("--node-idx", type=int, default=None,
                    help="要展示的节点 index, 默认 train_mask 中的第一个")
    ap.add_argument("--output", type=str, default=_SENTINEL_DEFAULT_OUTPUT,
                    help="把报告写入此文件 (默认: <dataset>_checkpoints/thought_analysis.txt; "
                         "传空串 '' 可关闭文件输出)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from cluster_prompt import DATASET_LABEL_MAP
        label_names = DATASET_LABEL_MAP.get(args.dataset)
    except Exception:
        label_names = None

    # 决定最终落盘路径:
    #   未传 --output         → 默认 <dataset>_checkpoints/thought_analysis.txt
    #   --output ""           → 只打印, 不写文件
    #   --output some/path.txt → 写到指定路径
    if args.output == _SENTINEL_DEFAULT_OUTPUT:
        output_path: Optional[str] = f"{args.dataset}_checkpoints/thought_analysis.txt"
    elif args.output == "":
        output_path = None
    else:
        output_path = args.output

    analyze(
        dataset_name=args.dataset,
        node_idx_override=args.node_idx,
        output_path=output_path,
        label_names=label_names,
    )


if __name__ == "__main__":
    main()
