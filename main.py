"""End-to-end pipeline that matches the specified algorithmic framework.

Stages
------
1. SAGE contrastive self-supervised pre-training  →  ``first_model.pt``
2. Downstream CoT inference on the source domain:
     (1) X → GNN → H ; Kmeans(H) ; H_token = Linear(H)
     (2) K cluster-center prompts → LLM
     (3) Per-node neighbor-selection thoughts (re-runs step(2) each iteration)
     (4) Final LLM classification using <Token_1 … Token_K> + last thought
3. (Optional) PCA-anchor alignment training for zero-shot generalisation
4. Zero-shot inference on new domains via instruction-file swapping

CLI
---
    python main.py                       # use defaults (seed=42, eval on test)
    python main.py --seed 7              # change seed (for multi-seed sweep)
    python main.py --eval-on val         # evaluate on val_mask (use when tuning)
    python main.py --eval-on both        # report val and test
    python main.py --tag baseline        # add a tag to the JSONL log row
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from seed_utils import seed_everything   # MUST be before config.py imports

from config import (
    DATASET_NAME,
    EVAL_ON,
    FIRST_MODEL_PATH,
    GLOBAL_SEED,
    INSTRUCTION_DIR,
    LLM_HIDDEN_DIM,
    MAX_CANDIDATES,
    MAX_NEIGHBORS_PER_SEQ,
    MAX_SOURCE_NODES,
    NUM_CLUSTER_THOUGHTS,
    NUM_CLUSTERS,
    NUM_PCA_ANCHORS,
    NUM_REPR_TOKENS,
    PCA_ALIGN_BATCH_SIZE,
    PCA_ALIGN_EPOCHS,
    PCA_ALIGN_LR,
    PRETRAIN_DATASET,
    PRETRAIN_MAX_NODES,
    SAGE_PRETRAIN_EPOCHS,
    SAGE_PRETRAIN_LR,
    SELECT_K,
    SOURCE_DOMAIN,
    SUBGRAPH_SEED,
    TARGET_DOMAINS,
    TRAIN_MODE,
    USE_ABSTRACT,
    USE_CONSTRAINED_DECODING,
    USE_SOFT_TOKEN,
    WARMUP_EPOCHS,
    canonical_dataset_name,
    resolve_pretrain_dataset,
)
from dataloader import load_gnn_dataset
from gcn import pca_compression
from graphsage import PrePromptSAGE, pretrain_sage
from cluster_prompt import (
    cluster_and_generate_prompts,
    llm_classify_nodes,
    load_answer_candidates,
    load_node_abstracts,
    load_node_titles,
    run_thought_loop,
    send_cluster_prompts_to_llm,
)
from cot_trainer import (
    CoTGraphLLM,
    load_instruction_template,
    train_cot,
    train_pca_alignment,
    zero_shot_eval_cross_domain,
    zero_shot_eval_with_template,
)


# ── Stage 1 ─────────────────────────────────────────────────


def stage1_pretrain_sage(data, n_in, n_h, num_layers_num, dropout, save_path):
    """Contrastive SAGE pretrain → first_model."""
    print("\n" + "=" * 60)
    print("--- Stage 1: GraphSAGE contrastive pre-training ---")
    print("=" * 60)

    if os.path.exists(save_path):
        print(f"✅ first_model already exists at {save_path}, skip pre-training.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PrePromptSAGE(
            n_in=n_in, n_h=n_h, num_layers_num=num_layers_num,
            dropout=dropout, projector_out_dim=LLM_HIDDEN_DIM,
        ).to(device)
        state = torch.load(save_path, map_location=device)
        model.load_state_dict(state, strict=False)
        return model

    return pretrain_sage(
        data=data, n_in=n_in, n_h=n_h, num_layers_num=num_layers_num,
        dropout=dropout, negative_sample_num=2,
        epochs=SAGE_PRETRAIN_EPOCHS, lr=SAGE_PRETRAIN_LR,
        projector_out_dim=LLM_HIDDEN_DIM, save_path=save_path,
    )


# ── Stage 2: downstream CoT pipeline (steps 1–4 of the framework) ──


def stage2_cot_pipeline(sage_model, data, dataset_name):
    """Run the complete CoT downstream-task pipeline on one domain."""
    print("\n" + "=" * 60)
    print("--- Stage 2: Downstream CoT pipeline ---")
    print("=" * 60)

    # cluster_and_generate_prompts uses .gcn attribute
    sage_model.gcn = sage_model.sage

    # Step (1)+(2) initial: K-means + K cluster-center prompts → LLM
    prompts = cluster_and_generate_prompts(
        sage_model, data, K=NUM_CLUSTERS,
        dataset_name=dataset_name, max_neighbors=MAX_NEIGHBORS_PER_SEQ,
    )
    cluster_results = send_cluster_prompts_to_llm(prompts, dataset_name=dataset_name)

    # Step (3) multi-thought: per-node selection + step(2) re-run each iteration
    optimized_edge_index, thought_results = run_thought_loop(
        sage_model, data, K=NUM_CLUSTERS,
        num_thoughts=NUM_CLUSTER_THOUGHTS,
        cluster_results=cluster_results,
        cluster_prompts=prompts,
        dataset_name=dataset_name,
        max_candidates=MAX_CANDIDATES,
        select_k=SELECT_K,
        max_neighbors_per_seq=MAX_NEIGHBORS_PER_SEQ,
    )
    print(f"✅ optimised edges: {optimized_edge_index.shape[1]}")

    # Step (4) final classification:  <Token_1 … Token_K> + last-thought text
    cls_results = llm_classify_nodes(
        sage_model, data, thought_results,
        optimized_edge_index, NUM_CLUSTER_THOUGHTS,
        dataset_name=dataset_name, num_tokens=NUM_REPR_TOKENS,
    )

    cluster_labels_path = f"dataset/{dataset_name}/cluster/cluster_labels.pt"
    cluster_labels = torch.load(cluster_labels_path).numpy()
    cluster_llm_text = {int(r["cluster_id"]): r["llm_response"] for r in cluster_results}

    return {
        "optimized_edge_index": optimized_edge_index,
        "thought_results": thought_results,
        "cls_results": cls_results,
        "cluster_labels": cluster_labels,
        "cluster_llm_text": cluster_llm_text,
    }


# ── main ────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED,
                    help="Global random seed (also accepted via CFG_SEED env var).")
    ap.add_argument("--eval-on", type=str, default=EVAL_ON,
                    choices=["val", "test", "both"],
                    help="Which mask to evaluate the LLM on.")
    ap.add_argument("--tag", type=str, default="run",
                    help="Free-form label written into tune_runs.jsonl.")
    ap.add_argument("--log-path", type=str, default="tune_runs.jsonl",
                    help="JSONL file appended with one row per run.")
    return ap.parse_args()


def _append_jsonl(path: str, row: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"⚠️ Could not write log row to {path}: {e}")


def _hyperparam_snapshot() -> dict:
    """Capture the active hyper-parameter values for the run log."""
    return {
        "DATASET_NAME":         DATASET_NAME,
        "SOURCE_DOMAIN":        SOURCE_DOMAIN,
        "PRETRAIN_DATASET":     PRETRAIN_DATASET,
        "MAX_SOURCE_NODES":     MAX_SOURCE_NODES,
        "NUM_CLUSTERS":         NUM_CLUSTERS,
        "NUM_CLUSTER_THOUGHTS": NUM_CLUSTER_THOUGHTS,
        "NUM_REPR_TOKENS":      NUM_REPR_TOKENS,
        "MAX_NEIGHBORS_PER_SEQ": MAX_NEIGHBORS_PER_SEQ,
        "MAX_CANDIDATES":       MAX_CANDIDATES,
        "SELECT_K":             SELECT_K,
        "WARMUP_EPOCHS":        WARMUP_EPOCHS,
        "SAGE_PRETRAIN_EPOCHS": SAGE_PRETRAIN_EPOCHS,
        "SAGE_PRETRAIN_LR":     SAGE_PRETRAIN_LR,
        "PCA_ALIGN_EPOCHS":     PCA_ALIGN_EPOCHS,
        "PCA_ALIGN_LR":         PCA_ALIGN_LR,
        "PCA_ALIGN_BATCH_SIZE": PCA_ALIGN_BATCH_SIZE,
        "NUM_PCA_ANCHORS":      NUM_PCA_ANCHORS,
        "TRAIN_MODE":           TRAIN_MODE,
        "USE_SOFT_TOKEN":       USE_SOFT_TOKEN,
        "USE_ABSTRACT":         USE_ABSTRACT,
        "USE_CONSTRAINED_DECODING": USE_CONSTRAINED_DECODING,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # =====================================================================
    # Downstream (Stage 2-4) data — Cora by default
    # =====================================================================
    source = canonical_dataset_name(SOURCE_DOMAIN or DATASET_NAME)
    print(f"\n📦 Downstream source domain: {source}")
    data = load_gnn_dataset(
        source,
        task="nc",
        max_nodes=MAX_SOURCE_NODES,
        seed=SUBGRAPH_SEED,
    )
    print(
        f"📊 Active graph: nodes={data.num_nodes}, "
        f"edges={data.edge_index.shape[1]}, max_source_nodes={MAX_SOURCE_NODES}"
    )
    n_in, n_h, num_layers_num, dropout = 128, 128, 2, 0.0
    data.x = torch.FloatTensor(pca_compression(data.x, k=n_in))

    # =====================================================================
    # Stage 1 — pre-train GraphSAGE with contrastive learning
    # auto: Citation Domain -> ogbn-arxiv, E-Commerce Domain -> Computers.
    # =====================================================================
    pretrain_source = resolve_pretrain_dataset(source)
    print(f"🧭 Pretrain switch: CFG_PRETRAIN_DATASET={PRETRAIN_DATASET!r} -> {pretrain_source}")
    if pretrain_source != source:
        print(f"\n📦 Pretrain source domain: {pretrain_source}  "
              f"(decoupled from downstream {source})")
        try:
            pretrain_data = load_gnn_dataset(
                pretrain_source,
                task="nc",
                max_nodes=PRETRAIN_MAX_NODES,
                seed=SUBGRAPH_SEED,
            )
            print(
                f"📊 Pretrain graph: nodes={pretrain_data.num_nodes}, "
                f"edges={pretrain_data.edge_index.shape[1]}, "
                f"max_nodes={PRETRAIN_MAX_NODES}"
            )
            # PCA 投影到与下游一致的 n_in 维, 保证预训练权重可在 cora 上复用。
            pretrain_data.x = torch.FloatTensor(pca_compression(pretrain_data.x, k=n_in))
        except FileNotFoundError as e:
            print(f"⚠️ Pretrain dataset '{pretrain_source}' 不可用 ({e}); "
                  f"回退到下游数据集 '{source}' 做预训练。")
            pretrain_source = source
            pretrain_data = data
    else:
        pretrain_data = data

    os.makedirs(f"{pretrain_source}_checkpoints", exist_ok=True)
    # 下游 source 的 checkpoint 目录也要建好, 用于落盘分析报告等。
    os.makedirs(f"{source}_checkpoints", exist_ok=True)
    first_model_path = f"{pretrain_source}_checkpoints/first_model.pt"

    sage_model = stage1_pretrain_sage(
        pretrain_data, n_in, n_h, num_layers_num, dropout, first_model_path
    )

    # 预训练数据可能很大（~170k 节点的 arxiv），用完立刻释放显存。
    if pretrain_source != source:
        del pretrain_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================================
    # Stage 2 — framework steps (1)-(4): cluster + thoughts + classification
    # 注意: stage2-4 全部使用下游 source (cora) 的 data, 但 sage_model 的
    # 权重来自 pretrain_source (arxiv) 的对比预训练。
    # =====================================================================
    cot_artefacts = stage2_cot_pipeline(sage_model, data, source)
    optimized_edge_index = cot_artefacts["optimized_edge_index"]
    cluster_labels      = cot_artefacts["cluster_labels"]
    cluster_llm_text    = cot_artefacts["cluster_llm_text"]
    data.edge_index     = optimized_edge_index   # feed back into downstream

    # =====================================================================
    # Stage 3 — zero-shot training (PCA-anchor alignment; label-free)
    # =====================================================================
    print("\n" + "=" * 60)
    print("--- Stage 3: Zero-shot training ---")
    print("=" * 60)
    cot_model = CoTGraphLLM(
        n_in=n_in, n_h=n_h, num_layers_num=num_layers_num, dropout=dropout,
        projector_out_dim=LLM_HIDDEN_DIM,
        first_model_path=first_model_path,
    ).to(device)

    title_dict = load_node_titles(source, data)
    try:
        abstract_dict = load_node_abstracts(source) or None
    except Exception as e:
        print(f"⚠️ load_node_abstracts failed ({e}); abstracts disabled in prompts")
        abstract_dict = None
    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None

    if TRAIN_MODE in ("pca_align", "both"):
        train_pca_alignment(
            model=cot_model, data=data,
            title_dict=title_dict, raw_texts=raw_texts,
            num_anchors=NUM_PCA_ANCHORS,
            epochs=PCA_ALIGN_EPOCHS,
            batch_size=PCA_ALIGN_BATCH_SIZE,
            lr=PCA_ALIGN_LR,
        )

    if TRAIN_MODE in ("sft", "both"):
        candidates_src = load_answer_candidates(source, data)
        train_mask = getattr(data, "train_mask", None)
        num_train_epochs = WARMUP_EPOCHS + NUM_CLUSTER_THOUGHTS + 5
        train_cot(
            model=cot_model, data=data,
            candidates=candidates_src, title_dict=title_dict,
            cluster_llm_text=cluster_llm_text,
            num_epochs=num_train_epochs,
            num_thoughts=NUM_CLUSTER_THOUGHTS,
            batch_size=2, lr=1e-4,
            train_mask=train_mask,
            cluster_labels=cluster_labels,
            dataset_name=source,
        )

    # =====================================================================
    # Stage 4 — Zero-shot inference: source sanity check + target domains
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"--- Stage 4: Zero-shot inference  (eval-on={args.eval_on}) ---")
    print("=" * 60)
    all_results = []

    candidates_src = load_answer_candidates(source, data)
    template_src = load_instruction_template(source, INSTRUCTION_DIR)

    val_mask  = getattr(data, "val_mask", None)
    test_mask = getattr(data, "test_mask", None)

    masks_to_run = []
    if args.eval_on in ("val", "both") and val_mask is not None:
        masks_to_run.append(("val", val_mask))
    if args.eval_on in ("test", "both") and test_mask is not None:
        masks_to_run.append(("test", test_mask))
    # Fallback if neither mask exists: evaluate on every node
    if not masks_to_run:
        masks_to_run.append(("all", None))

    val_acc = test_acc = None
    for split, mask in masks_to_run:
        res = zero_shot_eval_with_template(
            model=cot_model, data=data,
            candidates=candidates_src, title_dict=title_dict,
            template=template_src,
            dataset_name=f"{source}-{split}",
            edge_index=optimized_edge_index, test_mask=mask,
            abstract_dict=abstract_dict,
        )
        res["split"] = split
        all_results.append(res)
        if split == "val":
            val_acc = res["accuracy"]
        elif split == "test":
            test_acc = res["accuracy"]

    for tgt in TARGET_DOMAINS:
        if tgt == source:
            continue
        try:
            res = zero_shot_eval_cross_domain(
                model=cot_model,
                target_dataset=tgt,
                instruction_dir=INSTRUCTION_DIR,
                n_in=n_in,
            )
            res["split"] = "test"
            all_results.append(res)
        except Exception as e:
            print(f"⚠️ Zero-shot inference on '{tgt}' failed: {e}")

    print("\n=== Zero-shot summary ===")
    for r in all_results:
        print(
            f"  {r.get('dataset', source):>14s}  acc = {r['accuracy']:.4f}  "
            f"({r['correct']}/{r['total']})"
        )
    print("\n✅ Process Completed.")

    # ── Post-training analysis: per-thought homophily + demo node trajectory ──
    try:
        from cluster_prompt import DATASET_LABEL_MAP
        from scripts.analyze_thoughts import analyze as analyze_thoughts

        report_path = f"{source}_checkpoints/thought_analysis.txt"
        analyze_thoughts(
            dataset_name=source,
            node_idx_override=None,
            output_path=report_path,
            label_names=DATASET_LABEL_MAP.get(source),
        )
    except Exception as e:
        print(f"⚠️ Post-training analysis (homophily / demo node) failed: {e}")

    # ── Persist a structured run record for the tuner ──
    log_row = {
        "tag":           args.tag,
        "seed":          args.seed,
        "eval_on":       args.eval_on,
        "val_acc":       val_acc,
        "test_acc":      test_acc,
        "val_correct":   next((r["correct"] for r in all_results if r.get("split") == "val"), None),
        "val_total":     next((r["total"]   for r in all_results if r.get("split") == "val"), None),
        "test_correct":  next((r["correct"] for r in all_results if r.get("split") == "test"), None),
        "test_total":    next((r["total"]   for r in all_results if r.get("split") == "test"), None),
        "elapsed_sec":   round(time.time() - t_start, 1),
        "hparams":       _hyperparam_snapshot(),
    }
    _append_jsonl(args.log_path, log_row)
    print(f"📝 Run logged → {args.log_path}")


if __name__ == "__main__":
    main()
