import torch
import numpy as np
import os
import random
import pandas as pd
from sklearn.cluster import KMeans
from typing import Any, List, Dict, Tuple, Optional
from config import DATASET_NAME, GLOBAL_SEED, ROOT_PATH, USE_ABSTRACT


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> Dict[int, set]:
    adj = {i: set() for i in range(num_nodes)}
    edge_np = edge_index.cpu().numpy()
    for idx in range(edge_np.shape[1]):
        u, v = int(edge_np[0, idx]), int(edge_np[1, idx])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def get_neighbor_sequence(node_idx: int, adj: Dict[int, set], max_neighbors: int = 5) -> List[int]:
    """
    1-hop neighbors first; if fewer than max_neighbors, supplement with 2-hop.
    """
    hop1 = sorted(adj[node_idx])
    if len(hop1) >= max_neighbors:
        return hop1[:max_neighbors]

    sequence = list(hop1)
    hop1_set = set(hop1)
    hop2 = set()
    for n in hop1:
        for nn in adj[n]:
            if nn != node_idx and nn not in hop1_set:
                hop2.add(nn)

    remaining = max_neighbors - len(sequence)
    sequence.extend(sorted(hop2)[:remaining])
    return sequence[:max_neighbors]


def format_node_token(
    node_idx: int,
    H: torch.Tensor,
    raw_texts,
    adj: Dict[int, set],
    abstract_dict: Optional[Dict[str, str]] = None,
    abstract_max_len: int = 200,
    H_token: Optional[torch.Tensor] = None,
) -> str:
    """Format ``[Node ID, Node representation, (Node Abstract), Node degree]``.

    The "Node Abstract" field is omitted when the global ``USE_ABSTRACT``
    switch is off (or no ``abstract_dict`` is supplied).  This keeps prompts
    short and dramatically speeds up LLM calls on cora-sized graphs.

    If ``H_token`` (= Linear(H), LLM-aligned projection) is given, the
    ``Node representation`` field is serialised from H_token instead of H.
    """
    node_id = raw_texts[node_idx] if raw_texts is not None else str(node_idx)
    h_src = H_token if H_token is not None else H
    h = h_src[node_idx].detach().cpu().numpy()
    repr_str = f"dim={len(h)}, mean={h.mean():.4f}, std={h.std():.4f}"
    degree = len(adj.get(node_idx, set()))

    parts = [
        f"Node ID: {node_id}",
        f"Node representation: ({repr_str})",
    ]
    if USE_ABSTRACT and abstract_dict:
        abstract = abstract_dict.get(str(node_id), "N/A")
        if len(abstract) > abstract_max_len:
            abstract = abstract[:abstract_max_len] + "..."
        parts.append(f"Node Abstract: {abstract}")
    parts.append(f"Node degree: {degree}")
    return "[" + ", ".join(parts) + "]"


def _resolve_node_info_path(dataset_name: str) -> Optional[str]:
    """Locate ``node_info.csv``.

    Datasets shipped under ``dataset/<name>/`` use that path; some (e.g. cora)
    only have a project-root ``<name>/`` folder.  AutoDL puts the same files
    under ``/root/autodl-tmp/<name>/``.
    """
    candidates = [
        f"dataset/{dataset_name}/node_info.csv",
        f"{dataset_name}/node_info.csv",
        f"/root/autodl-tmp/dataset/{dataset_name}/node_info.csv",
        f"/root/autodl-tmp/{dataset_name}/node_info.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_node_abstracts(dataset_name: str) -> Dict[str, str]:
    """Return ``paper_id -> abstract`` mapping.

    When the global ``USE_ABSTRACT`` switch is off we skip CSV I/O entirely
    and return an empty dict — every downstream consumer guards against
    missing entries with ``"N/A"``, so this is safe.
    """
    if not USE_ABSTRACT:
        print("ℹ️ USE_ABSTRACT=False — skipping abstract loading (titles only)")
        return {}

    info_path = _resolve_node_info_path(dataset_name)
    abstract_dict = {}
    if info_path is not None:
        df = pd.read_csv(info_path, dtype={"paper_id": str})
        for _, row in df.iterrows():
            pid = str(row["paper_id"])
            abstract = str(row.get("abstract", row.get("input_text", "N/A")))
            abstract_dict[pid] = abstract
    else:
        print(
            f"⚠️ node_info.csv not found for dataset '{dataset_name}' — "
            f"abstracts will fall back to 'N/A'"
        )
    return abstract_dict


# ── main pipeline ────────────────────────────────────────────

def cluster_and_generate_prompts(
    gcn_model,
    data,
    K: int,
    dataset_name: str = DATASET_NAME,
    max_neighbors: int = 5,
) -> List[dict]:
    """
    1. Encode features X with pre-trained GNN → H
    2. K-means on H → K clusters
    3. Per cluster: find the most-central node
    4. Get its neighbor sequence (1-hop first, 2-hop if < max_neighbors)
    5. Build K prompts ready for the LLM
    """
    device = next(gcn_model.gcn.parameters()).device
    x_dev = data.x.to(device)
    ei_dev = data.edge_index.to(device)

    # ── Step 1: GNN encode (and projected H_token = Linear(H)) ──
    print("\n" + "=" * 60)
    print("--- Step 1: Encoding features with pre-trained GNN ---")
    with torch.no_grad():
        if hasattr(gcn_model, "embed_with_projection"):
            H, H_token = gcn_model.embed_with_projection(x_dev, ei_dev)
        else:
            H = gcn_model.embed(x_dev, ei_dev)
            H_token = H
    print(f"✅ H shape: {tuple(H.shape)},  H_token shape: {tuple(H_token.shape)}")

    # ── Step 2: K-means on H (un-projected) ──
    print(f"\n--- Step 2: K-means clustering (K={K}) on H ---")
    H_np = H.cpu().numpy()
    kmeans = KMeans(n_clusters=K, random_state=GLOBAL_SEED, n_init=10)
    labels = kmeans.fit_predict(H_np)
    centroids = kmeans.cluster_centers_

    for k in range(K):
        print(f"  Cluster {k}: {(labels == k).sum()} nodes")

    # ── Step 3: central nodes ──
    print("\n--- Step 3: Finding central nodes ---")
    central_nodes: List[Tuple[int, int]] = []  # (cluster_id, node_idx)
    for k in range(K):
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) == 0:
            continue
        dists = np.linalg.norm(H_np[cluster_indices] - centroids[k], axis=1)
        closest_idx = int(cluster_indices[np.argmin(dists)])
        central_nodes.append((k, closest_idx))
    print(f"✅ Found {len(central_nodes)} central nodes")

    # ── Step 4: neighbor sequences ──
    print("\n--- Step 4: Getting neighbor sequences ---")
    adj = build_adjacency(data.edge_index, data.num_nodes)
    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None

    neighbor_sequences: List[List[int]] = []
    for k, node_idx in central_nodes:
        seq = get_neighbor_sequence(node_idx, adj, max_neighbors)
        neighbor_sequences.append(seq)
        rid = raw_texts[node_idx] if raw_texts else node_idx
        print(f"  Cluster {k} central node {rid}: {len(seq)} neighbors")

    # ── Step 5: build prompts ──
    print("\n--- Step 5: Building prompts ---")
    abstract_dict = load_node_abstracts(dataset_name)

    prompts: List[dict] = []
    for idx, ((k, node_idx), neighbors) in enumerate(
        zip(central_nodes, neighbor_sequences)
    ):
        # Tokens use H_token (= Linear(H), LLM-aligned) per the framework
        center_token = format_node_token(
            node_idx, H, raw_texts, adj, abstract_dict, H_token=H_token
        )

        node_id = raw_texts[node_idx] if raw_texts else str(node_idx)

        nb_parts = []
        for j, nb_idx in enumerate(neighbors):
            nb_token = format_node_token(
                nb_idx, H, raw_texts, adj, abstract_dict, H_token=H_token
            )
            nb_parts.append(f"Token_{j + 1}: {nb_token}")
        neighbor_seq_str = ", ".join(nb_parts)

        # Build the optional "Abstract: ..." sentence only when enabled.
        info_clause = ""
        if USE_ABSTRACT and abstract_dict:
            abstract = abstract_dict.get(str(node_id), "N/A")
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            info_clause = f"with the following information: Abstract: {abstract}. "

        prompt = (
            f"Given the central node {center_token} as <Token_{idx + 1}>, "
            f"{info_clause}"
            f"Please analyze the node's preferenced neighbors in a short paragraph "
            f"based on the <neighbor sequence>: {neighbor_seq_str}."
        )
        prompts.append(
            {
                "cluster_id": k,
                "central_node_idx": node_idx,
                "central_node_id": str(node_id),
                "prompt": prompt,
                "neighbor_indices": neighbors,
            }
        )
        print(f"  Prompt {idx + 1}/{len(central_nodes)} built for cluster {k}")

    # ── save artefacts ──
    save_dir = f"dataset/{dataset_name}/cluster"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(torch.tensor(labels), os.path.join(save_dir, "cluster_labels.pt"))
    torch.save(H, os.path.join(save_dir, "H_embeddings.pt"))
    torch.save(H_token, os.path.join(save_dir, "H_token_embeddings.pt"))

    prompt_df = pd.DataFrame(
        [
            {
                "cluster_id": p["cluster_id"],
                "central_node_id": p["central_node_id"],
                "prompt_text": p["prompt"],
            }
            for p in prompts
        ]
    )
    csv_path = os.path.join(save_dir, f"{dataset_name}_cluster_prompts.csv")
    prompt_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(prompts)} cluster prompts to: {csv_path}")

    return prompts


def send_cluster_prompts_to_llm(
    prompts: List[dict],
    dataset_name: str = DATASET_NAME,
) -> List[dict]:
    """Send each of the K cluster prompts to the LLM and collect responses."""
    from use_llm_API import LLM_Predictor

    print("\n" + "=" * 60)
    print(f"--- Sending {len(prompts)} cluster prompts to LLM ---")
    print("=" * 60)

    predictor = LLM_Predictor()
    results: List[dict] = []

    for i, p in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Processing cluster {p['cluster_id']} ...")
        response = predictor.predict(p["prompt"])
        results.append(
            {
                "cluster_id": p["cluster_id"],
                "central_node_id": p["central_node_id"],
                "prompt": p["prompt"],
                "llm_response": response,
            }
        )
        print(f"  ✅ Response received (length: {len(response)})")

    save_dir = f"dataset/{dataset_name}/cluster"
    os.makedirs(save_dir, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_path = os.path.join(save_dir, f"{dataset_name}_cluster_llm_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"\n✅ Saved LLM results to: {result_path}")

    return results


# ── Thought‑loop utilities ───────────────────────────────────


def _extract_title_from_text(text: str, max_len: int = 160) -> str:
    """Best-effort: pull a one-line ``title`` out of a free-form node text."""
    if not text:
        return "Unknown"
    s = str(text).strip().replace("\r", " ").replace("\n", " ")

    # Common patterns in TAG datasets: "Title: ... Abstract: ..."
    lower = s.lower()
    if lower.startswith("title:"):
        s = s[len("title:"):].lstrip()
        for sep in (" abstract:", " summary:", "."):
            i = s.lower().find(sep)
            if i > 0:
                s = s[:i]
                break

    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    return s or "Unknown"


def load_node_titles(dataset_name: str, data: Any = None) -> Dict[str, str]:
    """Load ``paper_id -> title`` mapping.

    Resolution order:
      1. ``node_info.csv`` (preferred, if available).
      2. ``data.raw_texts`` (fallback) — auto-extract the title portion.
      3. Empty dict (caller defaults to "Unknown").
    """
    info_path = _resolve_node_info_path(dataset_name)
    title_dict: Dict[str, str] = {}
    if info_path is not None:
        df = pd.read_csv(info_path, dtype={"paper_id": str})
        for _, row in df.iterrows():
            pid = str(row["paper_id"])
            title_dict[pid] = str(row.get("title", "Unknown"))
        return title_dict

    if data is not None and getattr(data, "raw_texts", None):
        raw = list(data.raw_texts)
        for i, txt in enumerate(raw):
            title_dict[str(txt)] = _extract_title_from_text(txt)
            title_dict[str(i)] = _extract_title_from_text(txt)
        print(
            f"ℹ️ node_info.csv not found for '{dataset_name}' — "
            f"derived {len(raw)} titles from data.raw_texts"
        )
        return title_dict

    print(
        f"⚠️ node_info.csv not found for dataset '{dataset_name}' and no "
        f"raw_texts available — titles default to 'Unknown'"
    )
    return title_dict


def get_candidates(
    node_idx: int, adj: Dict[int, set], max_candidates: int = 10
) -> List[int]:
    """1-hop + 2-hop neighbors; random sample down to *max_candidates* if needed."""
    hop1 = sorted(adj.get(node_idx, set()))
    hop1_set = set(hop1)
    hop2 = set()
    for n in hop1:
        for nn in adj.get(n, set()):
            if nn != node_idx and nn not in hop1_set:
                hop2.add(nn)
    all_cands = hop1 + sorted(hop2)
    if len(all_cands) > max_candidates:
        all_cands = random.sample(all_cands, max_candidates)
    return all_cands


def format_token_with_title(
    node_idx: int,
    raw_texts,
    title_dict: Dict[str, str],
    adj: Dict[int, set],
) -> str:
    """Compact token: <Token> + Title."""
    node_id = raw_texts[node_idx] if raw_texts is not None else str(node_idx)
    degree = len(adj.get(node_idx, set()))
    title = title_dict.get(str(node_id), "Unknown")
    return f"[Node ID: {node_id}, Degree: {degree}] Title: {title}"


def build_instruction_prefix(
    cluster_result: dict,
    cluster_prompt_info: dict,
    raw_texts,
    title_dict: Dict[str, str],
    adj: Dict[int, set],
) -> str:
    """
    Instruction Prefix =
      Prefered neighbors: <nb1>, …, <nbK>.
      Reasons: <cluster‑center LLM output>.
    """
    neighbor_indices = cluster_prompt_info.get("neighbor_indices", [])
    nb_strs = [
        format_token_with_title(nb, raw_texts, title_dict, adj)
        for nb in neighbor_indices
    ]
    neighbors_text = ", ".join(nb_strs) if nb_strs else "None"
    reasons = cluster_result.get("llm_response", "N/A")
    return f"Prefered neighbors: {neighbors_text}. Reasons: {reasons}."


def build_node_selection_prompt(
    instruction_prefix: str,
    node_idx: int,
    candidates: List[int],
    raw_texts,
    title_dict: Dict[str, str],
    adj: Dict[int, set],
    is_last_thought: bool,
) -> str:
    """
    [Instruction Prefix] [node i].
    please select 5 neighbors for the target node from the candidate set: …
    (last thought adds: Please analyze the neighbor's preferenced nodes …)
    """
    node_id = raw_texts[node_idx] if raw_texts is not None else str(node_idx)
    degree = len(adj.get(node_idx, set()))
    title = title_dict.get(str(node_id), "Unknown")
    target_str = f"Target node: [Node ID: {node_id}, Degree: {degree}], Title: {title}"

    cand_strs = [
        format_token_with_title(c, raw_texts, title_dict, adj)
        for c in candidates
    ]
    candidates_str = ", ".join(cand_strs)

    prompt = (
        f"{instruction_prefix} "
        f"{target_str}. "
        f"Please select 5 neighbors for the target node from the candidate set: "
        f"{candidates_str}."
    )
    if is_last_thought:
        prompt += (
            " Please analyze the neighbor's preferenced nodes "
            "in a short paragraph."
        )
    return prompt


def parse_selected_neighbors(
    response: str,
    candidates: List[int],
    raw_texts,
    select_k: int = 5,
) -> List[int]:
    """Match candidate node‑IDs that appear in the LLM response; pad randomly if needed."""
    matched: List[int] = []
    for c_idx in candidates:
        nid = str(raw_texts[c_idx]) if raw_texts is not None else str(c_idx)
        if nid in response and c_idx not in matched:
            matched.append(c_idx)
            if len(matched) >= select_k:
                return matched

    remaining = [c for c in candidates if c not in matched]
    random.shuffle(remaining)
    matched.extend(remaining[: select_k - len(matched)])
    return matched[:select_k]


def build_optimized_edge_index(
    selections: Dict[int, List[int]],
    num_nodes: int,
    original_edge_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a new *bidirectional* edge_index from per‑node selections.

    Nodes that were NOT processed keep their original edges.
    """
    edges: set = set()
    processed = set(selections.keys())

    for node_idx, neighbors in selections.items():
        for nb in neighbors:
            u, v = min(node_idx, nb), max(node_idx, nb)
            edges.add((u, v))

    if original_edge_index is not None:
        orig = original_edge_index.cpu().numpy()
        for i in range(orig.shape[1]):
            u, v = int(orig[0, i]), int(orig[1, i])
            if u not in processed and v not in processed:
                edges.add((min(u, v), max(u, v)))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_list = sorted(edges)
    row = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    col = [e[1] for e in edge_list] + [e[0] for e in edge_list]
    return torch.tensor([row, col], dtype=torch.long)


# ── Main thought loop ───────────────────────────────────────


def _find_central_nodes(H: torch.Tensor, cluster_labels: np.ndarray, K: int) -> List[Tuple[int, int]]:
    """Node closest to each cluster's centroid in the current H."""
    H_np = H.detach().cpu().numpy()
    out: List[Tuple[int, int]] = []
    for k in range(K):
        cluster_indices = np.where(cluster_labels == k)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = H_np[cluster_indices].mean(axis=0)
        dists = np.linalg.norm(H_np[cluster_indices] - centroid, axis=1)
        closest_idx = int(cluster_indices[np.argmin(dists)])
        out.append((k, closest_idx))
    return out


def _build_cluster_prompt(
    idx: int,
    node_idx: int,
    neighbors: List[int],
    H: torch.Tensor,
    H_token: torch.Tensor,
    raw_texts,
    adj: Dict[int, set],
    abstract_dict: Optional[Dict[str, str]] = None,
) -> str:
    center_token = format_node_token(
        node_idx, H, raw_texts, adj, abstract_dict, H_token=H_token
    )
    node_id = raw_texts[node_idx] if raw_texts else str(node_idx)

    nb_parts = []
    for j, nb_idx in enumerate(neighbors):
        nb_token = format_node_token(
            nb_idx, H, raw_texts, adj, abstract_dict, H_token=H_token
        )
        nb_parts.append(f"Token_{j + 1}: {nb_token}")
    neighbor_seq_str = ", ".join(nb_parts) if nb_parts else "None"

    info_clause = ""
    if USE_ABSTRACT and abstract_dict:
        abstract = abstract_dict.get(str(node_id), "N/A")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        info_clause = f"with the following information: Abstract: {abstract}. "

    return (
        f"Given the central node {center_token} as <Token_{idx + 1}>, "
        f"{info_clause}"
        f"Please analyze the node's preferenced neighbors in a short paragraph "
        f"based on the <neighbor sequence>: {neighbor_seq_str}."
    )


def run_thought_loop(
    gcn_model,
    data,
    K: int,
    num_thoughts: int,
    cluster_results: List[dict],
    cluster_prompts: List[dict],
    dataset_name: str = DATASET_NAME,
    max_candidates: int = 10,
    select_k: int = 5,
    max_neighbors_per_seq: int = 5,
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Multi-thought loop.  One thought = step(2) + step(3) + GNN re-encode.

        t = 0       — use ``cluster_results`` / ``cluster_prompts`` built from
                      1-hop (+ 2-hop supplement) neighbors of each centre.
        t ≥ 1       — re-run step(2): build new cluster prompts whose neighbor
                      sequence comes from the **previous thought's selections**
                      for that centre (supplemented with current adj if < 5).
                      The cluster-center LLM output becomes the new Instruction
                      Prefix for step(3).

    Last thought additionally asks the per-node LLM to output a short analysis
    paragraph along with the selected neighbors.

    Returns ``(final_edge_index, all_thought_results)``.
    """
    from use_llm_API import LLM_Predictor

    device = next(gcn_model.gcn.parameters()).device
    predictor = LLM_Predictor()

    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None
    title_dict = load_node_titles(dataset_name, data)
    abstract_dict = load_node_abstracts(dataset_name)

    cluster_labels = torch.load(
        f"dataset/{dataset_name}/cluster/cluster_labels.pt"
    ).numpy()

    cr_map: Dict[int, dict] = {int(r["cluster_id"]): r for r in cluster_results}
    cp_map: Dict[int, dict] = {int(p["cluster_id"]): p for p in cluster_prompts}

    current_edge_index = data.edge_index.clone()
    all_thought_results: List[dict] = []
    previous_selections: Dict[int, List[int]] = {}   # updated each thought

    for t in range(num_thoughts):
        is_last = t == num_thoughts - 1
        print(f"\n{'=' * 60}")
        tag = "(LAST – with analysis)" if is_last else ""
        print(f"--- Thought {t + 1}/{num_thoughts} {tag} ---")
        print(f"{'=' * 60}")

        # ── GNN encode on current graph ──
        with torch.no_grad():
            if hasattr(gcn_model, "embed_with_projection"):
                H, H_token = gcn_model.embed_with_projection(
                    data.x.to(device), current_edge_index.to(device)
                )
            else:
                H = gcn_model.embed(
                    data.x.to(device), current_edge_index.to(device)
                )
                H_token = H

        # Adjacency of the CURRENT graph — drives step(2) & step(3)
        current_adj = build_adjacency(current_edge_index, data.num_nodes)

        # ── STEP (2) re-run on iteration t ≥ 1 ─────────────────────
        if t >= 1:
            print(f"--- Thought {t + 1}: Step (2) re-running cluster prompts ---")
            new_centrals = _find_central_nodes(H, cluster_labels, K)
            for k, node_idx in new_centrals:
                prev_sel = previous_selections.get(node_idx, [])
                # Representative sequence: prefer prev-thought selections,
                # fall back to current 1-hop/2-hop if not enough.
                neighbor_seq: List[int] = []
                for nb in prev_sel:
                    if len(neighbor_seq) >= max_neighbors_per_seq:
                        break
                    if nb not in neighbor_seq:
                        neighbor_seq.append(nb)
                if len(neighbor_seq) < max_neighbors_per_seq:
                    backup = get_neighbor_sequence(
                        node_idx, current_adj, max_neighbors_per_seq
                    )
                    for nb in backup:
                        if len(neighbor_seq) >= max_neighbors_per_seq:
                            break
                        if nb not in neighbor_seq:
                            neighbor_seq.append(nb)

                prompt = _build_cluster_prompt(
                    k, node_idx, neighbor_seq, H, H_token,
                    raw_texts, current_adj, abstract_dict,
                )
                response = predictor.predict(prompt)
                nid = raw_texts[node_idx] if raw_texts else str(node_idx)
                cr_map[k] = {
                    "cluster_id": k,
                    "central_node_id": str(nid),
                    "prompt": prompt,
                    "llm_response": response,
                }
                cp_map[k] = {
                    "cluster_id": k,
                    "central_node_idx": node_idx,
                    "central_node_id": str(nid),
                    "prompt": prompt,
                    "neighbor_indices": neighbor_seq,
                }
            # Persist iteration-specific cluster prompts / results
            save_dir = f"dataset/{dataset_name}/cluster"
            os.makedirs(save_dir, exist_ok=True)
            pd.DataFrame(cr_map.values()).to_csv(
                os.path.join(save_dir, f"thought_{t + 1}_cluster_llm_results.csv"),
                index=False,
            )

        # ── STEP (3) per-node neighbor selection ───────────────────
        all_selections: Dict[int, List[int]] = {}
        thought_results: List[dict] = []
        node_count = 0

        for k in range(K):
            cluster_node_indices = np.where(cluster_labels == k)[0]
            if len(cluster_node_indices) == 0:
                continue
            cr = cr_map.get(k)
            cp = cp_map.get(k)
            if cr is None or cp is None:
                node_count += len(cluster_node_indices)
                continue

            instruction_prefix = build_instruction_prefix(
                cr, cp, raw_texts, title_dict, current_adj
            )

            for node_idx in cluster_node_indices:
                node_idx = int(node_idx)
                candidates = get_candidates(
                    node_idx, current_adj, max_candidates
                )
                if not candidates:
                    node_count += 1
                    continue

                prompt = build_node_selection_prompt(
                    instruction_prefix,
                    node_idx,
                    candidates,
                    raw_texts,
                    title_dict,
                    current_adj,
                    is_last,
                )
                response = predictor.predict(prompt)
                selected = parse_selected_neighbors(
                    response, candidates, raw_texts, select_k
                )
                all_selections[node_idx] = selected

                nid = raw_texts[node_idx] if raw_texts else str(node_idx)
                sel_ids = [
                    (raw_texts[s] if raw_texts else str(s)) for s in selected
                ]
                thought_results.append(
                    {
                        "thought": t + 1,
                        "cluster_id": k,
                        "node_idx": node_idx,
                        "node_id": str(nid),
                        "selected_neighbors": str(sel_ids),
                        "llm_response": response,
                    }
                )
                node_count += 1
                if node_count % 100 == 0:
                    print(f"  Processed {node_count}/{data.num_nodes} nodes")

        save_dir = f"dataset/{dataset_name}/cluster"
        os.makedirs(save_dir, exist_ok=True)
        if thought_results:
            pd.DataFrame(thought_results).to_csv(
                os.path.join(save_dir, f"thought_{t + 1}_node_results.csv"),
                index=False,
            )

        # Build new edge_index from step-(3) selections
        current_edge_index = build_optimized_edge_index(
            all_selections, data.num_nodes, data.edge_index
        )
        torch.save(
            current_edge_index,
            os.path.join(save_dir, f"thought_{t + 1}_edge_index.pt"),
        )

        print(
            f"✅ Thought {t + 1} complete: {node_count} nodes processed, "
            f"new edges: {current_edge_index.shape[1]}"
        )
        all_thought_results.extend(thought_results)
        previous_selections = all_selections

    return current_edge_index, all_thought_results


# ── Final LLM classification ────────────────────────────────


DATASET_LABEL_MAP: Dict[str, List[str]] = {
    "cora": [
        "Case_Based",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Reinforcement_Learning",
        "Rule_Learning",
        "Theory",
    ],
    # PubMed-Diabetes labels 1/2/3 mapped to 0/1/2 by prepare_pubmed.py
    "pubmed": [
        "Diabetes Mellitus, Experimental",
        "Diabetes Mellitus Type 1",
        "Diabetes Mellitus Type 2",
    ],
    # graph_data_paper.pt is the OGBN-Arxiv paper graph.
    # Order = OGB official label-id mapping (`labelidx2arxivcategeory.csv`).
    "paper": [
        "arxiv cs.NA",  "arxiv cs.MM",  "arxiv cs.LO",  "arxiv cs.CY",
        "arxiv cs.CR",  "arxiv cs.DC",  "arxiv cs.HC",  "arxiv cs.CE",
        "arxiv cs.NI",  "arxiv cs.CC",  "arxiv cs.AI",  "arxiv cs.MA",
        "arxiv cs.GL",  "arxiv cs.NE",  "arxiv cs.SC",  "arxiv cs.AR",
        "arxiv cs.CV",  "arxiv cs.GR",  "arxiv cs.ET",  "arxiv cs.SY",
        "arxiv cs.CG",  "arxiv cs.OH",  "arxiv cs.PL",  "arxiv cs.SE",
        "arxiv cs.LG",  "arxiv cs.SD",  "arxiv cs.SI",  "arxiv cs.RO",
        "arxiv cs.IT",  "arxiv cs.PF",  "arxiv cs.CL",  "arxiv cs.IR",
        "arxiv cs.MS",  "arxiv cs.FL",  "arxiv cs.DS",  "arxiv cs.OS",
        "arxiv cs.GT",  "arxiv cs.DB",  "arxiv cs.DL",  "arxiv cs.DM",
    ],
}


def format_node_representation_tokens(
    H: torch.Tensor, node_idx: int, num_tokens: int = 8
) -> str:
    """Split the node embedding into *num_tokens* chunks; each chunk → 1 token."""
    h = H[node_idx].detach().cpu().numpy()
    chunks = np.array_split(h, num_tokens)
    tokens = []
    for i, chunk in enumerate(chunks):
        tokens.append(
            f"<Token_{i + 1}: mean={chunk.mean():.3f}, std={chunk.std():.3f}>"
        )
    return " ".join(tokens)


def load_answer_candidates(dataset_name: str, data=None) -> List[str]:
    """Resolve candidate label names for the classification question."""
    if data is not None and hasattr(data, "label_names") and data.label_names:
        return list(data.label_names)
    if data is not None and hasattr(data, "category_names") and data.category_names:
        return list(data.category_names)
    if dataset_name in DATASET_LABEL_MAP:
        return DATASET_LABEL_MAP[dataset_name]
    if data is not None and hasattr(data, "y") and data.y is not None:
        uniq = sorted({int(v) for v in data.y.cpu().tolist()})
        return [f"class_{i}" for i in uniq]
    return []


def build_classification_prompt(
    node_idx: int,
    H: torch.Tensor,
    title: str,
    neighbor_text: str,
    candidates: List[str],
    dataset_name: str = DATASET_NAME,
    num_tokens: int = 8,
) -> str:
    repr_tokens = format_node_representation_tokens(H, node_idx, num_tokens)
    candidates_str = ", ".join(candidates)
    domain_key = str(dataset_name).lower()
    entity = "product" if domain_key in {
        "children", "computers", "history", "photo", "sports",
    } else "paper"
    question = (
        f"Which e-commerce category does this {entity} belong to?"
        if entity == "product"
        else "Which research category does this paper belong to?"
    )
    return (
        f"Given the representation of a {entity}: {repr_tokens}, "
        f"with the following information: Title: {title}. "
        f"Neighbor: {neighbor_text}. "
        f"Question: {question} "
        f"Please directly give the most likely answer from the following "
        f"categories: {candidates_str}."
    )


def parse_classification_response(response: str, candidates: List[str]) -> str:
    """Return the candidate label appearing earliest in the response.

    If no candidate string is found we return an empty string, which counts
    as wrong.  Defaulting to ``candidates[0]`` here used to mask prompt
    failures and inflate the apparent accuracy of the dominant class.
    """
    if not response or not candidates:
        return ""
    resp_lower = response.lower()
    aliases: List[Tuple[str, str]] = []
    for c in candidates:
        aliases.append((c, c.lower()))
        if "_" in c:
            aliases.append((c, c.replace("_", " ").lower()))
        if " " in c:
            aliases.append((c, c.replace(" ", "_").lower()))
    best, best_pos = "", float("inf")
    for canon, alias in aliases:
        pos = resp_lower.find(alias)
        if 0 <= pos < best_pos:
            best, best_pos = canon, pos
    return best


def llm_classify_nodes(
    gcn_model,
    data,
    thought_results: List[dict],
    final_edge_index: torch.Tensor,
    num_thoughts: int,
    dataset_name: str = DATASET_NAME,
    num_tokens: int = 8,
) -> List[dict]:
    """
    Final step: re-encode with *final_edge_index* → H (+ H_token),
    combine with the last-thought text per node, ask the LLM for the class.
    Representation tokens are serialised from H_token = Linear(H).
    """
    from use_llm_API import LLM_Predictor

    print("\n" + "=" * 60)
    print("--- Final LLM classification ---")
    print("=" * 60)

    device = next(gcn_model.gcn.parameters()).device
    with torch.no_grad():
        if hasattr(gcn_model, "embed_with_projection"):
            H, H_token = gcn_model.embed_with_projection(
                data.x.to(device), final_edge_index.to(device)
            )
        else:
            H = gcn_model.embed(
                data.x.to(device), final_edge_index.to(device)
            )
            H_token = H
    print(f"✅ Final H shape: {tuple(H.shape)},  H_token shape: {tuple(H_token.shape)}")

    last_thought_map: Dict[int, str] = {}
    for r in thought_results:
        if int(r.get("thought", -1)) == num_thoughts:
            last_thought_map[int(r["node_idx"])] = str(r.get("llm_response", ""))

    title_dict = load_node_titles(dataset_name, data)
    candidates = load_answer_candidates(dataset_name, data)
    if not candidates:
        print("⚠️ No answer candidates found, skipping classification.")
        return []
    print(f"Candidates: {candidates}")

    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None
    predictor = LLM_Predictor()

    results: List[dict] = []
    for node_idx in range(data.num_nodes):
        nid = raw_texts[node_idx] if raw_texts else str(node_idx)
        title = title_dict.get(str(nid), "Unknown")
        neighbor_text = last_thought_map.get(node_idx, "")

        prompt = build_classification_prompt(
            node_idx, H_token, title, neighbor_text, candidates, dataset_name, num_tokens
        )
        response = predictor.predict(prompt)
        predicted = parse_classification_response(response, candidates)

        true_label: Optional[str] = None
        if hasattr(data, "y") and data.y is not None:
            try:
                label_idx = int(data.y[node_idx].item())
                if 0 <= label_idx < len(candidates):
                    true_label = candidates[label_idx]
            except Exception:
                true_label = None

        results.append(
            {
                "node_idx": node_idx,
                "node_id": str(nid),
                "title": title,
                "predicted": predicted,
                "true_label": true_label,
                "llm_response": response,
            }
        )

        if (node_idx + 1) % 100 == 0:
            print(f"  {node_idx + 1}/{data.num_nodes} classified")

    save_dir = f"dataset/{dataset_name}/cluster"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, f"{dataset_name}_classification_results.csv"
    )
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n✅ Saved classification results to: {out_path}")

    if any(r["true_label"] is not None for r in results):
        correct = sum(
            1 for r in results
            if r["true_label"] is not None and r["predicted"] == r["true_label"]
        )
        total = sum(1 for r in results if r["true_label"] is not None)
        acc = correct / total if total > 0 else 0.0
        print(f"✅ LLM classification accuracy: {acc:.4f} ({correct}/{total})")

    return results
