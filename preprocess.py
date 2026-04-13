import torch
import os
import csv
import random
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Final, Optional
from torch_geometric.data import Data

M_STRUCTURAL: Final[int] = 5
M_KNN: Final[int] = 5


class PromptConfig:
    def __init__(self,
                 ROOT_PATH,
                 DATASET_NAME,
                 thought,
                 use_structural_prompt: bool,
                 use_original_knn_prompt: bool,
                 epoch: int
                 ):
        self.use_structural_prompt = use_structural_prompt
        self.use_original_knn_prompt = use_original_knn_prompt
        self.ROOT_PATH = ROOT_PATH
        self.DATASET_NAME = DATASET_NAME
        self.thought = thought
        self.epoch = epoch


def load_pyg_data_for_prompt(dataset_name: str, root_path: str):
    pt_path = os.path.join(root_path, dataset_name, "processed_data.pt")

    if not os.path.exists(pt_path):
        print(f"❌ Error: File not found: {pt_path}")
        return None, None, None

    try:
        data_list = torch.load(pt_path, weights_only=False)
        data = data_list[0] if isinstance(data_list, list) else data_list

        if not hasattr(data, 'y') or not hasattr(data, 'raw_texts') or not hasattr(data, 'x'):
            print("❌ Error: Data object missing attributes.")
            return None, None, None

        all_node_ids = data.raw_texts
        labels = data.y.tolist()
        id2label = dict(zip(all_node_ids, [str(l) for l in labels]))

        print(f"✅ Loaded PyG Data. Nodes: {data.num_nodes}, Features: {data.x.shape}")
        return data, id2label, all_node_ids

    except Exception as e:
        print(f"❌ Error loading processed_data.pt: {e}")
        return None, None, None


def build_graph_from_pyg(data: Data, all_node_ids: List[str]) -> Optional[nx.Graph]:
    print(f"-> Building NetworkX graph from PyG edge_index...")
    try:
        edge_index = data.edge_index.cpu().numpy()

        src_nodes = [all_node_ids[i] for i in edge_index[0]]
        dst_nodes = [all_node_ids[i] for i in edge_index[1]]

        edge_list = list(zip(src_nodes, dst_nodes))

        G = nx.Graph()
        G.add_nodes_from(all_node_ids)
        G.add_edges_from(edge_list)

        print(f"   - Graph built. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        return G

    except Exception as e:
        print(f"❌ Error building graph: {e}")
        return None


def serialize_graph_tokens(
        data: Data, G: nx.Graph, all_node_ids: List[str]
) -> Dict[str, dict]:
    """
    对图中所有节点进行序列化
    Token_i = [Node_feature, Node_id, Node_degree]
    返回 node_id -> token 字典
    """
    features = data.x.cpu().numpy() if torch.is_tensor(data.x) else data.x
    token_map: Dict[str, dict] = {}

    for i, node_id in enumerate(all_node_ids):
        degree = G.degree(node_id) if node_id in G else 0
        token_map[node_id] = {
            'node_feature': features[i],
            'node_id': node_id,
            'node_degree': degree,
            'node_index': i,
        }

    print(f"✅ Graph serialized into tokens. Total: {len(token_map)}")
    return token_map


def format_token(token: dict, include_feature_summary: bool = True) -> str:
    """
    将单个 Token 格式化为可读字符串
    格式: [Node_feature: <summary>, Node_id: <id>, Node_degree: <degree>]
    """
    node_id = token['node_id']
    degree = token['node_degree']

    if include_feature_summary:
        feat = token['node_feature']
        feat_str = f"dim={len(feat)}, mean={feat.mean():.4f}, std={feat.std():.4f}"
        return f"[Node_feature: ({feat_str}), Node_id: {node_id}, Node_degree: {degree}]"

    return f"[Node_id: {node_id}, Node_degree: {degree}]"


def load_embeddings(DATA_PATH: str, thought_num: int, epoch: int):
    num_str = str(thought_num)
    filename = f"{epoch}/{num_str}_thought_embeddings.pt"
    full_path = os.path.join("dataset", DATA_PATH, filename)

    if not os.path.exists(full_path):
        print(f"❌ Error: Embeddings file not found: {full_path}")
        return None

    try:
        embed = torch.load(full_path).detach().cpu().numpy().astype(np.float32)
        print(f"✅ Embeddings loaded: {full_path}, Shape: {embed.shape}")
        return embed
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return None


def find_structural_neighbors(G: nx.Graph, start_node: str):
    if start_node not in G:
        return [], []

    first_hop = list(G.neighbors(start_node))
    second_hop = set()

    for node in first_hop:
        for connected_node in G.neighbors(node):
            if connected_node != start_node and connected_node not in first_hop:
                second_hop.add(connected_node)

    return first_hop, list(second_hop)


def get_knn_neighbors(target_node, features_matrix, all_node_ids, k_neighbors):
    id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
    if target_node not in id_to_idx:
        return []

    target_idx = id_to_idx[target_node]
    target_feature_vector = features_matrix[target_idx].reshape(1, -1)

    knn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    knn_model.fit(features_matrix)

    indices = knn_model.kneighbors(target_feature_vector, n_neighbors=k_neighbors + 1, return_distance=False)
    global_knn = [all_node_ids[idx] for idx in indices.flatten()[1:]]

    return global_knn


def process_node_and_generate_prompt(
        target_node, G, id2label, original_X, fusion_feature, all_node_ids,
        config, token_map: Dict[str, dict]
):
    result = {
        'target_node': target_node,
        'output_text': id2label.get(target_node, "Category not found")
    }

    if result['output_text'] == "Category not found":
        return None

    target_token_str = format_token(token_map[target_node])

    structural_neighbors = []
    structural_set = set()
    result['prompt_structural'] = ""
    result['structural_neighbors'] = []

    if config.use_structural_prompt:
        hop_1_all, hop_2_all = find_structural_neighbors(G, target_node)
        selected_hop_1 = random.sample(hop_1_all, min(len(hop_1_all), M_STRUCTURAL))
        selected_hop_2 = random.sample(hop_2_all, min(len(hop_2_all), M_STRUCTURAL))
        structural_neighbors = selected_hop_1 + selected_hop_2
        structural_set = set(structural_neighbors)

        structural_str = [format_token(token_map[n]) for n in structural_neighbors]
        result['prompt_structural'] = (
            f"Central node: {target_token_str}. "
            f"Selected structural neighbors (1-hop and 2-hop): [{', '.join(structural_str)}]."
        )
        result['structural_neighbors'] = structural_neighbors

    all_knn_F = get_knn_neighbors(target_node, fusion_feature, all_node_ids, M_KNN)

    content_knn_neighbors_F = []
    for node in all_knn_F:
        if not config.use_structural_prompt or node not in structural_set:
            content_knn_neighbors_F.append(node)
            if len(content_knn_neighbors_F) >= M_KNN:
                break

    fusion_knn_str = [format_token(token_map[n]) for n in content_knn_neighbors_F]
    result['prompt_fusion_knn'] = (
        f"Central node: {target_token_str}. "
        f"Selected content-based neighbors (from fusion features, non-structural): [{', '.join(fusion_knn_str)}]."
    )
    result['knn_fusion'] = content_knn_neighbors_F

    result['prompt_original_knn'] = ""
    result['knn_original_X'] = []

    if config.use_original_knn_prompt:
        if original_X is None:
            return result

        all_knn_X = get_knn_neighbors(target_node, original_X, all_node_ids, M_KNN)
        original_knn_str = [format_token(token_map[n]) for n in all_knn_X]
        result['prompt_original_knn'] = (
            f"Central node: {target_token_str}. "
            f"Selected global neighbors (from original features): [{', '.join(original_knn_str)}]."
        )
        result['knn_original_X'] = all_knn_X

    return result


def generate_prompts_dataset(original_X, config: PromptConfig):
    prompt_dir = os.path.join("dataset", config.DATASET_NAME, "prompt")
    os.makedirs(prompt_dir, exist_ok=True)

    fusion_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_fusion_knn_prompts.csv")
    structural_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_structural_prompts.csv")
    original_knn_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_original_knn_prompts.csv")
    thought_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_prompts_thought_{config.thought}.csv")

    pyg_data, id2label, all_node_ids = load_pyg_data_for_prompt(config.DATASET_NAME, config.ROOT_PATH)
    if pyg_data is None: return

    G = build_graph_from_pyg(pyg_data, all_node_ids)
    if G is None: return

    token_map = serialize_graph_tokens(pyg_data, G, all_node_ids)

    emb = load_embeddings(config.DATASET_NAME, config.thought, config.epoch)
    if emb is None:
        print("❌ Error: Fusion features not loaded.")
        return

    if config.thought == 1:
        for flag, path, use_struct, use_orig in [
            ('fusion', fusion_path, False, False),
            ('structural', structural_path, config.use_structural_prompt, False),
            ('original_knn', original_knn_path, False, config.use_original_knn_prompt)
        ]:
            if os.path.exists(path):
                print(f"⚠️ File exists, skipping: {path}")
                continue
            print(f"-> Generating {flag} prompts: {path}")
            write_prompts_csv(all_node_ids, G, id2label, emb, original_X, config, path, use_struct, use_orig, token_map)
    else:
        print(f"-> Generating Thought {config.thought} fusion prompts: {thought_path}")
        if os.path.exists(thought_path):
            print(f"⚠️ File exists, skipping: {thought_path}")
            return thought_path

        write_prompts_csv(all_node_ids, G, id2label, emb, original_X, config, thought_path, False, False, token_map)

    return thought_path


def write_prompts_csv(all_node_ids, G, id2label, emb0, original_X, config, output_path, use_structural,
                      use_original_knn, token_map: Dict[str, dict]):
    processed_count = 0
    skipped_not_in_graph = 0
    fieldnames = ['paper_id', 'output_text', 'prompt_text']

    temp_config = PromptConfig(config.ROOT_PATH, config.DATASET_NAME, config.thought, use_structural, use_original_knn, config.epoch)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, node_id in enumerate(all_node_ids):
            if node_id not in G:
                skipped_not_in_graph += 1
                continue

            result = process_node_and_generate_prompt(node_id, G, id2label, original_X, emb0, all_node_ids,
                                                      temp_config, token_map)

            if result is None: continue

            if use_structural:
                prompt_text = result['prompt_structural']
            elif use_original_knn:
                prompt_text = result['prompt_original_knn']
            else:
                prompt_text = result['prompt_fusion_knn']

            row = {
                'paper_id': node_id,
                'output_text': id2label.get(node_id, ''),
                'prompt_text': prompt_text
            }
            writer.writerow(row)
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"  -> Processed {processed_count} / {len(all_node_ids)}")

    print(f"Written to {output_path}, Total: {processed_count}, Skipped: {skipped_not_in_graph}")