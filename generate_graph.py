import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List


FILE_PATH = 'dataset/cora/graph.csv'
COLUMN_NAMES = ['source_id', 'target_id']
GRAPH_DELIMITER = ','


def generate_and_save_adjacency_matrix_unmatched(
        csv_filepath: str,
        column_names: List[str] = COLUMN_NAMES,
        delimiter: str = GRAPH_DELIMITER
) -> Tuple[np.ndarray, Dict[str, int]]:


    output_dir = os.path.dirname(csv_filepath)
    if not output_dir:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    adj_matrix_path = os.path.join(output_dir, 'cora_adj_matrix.npy')
    id_map_path = os.path.join(output_dir, 'cora_id_to_index_map.csv')

    try:
        df = pd.read_csv(
            csv_filepath,
            sep=delimiter,
            header=0,  # 明确指定第一行为表头
            names=column_names,  # 使用自定义列名
            dtype=str
        )
        print(f"✅ 成功读取文件: {csv_filepath} (已跳过表头)")
    except FileNotFoundError:
        print(f"❌ 错误：文件未找到于路径 {csv_filepath}。请检查路径。")
        return None, None
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return None, None

    source_col = column_names[0]
    target_col = column_names[1]
    unique_ids = pd.unique(df[[source_col, target_col]].values.ravel())
    N = len(unique_ids)
    id_to_index = {id_val: i for i, id_val in enumerate(unique_ids)}

    print(f"\n✨ 节点总数 N (基于 graph.csv): {N}")
    adj_matrix = np.zeros((N, N), dtype=np.int8)
    for _, row in df.iterrows():
        source_id = row[source_col]
        target_id = row[target_col]
        source_idx = id_to_index[source_id]
        target_idx = id_to_index[target_id]
        adj_matrix[source_idx, target_idx] = 1
    np.save(adj_matrix_path, adj_matrix)
    print(f"\n💾 邻接矩阵 (NumPy array) 已保存至: {adj_matrix_path}")
    id_map_df = pd.DataFrame(
        list(id_to_index.items()),
        columns=['paper_id', 'matrix_index']
    )
    id_map_df.to_csv(id_map_path, index=False)
    print(f"💾 ID 映射 (CSV) 已保存至: {id_map_path}")
    return adj_matrix, id_to_index
adj_matrix, id_map = generate_and_save_adjacency_matrix_unmatched(
    csv_filepath=FILE_PATH,
    delimiter=GRAPH_DELIMITER
)

if adj_matrix is not None:
    print("\n--- 结果形状打印 ---")
    print("邻接矩阵 A 的形状 (基于 graph.csv):", adj_matrix.shape)