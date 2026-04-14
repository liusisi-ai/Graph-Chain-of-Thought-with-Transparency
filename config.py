import os
from typing import List, Tuple, Union, Final

K_FILTER = 2
ROOT_PATH = 'dataset'
DATASET_NAME = 'cora'
ADJ_PATH = os.path.join(ROOT_PATH, DATASET_NAME, f'{DATASET_NAME}_adj_matrix.npy')
CONTENT_PATH = os.path.join(ROOT_PATH, DATASET_NAME, f'{DATASET_NAME}.content')
CKPT_PATH = f'{DATASET_NAME}_checkpoints/preprompt_gcn.pt'
EMBED0_PATH = f'{DATASET_NAME}_checkpoints/filtered_feature.pt'
MODEL_PATH = "/home/all-mpnet-base-v2"
global_model = None