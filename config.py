import os
from typing import List, Tuple, Union, Final


_CANONICAL_DATASET_NAMES = {
    'cora': 'cora',
    'pubmed': 'pubmed',
    'children': 'Children',
    'computers': 'Computers',
    'history': 'History',
    'photo': 'Photo',
    'sports': 'Sports',
    'ogbn-arxiv': 'ogbn-arxiv',
    'ogbn_arxiv': 'ogbn-arxiv',
    'arxiv': 'ogbn-arxiv',
    'paper': 'ogbn-arxiv',
}

CITATION_DOMAIN_DATASETS: Final[Tuple[str, ...]] = ('cora', 'pubmed')
ECOMMERCE_DOMAIN_DATASETS: Final[Tuple[str, ...]] = (
    'children', 'computers', 'history', 'photo', 'sports',
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def canonical_dataset_name(name: str) -> str:
    """Normalize common dataset aliases while keeping unknown names unchanged."""
    key = str(name or '').strip()
    return _CANONICAL_DATASET_NAMES.get(key.lower(), key)


def resolve_pretrain_dataset(source_dataset: str) -> str:
    """Resolve Stage-1 pretraining source.

    ``CFG_PRETRAIN_DATASET=auto`` selects by domain:
      * Citation domain (Cora/PubMed) -> ogbn-arxiv
      * E-commerce domain -> Computers

    Set ``CFG_PRETRAIN_DATASET`` to a concrete dataset name to override, or to
    an empty string to pretrain on the downstream source itself.
    """
    source = canonical_dataset_name(source_dataset)
    requested = _env_str('CFG_PRETRAIN_DATASET', 'auto').strip()
    if requested == '':
        return source
    if requested.lower() != 'auto':
        return canonical_dataset_name(requested)

    source_key = source.lower()
    if source_key in CITATION_DOMAIN_DATASETS:
        return 'ogbn-arxiv'
    if source_key in ECOMMERCE_DOMAIN_DATASETS:
        return 'Computers'
    return source


# Global RNG seed – every random source in the project is funnelled through
# ``seed_utils.seed_everything(GLOBAL_SEED)`` at the top of main.py.
GLOBAL_SEED = _env_int('CFG_SEED', 42)

K_FILTER = 2
ROOT_PATH = 'dataset'
# 仅使用 cora 文件夹下的数据进行训练
DATASET_NAME = canonical_dataset_name(_env_str('CFG_DATASET', 'cora'))
ADJ_PATH = os.path.join(ROOT_PATH, DATASET_NAME, f'{DATASET_NAME}_adj_matrix.npy')
CONTENT_PATH = os.path.join(ROOT_PATH, DATASET_NAME, f'{DATASET_NAME}.content')

# ── GNN pretrain (Stage 1 GraphSAGE contrastive) ──
# Was 1000; 200 is enough for small graphs like Cora. Override: CFG_SAGE_EPOCHS / CFG_SAGE_LR
SAGE_PRETRAIN_EPOCHS = _env_int('CFG_SAGE_EPOCHS', 200)
SAGE_PRETRAIN_LR     = _env_float('CFG_SAGE_LR', 0.01)
CKPT_PATH = f'{DATASET_NAME}_checkpoints/preprompt_gcn.pt'
EMBED0_PATH = f'{DATASET_NAME}_checkpoints/filtered_feature.pt'
FIRST_MODEL_PATH = f'{DATASET_NAME}_checkpoints/first_model.pt'

# 用于 Stage 1 GraphSAGE 对比预训练的数据集（与下游训练数据集解耦）。
# auto: Citation Domain -> ogbn-arxiv, E-Commerce Domain -> Computers。
# 设置 CFG_PRETRAIN_DATASET 为空字符串可退回到「预训练与下游同数据集」的旧行为。
PRETRAIN_DATASET   = _env_str('CFG_PRETRAIN_DATASET', 'auto')
# 子图采样上限, 避免在 ~170k 节点的 arxiv 上 OOM。0 = 不采样, 使用整图。
PRETRAIN_MAX_NODES = _env_int('CFG_PRETRAIN_MAX_NODES', 20000)

# ── Local models / HuggingFace fallback ──────────────────────────
# Resolution order:
#   1. AutoDL local copy at /root/autodl-tmp/<name>/
#   2. In-project local folder ./<name>/
#   3. HuggingFace Hub model id (auto-download on first use)
#
# transformers.from_pretrained() and SentenceTransformer() both accept HF ids
# as input — when you pass a string that isn't a local dir, they resolve it
# against the Hub and cache the weights under HF_HOME.
_AUTODL_BASE = '/root/autodl-tmp'
HF_LLM_ID = 'lmsys/vicuna-7b-v1.5-16k'
HF_SBERT_ID = 'sentence-transformers/all-mpnet-base-v2'


def _resolve_model_path(local_name: str, hf_id: str) -> str:
    """Pick the first existing local dir, otherwise return the HF id."""
    for cand in (
        os.path.join(_AUTODL_BASE, local_name),
        local_name,
        f'./{local_name}',
    ):
        if os.path.isdir(cand):
            return cand
    return hf_id


SENTENCE_MODEL_PATH = _resolve_model_path('all-mpnet-base-v2', HF_SBERT_ID)
LOCAL_LLM_PATH = _resolve_model_path('vicuna-7b-v1.5-16k', HF_LLM_ID)

# Keep HuggingFace cache off the small system disk.  These are picked up by
# huggingface_hub / transformers / sentence-transformers automatically.
_HF_CACHE = (
    f'{_AUTODL_BASE}/.cache/huggingface'
    if os.path.isdir(_AUTODL_BASE)
    else os.path.expanduser('~/.cache/huggingface')
)
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault('HF_HOME', _HF_CACHE)
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', _HF_CACHE)

# ── HuggingFace mirror ───────────────────────────────────────────
# AutoDL/中国大陆机器通常无法直连 huggingface.co (会出现
# "We couldn't connect to 'https://huggingface.co'" / TimeoutError)。
#
# 默认强制走 hf-mirror.com 镜像。若用户已显式指定一个非默认的 HF_ENDPOINT,
# 则尊重用户设置；否则覆盖。可通过 HF_DISABLE_MIRROR=1 关闭此强制行为。
_DEFAULT_HF = 'https://huggingface.co'
_MIRROR_HF = 'https://hf-mirror.com'
_user_endpoint = os.environ.get('HF_ENDPOINT', '').rstrip('/')

if os.environ.get('HF_DISABLE_MIRROR', '0').lower() in ('1', 'true', 'yes'):
    pass  # user opted out; honour their HF_ENDPOINT (or default)
elif _user_endpoint in ('', _DEFAULT_HF):
    os.environ['HF_ENDPOINT'] = _MIRROR_HF
# else: user pointed to some custom mirror — keep it

# hf_transfer 会启用并行加速下载, 要求 hf-mirror 支持 (目前支持)。
# 若机器没装 hf_transfer 包会回退，可主动关闭: HF_HUB_ENABLE_HF_TRANSFER=0
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

# Banner so the user sees immediately which endpoint is in use.
print(f"🌐 HF_ENDPOINT = {os.environ['HF_ENDPOINT']}  (HF_HOME = {_HF_CACHE})")

# Back-compat alias (other files still import MODEL_PATH)
MODEL_PATH = SENTENCE_MODEL_PATH

# ── Clustering / CoT prompt ──
# Every value here can be overridden from the environment by ``scripts/tune.py``.
NUM_CLUSTERS          = _env_int('CFG_NUM_CLUSTERS',          7)
MAX_NEIGHBORS_PER_SEQ = _env_int('CFG_MAX_NEIGHBORS_PER_SEQ', 5)
NUM_CLUSTER_THOUGHTS  = _env_int('CFG_NUM_THOUGHTS',          2)
MAX_CANDIDATES        = _env_int('CFG_MAX_CANDIDATES',       10)
SELECT_K              = _env_int('CFG_SELECT_K',              5)
NUM_REPR_TOKENS       = _env_int('CFG_NUM_REPR_TOKENS',       8)

# Whether to include the (often long) node abstract in cluster prompts.
# Default ON: removing it gives only minor speed-up but loses information.
# Override with: CFG_USE_ABSTRACT=0 to disable.
USE_ABSTRACT = _env_str('CFG_USE_ABSTRACT', '1').lower() in ('1', 'true', 'yes')

# ── CoT training ──
WARMUP_EPOCHS      = _env_int('CFG_WARMUP_EPOCHS',     5)
COT_START_EPOCH    = _env_int('CFG_COT_START_EPOCH',   5)
COT_WEIGHT         = _env_float('CFG_COT_WEIGHT',     0.5)
CONDITION_TEXT_DIM = 768
LLM_HIDDEN_DIM     = 4096
USE_LORA           = _env_str('CFG_USE_LORA', '1').lower() in ('1', 'true', 'yes')

# ── Zero-shot training (PCA anchor alignment) ──
SOURCE_DOMAIN        = canonical_dataset_name(_env_str('CFG_SOURCE_DOMAIN', 'cora'))
TARGET_DOMAINS       = []
NUM_PCA_ANCHORS      = _env_int('CFG_NUM_PCA_ANCHORS',     256)
PCA_ALIGN_EPOCHS     = _env_int('CFG_PCA_EPOCHS',            3)
PCA_ALIGN_LR         = _env_float('CFG_PCA_LR',          1e-4)
PCA_ALIGN_BATCH_SIZE = _env_int('CFG_PCA_BATCH_SIZE',      512)
INSTRUCTION_DIR      = 'instructions'

# Training mode: 'pca_align' | 'sft' | 'both'
TRAIN_MODE = _env_str('CFG_TRAIN_MODE', 'pca_align')

# Whether Stage-4 zero-shot inference prepends the GNN soft token to the LLM
# prompt.  When PCA alignment is under-trained the soft token is essentially
# noise and *hurts* accuracy (we observed 25% → 1.7% on cora).  Default OFF.
# Re-enable once `PCA-align loss` has converged (e.g. < 1.5):
#     CFG_USE_SOFT_TOKEN=1 python main.py …
USE_SOFT_TOKEN = _env_str('CFG_USE_SOFT_TOKEN', '0').lower() in ('1', 'true', 'yes')

# Constrained decoding for closed-set classification: instead of free
# generation + regex post-match, score each candidate's per-token log-
# likelihood given the prompt and pick argmax.  Strictly stronger for
# label-classification tasks; turn off only to debug raw generation.
USE_CONSTRAINED_DECODING = _env_str('CFG_CONSTRAINED', '1').lower() in ('1', 'true', 'yes')

# ── Subgraph sampling (for very large source graphs) ──
MAX_SOURCE_NODES = _env_int('CFG_MAX_SOURCE_NODES', 0)
SUBGRAPH_SEED    = GLOBAL_SEED

# ── Tuning evaluation mode ──
# 'val'  : use val_mask  (use this when *selecting* hyperparameters)
# 'test' : use test_mask (only use when *reporting* the final number)
# 'both' : evaluate on val first, then test
EVAL_ON = _env_str('CFG_EVAL_ON', 'test')

global_model = None
