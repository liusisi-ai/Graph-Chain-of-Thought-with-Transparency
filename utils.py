from use_llm_API import *
from preprocess import *
from config import ROOT_PATH, DATASET_NAME, MODEL_PATH, global_model
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union, Final

_cached_token_map = None


def init_token_map(data):
    """从 Data 对象构建并缓存全局 token_map"""
    global _cached_token_map
    from dataloader import build_token_map
    _cached_token_map = build_token_map(data)
    return _cached_token_map


def get_token_map():
    """获取缓存的 token_map"""
    return _cached_token_map


def generate_prompt(
        thoughts: int,
        structural_prompt_enabled: bool,
        original_knn_enabled: bool,
        epoch: int
):
    output_path = None

    print("=======================================================================")
    print("--- 启动预处理 (Generate Prompt by KNN) ---")
    print(f"--- 目标配置: 结构邻居 Prompt: {structural_prompt_enabled}, 原始 KNN Prompt: {original_knn_enabled} ---")
    print("=======================================================================")

    try:
        config = PromptConfig(
            ROOT_PATH,
            DATASET_NAME,
            thoughts,
            use_structural_prompt=structural_prompt_enabled,
            use_original_knn_prompt=original_knn_enabled,
            epoch=epoch,
        )
        original_X = torch.load(f'{DATASET_NAME}_checkpoints/original_x.pt')
        output_path = generate_prompts_dataset(original_X, config, token_map=_cached_token_map)

        print("\n✅ 预处理任务全部完成。")

    except NameError as e:
        print(f"\n❌ 错误: 缺少必要的函数或类定义。请确保 PromptConfig, generate_prompts_dataset 等已定义。错误信息: {e}")
    except Exception as e:
        print(f"\n❌ 预处理过程中发生未知错误: {e}")
    return output_path


def use_llm(structural: bool, original_knn: bool, read_path, thought, epoch):
    print("\n--- 调用 use_llm() 函数开始 LLM 推理 ---")
    output_path = run_llm_inference(
        ROOT_PATH,
        DATASET_NAME,
        enable_structural=structural,
        enable_original_knn=original_knn,
        read_path=read_path,
        thought=thought,
        epoch=epoch  # 传入 epoch
    )
    return output_path


def create_path(x, thought_num, epoch, dataset_name=DATASET_NAME):

    num_str = str(thought_num)
    # 路径结构: dataset/{dataset_name}/{epoch}/{thought_num}_thought_embeddings.pt
    filename = f"{epoch}/{num_str}_thought_embeddings.pt"
    full_path = os.path.join("dataset", dataset_name, filename)

    parent_dir = os.path.dirname(full_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"📁 Created missing directory: {parent_dir}")
    torch.save(x, full_path)
    print(f"✅ Saved embedding to: {full_path}")
    return None


def load_sentence_transformer(model_path: str) -> Union[SentenceTransformer, None]:
    global global_model
    if global_model is not None:
        return global_model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Sentence Transformer to {device}...")
        model = SentenceTransformer(model_path, device=device)
        global_model = model
        return model
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None


def generate_embeddings(data_name, read_path, thought, epoch):
    base_dir = f"dataset/{data_name}"
    epoch_dir = os.path.join(base_dir, str(epoch))

    tasks = [
        {
            "name": "Original Refined",
            "pt_path": os.path.join(epoch_dir, f"{data_name}_refined_original_knn_emb.pt"),
            "csv_path": os.path.join(epoch_dir, f"{data_name}_refined_text_original_knn_local_llm.csv"),
            "col": "refined_text"
        },
        {
            "name": "Structural Refined",
            "pt_path": os.path.join(epoch_dir, f"{data_name}_refined_structural_emb.pt"),
            "csv_path": os.path.join(epoch_dir, f"{data_name}_refined_text_structural_local_llm.csv"),
            "col": "refined_text"
        },
        {
            "name": "Fusion Refined",
            "pt_path": os.path.join(epoch_dir, f"{data_name}_refined_fusion_knn_emb.pt"),
            "csv_path": os.path.join(epoch_dir, f"{data_name}_refined_text_fusion_knn_local_llm.csv"),
            "col": "refined_text"
        }
    ]

    emb_model_path: str = MODEL_PATH
    model = None

    print(f"\n--- Checking Embeddings (Epoch={epoch}) ---")

    for task in tasks:
        pt_path = task['pt_path']
        csv_path = task['csv_path']
        col_name = task['col']

        if os.path.exists(pt_path):
            print(f"✅ Exists: {os.path.basename(pt_path)}")
            continue

        if not os.path.exists(csv_path):
            print(f"❌ Missing CSV: {csv_path}")
            continue

        print(f"⚠️ Generating: {os.path.basename(pt_path)} from {os.path.basename(csv_path)}")

        if model is None:
            model = load_sentence_transformer(emb_model_path)
            if model is None: return

        try:
            df = pd.read_csv(csv_path)
            texts = df[col_name].astype(str).tolist()
            
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=model.device
            )

            torch.save(embeddings, pt_path)
            print(f"   💾 Saved: {os.path.basename(pt_path)}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("🎯 All tasks processed.")


def load_thought(data, device, thought, epoch):
    epoch_dir = f"dataset/{data}/{epoch}"

    p_fusion = os.path.join(epoch_dir, f"{data}_refined_fusion_knn_emb.pt")
    p_original = os.path.join(epoch_dir, f"{data}_refined_original_knn_emb.pt")
    p_structural = os.path.join(epoch_dir, f"{data}_refined_structural_emb.pt")

    try:
        emb_f = torch.load(p_fusion, map_location=device)
        emb_o = torch.load(p_original, map_location=device)
        emb_s = torch.load(p_structural, map_location=device)

        print(f"✅ Loaded 3 Embeddings (Epoch {epoch})")
        return emb_f, emb_o, emb_s

    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return None, None, None