import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import warnings
from typing import Dict, Any, Tuple

from config import LOCAL_LLM_PATH

warnings.filterwarnings("ignore", category=UserWarning)

# Default to local LLM (vicuna) – set to True only when an OpenAI-compatible
# endpoint and key are available. Can also be overridden via env var ``USE_API``.
USE_API = os.environ.get("USE_API", "0").lower() in ("1", "true", "yes")

API_CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY", ""),
    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "model_name": os.environ.get("OPENAI_MODEL", "gpt-4.1"),
    "temperature": 0.2,
}

# Local dir or HuggingFace model id (e.g. 'lmsys/vicuna-7b-v1.5-16k').
LOCAL_MODEL_PATH = LOCAL_LLM_PATH


def _try_import_openai():
    """Lazy import of ``openai`` so the module works without the package
    installed when only the local LLM is used."""
    try:
        from openai import OpenAI  # pip install openai
        return OpenAI
    except ImportError as e:
        raise ImportError(
            "USE_API=True but the `openai` package is not installed. "
            "Either run `pip install openai`, set USE_API=0, or call "
            "LLM_Predictor(use_api=False) to use the local LLM."
        ) from e


def _print_vram_summary():
    """Pretty-print available CUDA memory before loading the LLM."""
    if not torch.cuda.is_available():
        print("ℹ️ CUDA not available — model will load on CPU (very slow).")
        return None, 0.0
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    free = (torch.cuda.mem_get_info()[0] / 1024 ** 3) if hasattr(torch.cuda, "mem_get_info") else total
    print(f"🖥️ GPU[{idx}] {name}  total={total:.1f} GB  free={free:.1f} GB")
    return name, free


def _need_4bit(local_path: str, free_gib: float) -> bool:
    """Return True when fp16 loading is unlikely to fit in the current GPU."""
    is_7b = "7b" in local_path.lower() or "7B" in local_path
    # rough fp16 weight footprint + activation overhead
    needed_fp16 = 14.0 if is_7b else 26.0   # 13b ≈ 26 GB
    return free_gib < needed_fp16 * 0.9



def load_file(DATASET_NAME, epoch):
    PROMPT_FILES = {
        'fusion_knn': f'dataset/{DATASET_NAME}/prompt/{DATASET_NAME}_fusion_knn_prompts.csv',
        'structural': f'dataset/{DATASET_NAME}/prompt/{DATASET_NAME}_structural_prompts.csv',
        'original_knn': f'dataset/{DATASET_NAME}/prompt/{DATASET_NAME}_original_knn_prompts.csv',
    }
    NODE_INFO_PATH = f'dataset/{DATASET_NAME}/node_info.csv'
    FIXED_SUMMARY_PATH = f'dataset/{DATASET_NAME}/node_summaries.csv'
    OUTPUT_PATH_TEMPLATE = f'dataset/{DATASET_NAME}/{epoch}/{DATASET_NAME}_refined_text_{{type}}_local_llm.csv'
    return PROMPT_FILES, NODE_INFO_PATH, OUTPUT_PATH_TEMPLATE, FIXED_SUMMARY_PATH


class LLM_Predictor:
    def __init__(self, use_api=USE_API, local_path=LOCAL_MODEL_PATH):
        self.use_api = use_api
        self.tokenizer = None
        self.model = None
        self.device = None
        self.client = None

        if self.use_api:
            print(f"🚀 [Init] 初始化 API 客户端 ({API_CONFIG['model_name']})...")
            try:
                OpenAI = _try_import_openai()
                self.client = OpenAI(
                    api_key=API_CONFIG["api_key"],
                    base_url=API_CONFIG["base_url"]
                )
                print("✅ API 客户端就绪。")
            except Exception as e:
                print(f"❌ API 初始化失败: {e}")
        else:
            is_hf_id = not os.path.isdir(local_path)
            src = "HuggingFace Hub" if is_hf_id else "local dir"
            print(f"🚀 [Init] Loading LLM from {src}: {local_path} ...")

            _, free_gib = _print_vram_summary()
            use_4bit = os.environ.get("LLM_4BIT", "auto")
            if use_4bit == "auto":
                use_4bit = _need_4bit(local_path, free_gib)
            else:
                use_4bit = use_4bit.lower() in ("1", "true", "yes")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"检测到的设备: {self.device}  (4bit={'on' if use_4bit else 'off'})")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                model_kwargs: Dict[str, Any] = dict(
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                if use_4bit:
                    try:
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        print("ℹ️ Using 4-bit (NF4) quantization to fit the GPU.")
                    except Exception as e:
                        print(f"⚠️ Cannot enable 4-bit quantisation ({e}); falling back to fp16.")
                        model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["torch_dtype"] = torch.float16

                self.model = AutoModelForCausalLM.from_pretrained(local_path, **model_kwargs)
                self.model.eval()
                print(f"✅ 本地模型加载成功!")
            except Exception as e:
                msg = str(e).lower()
                hint = ""
                if "out of memory" in msg or "cuda oom" in msg or "cublas" in msg:
                    hint = (
                        "\n💡 GPU 显存不足。Vicuna-7B fp16 需要 ~14 GB 可用显存。"
                        "可选方案：\n"
                        "  1) 启用 4-bit 量化：export LLM_4BIT=1 (需要 bitsandbytes==0.41.x)，仅需 ~5 GB 显存\n"
                        "  2) 升级到 24 GB 显卡 (RTX 3090 / 4090 / A5000 / A6000)\n"
                        "  3) 改用更小模型：在 config.py 把 HF_LLM_ID 改成 'lmsys/vicuna-7b-v1.5' 之外的小模型，例如 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'"
                    )
                elif "no module named 'bitsandbytes'" in msg:
                    hint = "\n💡 请安装 bitsandbytes：pip install bitsandbytes==0.41.3"
                raise RuntimeError(f"❌ 本地模型加载失败: {e}{hint}") from e

    def predict(self, prompt: str) -> str:
        if self.use_api:
            return self._predict_api(prompt)
        else:
            return self._predict_local(prompt)

    def _predict_api(self, prompt: str) -> str:
        if not self.client:
            return "Error: API client not initialized."
        try:
            # 将 Prompt 包装成 Chat 格式
            messages = [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=API_CONFIG["model_name"],
                messages=messages,
                temperature=API_CONFIG["temperature"],
                max_tokens=512  # 控制输出长度
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ API 请求异常: {e}")
            return ""

    def _predict_local(self, prompt: str) -> str:
        if not self.model or not self.tokenizer:
            return "Error: Local LLM not initialized."

        try:
            encoded_input = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16000  # Vicuna 16k context
            ).to(self.device)

            generation_config = GenerationConfig(
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.2,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )

            with torch.no_grad():
                output = self.model.generate(
                    encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    generation_config=generation_config
                )

            generated_text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)

            # 清洗 Vicuna 的 Prompt 标记
            start_tag = "ASSISTANT:"
            if start_tag in generated_text:
                return generated_text.split(start_tag, 1)[-1].strip()
            else:
                return generated_text.strip()

        except Exception as e:
            print(f"❌ 本地推理错误: {e}")
            return "Error during generation."




VICUNA_PROMPT_TEMPLATE = (
    "A chat between a helpful research assistant and a curious user.\n\n"
    "USER: {user_input}\n"
    "ASSISTANT:"
)


def build_summarize_prompt(title: str, abstract: str) -> str:
    user_input = (
        f'The title of the paper is "{title}", '
        f'the abstract of the paper is "{abstract}". '
        f'Please summarize the paper.'
    )
    return VICUNA_PROMPT_TEMPLATE.format(user_input=user_input)


def build_full_analysis_prompt(summary_text: str, neighbor_prompt_text: str) -> str:
    semantic_context = (
        "The core semantic content of the central node is summarized as follows: "
        f'"{summary_text}"\n\n'
    )
    analysis_instruction = (
        f"{neighbor_prompt_text}\n\n"
        'Based *strictly* on the semantic content of the central node and the presence of these neighbor IDs. '
        '**Do not attempt to interpret or assume the content of the neighbor IDs**.'
        'Similar to cluster assignment in K-means, identify the shared aspects that contribute to their feature-space similarity, and discard nodes exhibiting low similarity. '
        'Similar to moving centroids in K-means, state the derived insights in a **single, concise, and dense paragraph**.'
        'Finally, integrate these insights into a compact, refined representation for the target node.'
    )
    user_input = semantic_context + analysis_instruction
    return VICUNA_PROMPT_TEMPLATE.format(user_input=user_input)
def run_llm_inference(
        ROOT_PATH,
        DATASET_NAME,
        enable_structural: bool,
        enable_original_knn: bool,
        read_path: str,
        thought,
        epoch
):
    print("==========================================================")
    print(f"--- 启动 LLM 推理任务 (Thought = {thought}, Epoch = {epoch}) ---")
    print(f"--- 模式: {'API' if USE_API else 'LOCAL'} ---")
    print("==========================================================")
    PROMPT_FILES, NODE_INFO_PATH, OUTPUT_PATH_TEMPLATE, FIXED_SUMMARY_PATH = load_file(DATASET_NAME, epoch)
    predictor = LLM_Predictor()
    if not USE_API and not predictor.model:
        print("❌ 本地模型加载失败，任务终止。")
        return

    # 3. 检查并生成全局 Summary
    if not os.path.exists(FIXED_SUMMARY_PATH):
        print(f"📢 [Pre-check] 摘要文件不存在，开始生成: {FIXED_SUMMARY_PATH}")
        if not os.path.exists(NODE_INFO_PATH):
            print(f"❌ 节点信息文件不存在: {NODE_INFO_PATH}")
            return

        node_info_df = pd.read_csv(NODE_INFO_PATH)
        sum_results = []
        for _, row in tqdm(node_info_df.iterrows(), total=len(node_info_df), desc="Generating Summaries"):
            title = str(row.get('title', "Unknown Title"))
            abstract = str(row.get('abstract', row.get('input_text', "No content available.")))

            prompt = build_summarize_prompt(title, abstract)
            summary = predictor.predict(prompt)  # 使用封装接口

            sum_results.append({
                'paper_id': str(row['paper_id']),
                'summarize_text': summary
            })

        pd.DataFrame(sum_results).to_csv(FIXED_SUMMARY_PATH, index=False)
        print("✅ 全局摘要生成完毕。")

    # 4. 预加载 Summary
    print(f"-> Loading summaries from {FIXED_SUMMARY_PATH}")
    summary_df = pd.read_csv(FIXED_SUMMARY_PATH, dtype={'paper_id': str})
    summary_dict = dict(zip(summary_df['paper_id'], summary_df['summarize_text']))

    # 5. 推理逻辑
    if thought == 1:
        PROMPT_DIR = os.path.join(ROOT_PATH, DATASET_NAME, "prompt")
        EPOCH_DIR = os.path.join(ROOT_PATH, DATASET_NAME, str(epoch))

        PROMPT_FILES_MAP = {
            'structural': f"{DATASET_NAME}_structural_prompts.csv",
            'original_knn': f"{DATASET_NAME}_original_knn_prompts.csv",
            'fusion_knn': f"{DATASET_NAME}_fusion_knn_prompts.csv"
        }
        REFINED_EMB = {
            'structural': f"{DATASET_NAME}_refined_text_structural_local_llm_refined_emb.pt",
            'original_knn': f"{DATASET_NAME}_refined_text_original_knn_local_llm_refined_emb.pt",
            'fusion_knn': f"{DATASET_NAME}_refined_text_fusion_knn_local_llm_refined_emb.pt"
        }

        files_to_process = {}
        for key in ['structural', 'original_knn', 'fusion_knn']:
            emb_path = os.path.join(EPOCH_DIR, REFINED_EMB[key])
            csv_out_path = OUTPUT_PATH_TEMPLATE.format(type=key)

            # 简单的跳过逻辑：如果embedding存在 或 csv存在，就跳过
            if os.path.exists(emb_path):
                print(f"⚠️ {key} 的 Embedding 已存在，跳过。")
                continue
            if os.path.exists(csv_out_path):
                print(f"⚠️ {key} 的 CSV 结果已存在，跳过。")
                continue
            files_to_process[key] = PROMPT_FILES_MAP[key]

        if not files_to_process:
            print(f"✅ Epoch {epoch} (Thought=1) 所有任务已完成。")
            return

        for prompt_type, fname in files_to_process.items():
            full_path = os.path.join(PROMPT_DIR, fname)
            print(f"\n=== 处理 {prompt_type}: {fname} ===")
            if not os.path.exists(full_path):
                continue

            results = []
            prompt_df = pd.read_csv(full_path, dtype={'paper_id': str})

            for _, row in tqdm(prompt_df.iterrows(), total=len(prompt_df)):
                node_id = str(row['paper_id'])
                summary = summary_dict.get(node_id, "")
                neighbor_prompt = str(row.get('prompt_text', row.get(f'prompt_{prompt_type}', '')))
                full_prompt = build_full_analysis_prompt(summary, neighbor_prompt)
                refined = predictor.predict(full_prompt)

                results.append({
                    'paper_id': node_id,
                    'output_label': row.get('output_text', ''),
                    'summarize_text': summary,
                    'refined_text': refined,
                    'neighbor_prompt': neighbor_prompt
                })

            out_path = OUTPUT_PATH_TEMPLATE.format(type=prompt_type)
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"✅ 保存: {out_path}")

    else:
        if not os.path.exists(read_path):
            print(f"❌ 输入文件不存在: {read_path}")
            return
        prompt_type = 'fusion_knn'
        output_type = f'fusion_knn_thought_{thought}'
        output_path = OUTPUT_PATH_TEMPLATE.format(type=output_type)
        if os.path.exists(output_path):
            print(f"⚠️ 文件已存在: {output_path}")
            return output_path
        print(f"🚀 处理 Thought {thought}...")
        neighbors_df = pd.read_csv(read_path, dtype={'paper_id': str})
        current_results = []
        for _, neighbor_row in tqdm(neighbors_df.iterrows(), total=len(neighbors_df)):
            node_id = str(neighbor_row['paper_id'])
            summary_text = summary_dict.get(node_id, '')
            neighbor_prompt_text = str(neighbor_row.get('prompt_fusion_knn', neighbor_row.get('prompt_text', '')))
            if not neighbor_prompt_text:
                continue
            analysis_prompt = build_full_analysis_prompt(summary_text, neighbor_prompt_text)
            refined_text = predictor.predict(analysis_prompt)
            current_results.append({
                'paper_id': node_id,
                'output_label': neighbor_row.get('output_text', ''),
                'summarize_text': summary_text,
                'refined_text': refined_text,
                'neighbor_prompt': neighbor_prompt_text
            })
        if current_results:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(current_results).to_csv(output_path, index=False)
            print(f'✅ 结果保存成功: {output_path}')
        return output_path