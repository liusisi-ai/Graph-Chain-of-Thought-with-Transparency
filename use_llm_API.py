import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import warnings
from typing import Dict, Any, Tuple
from openai import OpenAI  # pip install openai

warnings.filterwarnings("ignore", category=UserWarning)

USE_API = True

API_CONFIG = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4.1",
    "temperature": 0.2
}

LOCAL_MODEL_PATH = "vicuna-7b-v1.5-16k"



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
                self.client = OpenAI(
                    api_key=API_CONFIG["api_key"],
                    base_url=API_CONFIG["base_url"]
                )
                print("✅ API 客户端就绪。")
            except Exception as e:
                print(f"❌ API 初始化失败: {e}")
        else:
            print(f"🚀 [Init] 正在加载本地模型: {local_path} ...")
            if not os.path.isdir(local_path):
                print(f"❌ 错误：模型路径不存在: {local_path}")
                return

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"检测到的设备: {self.device}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                print(f"✅ 本地模型加载成功!")
            except Exception as e:
                print(f"❌ 本地模型加载失败: {e}")

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