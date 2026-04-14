import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import warnings
from typing import Dict, Any, Tuple
import os.path as osp

warnings.filterwarnings("ignore", category=UserWarning)

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


global_tokenizer, global_model, global_device = None, None, None


def load_local_llm(model_path: str):
    global global_tokenizer, global_model, global_device

    if global_model is not None:
        print("✅ LLM 已加载，跳过重复初始化。")
        return global_tokenizer, global_model, global_device

    if not os.path.isdir(model_path):
        print(f"❌ 错误：模型路径不存在或不是一个目录: {model_path}")
        return None, None, None

    try:
        print("🚀 正在加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("🚀 正在加载模型...")
        start_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"检测到的设备: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.eval()

        end_time = time.time()
        print(f"✅ 模型加载成功！耗时: {end_time - start_time:.2f} 秒, 设备: {model.device}")

        global_tokenizer, global_model, global_device = tokenizer, model, device
        return tokenizer, model, device

    except Exception as e:
        print(f"❌ 模型加载失败，请检查您的文件和环境配置：{e}")
        return None, None, None


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


def ask_llm_local(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.device) -> str:
    if not model or not tokenizer:
        return "Error: Local LLM not initialized."

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000
    ).to(device)

    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )

        generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        start_tag = "ASSISTANT:"
        if start_tag in generated_text:
            response_text = generated_text.split(start_tag, 1)[-1].strip()
        else:
            response_text = generated_text.strip()

        return response_text

    except Exception as e:
        print(f"LLM 推理发生错误: {e}")
        return "Error during local model generation."


def run_llm_inference(
        ROOT_PATH,
        DATASET_NAME,
        enable_structural: bool,
        enable_original_knn: bool,
        read_path: str,
        thought,
        epoch  # 新增 epoch 参数
):


    print("==========================================================")
    print(f"--- 启动 LLM 推理任务 (Thought = {thought}, Epoch = {epoch}) ---")
    print(f"--- 启用 Structural: {enable_structural}, 启用 Original KNN: {enable_original_knn} ---")
    print("==========================================================")

    PROMPT_FILES, NODE_INFO_PATH, OUTPUT_PATH_TEMPLATE, FIXED_SUMMARY_PATH = load_file(DATASET_NAME, epoch)
    if not os.path.exists(FIXED_SUMMARY_PATH):
        print(f"📢 [Pre-check] 摘要文件不存在，开始生成: {FIXED_SUMMARY_PATH}")
        if not os.path.exists(NODE_INFO_PATH):
            print(f"❌ 节点信息文件不存在: {NODE_INFO_PATH}")
            return

        tokenizer, model, device = load_local_llm(LOCAL_MODEL_PATH)
        node_info_df = pd.read_csv(NODE_INFO_PATH)

        sum_results = []
        for _, row in tqdm(node_info_df.iterrows(), total=len(node_info_df), desc="Generating Summaries"):
            title = str(row.get('title', "Unknown Title"))
            abstract = str(row.get('abstract', row.get('input_text', "No content available.")))
            summary = ask_llm_local(build_summarize_prompt(title, abstract), tokenizer, model, device)
            sum_results.append({
                'paper_id': str(row['paper_id']),
                'summarize_text': summary
            })

        pd.DataFrame(sum_results).to_csv(FIXED_SUMMARY_PATH, index=False)
        print("✅ 全局摘要生成完毕。")
    print(f"-> Loading summaries from {FIXED_SUMMARY_PATH}")
    summary_df = pd.read_csv(FIXED_SUMMARY_PATH, dtype={'paper_id': str})
    summary_dict = dict(zip(summary_df['paper_id'], summary_df['summarize_text']))
    tokenizer, model, device = load_local_llm(LOCAL_MODEL_PATH)
    if not model:
        print("❌ LLM 未成功加载，无法进行推理。")
        return

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
            if os.path.exists(emb_path):
                print(f"⚠️ {key} 的 Embedding 已存在于 Epoch {epoch} 目录，跳过: {emb_path}")
                continue
            csv_out_path = OUTPUT_PATH_TEMPLATE.format(type=key)
            if os.path.exists(csv_out_path):
                print(f"⚠️ {key} 的 CSV 结果已存在于 Epoch {epoch} 目录，跳过: {csv_out_path}")
                continue

            files_to_process[key] = PROMPT_FILES_MAP[key]

        if not files_to_process:
            print(f"✅ Epoch {epoch} (Thought=1) 的所有任务均已完成，无需推理。")
            return

        for prompt_type, fname in files_to_process.items():
            full_path = os.path.join(PROMPT_DIR, fname)
            print(f"\n=== 处理 {prompt_type}: {fname} ===")
            if not os.path.exists(full_path):
                print(f"跳过：文件不存在 -> {full_path}")
                continue

            results = []
            prompt_df = pd.read_csv(full_path, dtype={'paper_id': str})

            for _, row in tqdm(prompt_df.iterrows(), total=len(prompt_df)):
                node_id = str(row['paper_id'])
                summary = summary_dict.get(node_id, "")
                if not summary:
                    pass

                neighbor_prompt = str(row.get('prompt_text', row.get(f'prompt_{prompt_type}', '')))
                refined = ask_llm_local(build_full_analysis_prompt(summary, neighbor_prompt), tokenizer, model, device)

                results.append({
                    'paper_id': node_id,
                    'output_label': row.get('output_text', ''),
                    'summarize_text': summary,
                    'refined_text': refined,
                    'neighbor_prompt': neighbor_prompt
                })

            if not results:
                print(f"跳过：{prompt_type} 无生成结果")
                continue

            out_path = OUTPUT_PATH_TEMPLATE.format(type=prompt_type)
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"✅ 保存成功：{out_path} (总样本 {len(results)})")

        return

    else:
        if not os.path.exists(read_path):
            print(f"❌ 错误：Thought > 1 模式下的输入文件不存在: {read_path}")
            return

        prompt_type = 'fusion_knn'
        output_type = f'fusion_knn_thought_{thought}'
        output_path = OUTPUT_PATH_TEMPLATE.format(type=output_type)

        if os.path.exists(output_path):
            print(f"⚠️ 文件已存在，跳过推理: {output_path}")
            return output_path

        print(f"\n==================================================")
        print(f"🚀 开始处理数据集: {prompt_type} (Thought {thought}, Epoch {epoch}, 文件: {read_path})")
        print(f"==================================================")

        neighbors_df = pd.read_csv(read_path, dtype={'paper_id': str})
        current_results = []

        for _, neighbor_row in tqdm(neighbors_df.iterrows(), total=len(neighbors_df),
                                    desc=f"Processing {prompt_type}"):

            node_id = str(neighbor_row['paper_id'])
            summary_text = summary_dict.get(node_id, '')
            neighbor_prompt_text = str(neighbor_row.get('prompt_fusion_knn', neighbor_row.get('prompt_text', '')))

            if not neighbor_prompt_text:
                continue

            analysis_prompt = build_full_analysis_prompt(summary_text, neighbor_prompt_text)
            refined_text = ask_llm_local(analysis_prompt, tokenizer, model, device)

            current_results.append({
                'paper_id': node_id,
                'output_label': neighbor_row.get('output_text', ''),
                'summarize_text': summary_text,
                'refined_text': refined_text,
                'neighbor_prompt': neighbor_prompt_text
            })

        if not current_results:
            print(f"⚠️ 警告：{prompt_type} (Thought {thought}) 类型没有生成任何结果，跳过保存。")
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df = pd.DataFrame(current_results)
            output_df.to_csv(output_path, index=False)
            print(f'✅ {output_type.upper()} 结果保存成功 (总计 {len(output_df)} 个样本): {output_path}')

        print("\n🎉 LLM 推理任务执行完毕。")
        return output_path