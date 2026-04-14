import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union, Final

MODEL_PATH: Final[str] = "./all-mpnet-base-v2"
global_model = None


def load_file(data_name) -> Tuple[str, List[str]]:
    BASE_DIR = f"dataset/{data_name}/preprocessed"
    DEFAULT_FILES = [
        f"{data_name}_refined_text_fusion_knn_local_llm.csv",
        f"{data_name}_refined_text_original_knn_local_llm.csv",
        f"{data_name}_refined_text_structural_local_llm.csv",
    ]
    return BASE_DIR, DEFAULT_FILES


def load_sentence_transformer(model_path: str) -> Union[SentenceTransformer, None]:
    global global_model
    
    if global_model is not None:
        print("✅ Sentence Transformer 模型已加载，跳过重复初始化。")
        return global_model
        
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 正在加载 Sentence Transformer 模型到 {device}...")
        
        model = SentenceTransformer(model_path, device=device)
        print("✅ 模型加载成功！")
        
        global_model = model
        return model
    except Exception as e:
        print(f"❌ Sentence Transformer 模型加载失败，请检查路径和依赖: {e}")
        return None


def generate_embeddings(data_name, read_path, thought):
    base_directory, data = load_file(data_name)
    
    if thought == 1:
        input_files = data
    else:
        filename = os.path.basename(read_path)
        base_directory = os.path.dirname(read_path)
        input_files = [filename]

    emb_model_path: str = MODEL_PATH

    print("\n========================================================")
    print("--- 启动文本嵌入生成任务 (Sentence-BERT) ---")
    print("========================================================")

    model = load_sentence_transformer(emb_model_path)
    if model is None:
        return

    processed_count = 0
    for file in input_files:
        input_path = os.path.join(base_directory, file)
        suffix_t = "" if thought == 1 else f"_t{thought}"
        out1 = os.path.join(base_directory, file.replace(".csv", f"_summarize{suffix_t}_emb.pt"))
        out2 = os.path.join(base_directory, file.replace(".csv", f"_refined{suffix_t}_emb.pt"))

        if os.path.exists(out1) and os.path.exists(out2):
            print(f"⚠️ 输出文件已存在，跳过生成: {out1}, {out2}")
            continue
        
        if not os.path.exists(input_path):
            print(f"⚠️ 警告: 输入文件不存在，跳过: {input_path}")
            continue

        try:
            df = pd.read_csv(input_path)
            if "summarize_text" not in df.columns or "refined_text" not in df.columns:
                 print(f"❌ 错误: 文件 {file} 缺少必要的列，跳过。")
                 continue
                 
            texts1 = df["summarize_text"].astype(str).tolist()
            texts2 = df["refined_text"].astype(str).tolist()
            
            print(f"\nProcessing {file} (样本数: {len(df)})...")
            
            device = model.device
            emb1 = model.encode(texts1, batch_size=32, show_progress_bar=True,
                                convert_to_tensor=True, device=device)
            emb2 = model.encode(texts2, batch_size=32, show_progress_bar=True,
                                convert_to_tensor=True, device=device)
            
            os.makedirs(os.path.dirname(out1) or '.', exist_ok=True)
            torch.save(emb1, out1)
            torch.save(emb2, out2)
            print(f"   -> 总结文本嵌入保存到: {out1}")
            print(f"   -> 精炼文本嵌入保存到: {out2}")
            print(f"✅ Saved embeddings for {file} (Shape: {emb1.shape})")
            processed_count += 1
            
        except KeyError as e:
            print(f"❌ 错误: 文件 {file} 缺少必要的列 {e}，跳过。")
        except Exception as e:
            print(f"❌ 处理文件 {file} 时发生错误: {e}")

    print("\n🎯 所有文件处理完成！")
    print(f"成功处理的文件数量: {processed_count}")


def load_thought(data, device, thought) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    base_dir, csv_files = load_file(data)
    
    path_summarize = os.path.join(base_dir, csv_files[0].replace(".csv", "_summarize_emb.pt"))
    if thought == 1:
        path_fusion_refined = os.path.join(base_dir, csv_files[0].replace(".csv", "_refined_emb.pt"))
    else:
        suffix_t = f"_t{thought}"
        fusion_dir = os.path.join(base_dir, f"thought_{thought}")
        os.makedirs(fusion_dir, exist_ok=True)

        fname = csv_files[0].replace("_local_llm.csv", f"_thought_{thought}_local_llm_refined{suffix_t}_emb.pt")
        path_fusion_refined = os.path.join(fusion_dir, fname)
        
    path_original_refined = os.path.join(base_dir, csv_files[1].replace(".csv", "_refined_emb.pt"))
    path_structural_refined = os.path.join(base_dir, csv_files[2].replace(".csv", "_refined_emb.pt"))

    try:
        emb_fusion_refined = torch.load(path_fusion_refined).to(device)
        emb_original_refined = torch.load(path_original_refined).to(device)
        emb_structural_refined = torch.load(path_structural_refined).to(device)
        
        print(f"✅ 所有嵌入文件加载成功，并移动到 {device}。")
        
        return emb_fusion_refined, emb_original_refined, emb_structural_refined #emb_summarize, 
    
    except Exception as e:
        print(f"❌ 嵌入文件加载失败: {e}")
        print(f"检查路径是否存在：\n- summarize: {path_summarize}\n- fusion_refined: {path_fusion_refined}\n- original_refined: {path_original_refined}\n- structural_refined: {path_structural_refined}")
        return None, None, None, None



class FusionMLP(nn.Module):
    
    def __init__(self, hidden_dim, n_h, n_in, dropout, num_classes):
        super(FusionMLP, self).__init__()
        self.embed_dim = 768
        self.fc1 = nn.Linear(self.embed_dim * 3, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, n_h)
        self.fc3 = nn.Linear(n_h, n_in)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, emb1, emb2, emb3, emb4, emb0, label):
        
        x = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        
        x = self.fc1(x)
        x = F.relu(self.norm1(x))
        
        thought = self.fc2(self.dropout(x))
        
        new = torch.mul(F.normalize(thought, dim=1), F.normalize(emb0, dim=1))

        newer = self.fc3(new)
        logits = self.classifier(newer)
        # loss_cls = self.criterion(logits, label)

        return newer, logits
        
    def eva(self, emb1, emb2, emb3, emb4, emb0) -> torch.Tensor:
        x = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        x = self.fc1(x)
        x = F.relu(self.norm1(x))
        thought = self.fc2(self.dropout(x))
        new = torch.mul(F.normalize(thought, dim=1), F.normalize(emb0, dim=1))
        newer = self.fc3(new)
        logits = self.classifier(newer)

        return logits