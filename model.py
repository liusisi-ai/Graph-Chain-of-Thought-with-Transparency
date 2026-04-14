import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from preprocess import *
from use_llm_API import *
from utils import *
from config import *

class ConditionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.):
        super(ConditionNet, self).__init__()
        self.input_fc = nn.Linear(768*2, hidden_dim)
        self.hidden_fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        #x = F.elu(self.input_fc(x))
        x = self.input_fc(x)
        for layer in self.hidden_fc:
            x = F.elu(layer(x))
        output = self.output_fc(x)
        return output


class FusionMLP(nn.Module):
    def __init__(self, hidden_dim, n_h, n_in, dropout, num_classes, num_nodes, data=None, think_layer_num=1):
        super(FusionMLP, self).__init__()
        self.think_layer_num = think_layer_num
        self.condition_layers = nn.ModuleList(
            [ConditionNet(hidden_dim, n_h, n_h, 1) for _ in range(think_layer_num)])

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.cached_thoughts = None
        self.res_weights = nn.Parameter(torch.ones(think_layer_num))

        self.learnable_x = nn.Parameter(torch.empty(num_nodes, n_in))
        self.is_initialized = False
        self.input_proj = nn.Linear(n_in, 128) if n_in != 128 else nn.Identity()

        if data is not None and hasattr(data, 'raw_texts'):
            init_token_map(data)

    def forward(self, gcn, edge_index, x, update_thought, epoch, a):
        x = self.input_proj(x)
        origin_x = x.clone()
        current_pass_thoughts = []

        for i, condition_net in enumerate(self.condition_layers):
            embed_1 = gcn.convs[0](x, edge_index)
            embed_2 = gcn.convs[1](embed_1, edge_index) + embed_1
            if update_thought or self.cached_thoughts is None:
                thought = self.use_thought(embed_2, i, epoch, a)
                current_pass_thoughts.append(thought.detach())

            else:
                if self.cached_thoughts is None or i >= len(self.cached_thoughts):
                    # 如果发生这种情况，说明之前的缓存逻辑有错，强制重新计算或报错
                    print(f"警告: Cache 失效或索引 {i} 越界，尝试重新计算...")
                    thought = self.use_thought(embed_2, i, epoch, a)
                    current_pass_thoughts.append(thought.detach())
                else:
                    thought = self.cached_thoughts[i]
                    current_pass_thoughts.append(thought)
            prompt = condition_net(thought)
            thought_emb = prompt * origin_x
            x = origin_x + thought_emb
        if update_thought or self.cached_thoughts is None:
            self.cached_thoughts = current_pass_thoughts
        x = self.learnable_x*origin_x

        embed = gcn(x, edge_index)
        logits = self.classifier(embed)

        return x, logits

    def use_thought(self, x, thought_counter, epoch,a):
        num_thoughts = thought_counter + 1
        print(f"更新思考 #{num_thoughts} (Epoch {epoch})")
        create_path(x, num_thoughts, epoch)
        p1 = generate_prompt(num_thoughts, False, False, epoch)
        p2 = use_llm(False, False, p1, num_thoughts, epoch)
        generate_embeddings(DATASET_NAME, p2, num_thoughts, epoch)
        device = x.device
        emb1, emb2, emb3 = load_thought(DATASET_NAME, device, num_thoughts, epoch)
        if emb1 is None:
            raise RuntimeError(f"Failed to load embeddings for Thought {num_thoughts} at Epoch {epoch}")
        emb = torch.cat([emb1,emb3], dim=1)
        return emb