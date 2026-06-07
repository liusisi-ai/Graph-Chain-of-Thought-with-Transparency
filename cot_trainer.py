"""CoT training stage.

Architecture
============

    (Graph)                              (CoT text)
       |                                     |
   SAGE encoder  ──►  graph_h  ◄── ConditionNet(text_emb) * cot_weight
                                     (injected back each thought)
       |
   Projector (n_h → LLM hidden dim)
       |                               ┌───────────────────────────┐
       ├─► soft tokens (one per node) ─┤  Frozen / LoRA LLM        │
                                       │  input = soft_tokens +    │
                                       │          text prompt ids  │
                                       └───────────────────────────┘
                                                     │
                                    masked SFT loss (only on answer tokens)

One thought == one LLM forward + one GNN forward.  The very first GNN forward
(before any LLM call) does NOT count as a thought.  After ``NUM_CLUSTER_THOUGHTS``
iterations the final answer is produced.

Training loop
=============

* Phase A — ``WARMUP_EPOCHS``: train only the projector (and optionally the
  LoRA adapters if enabled) so the graph hidden space aligns with the LLM
  embedding space.  No CoT yet.

* Phase B — starting at ``COT_START_EPOCH``: activate CoT.  Each mini-batch:
    1. Encode graph with the current ``edge_index`` → ``graph_h``
    2. For ``T`` thoughts:
         - Encode historical CoT text with sentence-BERT → condition vector
         - ``graph_h = graph_h + condition_vector * cot_weight``
         - Sample an LLM response (teacher-forced against ground truth)
         - Append to history (this counts as a thought)
    3. Build a supervised prompt ``P`` ending in the final answer ``A``
    4. Mask ``P`` tokens, only compute CE loss on ``A`` tokens.
    5. Sum per-sample losses and divide by ``batch_size``.

Evaluation is zero-shot: no labels are given inside the prompt, the LLM is
asked to answer directly.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    DATASET_NAME,
    LOCAL_LLM_PATH,
    SENTENCE_MODEL_PATH,
    FIRST_MODEL_PATH,
    CONDITION_TEXT_DIM,
    LLM_HIDDEN_DIM,
    COT_WEIGHT,
    WARMUP_EPOCHS,
    COT_START_EPOCH,
    USE_LORA,
    USE_SOFT_TOKEN,
    USE_CONSTRAINED_DECODING,
)
from graphsage import SAGELayers, PrePromptSAGE


# ── ConditionNet ─────────────────────────────────────────────


class ConditionNet(nn.Module):
    """Maps sentence-BERT CoT text embedding → graph-hidden-dim condition vector."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── CoT Graph-LLM module ─────────────────────────────────────


class CoTGraphLLM(nn.Module):
    """
    Wraps
        * SAGE encoder (loaded from first_model)
        * Projector  (graph hidden → LLM hidden)
        * ConditionNet (CoT text → graph hidden, additive)
        * Vicuna-7B LLM (frozen or LoRA)
    """

    def __init__(
        self,
        n_in: int,
        n_h: int,
        num_layers_num: int,
        dropout: float,
        projector_out_dim: int = LLM_HIDDEN_DIM,
        cot_text_dim: int = CONDITION_TEXT_DIM,
        cot_weight: float = COT_WEIGHT,
        first_model_path: str = FIRST_MODEL_PATH,
        local_llm_path: str = LOCAL_LLM_PATH,
        use_lora: bool = USE_LORA,
    ):
        super().__init__()
        self.n_h = n_h

        # ── Graph encoder from first_model ──
        self.sage = SAGELayers(n_in, n_h, num_layers_num, dropout)
        self.projector = nn.Sequential(
            nn.Linear(n_h, projector_out_dim),
            nn.GELU(),
            nn.Linear(projector_out_dim, projector_out_dim),
        )
        self._load_first_model(first_model_path)

        # ── ConditionNet (text → graph hidden) ──
        self.condition_net = ConditionNet(cot_text_dim, n_h, n_h)
        self.cot_weight = nn.Parameter(torch.tensor(float(cot_weight)))

        # ── LLM (frozen or LoRA) ──
        self.tokenizer, self.llm = self._load_llm(local_llm_path, use_lora)
        # Cache LLM input-embedding matrix for soft prompt insertion
        self.llm_hidden_dim = self.llm.get_input_embeddings().embedding_dim

        # ── Sentence transformer for CoT text encoding ──
        self.sent_model = None  # lazy load; see encode_cot_text

    # ── loaders ────────────────────────────────────────────

    def _load_first_model(self, path: str):
        if not os.path.exists(path):
            print(f"⚠️ first_model not found at {path} – using random init.")
            return
        state = torch.load(path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        loaded = [k for k in state.keys() if k.startswith(("sage.", "projector."))]
        print(f"✅ Loaded {len(loaded)} tensors from first_model.")
        if unexpected:
            print(f"   (skipped unexpected keys: {len(unexpected)})")

    @staticmethod
    def _raise_llm_load_error(exc: Exception, path: str):
        """Inspect the raised exception and give actionable hints."""
        msg = str(exc).lower()
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        hint_lines = []
        if any(k in msg for k in (
            "couldn't connect", "max retries exceeded", "timed out",
            "connection error", "name resolution", "localentrynotfounderror",
            "we couldn't connect",
        )):
            hint_lines += [
                "💡 网络无法访问 HuggingFace。",
                f"   当前 HF_ENDPOINT = {endpoint}",
                "   可选方案：",
                "   1) 镜像未生效 → 终端先 export 再运行：",
                "        export HF_ENDPOINT=https://hf-mirror.com",
                "        unset  HF_DISABLE_MIRROR",
                "        python main.py",
                "   2) 镜像也不通 → 离线预下载到 /root/autodl-tmp/：",
                "        pip install -U huggingface_hub modelscope",
                "        export HF_ENDPOINT=https://hf-mirror.com",
                "        huggingface-cli download lmsys/vicuna-7b-v1.5-16k \\",
                "            --local-dir /root/autodl-tmp/vicuna-7b-v1.5-16k",
                "        huggingface-cli download sentence-transformers/all-mpnet-base-v2 \\",
                "            --local-dir /root/autodl-tmp/all-mpnet-base-v2",
                "      代码会自动检测到本地副本，不再访问网络。",
                "   3) 或改用 OpenAI 兼容 API：",
                "        export USE_API=1; export OPENAI_API_KEY=sk-...; pip install openai",
            ]
        elif "out of memory" in msg or "cuda oom" in msg:
            hint_lines += [
                "💡 GPU 显存不足。Vicuna-7B fp16 约需 14 GB。",
                "   1) 启用 4-bit：export LLM_4BIT=1; pip install bitsandbytes==0.41.3",
                "   2) 升级 24 GB 卡 (3090/4090/A5000)",
            ]
        elif "no module named 'bitsandbytes'" in msg:
            hint_lines += [
                "💡 pip install bitsandbytes==0.41.3",
            ]
        if hint_lines:
            raise RuntimeError(f"❌ LLM 加载失败 ({path}): {exc}\n" + "\n".join(hint_lines)) from exc
        raise

    def _load_llm(self, path: str, use_lora: bool):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        is_hf_id = not os.path.isdir(path)
        src = "HuggingFace Hub" if is_hf_id else "local dir"
        print(f"🚀 Loading LLM from {src}: {path} ...")
        if is_hf_id:
            print(f"   (HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'default')})")

        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception as e:
            self._raise_llm_load_error(e, path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # ── 4-bit fallback when free GPU memory is too low for fp16 ──
        model_kwargs = dict(
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        use_4bit = os.environ.get("LLM_4BIT", "auto")
        if use_4bit == "auto":
            free = (
                torch.cuda.mem_get_info()[0] / 1024 ** 3
                if torch.cuda.is_available() and hasattr(torch.cuda, "mem_get_info")
                else 0.0
            )
            use_4bit = torch.cuda.is_available() and free < 12.6
        else:
            use_4bit = use_4bit.lower() in ("1", "true", "yes")
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
                print(f"⚠️ Cannot enable 4-bit ({e}); falling back to fp16.")
                model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype

        try:
            llm = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except Exception as e:
            self._raise_llm_load_error(e, path)

        # Always freeze the base LLM
        for p in llm.parameters():
            p.requires_grad = False

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj"],
                )
                llm = get_peft_model(llm, lora_cfg)
                llm.print_trainable_parameters()
                print("✅ LoRA adapters attached.")
            except Exception as e:
                print(f"⚠️ LoRA unavailable ({e}); LLM kept fully frozen.")
        else:
            print("✅ LLM fully frozen (no LoRA).")

        llm.eval()  # base is eval; LoRA adapters still trainable
        return tokenizer, llm

    def _load_sentence_model(self):
        if self.sent_model is not None:
            return self.sent_model
        from sentence_transformers import SentenceTransformer
        device = next(self.parameters()).device
        self.sent_model = SentenceTransformer(SENTENCE_MODEL_PATH, device=str(device))
        return self.sent_model

    # ── components ─────────────────────────────────────────

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        condition_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        graph_h : [N, n_h] graph hidden after optional CoT injection
        soft_tokens : [N, llm_hidden] projected to LLM space
        """
        h = self.sage(x, edge_index)
        if condition_vector is not None:
            if condition_vector.dim() == 1:
                condition_vector = condition_vector.unsqueeze(0)
            if condition_vector.size(0) == 1 and h.size(0) > 1:
                condition_vector = condition_vector.expand_as(h)
            h = h + condition_vector * self.cot_weight
        soft_tokens = self.projector(h)
        return h, soft_tokens

    def encode_cot_text(self, texts: List[str]) -> torch.Tensor:
        """Sentence-BERT encode CoT text (one string per node, or a global one)."""
        if not texts:
            return torch.zeros(
                (1, CONDITION_TEXT_DIM), device=next(self.parameters()).device
            )
        model = self._load_sentence_model()
        with torch.no_grad():
            emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return emb.to(next(self.parameters()).device)

    # ── LLM forward with soft-prompt + answer-masked CE loss ──

    def sft_forward_masked(
        self,
        prompts: List[str],
        answers: List[str],
        soft_tokens: Optional[torch.Tensor] = None,
        max_length: int = 1024,
    ) -> torch.Tensor:
        """Compute CE loss only on answer tokens (Vicuna SFT style).

        Parameters
        ----------
        prompts : list of strings (length B)
        answers : list of strings (length B)
        soft_tokens : optional tensor [B, 1, llm_hidden] prepended to input embeds.
        """
        assert len(prompts) == len(answers)
        llm = self.llm
        tok = self.tokenizer
        device = next(llm.parameters()).device

        total_loss = 0.0
        for i, (p, a) in enumerate(zip(prompts, answers)):
            full = p + a
            full_ids = tok(
                full, return_tensors="pt", truncation=True, max_length=max_length
            ).input_ids.to(device)
            prompt_ids = tok(
                p, return_tensors="pt", truncation=True, max_length=max_length
            ).input_ids.to(device)
            plen = prompt_ids.size(1)

            # Build labels: mask prompt tokens (-100), keep answer tokens
            labels = full_ids.clone()
            labels[:, :plen] = -100

            if soft_tokens is not None:
                # Prepend soft token(s) as virtual embeddings
                embed_layer = llm.get_input_embeddings()
                text_embeds = embed_layer(full_ids)            # [1, L, d]
                st_i = soft_tokens[i].to(
                    device=text_embeds.device,
                    dtype=text_embeds.dtype,
                ).unsqueeze(0)                                  # [1, k, d] or [1, d]
                if st_i.dim() == 2:
                    st_i = st_i.unsqueeze(1)                   # [1, 1, d]
                inputs_embeds = torch.cat([st_i, text_embeds], dim=1)

                # Pad labels on left so they align with inputs_embeds
                pad = torch.full(
                    (1, st_i.size(1)), -100, dtype=labels.dtype, device=device
                )
                labels = torch.cat([pad, labels], dim=1)

                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                out = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            else:
                attention_mask = torch.ones_like(full_ids, dtype=torch.long)
                out = llm(input_ids=full_ids, attention_mask=attention_mask, labels=labels)

            total_loss = total_loss + out.loss

        return total_loss / max(1, len(prompts))

    @torch.no_grad()
    def generate(self, prompt: str, soft_token: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 32) -> str:
        tok = self.tokenizer
        device = next(self.llm.parameters()).device
        encoded = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        if soft_token is not None:
            embed_layer = self.llm.get_input_embeddings()
            text_embeds = embed_layer(ids)
            st = soft_token.to(device=text_embeds.device, dtype=text_embeds.dtype)
            if st.dim() == 1:
                st = st.unsqueeze(0).unsqueeze(0)
            elif st.dim() == 2:
                st = st.unsqueeze(1)
            inputs_embeds = torch.cat([st, text_embeds], dim=1)
            soft_mask = torch.ones(
                (attention_mask.size(0), st.size(1)),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            generation_mask = torch.cat([soft_mask, attention_mask], dim=1)
            out = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=generation_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        else:
            out = self.llm.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    @torch.no_grad()
    def score_candidates(
        self,
        prompt: str,
        candidates: List[str],
        soft_token: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Per-candidate length-normalised log-likelihood under the LLM.

        Computes ``mean_t log p(cand_t | prompt, cand_<t)`` for every
        ``cand`` in ``candidates`` *in a single batched forward* and returns
        the list of mean log-probs.  ``argmax`` over the returned list is
        the constrained-decoding prediction; it is strictly stronger than
        free generation + substring matching for closed-set labels.
        """
        tok = self.tokenizer
        device = next(self.llm.parameters()).device
        embed_layer = self.llm.get_input_embeddings()

        # Match BPE merges of the natural continuation: if the prompt ends
        # without trailing whitespace, prefix " " to each candidate so the
        # first token aligns the way it would in real generation.
        space_prefix = "" if prompt.endswith((" ", "\t", "\n")) else " "

        prompt_enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        prompt_ids = prompt_enc.input_ids.to(device)
        prompt_attn = prompt_enc.attention_mask.to(device)
        P = prompt_ids.size(1)
        K = len(candidates)
        if K == 0:
            return []

        cand_ids_list = [
            tok(space_prefix + c, return_tensors="pt",
                add_special_tokens=False).input_ids.to(device)
            for c in candidates
        ]
        cand_lens = [c.size(1) for c in cand_ids_list]
        max_cand_len = max(max(cand_lens), 1)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
            tok.eos_token_id if tok.eos_token_id is not None else 0
        )

        # Build batched (K, P + max_cand_len) input ids + attention mask
        prompt_ids_b = prompt_ids.expand(K, -1)
        prompt_attn_b = prompt_attn.expand(K, -1)
        cand_padded = torch.full(
            (K, max_cand_len), pad_id, device=device, dtype=prompt_ids.dtype,
        )
        cand_attn = torch.zeros(
            (K, max_cand_len), device=device, dtype=prompt_attn.dtype,
        )
        for i, cids in enumerate(cand_ids_list):
            L = cids.size(1)
            if L == 0:
                continue
            cand_padded[i, :L] = cids[0]
            cand_attn[i, :L] = 1
        full_ids = torch.cat([prompt_ids_b, cand_padded], dim=1)
        full_attn = torch.cat([prompt_attn_b, cand_attn], dim=1)
        full_embeds = embed_layer(full_ids)

        # Optional soft-token prepend (same one for all K candidates)
        if soft_token is not None:
            st = soft_token.to(device=full_embeds.device, dtype=full_embeds.dtype)
            if st.dim() == 1:
                st = st.unsqueeze(0).unsqueeze(0)
            elif st.dim() == 2:
                st = st.unsqueeze(1)
            st_b = st.expand(K, -1, -1)
            full_embeds = torch.cat([st_b, full_embeds], dim=1)
            full_attn = torch.cat([
                torch.ones(
                    (K, st_b.size(1)),
                    dtype=full_attn.dtype, device=device,
                ),
                full_attn,
            ], dim=1)
            prompt_offset = st_b.size(1) + P
        else:
            prompt_offset = P

        out = self.llm(inputs_embeds=full_embeds, attention_mask=full_attn)
        # Cast to float32 before softmax for numerical stability under fp16/4bit
        log_probs = F.log_softmax(out.logits.float(), dim=-1)

        scores: List[float] = []
        for i, L in enumerate(cand_lens):
            if L == 0:
                scores.append(float("-inf"))
                continue
            # Predicted distribution at position t-1 generates token t.
            # Candidate tokens occupy positions [prompt_offset .. prompt_offset+L).
            idx_pos = torch.arange(prompt_offset - 1, prompt_offset - 1 + L,
                                   device=device)
            tok_ids = cand_ids_list[i][0]               # [L]
            ll = log_probs[i, idx_pos, tok_ids].sum().item()
            scores.append(ll / L)
        return scores


# ── training loop ────────────────────────────────────────────


def build_sft_samples(
    data,
    candidates: List[str],
    title_dict: Dict[str, str],
    raw_texts,
    mask: Optional[torch.Tensor] = None,
    dataset_name: str = DATASET_NAME,
) -> List[Tuple[int, str, str]]:
    """
    Build (node_idx, prompt, answer) triples for Vicuna-style SFT.
    ``answer`` is the ground-truth label string followed by EOS.
    """
    samples = []
    indices = (
        torch.nonzero(mask).squeeze(-1).tolist()
        if mask is not None
        else list(range(data.num_nodes))
    )
    for idx in indices:
        if not hasattr(data, "y") or data.y is None:
            continue
        label_idx = int(data.y[idx].item())
        if not (0 <= label_idx < len(candidates)):
            continue
        answer = candidates[label_idx]
        nid = raw_texts[idx] if raw_texts is not None else str(idx)
        title = title_dict.get(str(nid), "Unknown")
        domain_key = str(dataset_name).lower()
        entity = "product" if domain_key in {
            "children", "computers", "history", "photo", "sports",
        } else "paper"
        question = (
            f"Which e-commerce category does this {entity} belong to?"
            if entity == "product"
            else "Which research category does this paper belong to?"
        )

        prompt = (
            f"Title: {title}. "
            f"{question} "
            f"Options: {', '.join(candidates)}. Answer: "
        )
        samples.append((idx, prompt, f"{answer}</s>"))
    return samples


def train_cot(
    model: CoTGraphLLM,
    data,
    candidates: List[str],
    title_dict: Dict[str, str],
    cluster_llm_text: Dict[int, str],
    num_epochs: int,
    num_thoughts: int,
    batch_size: int = 4,
    lr: float = 1e-4,
    train_mask: Optional[torch.Tensor] = None,
    cluster_labels: Optional[np.ndarray] = None,
    dataset_name: str = DATASET_NAME,
) -> None:
    """
    Two-phase training:
      Epochs 1 … WARMUP_EPOCHS-1 : projector warmup only (no CoT)
      Epochs  ≥ COT_START_EPOCH  : full CoT with condition injection
    """
    import torch.optim as optim

    device = next(model.parameters()).device
    data = data.to(device)

    # Collect trainable parameters
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainables, lr=lr)

    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None
    all_samples = build_sft_samples(
        data, candidates, title_dict, raw_texts, train_mask, dataset_name
    )
    print(f"📊 Total SFT samples: {len(all_samples)}")

    history: List[str] = []  # accumulated CoT text across thoughts
    current_edge_index = data.edge_index

    for epoch in range(1, num_epochs + 1):
        use_cot = epoch >= COT_START_EPOCH
        phase = "CoT" if use_cot else "Warm-up"
        print(f"\n── Epoch {epoch}/{num_epochs} ({phase}) ──")

        # ── 1. build condition vector from accumulated CoT history ──
        condition_vector = None
        if use_cot and history:
            text_embs = model.encode_cot_text(history[-num_thoughts:])
            # mean-pool over historical thoughts → single global condition
            pooled = text_embs.mean(dim=0, keepdim=True)
            condition_vector = model.condition_net(pooled).squeeze(0)

        # ── 2. GNN encode (with or without condition) ──
        graph_h, soft_tokens = model.encode_graph(
            data.x, current_edge_index, condition_vector
        )

        # ── 3. mini-batch SFT with answer masking ──
        np.random.shuffle(all_samples)
        total = 0.0
        n_batches = 0
        for b in tqdm(
            range(0, len(all_samples), batch_size),
            desc=f"Epoch {epoch}",
            leave=False,
        ):
            batch = all_samples[b : b + batch_size]
            idxs, prompts, answers = zip(*batch)
            st = soft_tokens[list(idxs)]
            loss = model.sft_forward_masked(
                list(prompts), list(answers), soft_tokens=st
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1

        print(f"   mean SFT loss: {total / max(1, n_batches):.4f}")

        # ── 4. if in CoT phase, run ONE "thought" to expand history ──
        #        (one LLM call + one GNN re-encode == one thought)
        if use_cot and cluster_labels is not None and len(history) < num_thoughts:
            # Use cluster-center LLM texts already produced upstream as the
            # next thought; this avoids an extra heavy LLM call inside training.
            for k, text in cluster_llm_text.items():
                if text and text not in history:
                    history.append(text)
                    break
            # graph is re-encoded next epoch with the new condition vector

    print("✅ CoT training finished.")


# ── zero-shot evaluation ─────────────────────────────────────


def zero_shot_eval(
    model: CoTGraphLLM,
    data,
    candidates: List[str],
    title_dict: Dict[str, str],
    test_mask: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Zero-shot classification: NO ground-truth in the prompt."""
    device = next(model.parameters()).device
    ei = (edge_index if edge_index is not None else data.edge_index).to(device)
    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None

    with torch.no_grad():
        _, soft_tokens = model.encode_graph(data.x.to(device), ei, None)

    indices = (
        torch.nonzero(test_mask).squeeze(-1).tolist()
        if test_mask is not None
        else list(range(data.num_nodes))
    )

    results = []
    correct = 0
    total = 0
    for idx in tqdm(indices, desc="Zero-shot eval"):
        nid = raw_texts[idx] if raw_texts is not None else str(idx)
        title = title_dict.get(str(nid), "Unknown")
        prompt = (
            f"Title: {title}. "
            f"Which arXiv CS sub-category does this paper belong to? "
            f"Options: {', '.join(candidates)}. Answer: "
        )
        pred_text = model.generate(prompt, soft_token=soft_tokens[idx])
        pred = _match_candidate(pred_text, candidates)

        true_label = None
        if hasattr(data, "y") and data.y is not None:
            li = int(data.y[idx].item())
            if 0 <= li < len(candidates):
                true_label = candidates[li]
        if true_label is not None:
            total += 1
            if pred == true_label:
                correct += 1

        results.append(
            {
                "node_idx": idx,
                "node_id": str(nid),
                "title": title,
                "predicted": pred,
                "true_label": true_label,
                "raw": pred_text,
            }
        )

    save_dir = f"dataset/{DATASET_NAME}/cluster"
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(save_dir, f"{DATASET_NAME}_zeroshot_results.csv"), index=False
    )
    acc = correct / total if total else 0.0
    print(f"✅ Zero-shot accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}


class _SafeDict(dict):
    """dict that returns "" for missing keys so template.format_map is forgiving."""

    def __missing__(self, key):  # pragma: no cover
        return ""


def _build_neighbor_text(
    node_idx: int,
    edge_index: torch.Tensor,
    raw_texts,
    title_dict: Dict[str, str],
    max_neighbors: int = 5,
) -> str:
    """Concatenate up to ``max_neighbors`` neighbor titles into a short string."""
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    mask = src == node_idx
    neigh = dst[mask].tolist()[:max_neighbors]
    parts = []
    for nb in neigh:
        key = str(raw_texts[nb]) if raw_texts is not None else str(nb)
        parts.append(title_dict.get(key, "Unknown"))
    return "; ".join(parts) if parts else "(no neighbor)"


def _match_candidate(text: str, candidates: List[str]) -> str:
    """Find the candidate label that appears earliest in ``text``.

    Important: when *no* candidate string is found we return an empty string
    rather than ``candidates[0]``.  Returning the first candidate biases the
    accuracy estimate (e.g. cora's "Case_Based" is dominant) and hides
    underlying problems in the prompt / generation.  Empty answers count as
    incorrect, which is the honest behaviour during tuning.
    """
    if not text or not candidates:
        return ""
    # Try both the canonical form and a "_" → " " normalised form.
    aliases = []
    for c in candidates:
        aliases.append((c, c.lower()))
        if "_" in c:
            aliases.append((c, c.replace("_", " ").lower()))
        if " " in c:
            aliases.append((c, c.replace(" ", "_").lower()))
    low = text.lower()
    best, best_pos = "", float("inf")
    for canon, alias in aliases:
        pos = low.find(alias)
        if 0 <= pos < best_pos:
            best, best_pos = canon, pos
    return best


# ═══════════════════════════════════════════════════════════════
#  Zero-shot training via PCA anchors + cross-domain inference
# ═══════════════════════════════════════════════════════════════
#
# Motivation
# ----------
# Supervised SFT on one domain overfits.  Instead, we pre-align the graph
# encoder's output space with the PCA subspace of the LLM's input-embedding
# matrix — these principal directions capture the dominant structure of the
# LLM token space and act as universal *anchors*.  Because the anchors are
# computed from the LLM only (domain-agnostic), the encoder acquires a truly
# transferable mapping: train once on a single source domain, then switch
# the instruction file to a new domain for zero-shot inference.


def compute_llm_embedding_pca_anchors(llm, k: int = 256) -> torch.Tensor:
    """Top-``k`` principal components of the LLM input-embedding matrix."""
    W = llm.get_input_embeddings().weight.detach().float().cpu().numpy()
    Wc = W - W.mean(axis=0, keepdims=True)
    # SVD: Wc = U · S · V^T ;  rows of V^T are principal directions in R^d
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    anchors = Vt[:k]                      # [k, d]
    return torch.tensor(anchors, dtype=torch.float32)


def get_llm_text_embeddings(
    llm, tokenizer, texts: List[str], max_length: int = 64
) -> torch.Tensor:
    """Encode each text via LLM input-embedding layer, mean-pool token dim."""
    device = next(llm.parameters()).device
    embed_layer = llm.get_input_embeddings()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(
                t, return_tensors="pt", truncation=True, max_length=max_length
            ).input_ids.to(device)
            emb = embed_layer(ids).float().mean(dim=1)   # [1, d]
            out.append(emb.cpu())
    return torch.cat(out, dim=0)


def pca_alignment_loss(
    h_proj: torch.Tensor,
    target_emb: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE between encoder output and target, after projection onto anchors.

    * ``h_proj``    : [B, d] — encoder's projected graph embedding
    * ``target_emb``: [B, d] — LLM embedding of the node's text
    * ``anchors``   : [K, d] — PCA principal components of LLM embedding matrix
    """
    h_sub = h_proj @ anchors.T                # [B, K]
    t_sub = target_emb @ anchors.T            # [B, K]
    h_n = F.normalize(h_sub, dim=-1)
    t_n = F.normalize(t_sub, dim=-1)
    logits = h_n @ t_n.T / temperature        # [B, B]
    labels = torch.arange(h_proj.size(0), device=h_proj.device)
    return F.cross_entropy(logits, labels)


def train_pca_alignment(
    model: CoTGraphLLM,
    data,
    title_dict: Dict[str, str],
    raw_texts,
    num_anchors: int = 256,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-4,
) -> None:
    """Pre-align encoder's projector output with the LLM-PCA anchor subspace.

    Trains only on a single source domain; no supervised labels are used.
    """
    device = next(model.parameters()).device
    data = data.to(device)

    print("\n" + "=" * 60)
    print(f"--- Zero-shot training: PCA anchor alignment ({epochs} ep) ---")
    print("=" * 60)
    print("📐 Computing LLM-embedding PCA anchors ...")
    anchors = compute_llm_embedding_pca_anchors(model.llm, k=num_anchors).to(device)
    print(f"   anchors: {tuple(anchors.shape)}")

    print("📐 Computing per-node title embeddings (LLM embedding layer) ...")
    titles = [
        title_dict.get(
            str(raw_texts[i]) if raw_texts is not None else str(i),
            "Unknown",
        )
        for i in range(data.num_nodes)
    ]
    target_emb = get_llm_text_embeddings(model.llm, model.tokenizer, titles).to(device)
    print(f"   target_emb: {tuple(target_emb.shape)}")

    trainables = [p for p in model.parameters() if p.requires_grad]
    if not trainables:
        print("⚠️ No trainable parameters; enable LoRA or unfreeze projector.")
        return
    optimizer = torch.optim.AdamW(trainables, lr=lr)

    N = data.num_nodes
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        _, soft_tokens = model.encode_graph(data.x, data.edge_index, None)   # [N, d]

        # Sample a random subset each epoch to keep InfoNCE negatives diverse
        if N > batch_size:
            idx = torch.randperm(N, device=device)[:batch_size]
            h = soft_tokens[idx]
            tgt = target_emb[idx]
        else:
            h, tgt = soft_tokens, target_emb

        loss = pca_alignment_loss(h, tgt, anchors)
        loss.backward()
        optimizer.step()
        print(f"   Epoch {ep:03d}/{epochs}: PCA-align loss = {loss.item():.4f}")

    print("✅ PCA alignment training done.")


# ── Instruction-file driven zero-shot inference ─────────────────


DEFAULT_INSTRUCTION_TEMPLATE = (
    "Title: {title}. "
    "Which sub-category does this paper belong to? "
    "Options: {candidates}. Answer: "
)


def load_instruction_template(
    dataset_name: str, instruction_dir: str = "instructions"
) -> str:
    path = os.path.join(instruction_dir, f"{dataset_name}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            tmpl = f.read().strip()
        print(f"✅ Loaded instruction template: {path}")
        return tmpl
    print(f"⚠️ No instruction file for '{dataset_name}' → using default template.")
    return DEFAULT_INSTRUCTION_TEMPLATE


def zero_shot_eval_with_template(
    model: CoTGraphLLM,
    data,
    candidates: List[str],
    title_dict: Dict[str, str],
    template: str,
    dataset_name: str,
    edge_index: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
    abstract_dict: Optional[Dict[str, str]] = None,
    use_soft_token: Optional[bool] = None,
    abstract_max_len: int = 400,
) -> Dict[str, float]:
    """Zero-shot inference driven by an instruction template string.

    Parameters
    ----------
    abstract_dict : optional ``paper_id -> abstract`` mapping.  When provided,
        the ``{abstract}`` placeholder in the template is filled from it.
    use_soft_token : if False, the GNN soft token is *not* prepended to the
        LLM prompt — pure-text inference.  Defaults to ``config.USE_SOFT_TOKEN``.
    """
    device = next(model.parameters()).device
    ei = (edge_index if edge_index is not None else data.edge_index).to(device)
    raw_texts = data.raw_texts if hasattr(data, "raw_texts") else None

    with torch.no_grad():
        _, soft_tokens = model.encode_graph(data.x.to(device), ei, None)

    if use_soft_token is None:
        use_soft_token = USE_SOFT_TOKEN
    print(f"   ↳ use_soft_token={use_soft_token}  "
          f"abstract_in_prompt={abstract_dict is not None}  "
          f"constrained={USE_CONSTRAINED_DECODING}")

    indices = (
        torch.nonzero(test_mask).squeeze(-1).tolist()
        if test_mask is not None
        else list(range(data.num_nodes))
    )

    results = []
    correct, total = 0, 0
    for idx in tqdm(indices, desc=f"Zero-shot[{dataset_name}]"):
        nid = raw_texts[idx] if raw_texts is not None else str(idx)
        title = title_dict.get(str(nid), title_dict.get(str(idx), "Unknown"))

        abstract_str = ""
        if abstract_dict:
            abstract_str = abstract_dict.get(str(nid), abstract_dict.get(str(idx), ""))
            if abstract_str and len(abstract_str) > abstract_max_len:
                abstract_str = abstract_str[:abstract_max_len].rstrip() + "..."
        if not abstract_str:
            abstract_str = "N/A"

        neighbor_text = _build_neighbor_text(idx, ei, raw_texts, title_dict)
        fmt = _SafeDict(
            title=title,
            abstract=abstract_str,
            candidates=", ".join(candidates),
            domain=dataset_name,
            neighbor_text=neighbor_text,
        )
        prompt = template.format_map(fmt)
        st = soft_tokens[idx] if use_soft_token else None
        if USE_CONSTRAINED_DECODING:
            cand_scores = model.score_candidates(prompt, candidates, soft_token=st)
            best_i = max(range(len(candidates)), key=lambda j: cand_scores[j])
            pred = candidates[best_i]
            pred_text = (
                f"[scored] {pred} "
                f"(top3="
                + ", ".join(
                    f"{candidates[j]}={cand_scores[j]:.3f}"
                    for j in sorted(range(len(candidates)),
                                    key=lambda j: -cand_scores[j])[:3]
                )
                + ")"
            )
        else:
            pred_text = model.generate(prompt, soft_token=st)
            pred = _match_candidate(pred_text, candidates)

        true_label = None
        if hasattr(data, "y") and data.y is not None:
            li = int(data.y[idx].item())
            if 0 <= li < len(candidates):
                true_label = candidates[li]
        if true_label is not None:
            total += 1
            if pred == true_label:
                correct += 1

        results.append(
            {
                "node_idx": idx,
                "node_id": str(nid),
                "title": title,
                "predicted": pred,
                "true_label": true_label,
                "raw": pred_text,
            }
        )

    save_dir = f"dataset/{dataset_name}/cluster"
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(save_dir, f"{dataset_name}_zeroshot_results.csv"), index=False
    )
    acc = correct / total if total else 0.0
    print(f"✅ [{dataset_name}] zero-shot accuracy: {acc:.4f} ({correct}/{total})")
    return {"dataset": dataset_name, "accuracy": acc, "correct": correct, "total": total}


def zero_shot_eval_cross_domain(
    model: CoTGraphLLM,
    target_dataset: str,
    instruction_dir: str = "instructions",
    n_in: int = 128,
) -> Dict[str, float]:
    """
    Load a *new* domain's data + instruction file and run zero-shot inference.
    No re-training, no labels used – the model was trained only on the source domain.
    """
    from dataloader import load_gnn_dataset
    from gcn import pca_compression
    from cluster_prompt import (
        load_answer_candidates,
        load_node_titles,
        load_node_abstracts,
    )

    print("\n" + "=" * 60)
    print(f"=== Zero-shot cross-domain inference: {target_dataset} ===")
    print("=" * 60)

    data = load_gnn_dataset(target_dataset, task="nc")
    data.x = torch.FloatTensor(pca_compression(data.x, k=n_in))
    candidates = load_answer_candidates(target_dataset, data)
    title_dict = load_node_titles(target_dataset, data)
    try:
        abstract_dict = load_node_abstracts(target_dataset) or None
    except Exception:
        abstract_dict = None
    template = load_instruction_template(target_dataset, instruction_dir)

    test_mask = getattr(data, "test_mask", None)
    return zero_shot_eval_with_template(
        model=model, data=data, candidates=candidates,
        title_dict=title_dict, template=template,
        dataset_name=target_dataset, test_mask=test_mask,
        abstract_dict=abstract_dict,
    )
