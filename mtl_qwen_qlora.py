# %% [markdown]
# # 🔬 Variety-Aware Sentiment and Sarcasm Detection
# # Using Multi-Task Learning — MTL-Proposed with Qwen2.5-14B LoRA
#
# **Dataset:** BESSTIE (BEnchmark for Sentiment and Sarcasm for varieTIes of English)
#
# **Pipeline (Streamlined — chỉ MTL-Proposed):**
# 1. Setup & Data Loading
# 2. Qwen2.5-14B LoRA backbone + CTAI + NTN Fusion + Dynamic Loss
# 3. Training (2 varieties: en-AU, en-UK)
# 4. Cross-Variety Evaluation
# 5. Results Visualization

# %% [markdown]
# ## Section 1: Setup & Install

# %%
# ====================================================================
# SECTION 1: SETUP & INSTALL
# ====================================================================

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("transformers")
install("datasets")
install("accelerate")
install("scikit-learn")
install("seaborn")
install("peft")           # LoRA / QLoRA adapters
install("bitsandbytes")   # 4-bit quantization
install("torchao")        # Sửa lỗi incompatible torchao version của peft

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import time
from copy import deepcopy

warnings.filterwarnings("ignore")

# %%
# --- Reproducibility ---
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# --- Device ---
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda:0" if NUM_GPUS > 0 else "cpu")
print(f"🖥️ Default Device: {DEVICE}")
print(f"   GPUs Available: {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

# %%
# ====================================================================
# HYPERPARAMETERS — Qwen2.5-14B LoRA BF16 (A100 80GB / ~10h training)
# ====================================================================

# --- Backbone LLM ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # 14.7B params — GẤP ĐÔI Qwen2.5-7B

# ====================================================================
# LoRA Config — LẤP ĐẦY A100 80GB VRAM + TRAINING ~10 TIẾNG
# ====================================================================
#
# Chiến lược chọn model:
#   Qwen2.5-7B  (7.6B)  — BF16 chỉ dùng ~15 GB → lãng phí 65 GB
#   Qwen2.5-14B (14.7B) — BF16 dùng ~29 GB → lấp đầy 80 GB ← CHỌN
#   Qwen2.5-32B (32B)   — BF16 = 64 GB model → không đủ VRAM train
#   Qwen2.5-32B 4-bit   — mỗi variety ~6h → 4 runs = 24h → vượt 12h
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  BẢNG TÍNH VRAM CHI TIẾT — Qwen2.5-14B (A100 80GB)             │
# ├──────────────────────────┬───────────┬──────────────────────────┤
# │ Component                │ VRAM (GB) │ Ghi chú                  │
# ├──────────────────────────┼───────────┼──────────────────────────┤
# │ Base model (BF16 frozen) │   29.4    │ 14.7B × 2 bytes          │
# │ LoRA r=128 weights       │    1.1    │ ~551M params × BF16      │
# │ LoRA optimizer (AdamW)   │    4.4    │ 2× FP32 states           │
# │ LoRA gradients           │    1.1    │ same as weights           │
# │ Heads (CTAI+NTN+proj)    │    5.8    │ w+optimizer+gradients    │
# │ Activations (NO ckpt)    │  ~31.0    │ BS=8, seq=384, 48 layers │
# │ CUDA overhead            │    3.0    │ kernels, fragmentation   │
# ├──────────────────────────┼───────────┼──────────────────────────┤
# │ TỔNG DỰ KIẾN            │  ~76 GB   │ Dư ~4 GB an toàn         │
# └──────────────────────────┴───────────┴──────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  ƯỚC TÍNH THỜI GIAN TRAINING                                   │
# ├───────────────────────────────────┬────────────────────────────┤
# │ Thông số                         │ Giá trị                    │
# ├───────────────────────────────────┼────────────────────────────┤
# │ Per step (BS=8, no ckpt)          │ ~1.0 - 1.2s               │
# │ Steps/epoch (7500 samples / 8)    │ 937 steps                 │
# │ Optimizer steps (÷ grad_accum=4)  │ 234 / epoch               │
# │ Per epoch                         │ ~16 min                   │
# │ Epochs (early stop ~8-10)         │ ~2.5h / variety           │
# │ 4 runs (2 + 2 cross-variety)      │ ~10h tổng                 │
# └───────────────────────────────────┴────────────────────────────┘
#
# Tại sao 14B thay vì 7B?
#   → 14.7B params = gấp đôi capacity → biểu diễn phong phú hơn
#   → hidden_dim=5120 (vs 3584) → CTAI/NTN tương tác sâu hơn
#   → 48 layers (vs 28) → nhiều tầng trừu tượng hơn cho sarcasm
#
# Tại sao KHÔNG dùng gradient checkpointing?
#   → Tắt grad_ckpt → mỗi step nhanh hơn (không recompute forward)
#   → VRAM thừa để chứa activations (~31 GB)
#   → Tổng thời gian ~10h (cân bằng giữa VRAM và time budget)
#
# Tại sao BS=8 + grad_accum=4?
#   → BS=8 nhỏ → activations vừa đủ lấp VRAM (không OOM)
#   → grad_accum=4 → effective batch = 32 → gradient ổn định
#   → Nhiều steps/epoch → tận dụng hết thời gian 10h
# ====================================================================

LORA_R = 128              # LoRA rank cao — ~551M trainable params (14B model)
LORA_ALPHA = 256           # LoRA scaling = alpha/r = 2.0 (chuẩn)
LORA_DROPOUT = 0.05        # Dropout nhẹ cho LoRA adapters
LORA_TARGET_MODULES = [    # Áp LoRA lên TẤT CẢ linear layers trong attention + MLP
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (4 modules)
    "gate_proj", "up_proj", "down_proj",       # MLP SwiGLU (3 modules)
]  # → 7 modules × 48 layers = 336 LoRA adapters
USE_4BIT = False           # ❌ KHÔNG quantize — BF16 full precision (tốt nhất cho F1)
GRADIENT_CHECKPOINTING = False  # ❌ TẮT — dùng VRAM cho activations, tăng tốc mỗi step

# --- Training ---
MAX_LEN = 384              # 384 tokens — cân bằng context vs VRAM (14B model cần nhiều VRAM hơn)
BATCH_SIZE = 8             # BS nhỏ để activations không OOM (14B × 48 layers = VRAM lớn)
GRAD_ACCUM_STEPS = 4       # Effective batch = 8 × 4 = 32 (gradient ổn định)
LEARNING_RATE = 2e-4       # LR chuẩn cho LoRA
NUM_EPOCHS = 15
LAMBDA_SARCASM = 0.7
DROPOUT = 0.1
NTN_SLICES = 8             # 8 slices (hidden_dim=5120 → W=[8,5120,5120] = 210M params, vừa phải)
CTAI_HEADS = 16            # 16 heads cho CTAI (head_dim = 5120/16 = 320)
PATIENCE = 5
USE_AMP = torch.cuda.is_available()

# Lấy hidden size tự động từ config
_llm_config = AutoConfig.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
HIDDEN_DIM = _llm_config.hidden_size  # 5120 cho Qwen2.5-14B

# Kiểm tra CTAI_HEADS chia hết HIDDEN_DIM
assert HIDDEN_DIM % CTAI_HEADS == 0, \
    f"HIDDEN_DIM={HIDDEN_DIM} không chia hết cho CTAI_HEADS={CTAI_HEADS}."

# Tính số layers
NUM_LAYERS = _llm_config.num_hidden_layers  # 48 cho Qwen2.5-14B

print(f"\n📋 Config — Qwen2.5-14B tối ưu cho A100 80GB:")
print(f"   Model: {LLM_MODEL_NAME} (14.7B params)")
print(f"   Precision: {'4-bit NF4 QLoRA' if USE_4BIT else 'BF16 Full Precision (KHÔNG quantize)'}")
print(f"   Hidden Dim: {HIDDEN_DIM}")
print(f"   Layers: {NUM_LAYERS}")
print(f"   LoRA r={LORA_R}, alpha={LORA_ALPHA}, targets={len(LORA_TARGET_MODULES)} modules × {NUM_LAYERS} layers")
print(f"   Gradient Checkpointing: {'ON' if GRADIENT_CHECKPOINTING else 'OFF (dùng VRAM cho tốc độ)'}")
print(f"   Max Length: {MAX_LEN}")
print(f"   Batch Size: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
print(f"   LR: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS} (early stop patience={PATIENCE})")
print(f"   Lambda (sarcasm): {LAMBDA_SARCASM}")
print(f"   NTN Slices: {NTN_SLICES}")
print(f"   CTAI Heads: {CTAI_HEADS} (head_dim = {HIDDEN_DIM // CTAI_HEADS})")
print(f"   VRAM dự kiến: ~76 GB / 80 GB")
print(f"   Thời gian dự kiến: ~10h (4 runs × ~2.5h)")

# %% [markdown]
# ## Section 2: Load & Explore BESSTIE Dataset

# %%
# ====================================================================
# SECTION 2: LOAD DATASET (100% TRAIN — KHÔNG SPLIT, KHÔNG LỌC DOMAIN)
# ====================================================================

TRAIN_PATHS = {
    "en-AU": "/kaggle/input/datasets/khanh002010/alta-data-v4/en-AU/en-AU/train.csv",
    "en-UK": "/kaggle/input/datasets/khanh002010/alta-data-v4/en-UK/en-UK/train.csv",
}

VALID_PATH = "/kaggle/input/datasets/khanh002010/valid-chuan/valid.csv"

VARIETIES = ["en-AU", "en-UK"]

train_list = []

def prepare_df(df, variety=None):
    """Chuẩn hóa tên cột từ file dataset sang chuẩn notebook."""
    if "sentiment" in df.columns and "sentiment_label" not in df.columns:
        df["sentiment_label"] = df["sentiment"].astype(int)
    if "sarcasm" in df.columns and "sarcasm_label" not in df.columns:
        df["sarcasm_label"] = df["sarcasm"].astype(int)
    if variety and "variety" not in df.columns:
        df["variety"] = variety
    if "source" in df.columns and "domain" not in df.columns:
        df["domain"] = df["source"].str.upper()
    elif "domain" not in df.columns:
        df["domain"] = "UNKNOWN"
    else:
        df["domain"] = df["domain"].str.upper()
    return df

def load_csv_auto(path):
    """Tự động phát hiện dấu phân cách (; hoặc ,) để đọc file CSV an toàn."""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(path, sep=sep)

print("📦 Loading datasets (100% Train — không split, không lọc domain)...")
for v in VARIETIES:
    df_tr = load_csv_auto(TRAIN_PATHS[v])
    df_tr = prepare_df(df_tr, v)
    train_list.append(df_tr)
    print(f"   {v:6s} -> Train (100%): {len(df_tr):5d} dòng (Google + Reddit)")

print(f"\n📂 Đọc file validation: {VALID_PATH}")
df_val_raw = load_csv_auto(VALID_PATH)
df_val_raw = prepare_df(df_val_raw)
print(f"   Tổng dòng validation: {len(df_val_raw)}")

df_train_filtered = pd.concat(train_list, ignore_index=True)
df_val_filtered   = df_val_raw.copy()
df_test_filtered  = df_val_raw.copy()

print(f"\n📊 Tổng cộng:")
print(f"   Train: {len(df_train_filtered)} dòng")
print(f"   Val:   {len(df_val_filtered)} dòng")

# %% [markdown]
# ## Section 3: Data Preprocessing

# %%
# ====================================================================
# SECTION 3: DATA PREPROCESSING
# ====================================================================

import re

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for df in [df_train_filtered, df_val_filtered, df_test_filtered]:
    df["text_clean"] = df["text"].apply(clean_text)

print("Text cleaning done")
print(f"   Avg text length (train): {df_train_filtered['text_clean'].str.len().mean():.0f} chars")

# %%
# --- Tokenizer (Qwen2.5-14B) ---
print(f"\nLoading tokenizer: {LLM_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)

# Qwen2.5 không có pad_token mặc định — dùng eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"   ⚠️ pad_token set to eos_token: '{tokenizer.pad_token}'")

# Padding bên trái (chuẩn cho decoder-only model — last token = representation)
tokenizer.padding_side = "left"
print(f"   Padding side: {tokenizer.padding_side}")

# %%
# --- PyTorch Dataset ---
class BESSTIEDataset(Dataset):
    def __init__(self, texts, sentiment_labels, sarcasm_labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.sarcasm_labels = sarcasm_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = self.sentiment_labels[idx]
        sarcasm = self.sarcasm_labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "sentiment_label": torch.tensor(sentiment, dtype=torch.long),
            "sarcasm_label": torch.tensor(sarcasm, dtype=torch.long),
        }

# %%
# --- Helper to create datasets ---
def create_datasets(df_train, df_val, df_test, variety=None):
    tr = df_train.copy()
    va = df_val.copy()
    te = df_test.copy()

    if variety:
        tr = tr[tr["variety"] == variety]
        va = va[va["variety"] == variety]
        te = te[te["variety"] == variety]

    train_ds = BESSTIEDataset(
        tr["text_clean"].values, tr["sentiment_label"].values,
        tr["sarcasm_label"].values, tokenizer
    )
    val_ds = BESSTIEDataset(
        va["text_clean"].values, va["sentiment_label"].values,
        va["sarcasm_label"].values, tokenizer
    )
    test_ds = BESSTIEDataset(
        te["text_clean"].values, te["sentiment_label"].values,
        te["sarcasm_label"].values, tokenizer
    )
    return train_ds, val_ds, test_ds

# %%
# --- Compute class weights ---
def compute_class_weights(labels, device=None):
    if device is None:
        device = DEVICE
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights).to(device)

print("Data utilities defined")

# %% [markdown]
# ## Section 4: Model Definitions — Qwen2.5-14B LoRA + CTAI + NTN

# %%
# ====================================================================
# SECTION 4: MODEL DEFINITIONS
# ====================================================================

# --- 4a. Cross-Task Attention Interaction (CTAI) Module ---
class CrossTaskAttention(nn.Module):
    """
    Cross-Task Attention Interaction (CTAI).
    Allows sentiment representation to attend to sarcasm representation
    and vice versa, enabling information exchange between tasks.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=CTAI_HEADS, dropout=DROPOUT):
        super().__init__()
        self.sent_to_sarc_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.sarc_to_sent_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.layer_norm_sent = nn.LayerNorm(hidden_dim)
        self.layer_norm_sarc = nn.LayerNorm(hidden_dim)

    def forward(self, sent_repr, sarc_repr):
        sent_repr_3d = sent_repr.unsqueeze(1)
        sarc_repr_3d = sarc_repr.unsqueeze(1)

        sent_attended, _ = self.sent_to_sarc_attn(
            query=sent_repr_3d, key=sarc_repr_3d, value=sarc_repr_3d
        )
        sarc_attended, _ = self.sarc_to_sent_attn(
            query=sarc_repr_3d, key=sent_repr_3d, value=sent_repr_3d
        )

        enhanced_sent = self.layer_norm_sent(sent_repr + sent_attended.squeeze(1))
        enhanced_sarc = self.layer_norm_sarc(sarc_repr + sarc_attended.squeeze(1))

        return enhanced_sent, enhanced_sarc

print("CrossTaskAttention (CTAI) defined")

# %%
# --- 4b. Neural Tensor Network (NTN) Fusion ---
class NTNFusion(nn.Module):
    """
    Neural Tensor Network Fusion.
    s+ = tanh(s_sen^T * W[1:k] * s_sar + V * [s_sen; s_sar] + b)
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, k=NTN_SLICES):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.randn(k, hidden_dim, hidden_dim) * 0.01)
        self.V = nn.Linear(hidden_dim * 2, k, bias=True)

    def forward(self, sent_repr, sarc_repr):
        batch_size = sent_repr.size(0)

        bilinear = torch.zeros(batch_size, self.k).to(sent_repr.device)
        for i in range(self.k):
            bilinear[:, i] = torch.sum(
                sent_repr * torch.mm(sarc_repr, self.W[i].t()), dim=1
            )

        concat = torch.cat([sent_repr, sarc_repr], dim=1)
        linear = self.V(concat)

        fused = torch.tanh(bilinear + linear)
        return fused

print("NTNFusion defined")

# %%
# --- 4c. MTL-Proposed Model with Qwen2.5-14B LoRA ---
class MTLProposedModel(nn.Module):
    """
    MTL Proposed: Qwen2.5-14B (BF16 LoRA) + CTAI + NTN Fusion + Dynamic Loss.

    Architecture:
        Input -> Qwen2.5-14B LoRA Encoder -> Last-Token Pooling
        -> Task-specific projections (s_sen, s_sar)
        -> CTAI (cross-task attention interaction)
        -> NTN Fusion -> fused representation s+
        -> Sentiment Head: s_sen_enhanced -> FC -> 2 classes
        -> Sarcasm Head: [s_sar_enhanced + s+] -> FC -> 2 classes

    Key Differences vs RoBERTa version:
        - Decoder-only model -> dùng last-token pooling thay vì [CLS]
        - LoRA BF16 -> chỉ train LoRA adapters (~551M params)
        - CTAI/NTN/Heads train full (không qua LoRA)

    VRAM Budget (A100 80GB):
        - Base model BF16 (frozen): ~29.4 GB
        - LoRA r=128 + optimizer + gradients: ~6.6 GB
        - Heads + optimizer + gradients: ~5.8 GB
        - Activations (BS=8, seq=384, NO grad_ckpt, 48L): ~31 GB
        - Total: ~76 GB / 80 GB
    """
    def __init__(self, dropout=DROPOUT, ntn_slices=NTN_SLICES, ctai_heads=CTAI_HEADS):
        super().__init__()

        # --- Load Qwen2.5-14B ---
        if USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.encoder = AutoModel.from_pretrained(
                LLM_MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            # prepare_model_for_kbit_training CHỈ cần khi dùng quantization
            self.encoder = prepare_model_for_kbit_training(self.encoder)
            print("  ✅ 4-bit quantization + prepare_model_for_kbit_training")
        else:
            # BF16 full precision — KHÔNG quantize, KHÔNG cần prepare_model_for_kbit_training
            self.encoder = AutoModel.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            # Đóng băng toàn bộ base model (LoRA sẽ thêm trainable params sau)
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  ✅ BF16 full precision — base model frozen")

        # --- Gradient checkpointing ---
        if GRADIENT_CHECKPOINTING and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
            print("  ✅ Gradient checkpointing enabled")
        else:
            print("  ⚡ Gradient checkpointing OFF — trading VRAM for speed")

        # --- Wrap with LoRA ---
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()

        # In VRAM sau khi load model
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  📊 VRAM after model load: {vram_used:.1f} GB allocated, {vram_reserved:.1f} GB reserved")

        # --- Task heads (train FULL — không qua LoRA) ---
        self.dropout = nn.Dropout(dropout)

        self.sent_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(dropout)
        )
        self.sarc_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(dropout)
        )

        self.ctai = CrossTaskAttention(
            hidden_dim=HIDDEN_DIM, num_heads=ctai_heads, dropout=dropout
        )
        self.ntn = NTNFusion(hidden_dim=HIDDEN_DIM, k=ntn_slices)

        self.sentiment_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(HIDDEN_DIM // 4, 2)
        )
        self.sarcasm_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + ntn_slices, HIDDEN_DIM // 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(HIDDEN_DIM // 4, 2)
        )

    def _get_last_token_repr(self, hidden_states, attention_mask):
        """
        Trích xuất biểu diễn token cuối cùng KHÔNG phải padding.
        Với left-padding, token cuối = token cuối cùng trong sequence.
        Nếu padding_side='left', token có ý nghĩa nằm ở cuối → lấy token cuối cùng.
        """
        # Với left-padding: token cuối cùng luôn là token thực (non-pad)
        return hidden_states[:, -1, :]

    def forward_heads(self, pooled_output):
        cls_drop = self.dropout(pooled_output)
        s_sen = self.sent_proj(cls_drop)
        s_sar = self.sarc_proj(cls_drop)

        s_sen_enhanced, s_sar_enhanced = self.ctai(s_sen, s_sar)
        s_plus = self.ntn(s_sen_enhanced, s_sar_enhanced)

        sentiment_logits = self.sentiment_head(s_sen_enhanced)
        sarc_input = torch.cat([s_sar_enhanced, s_plus], dim=1)
        sarcasm_logits = self.sarcasm_head(sarc_input)

        return sentiment_logits, sarcasm_logits

    def forward(self, input_ids, attention_mask, rdrop=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Decoder-only: dùng last-token pooling thay vì [CLS]
        pooled_output = self._get_last_token_repr(outputs.last_hidden_state, attention_mask)

        if rdrop and self.training:
            sent_1, sarc_1 = self.forward_heads(pooled_output)
            sent_2, sarc_2 = self.forward_heads(pooled_output)
            return sent_1, sarc_1, sent_2, sarc_2

        return self.forward_heads(pooled_output)

print("MTLProposedModel (Qwen2.5-14B LoRA) defined")

# %% [markdown]
# ## Section 5: Training Utilities

# %%
# ====================================================================
# SECTION 5: TRAINING UTILITIES
# ====================================================================

class DynamicLossWeighter:
    """
    Dynamic Joint Loss [AMTF-Net]: automatically balances loss weights.
    alpha(t) = L_sen(t) / (L_sen(t) + L_sar(t))
    beta(t) = L_sar(t) / (L_sen(t) + L_sar(t))
    """
    def __init__(self, temperature=2.0):
        self.temperature = temperature
        self.sent_loss_history = []
        self.sarc_loss_history = []

    def get_weights(self, sent_loss_val, sarc_loss_val):
        self.sent_loss_history.append(sent_loss_val)
        self.sarc_loss_history.append(sarc_loss_val)
        if len(self.sent_loss_history) < 2:
            return 0.5, 0.5
        w_sent = np.exp(sent_loss_val / self.temperature)
        w_sarc = np.exp(sarc_loss_val / self.temperature)
        total = w_sent + w_sarc
        return w_sent / total, w_sarc / total

# %%
def train_mtl_epoch(model, dataloader, optimizer, scheduler,
                    sent_criterion, sarc_criterion,
                    lambda_sarc=LAMBDA_SARCASM, dynamic_weighter=None,
                    scaler=None, grad_accum_steps=GRAD_ACCUM_STEPS):
    model.train()
    device = next(model.parameters()).device
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    total_loss = 0
    total_sent_loss = 0
    total_sarc_loss = 0
    all_sent_preds, all_sent_labels = [], []
    all_sarc_preds, all_sarc_labels = [], []

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_labels = batch["sentiment_label"].to(device)
        sarc_labels = batch["sarcasm_label"].to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16):
            sent_logits, sarc_logits = model(input_ids, attention_mask)
            loss_sent = sent_criterion(sent_logits, sent_labels)
            loss_sarc = sarc_criterion(sarc_logits, sarc_labels)

            if dynamic_weighter:
                alpha, beta = dynamic_weighter.get_weights(
                    loss_sent.item(), loss_sarc.item()
                )
                loss = alpha * loss_sent + beta * loss_sarc
            else:
                loss = loss_sent + lambda_sarc * loss_sarc

            # Scale loss cho gradient accumulation
            loss = loss / grad_accum_steps

        if scaler is not None and USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step mỗi grad_accum_steps
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None and USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps  # Unscale lại
        total_sent_loss += loss_sent.item()
        total_sarc_loss += loss_sarc.item()

        sent_preds = torch.argmax(sent_logits, dim=1).cpu().numpy()
        sarc_preds = torch.argmax(sarc_logits, dim=1).cpu().numpy()
        all_sent_preds.extend(sent_preds)
        all_sent_labels.extend(sent_labels.cpu().numpy())
        all_sarc_preds.extend(sarc_preds)
        all_sarc_labels.extend(sarc_labels.cpu().numpy())

    n = len(dataloader)
    f1_sent = f1_score(all_sent_labels, all_sent_preds, average="macro")
    f1_sarc = f1_score(all_sarc_labels, all_sarc_preds, average="macro")
    return total_loss/n, total_sent_loss/n, total_sarc_loss/n, f1_sent, f1_sarc

# %%
@torch.no_grad()
def evaluate_mtl(model, dataloader, sent_criterion, sarc_criterion):
    model.eval()
    device = next(model.parameters()).device
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    total_loss = 0
    all_sent_preds, all_sent_labels = [], []
    all_sarc_preds, all_sarc_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_labels = batch["sentiment_label"].to(device)
        sarc_labels = batch["sarcasm_label"].to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16):
            sent_logits, sarc_logits = model(input_ids, attention_mask)
            loss_sent = sent_criterion(sent_logits, sent_labels)
            loss_sarc = sarc_criterion(sarc_logits, sarc_labels)
            loss = loss_sent + LAMBDA_SARCASM * loss_sarc
        total_loss += loss.item()

        all_sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
        all_sent_labels.extend(sent_labels.cpu().numpy())
        all_sarc_preds.extend(torch.argmax(sarc_logits, dim=1).cpu().numpy())
        all_sarc_labels.extend(sarc_labels.cpu().numpy())

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "sent_f1": f1_score(all_sent_labels, all_sent_preds, average="macro"),
        "sent_precision": precision_score(all_sent_labels, all_sent_preds, average="macro", zero_division=0),
        "sent_recall": recall_score(all_sent_labels, all_sent_preds, average="macro", zero_division=0),
        "sarc_f1": f1_score(all_sarc_labels, all_sarc_preds, average="macro"),
        "sarc_precision": precision_score(all_sarc_labels, all_sarc_preds, average="macro", zero_division=0),
        "sarc_recall": recall_score(all_sarc_labels, all_sarc_preds, average="macro", zero_division=0),
        "sent_preds": all_sent_preds, "sent_labels": all_sent_labels,
        "sarc_preds": all_sarc_preds, "sarc_labels": all_sarc_labels,
    }

print("Training utilities defined")

# %% [markdown]
# ## Section 6: Run MTL-Proposed Experiment

# %%
# ====================================================================
# EXPERIMENT: MTL-PROPOSED (Qwen2.5-14B LoRA)
# ====================================================================

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"📁 Checkpoint directory: {CHECKPOINT_DIR}")

def run_mtl_experiment(variety, num_epochs=NUM_EPOCHS, lambda_sarc=LAMBDA_SARCASM,
                       use_dynamic_loss=True, device=None):
    """Run MTL-Proposed experiment with Qwen2.5-14B LoRA."""
    if device is None:
        device = DEVICE
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    print(f"\n{'='*60}")
    print(f"  MTL-Proposed (Qwen2.5-14B LoRA) | {variety or 'Combined'} | Device: {device}")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = create_datasets(
        df_train_filtered, df_val_filtered, df_test_filtered,
        variety=variety
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Dọn dẹp VRAM
    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    # Khởi tạo model (QLoRA + CTAI + NTN)
    start_time = time.time()
    model = MTLProposedModel()
    # device_map="auto" đã tự đặt model lên GPU
    load_time = time.time() - start_time
    print(f"  ⏱️ Model loaded in {load_time:.1f}s")

    # Tính class weights
    train_subset = df_train_filtered[
        df_train_filtered["variety"] == variety
    ] if variety else df_train_filtered

    sent_weights = compute_class_weights(train_subset["sentiment_label"].values, device=device)
    sarc_weights = compute_class_weights(train_subset["sarcasm_label"].values, device=device)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)
    sarc_criterion = nn.CrossEntropyLoss(weight=sarc_weights)

    # --- Optimizer: 2 nhóm LR ---
    # LoRA adapters (trong encoder): LR = 2e-4 (chuẩn LoRA)
    # Task heads (CTAI, NTN, projections, classifiers): LR = 1e-4
    lora_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            lora_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW([
        {"params": lora_params, "lr": LEARNING_RATE},         # 2e-4 cho LoRA
        {"params": head_params, "lr": LEARNING_RATE / 2},     # 1e-4 cho heads
    ], weight_decay=0.01)

    total_optimizer_steps = (len(train_loader) // GRAD_ACCUM_STEPS + 1) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_optimizer_steps),
        num_training_steps=total_optimizer_steps
    )

    print(f"  📊 Optimizer groups:")
    print(f"     LoRA params: {sum(p.numel() for p in lora_params):,} @ LR={LEARNING_RATE}")
    print(f"     Head params: {sum(p.numel() for p in head_params):,} @ LR={LEARNING_RATE/2}")
    print(f"     Total optimizer steps: {total_optimizer_steps}")

    dynamic_weighter = DynamicLossWeighter() if use_dynamic_loss else None

    best_avg_f1 = 0
    best_model_state = None
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and "cuda" in str(device))

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, sent_loss, sarc_loss, train_f1_sent, train_f1_sarc = train_mtl_epoch(
            model, train_loader, optimizer, scheduler,
            sent_criterion, sarc_criterion, lambda_sarc, dynamic_weighter,
            scaler=scaler
        )
        val_metrics = evaluate_mtl(model, val_loader, sent_criterion, sarc_criterion)
        avg_f1 = (val_metrics["sent_f1"] + val_metrics["sarc_f1"]) / 2
        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s) | "
              f"Loss: {train_loss:.4f} | "
              f"Train Sent F1: {train_f1_sent:.4f} Sarc F1: {train_f1_sarc:.4f} | "
              f"Val Sent F1: {val_metrics['sent_f1']:.4f} Sarc F1: {val_metrics['sarc_f1']:.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            # Lưu state_dict chỉ cho phần trainable (LoRA + heads)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                                if any(k.startswith(prefix) for prefix in
                                       ["sent_proj", "sarc_proj", "ctai", "ntn",
                                        "sentiment_head", "sarcasm_head", "dropout"])
                                or "lora" in k.lower()}
            patience_counter = 0

            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_MTL-Proposed_{variety or 'Combined'}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": best_model_state,
                "best_avg_f1": best_avg_f1,
                "variety": variety or "Combined",
                "model_type": "MTL-Proposed-Qwen2.5-14B-QLoRA",
                "model_name": LLM_MODEL_NAME,
                "lora_config": {
                    "r": LORA_R, "alpha": LORA_ALPHA,
                    "targets": LORA_TARGET_MODULES,
                },
            }, ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path} (Avg F1={best_avg_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model và evaluate trên test set
    model.load_state_dict(best_model_state, strict=False)
    test_metrics = evaluate_mtl(model, test_loader, sent_criterion, sarc_criterion)

    print(f"\n  TEST Results ({variety or 'Combined'} | MTL-Proposed Qwen2.5-14B LoRA):")
    print(f"     Sentiment - F1: {test_metrics['sent_f1']:.4f} | "
          f"P: {test_metrics['sent_precision']:.4f} | R: {test_metrics['sent_recall']:.4f}")
    print(f"     Sarcasm   - F1: {test_metrics['sarc_f1']:.4f} | "
          f"P: {test_metrics['sarc_precision']:.4f} | R: {test_metrics['sarc_recall']:.4f}")

    result = {
        "variety": variety or "Combined", "model_type": "MTL-Proposed",
        "sent_f1": test_metrics["sent_f1"],
        "sent_precision": test_metrics["sent_precision"],
        "sent_recall": test_metrics["sent_recall"],
        "sarc_f1": test_metrics["sarc_f1"],
        "sarc_precision": test_metrics["sarc_precision"],
        "sarc_recall": test_metrics["sarc_recall"],
        "sent_preds": test_metrics["sent_preds"],
        "sent_labels": test_metrics["sent_labels"],
        "sarc_preds": test_metrics["sarc_preds"],
        "sarc_labels": test_metrics["sarc_labels"],
    }

    del model
    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result

print("Experiment runner defined")

# %% [markdown]
# ## Run Experiments

# %%
# ====================================================================
# RUN MTL-PROPOSED FOR BOTH VARIETIES
# ====================================================================

all_results = []

print("\n" + "="*80)
print("  🚀 BẮT ĐẦU EXPERIMENT: MTL-PROPOSED (Qwen2.5-14B LoRA)")
print("="*80)

total_start = time.time()

for variety in VARIETIES:
    result = run_mtl_experiment(
        variety=variety,
        use_dynamic_loss=True,
        device=DEVICE
    )
    all_results.append(result)

total_time = time.time() - total_start
print(f"\n⏱️ Tổng thời gian training: {total_time/3600:.1f} giờ ({total_time:.0f}s)")

# In bảng tổng kết
print("\n" + "="*75)
print("  🏆 KẾT QUẢ TỐT NHẤT - MTL-PROPOSED (Qwen2.5-14B LoRA)")
print("="*75)
for r in all_results:
    avg_f1 = (r['sent_f1'] + r['sarc_f1']) / 2
    print(f"  * Phương ngữ: {r['variety']:<6} | Sent F1: {r['sent_f1']:.4f} | Sarc F1: {r['sarc_f1']:.4f} | Avg F1: {avg_f1:.4f}")
    print(f"    - Sentiment: Precision = {r['sent_precision']:.4f}, Recall = {r['sent_recall']:.4f}")
    print(f"    - Sarcasm:   Precision = {r['sarc_precision']:.4f}, Recall = {r['sarc_recall']:.4f}")
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_MTL-Proposed_{r['variety']}.pt")
    if os.path.exists(ckpt):
        print(f"    💾 Checkpoint: {ckpt}")
print("="*75)



# %% [markdown]
# ## Results

# %%
# ====================================================================
# KẾT QUẢ (chỉ in điểm)
# ====================================================================

print("\n" + "="*70)
print("  MAIN RESULTS")
print("="*70)
print(f"  {'Variety':<8} {'Sent F1':>10} {'Sarc F1':>10} {'Avg F1':>10}")
print("-"*42)
for r in all_results:
    avg = (r['sent_f1'] + r['sarc_f1']) / 2
    print(f"  {r['variety']:<8} {r['sent_f1']:>10.4f} {r['sarc_f1']:>10.4f} {avg:>10.4f}")


print(f"  Training time: {total_time/3600:.1f}h")

# Save results to JSON
all_results_clean = []
for r in all_results:
    r_clean = {k: v for k, v in r.items()
               if k not in ["preds", "labels", "sent_preds", "sent_labels", "sarc_preds", "sarc_labels"]}
    for k, v in r_clean.items():
        if isinstance(v, (np.float32, np.float64)):
            r_clean[k] = float(v)
    all_results_clean.append(r_clean)

with open("experiment_results.json", "w") as f:
    json.dump({
        "model": LLM_MODEL_NAME,
        "main_results": all_results_clean,
    }, f, indent=2)

print("Saved: experiment_results.json")

