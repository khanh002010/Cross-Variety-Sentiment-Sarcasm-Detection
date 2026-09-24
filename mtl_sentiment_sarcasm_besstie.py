# %% [markdown]
# # 🔬 Variety-Aware Sentiment and Sarcasm Detection
# # Using Multi-Task Learning for Australian and British English
#
# **Dataset:** BESSTIE (BEnchmark for Sentiment and Sarcasm for varieTIes of English)
#
# **Pipeline:**
# 1. STL Baselines (Single-Task Learning)
# 2. MTL Baseline (Shared Encoder + 2 Heads)
# 3. MTL Proposed (+ CTAI + NTN Fusion + Dynamic Loss)
# 4. Cross-Variety Evaluation
# 5. Ablation Study

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

import os
# Dùng max_split_size_mb để chống phân mảnh bộ nhớ an toàn (tránh expandable_segments:True vì xung đột với multi-threading/empty_cache)
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

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
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

# --- Device & Multi-GPU Detection ---
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda:0" if NUM_GPUS > 0 else "cpu")
print(f"🖥️ Default Device: {DEVICE}")
print(f"   GPUs Available: {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

# --- Hyperparameters ---
MODEL_NAME = "roberta-large"  # "roberta-large" hoặc "roberta-base"
MAX_LEN = 384                 # Tăng từ 256 lên 384 để nắm trọn ngữ cảnh bài viết dài (Reddit / Google Review)
BATCH_SIZE = 16               # Giảm từ 24 xuống 16 để VRAM chỉ dùng ~10.2GB (dư hơn 4.3GB an toàn trên GPU T4 14.56GB, triệt tiêu hoàn toàn OOM)
LEARNING_RATE = 1e-5 if "large" in MODEL_NAME else 2e-5  # Large model cần LR nhỏ hơn (1e-5)
NUM_EPOCHS = 15
LAMBDA_SARCASM = 0.7  # weight cho sarcasm loss trong joint loss
DROPOUT = 0.1
NTN_SLICES = 8        # Tăng từ 4 lên 8 lát tensor để tăng cường tương tác phi tuyến tính Sentiment-Sarcasm
CTAI_HEADS = 16       # 16 attention heads cho CTAI (chuẩn theo kiến trúc 16 heads của RoBERTa-large, head_dim = 64)
PATIENCE = 5          # early stopping patience (tăng vì freeze layers khiến model học chậm hơn)
USE_AMP = torch.cuda.is_available()  # Bật PyTorch AMP (FP16) tiết kiệm ~50% VRAM và tăng tốc x2

# Tự động lấy kích thước hidden size (768 cho base, 1024 cho large)
_model_config = AutoConfig.from_pretrained(MODEL_NAME)
HIDDEN_DIM = _model_config.hidden_size

print(f"\n📋 Config:")
print(f"   Model: {MODEL_NAME}")
print(f"   Hidden Dim: {HIDDEN_DIM}")
print(f"   Max Length: {MAX_LEN}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   LR: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Lambda (sarcasm): {LAMBDA_SARCASM}")
print(f"   NTN Slices: {NTN_SLICES}")
print(f"   CTAI Heads: {CTAI_HEADS}")
print(f"   Mixed Precision (FP16): {USE_AMP}")

# %% [markdown]
# ## Section 2: Load & Explore BESSTIE Dataset

# %%
# ====================================================================
# SECTION 2: LOAD DATASET (100% TRAIN — KHÔNG SPLIT, KHÔNG LỌC DOMAIN)
# ====================================================================

# Đường dẫn dữ liệu mới (v4)
TRAIN_PATHS = {
    "en-AU": "/kaggle/input/datasets/khanh002010/alta-data-v4/en-AU/en-AU/train.csv",
    "en-UK": "/kaggle/input/datasets/khanh002010/alta-data-v4/en-UK/en-UK/train.csv",
}

# Tập validation dùng chung (đã gộp cả en-AU và en-UK)
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
    # Ánh xạ cột 'source' sang 'domain' và viết hoa ('reddit' -> 'REDDIT', 'google' -> 'GOOGLE')
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
    # Đọc 100% file train.csv (tự nhận diện dấu ; hoặc ,)
    df_tr = load_csv_auto(TRAIN_PATHS[v])
    df_tr = prepare_df(df_tr, v)
    train_list.append(df_tr)
    print(f"   {v:6s} -> Train (100%): {len(df_tr):5d} dòng (Google + Reddit)")

# Đọc file validation dùng chung (tự nhận diện dấu ; hoặc ,)
print(f"\n📂 Đọc file validation: {VALID_PATH}")
df_val_raw = load_csv_auto(VALID_PATH)
df_val_raw = prepare_df(df_val_raw)
print(f"   Tổng dòng validation: {len(df_val_raw)}")

# Gộp dữ liệu
df_train_filtered = pd.concat(train_list, ignore_index=True)
df_val_filtered   = df_val_raw.copy()
df_test_filtered  = df_val_raw.copy()  # Trỏ test vào val (để code phía sau in kết quả cuối cùng)

print(f"\n📊 Tổng cộng:")
print(f"   Train: {len(df_train_filtered)} (Toàn bộ Google + Reddit, 2 phương ngữ)")
print(f"   Val:   {len(df_val_filtered)} (Dùng để chọn checkpoint tốt nhất)")

# %%
# --- Statistics per variety and domain ---
print("\n📊 Label Distribution (Train):")
for variety in VARIETIES:
    for domain in df_train_filtered["domain"].unique():
        subset = df_train_filtered[
            (df_train_filtered["variety"] == variety) &
            (df_train_filtered["domain"] == domain)
        ]
        if len(subset) == 0:
            continue
        n = len(subset)
        sent_pos = (subset["sentiment_label"] == 1).sum()
        sarc_pos = (subset["sarcasm_label"] == 1).sum()
        print(f"   {variety} | {domain:8s} | N={n:5d} | "
              f"Sent+ {sent_pos/n*100:.0f}% | Sarc+ {sarc_pos/n*100:.0f}%")

# %%
# --- Visualize label distribution ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Label Distribution per Variety x Domain", fontsize=14, fontweight="bold")

idx = 0
for variety in VARIETIES:
    for label_col, label_name in [("sentiment_label", "Sentiment"), ("sarcasm_label", "Sarcasm")]:
        subset = df_train_filtered[df_train_filtered["variety"] == variety]
        counts = subset[label_col].value_counts().sort_index()
        colors = ["#e74c3c", "#2ecc71"] if label_name == "Sentiment" else ["#3498db", "#e67e22"]
        labels = ["Negative", "Positive"] if label_name == "Sentiment" else ["Not Sarcastic", "Sarcastic"]
        axes[idx].bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
        axes[idx].set_title(f"{variety}\n{label_name}")
        axes[idx].set_ylabel("Count")
        for j, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            axes[idx].text(j, v + 10, str(v), ha="center", fontweight="bold")
        idx += 1

plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: label_distribution.png")

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
# --- Tokenizer ---
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
def create_datasets(df_train, df_val, df_test, variety=None, domain=None):
    tr = df_train.copy()
    va = df_val.copy()
    te = df_test.copy()

    if variety:
        tr = tr[tr["variety"] == variety]
        va = va[va["variety"] == variety]
        te = te[te["variety"] == variety]
    # Không lọc domain nữa — dùng toàn bộ dữ liệu (Google + Reddit)

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

# Tính class weights trên toàn bộ tập train en-AU (cả Google + Reddit)
au_all = df_train_filtered[df_train_filtered["variety"] == "en-AU"]
sarc_weights = compute_class_weights(au_all["sarcasm_label"].values)
sent_weights = compute_class_weights(au_all["sentiment_label"].values)
print(f"   Sarcasm class weights (en-AU ALL): {sarc_weights}")
print(f"   Sentiment class weights (en-AU ALL): {sent_weights}")

# %% [markdown]
# ## Section 4: Model Definitions

# %%
# ====================================================================
# SECTION 4: MODEL DEFINITIONS
# ====================================================================

# --- 4a. Single-Task Learning (STL) Model ---
class STLModel(nn.Module):
    """Single-Task Learning: 1 encoder + 1 classification head."""
    def __init__(self, model_name=MODEL_NAME, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

print("STLModel defined")

# %%
# --- 4b. MTL Baseline Model ---
class MTLBaselineModel(nn.Module):
    """
    MTL Baseline: Shared Encoder + 2 Task-Specific Heads.
    Joint Loss: L_total = L_sentiment + lambda * L_sarcasm
    """
    def __init__(self, model_name=MODEL_NAME, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM // 2, 2)
        )
        self.sarcasm_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM // 2, 2)
        )

    def forward_heads(self, cls_output):
        cls_drop = self.dropout(cls_output)
        sentiment_logits = self.sentiment_head(cls_drop)
        sarcasm_logits = self.sarcasm_head(cls_drop)
        return sentiment_logits, sarcasm_logits

    def forward(self, input_ids, attention_mask, rdrop=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        if rdrop and self.training:
            sent_1, sarc_1 = self.forward_heads(cls_output)
            sent_2, sarc_2 = self.forward_heads(cls_output)
            return sent_1, sarc_1, sent_2, sarc_2
        return self.forward_heads(cls_output)

print("MTLBaselineModel defined")

# %%
# --- 4c. Cross-Task Attention Interaction (CTAI) Module ---
# Inspired by AMTF-Net [Lakshmi et al., 2026]

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
# --- 4d. Neural Tensor Network (NTN) Fusion ---
# From Majumder et al. [2019]

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
# --- 4e. MTL Proposed Model (Full Architecture) ---
class MTLProposedModel(nn.Module):
    """
    MTL Proposed: Shared Encoder + CTAI + NTN Fusion + Dynamic Loss.

    Architecture:
        Input -> Shared RoBERTa Encoder -> [CLS]
        -> Task-specific projections (s_sen, s_sar)
        -> CTAI (cross-task attention interaction)
        -> NTN Fusion -> fused representation s+
        -> Sentiment Head: s_sen_enhanced -> FC -> 2 classes
        -> Sarcasm Head: [s_sar_enhanced + s+] -> FC -> 2 classes
    """
    def __init__(self, model_name=MODEL_NAME, dropout=DROPOUT,
                 ntn_slices=NTN_SLICES, ctai_heads=CTAI_HEADS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
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

    def forward_heads(self, cls_output):
        cls_drop = self.dropout(cls_output)
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
        cls_output = outputs.last_hidden_state[:, 0, :]

        if rdrop and self.training:
            sent_1, sarc_1 = self.forward_heads(cls_output)
            sent_2, sarc_2 = self.forward_heads(cls_output)
            return sent_1, sarc_1, sent_2, sarc_2

        return self.forward_heads(cls_output)

print("MTLProposedModel defined")

# %%
# --- 4f. Ablation variants ---

class MTLNTNOnlyModel(nn.Module):
    """MTL with NTN Fusion only (no CTAI) - for ablation study."""
    def __init__(self, model_name=MODEL_NAME, dropout=DROPOUT, ntn_slices=NTN_SLICES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.sent_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(dropout)
        )
        self.sarc_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(dropout)
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

    def forward_heads(self, cls_output):
        cls_drop = self.dropout(cls_output)
        s_sen = self.sent_proj(cls_drop)
        s_sar = self.sarc_proj(cls_drop)
        s_plus = self.ntn(s_sen, s_sar)
        sentiment_logits = self.sentiment_head(s_sen)
        sarc_input = torch.cat([s_sar, s_plus], dim=1)
        sarcasm_logits = self.sarcasm_head(sarc_input)
        return sentiment_logits, sarcasm_logits

    def forward(self, input_ids, attention_mask, rdrop=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        if rdrop and self.training:
            sent_1, sarc_1 = self.forward_heads(cls_output)
            sent_2, sarc_2 = self.forward_heads(cls_output)
            return sent_1, sarc_1, sent_2, sarc_2
        return self.forward_heads(cls_output)


class MTLCTAIOnlyModel(nn.Module):
    """MTL with CTAI only (no NTN) - for ablation study."""
    def __init__(self, model_name=MODEL_NAME, dropout=DROPOUT, ctai_heads=CTAI_HEADS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
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
        self.sentiment_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(HIDDEN_DIM // 4, 2)
        )
        self.sarcasm_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(HIDDEN_DIM // 4, 2)
        )

    def forward_heads(self, cls_output):
        cls_drop = self.dropout(cls_output)
        s_sen = self.sent_proj(cls_drop)
        s_sar = self.sarc_proj(cls_drop)
        s_sen_enhanced, s_sar_enhanced = self.ctai(s_sen, s_sar)
        sentiment_logits = self.sentiment_head(s_sen_enhanced)
        sarcasm_logits = self.sarcasm_head(s_sar_enhanced)
        return sentiment_logits, sarcasm_logits

    def forward(self, input_ids, attention_mask, rdrop=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        if rdrop and self.training:
            sent_1, sarc_1 = self.forward_heads(cls_output)
            sent_2, sarc_2 = self.forward_heads(cls_output)
            return sent_1, sarc_1, sent_2, sarc_2
        return self.forward_heads(cls_output)

print("Ablation models defined (NTN-only, CTAI-only)")

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
def train_stl_epoch(model, dataloader, optimizer, scheduler, criterion, scaler=None, task="sentiment"):
    model.train()
    device = next(model.parameters()).device
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch[f"{task}_label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        if scaler is not None and USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1

# %%
def train_mtl_epoch(model, dataloader, optimizer, scheduler,
                    sent_criterion, sarc_criterion,
                    lambda_sarc=LAMBDA_SARCASM, dynamic_weighter=None,
                    scaler=None):
    model.train()
    device = next(model.parameters()).device
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    total_loss = 0
    total_sent_loss = 0
    total_sarc_loss = 0
    all_sent_preds, all_sent_labels = [], []
    all_sarc_preds, all_sarc_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_labels = batch["sentiment_label"].to(device)
        sarc_labels = batch["sarcasm_label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
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

        if scaler is not None and USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
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
def evaluate_stl(model, dataloader, criterion, task="sentiment"):
    model.eval()
    device = next(model.parameters()).device
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch[f"{task}_label"].to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, precision, recall, all_preds, all_labels

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

        with torch.cuda.amp.autocast(enabled=USE_AMP):
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
# ## Section 6-8: Run All Experiments

# %%
# ====================================================================
# EXPERIMENT RUNNERS
# ====================================================================

# Thư mục lưu checkpoint
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"📁 Checkpoint directory: {CHECKPOINT_DIR}")

def run_stl_experiment(variety, task, num_epochs=NUM_EPOCHS, device=None):
    """Run a full STL experiment for one variety and one task."""
    if device is None:
        device = DEVICE
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    print(f"\n{'='*60}")
    print(f"  STL | {variety} | {task.upper()} | Device: {device}")
    print(f"{'='*60}")

    # Không lọc domain — dùng toàn bộ dữ liệu (Google + Reddit)
    train_ds, val_ds, test_ds = create_datasets(
        df_train_filtered, df_val_filtered, df_test_filtered,
        variety=variety
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    model = STLModel().to(device)

    # Tính class weights trên toàn bộ tập train của variety (Google + Reddit)
    subset = df_train_filtered[df_train_filtered["variety"] == variety]
    labels_for_weights = subset[f"{task}_label"].values

    weights = compute_class_weights(labels_for_weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and "cuda" in str(device))

    for epoch in range(num_epochs):
        train_loss, train_f1 = train_stl_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler=scaler, task=task
        )
        val_loss, val_f1, val_p, val_r, _, _ = evaluate_stl(
            model, val_loader, criterion, task
        )
        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            # Lưu checkpoint tốt nhất ra file
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_STL_{task}_{variety}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": best_model_state,
                "best_f1": best_f1,
                "variety": variety,
                "task": task,
                "model_name": MODEL_NAME,
            }, ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path} (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_loss, test_f1, test_p, test_r, test_preds, test_labels = evaluate_stl(
        model, test_loader, criterion, task
    )

    print(f"\n  TEST Results ({variety} | {task}):")
    print(f"     F1: {test_f1:.4f} | Precision: {test_p:.4f} | Recall: {test_r:.4f}")

    del model
    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    return {
        "variety": variety, "task": task, "model_type": "STL",
        "f1": test_f1, "precision": test_p, "recall": test_r,
        "preds": test_preds, "labels": test_labels
    }

# %%
def run_mtl_experiment(variety, model_class, model_name_str,
                       num_epochs=NUM_EPOCHS, lambda_sarc=LAMBDA_SARCASM,
                       use_dynamic_loss=False, domain=None, device=None):
    """Run a full MTL experiment."""
    if device is None:
        device = DEVICE
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    print(f"\n{'='*60}")
    print(f"  {model_name_str} | {variety or 'Combined'} | Device: {device}")
    print(f"{'='*60}")

    # Không lọc domain — dùng toàn bộ dữ liệu (Google + Reddit)
    train_ds, val_ds, test_ds = create_datasets(
        df_train_filtered, df_val_filtered, df_test_filtered,
        variety=variety
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Dọn dẹp VRAM trước khi khởi tạo mô hình mới
    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    model = model_class().to(device)

    # Tính class weights trên toàn bộ tập train (Google + Reddit)
    train_subset = df_train_filtered[
        df_train_filtered["variety"] == variety
    ] if variety else df_train_filtered

    sent_weights = compute_class_weights(train_subset["sentiment_label"].values, device=device)
    sarc_weights = compute_class_weights(train_subset["sarcasm_label"].values, device=device)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)
    sarc_criterion = nn.CrossEntropyLoss(weight=sarc_weights)

    # === LLRD (Layerwise Learning Rate Decay) ===
    # Thay vì đóng băng cứng, dùng LR giảm dần theo tầng:
    #   Layer trên (gần output): LR cao → học đặc trưng task-specific
    #   Layer dưới (gần input):  LR rất thấp → giữ kiến thức pretrained
    # Công thức: lr_layer_i = base_lr × decay^(num_layers - i)
    # Với decay=0.85, layer 0 có LR = 1e-5 × 0.85^24 = 1.7e-7 (gần như đóng băng)
    if hasattr(model, 'encoder'):
        num_layers = len(model.encoder.encoder.layer)
        decay_factor = 0.85

        param_groups = []

        # Embeddings: LR thấp nhất
        embed_params = [p for p in model.encoder.embeddings.parameters() if p.requires_grad]
        if embed_params:
            param_groups.append({
                'params': embed_params,
                'lr': LEARNING_RATE * (decay_factor ** (num_layers + 1))
            })

        # Encoder layers: LR giảm dần từ dưới lên trên
        for i in range(num_layers):
            layer_params = [p for p in model.encoder.encoder.layer[i].parameters() if p.requires_grad]
            if layer_params:
                layer_lr = LEARNING_RATE * (decay_factor ** (num_layers - i))
                param_groups.append({
                    'params': layer_params,
                    'lr': layer_lr
                })

        # Task heads (projections, CTAI, NTN, classifiers): LR cao nhất
        encoder_param_ids = set(id(p) for p in model.encoder.parameters())
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': LEARNING_RATE
            })

        optimizer = AdamW(param_groups, weight_decay=0.01)

        # In LR range
        lr_bottom = LEARNING_RATE * (decay_factor ** (num_layers + 1))
        lr_top = LEARNING_RATE * (decay_factor ** 1)
        print(f"  📊 LLRD (decay={decay_factor}): Embed LR={lr_bottom:.2e} | "
              f"Layer 0={LEARNING_RATE * (decay_factor ** num_layers):.2e} | "
              f"Layer {num_layers-1}={lr_top:.2e} | Heads={LEARNING_RATE:.2e}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"     Trainable: {trainable:,} / {total_params:,} (100% — no freezing)")
    else:
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    dynamic_weighter = DynamicLossWeighter() if use_dynamic_loss else None

    best_avg_f1 = 0
    best_model_state = None
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and "cuda" in str(device))

    for epoch in range(num_epochs):
        train_loss, sent_loss, sarc_loss, train_f1_sent, train_f1_sarc = train_mtl_epoch(
            model, train_loader, optimizer, scheduler,
            sent_criterion, sarc_criterion, lambda_sarc, dynamic_weighter,
            scaler=scaler
        )
        val_metrics = evaluate_mtl(model, val_loader, sent_criterion, sarc_criterion)
        avg_f1 = (val_metrics["sent_f1"] + val_metrics["sarc_f1"]) / 2

        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Sent F1: {train_f1_sent:.4f} Sarc F1: {train_f1_sarc:.4f} | "
              f"Val Sent F1: {val_metrics['sent_f1']:.4f} Sarc F1: {val_metrics['sarc_f1']:.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            # Lưu checkpoint tốt nhất ra file
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name_str}_{variety or 'Combined'}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": best_model_state,
                "best_avg_f1": best_avg_f1,
                "variety": variety or "Combined",
                "model_type": model_name_str,
                "model_name": MODEL_NAME,
            }, ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path} (Avg F1={best_avg_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_metrics = evaluate_mtl(model, test_loader, sent_criterion, sarc_criterion)

    print(f"\n  TEST Results ({variety or 'Combined'} | {model_name_str}):")
    print(f"     Sentiment - F1: {test_metrics['sent_f1']:.4f} | "
          f"P: {test_metrics['sent_precision']:.4f} | R: {test_metrics['sent_recall']:.4f}")
    print(f"     Sarcasm   - F1: {test_metrics['sarc_f1']:.4f} | "
          f"P: {test_metrics['sarc_precision']:.4f} | R: {test_metrics['sarc_recall']:.4f}")

    result = {
        "variety": variety or "Combined", "model_type": model_name_str,
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

print("Experiment runners defined")

# %% [markdown]
# ## Experiment 1A: Single-Task Learning (STL) - Sentiment Analysis
# Huấn luyện riêng biệt mô hình RoBERTa-large cho tác vụ Phân tích Cảm xúc (Sentiment Analysis)

# %%
# ====================================================================
# EXPERIMENT 1A: STL - SENTIMENT ANALYSIS
# ====================================================================

if "all_results" not in globals():
    all_results = []

print("\n" + "="*80)
print("  🚀 [TASK 1A] BẮT ĐẦU EXPERIMENT: STL SENTIMENT ANALYSIS")
print("="*80)

stl_sent_results = []
USE_2GPU = False  # Chạy tuần tự trên GPU (an toàn tuyệt đối, không lo nghẽn CPU/Deadlock)

if USE_2GPU and NUM_GPUS >= 2 and len(VARIETIES) >= 2:
    print("  ⚡ KÍCH HOẠT 2 GPU CHO STL SENTIMENT:")
    print(f"     GPU 0 (cuda:0) -> Phương ngữ: {VARIETIES[0]}")
    print(f"     GPU 1 (cuda:1) -> Phương ngữ: {VARIETIES[1]}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(run_stl_experiment, variety=VARIETIES[0], task="sentiment", device=torch.device("cuda:0"))
        f1 = executor.submit(run_stl_experiment, variety=VARIETIES[1], task="sentiment", device=torch.device("cuda:1"))
        stl_sent_results.append(f0.result())
        stl_sent_results.append(f1.result())
else:
    for variety in VARIETIES:
        stl_sent_results.append(run_stl_experiment(variety=variety, task="sentiment", device=DEVICE))

# Cập nhật kết quả vào all_results (loại bỏ kết quả cũ của task này nếu chạy lại)
all_results = [r for r in all_results if not (r.get("model_type") == "STL" and r.get("task") == "sentiment")] + stl_sent_results

# In bảng tổng kết kết quả tốt nhất của STL Sentiment
print("\n" + "="*75)
print("  🏆 KẾT QUẢ TỐT NHẤT - EXPERIMENT 1A: STL SENTIMENT ANALYSIS")
print("="*75)
for r in stl_sent_results:
    print(f"  * Phương ngữ: {r['variety']:<6} | Sentiment F1: {r['f1']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f}")
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_STL_sentiment_{r['variety']}.pt")
    if os.path.exists(ckpt):
        print(f"    💾 Checkpoint: {ckpt}")
print("="*75)

# %% [markdown]
# ## Experiment 1B: Single-Task Learning (STL) - Sarcasm Detection
# Huấn luyện riêng biệt mô hình RoBERTa-large cho tác vụ Phát hiện Mỉa mai (Sarcasm Detection)

# %%
# ====================================================================
# EXPERIMENT 1B: STL - SARCASM DETECTION
# ====================================================================

if "all_results" not in globals():
    all_results = []

print("\n" + "="*80)
print("  🚀 [TASK 1B] BẮT ĐẦU EXPERIMENT: STL SARCASM DETECTION")
print("="*80)

stl_sarc_results = []
USE_2GPU = False  # Chạy tuần tự trên GPU (an toàn tuyệt đối, không lo nghẽn CPU/Deadlock)

if USE_2GPU and NUM_GPUS >= 2 and len(VARIETIES) >= 2:
    print("  ⚡ KÍCH HOẠT 2 GPU CHO STL SARCASM:")
    print(f"     GPU 0 (cuda:0) -> Phương ngữ: {VARIETIES[0]}")
    print(f"     GPU 1 (cuda:1) -> Phương ngữ: {VARIETIES[1]}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(run_stl_experiment, variety=VARIETIES[0], task="sarcasm", device=torch.device("cuda:0"))
        f1 = executor.submit(run_stl_experiment, variety=VARIETIES[1], task="sarcasm", device=torch.device("cuda:1"))
        stl_sarc_results.append(f0.result())
        stl_sarc_results.append(f1.result())
else:
    for variety in VARIETIES:
        stl_sarc_results.append(run_stl_experiment(variety=variety, task="sarcasm", device=DEVICE))

# Cập nhật kết quả vào all_results (loại bỏ kết quả cũ của task này nếu chạy lại)
all_results = [r for r in all_results if not (r.get("model_type") == "STL" and r.get("task") == "sarcasm")] + stl_sarc_results

# In bảng tổng kết kết quả tốt nhất của STL Sarcasm
print("\n" + "="*75)
print("  🏆 KẾT QUẢ TỐT NHẤT - EXPERIMENT 1B: STL SARCASM DETECTION")
print("="*75)
for r in stl_sarc_results:
    print(f"  * Phương ngữ: {r['variety']:<6} | Sarcasm F1: {r['f1']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f}")
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_STL_sarcasm_{r['variety']}.pt")
    if os.path.exists(ckpt):
        print(f"    💾 Checkpoint: {ckpt}")
print("="*75)

# %% [markdown]
# ## Experiment 2: Multi-Task Learning (MTL) Baseline
# Mô hình đa tác vụ chuẩn (Shared RoBERTa-large + Linear Classification Heads riêng biệt)

# %%
# ====================================================================
# EXPERIMENT 2: MULTI-TASK LEARNING (MTL) BASELINE
# ====================================================================

if "all_results" not in globals():
    all_results = []

print("\n" + "="*80)
print("  🚀 [EXPERIMENT 2] BẮT ĐẦU EXPERIMENT: MTL BASELINE")
print("="*80)

mtl_base_results = []
USE_2GPU = False  # Chạy tuần tự trên GPU (an toàn tuyệt đối, không lo nghẽn CPU/Deadlock)

if USE_2GPU and NUM_GPUS >= 2 and len(VARIETIES) >= 2:
    print("  ⚡ KÍCH HOẠT 2 GPU CHO MTL BASELINE:")
    print(f"     GPU 0 (cuda:0) -> Phương ngữ: {VARIETIES[0]}")
    print(f"     GPU 1 (cuda:1) -> Phương ngữ: {VARIETIES[1]}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(
            run_mtl_experiment,
            variety=VARIETIES[0],
            model_class=MTLBaselineModel,
            model_name_str="MTL-Baseline",
            lambda_sarc=LAMBDA_SARCASM,
            device=torch.device("cuda:0")
        )
        f1 = executor.submit(
            run_mtl_experiment,
            variety=VARIETIES[1],
            model_class=MTLBaselineModel,
            model_name_str="MTL-Baseline",
            lambda_sarc=LAMBDA_SARCASM,
            device=torch.device("cuda:1")
        )
        mtl_base_results.append(f0.result())
        mtl_base_results.append(f1.result())
else:
    for variety in VARIETIES:
        mtl_base_results.append(run_mtl_experiment(
            variety=variety,
            model_class=MTLBaselineModel,
            model_name_str="MTL-Baseline",
            lambda_sarc=LAMBDA_SARCASM,
            device=DEVICE
        ))

# Cập nhật kết quả vào all_results (loại bỏ kết quả cũ của task này nếu chạy lại)
all_results = [r for r in all_results if r.get("model_type") != "MTL-Baseline"] + mtl_base_results

# In bảng tổng kết kết quả tốt nhất của MTL Baseline
print("\n" + "="*75)
print("  🏆 KẾT QUẢ TỐT NHẤT - EXPERIMENT 2: MTL BASELINE")
print("="*75)
for r in mtl_base_results:
    avg_f1 = (r['sent_f1'] + r['sarc_f1']) / 2
    print(f"  * Phương ngữ: {r['variety']:<6} | Sent F1: {r['sent_f1']:.4f} | Sarc F1: {r['sarc_f1']:.4f} | Avg F1: {avg_f1:.4f}")
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_MTL-Baseline_{r['variety']}.pt")
    if os.path.exists(ckpt):
        print(f"    💾 Checkpoint: {ckpt}")
print("="*75)

# %% [markdown]
# ## Experiment 3: Proposed Multi-Task Learning (MTL) Model
# Mô hình Đề xuất: Shared RoBERTa-large + CTAI (16 heads) + NTN (8 slices) + Dynamic Loss Weighting

# %%
# ====================================================================
# EXPERIMENT 3: PROPOSED MTL MODEL (CTAI + NTN + DYNAMIC LOSS)
# ====================================================================

if "all_results" not in globals():
    all_results = []

print("\n" + "="*80)
print("  🚀 [EXPERIMENT 3] BẮT ĐẦU EXPERIMENT: PROPOSED MTL MODEL")
print("="*80)

mtl_prop_results = []
USE_2GPU = False  # Chạy tuần tự trên GPU (an toàn tuyệt đối, không lo nghẽn CPU/Deadlock)

if USE_2GPU and NUM_GPUS >= 2 and len(VARIETIES) >= 2:
    print("  ⚡ KÍCH HOẠT 2 GPU CHO MTL PROPOSED:")
    print(f"     GPU 0 (cuda:0) -> Phương ngữ: {VARIETIES[0]}")
    print(f"     GPU 1 (cuda:1) -> Phương ngữ: {VARIETIES[1]}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(
            run_mtl_experiment,
            variety=VARIETIES[0],
            model_class=MTLProposedModel,
            model_name_str="MTL-Proposed",
            use_dynamic_loss=True,
            device=torch.device("cuda:0")
        )
        f1 = executor.submit(
            run_mtl_experiment,
            variety=VARIETIES[1],
            model_class=MTLProposedModel,
            model_name_str="MTL-Proposed",
            use_dynamic_loss=True,
            device=torch.device("cuda:1")
        )
        mtl_prop_results.append(f0.result())
        mtl_prop_results.append(f1.result())
else:
    for variety in VARIETIES:
        mtl_prop_results.append(run_mtl_experiment(
            variety=variety,
            model_class=MTLProposedModel,
            model_name_str="MTL-Proposed",
            use_dynamic_loss=True,
            device=DEVICE
        ))

# Cập nhật kết quả vào all_results (loại bỏ kết quả cũ của task này nếu chạy lại)
all_results = [r for r in all_results if r.get("model_type") != "MTL-Proposed"] + mtl_prop_results

# In bảng tổng kết kết quả tốt nhất của MTL Proposed
print("\n" + "="*75)
print("  🏆 KẾT QUẢ TỐT NHẤT - EXPERIMENT 3: PROPOSED MTL MODEL")
print("="*75)
for r in mtl_prop_results:
    avg_f1 = (r['sent_f1'] + r['sarc_f1']) / 2
    print(f"  * Phương ngữ: {r['variety']:<6} | Sent F1: {r['sent_f1']:.4f} | Sarc F1: {r['sarc_f1']:.4f} | Avg F1: {avg_f1:.4f}")
    print(f"    - Sentiment: Precision = {r['sent_precision']:.4f}, Recall = {r['sent_recall']:.4f}")
    print(f"    - Sarcasm:   Precision = {r['sarc_precision']:.4f}, Recall = {r['sarc_recall']:.4f}")
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_MTL-Proposed_{r['variety']}.pt")
    if os.path.exists(ckpt):
        print(f"    💾 Checkpoint: {ckpt}")
print("="*75)


# %% [markdown]
# ## Cross-Variety Evaluation

# %%
# ====================================================================
# CROSS-VARIETY EVALUATION
# ====================================================================

print("\n" + "="*80)
print("  EXPERIMENT 4: CROSS-VARIETY EVALUATION")
print("="*80)

if "cross_variety_results" not in globals():
    cross_variety_results = []

def run_cross_variety(train_variety, test_variety, model_class, model_name_str, device=None):
    if device is None:
        device = DEVICE
    if "cuda" in str(device):
        torch.cuda.set_device(device)
    print(f"\n{'='*60}")
    print(f"  CROSS-VARIETY | Train: {train_variety} -> Test: {test_variety} | {model_name_str} | Device: {device}")
    print(f"{'='*60}")

    # Không lọc domain — dùng toàn bộ dữ liệu (Google + Reddit)
    train_sub = df_train_filtered[
        df_train_filtered["variety"] == train_variety
    ]
    val_sub = df_val_filtered[
        df_val_filtered["variety"] == train_variety
    ]
    test_sub = df_test_filtered[
        df_test_filtered["variety"] == test_variety
    ]

    train_ds = BESSTIEDataset(
        train_sub["text_clean"].values, train_sub["sentiment_label"].values,
        train_sub["sarcasm_label"].values, tokenizer
    )
    val_ds = BESSTIEDataset(
        val_sub["text_clean"].values, val_sub["sentiment_label"].values,
        val_sub["sarcasm_label"].values, tokenizer
    )
    test_ds = BESSTIEDataset(
        test_sub["text_clean"].values, test_sub["sentiment_label"].values,
        test_sub["sarcasm_label"].values, tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    model = model_class().to(device)

    sent_weights = compute_class_weights(train_sub["sentiment_label"].values, device=device)
    sarc_weights = compute_class_weights(train_sub["sarcasm_label"].values, device=device)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)
    sarc_criterion = nn.CrossEntropyLoss(weight=sarc_weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_avg_f1 = 0
    best_model_state = None
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and "cuda" in str(device))

    for epoch in range(NUM_EPOCHS):
        train_mtl_epoch(model, train_loader, optimizer, scheduler,
                        sent_criterion, sarc_criterion, LAMBDA_SARCASM, scaler=scaler)
        val_metrics = evaluate_mtl(model, val_loader, sent_criterion, sarc_criterion)
        avg_f1 = (val_metrics["sent_f1"] + val_metrics["sarc_f1"]) / 2

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Val Sent F1: {val_metrics['sent_f1']:.4f} Sarc F1: {val_metrics['sarc_f1']:.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_metrics = evaluate_mtl(model, test_loader, sent_criterion, sarc_criterion)

    print(f"\n  CROSS-VARIETY TEST ({train_variety} -> {test_variety}):")
    print(f"     Sentiment F1: {test_metrics['sent_f1']:.4f}")
    print(f"     Sarcasm F1: {test_metrics['sarc_f1']:.4f}")

    del model
    import gc
    gc.collect()
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    return {
        "train_variety": train_variety, "test_variety": test_variety,
        "model_type": model_name_str,
        "sent_f1": test_metrics["sent_f1"], "sarc_f1": test_metrics["sarc_f1"],
        "sent_precision": test_metrics["sent_precision"],
        "sent_recall": test_metrics["sent_recall"],
        "sarc_precision": test_metrics["sarc_precision"],
        "sarc_recall": test_metrics["sarc_recall"],
    }

# %%
if NUM_GPUS >= 2 and len(VARIETIES) >= 2:
    print("\n⚡ KÍCH HOẠT 2 GPU CHO CROSS-VARIETY:")
    print("   GPU 0 -> en-AU -> en-UK")
    print("   GPU 1 -> en-UK -> en-AU")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_cv0 = executor.submit(run_cross_variety, "en-AU", "en-UK", MTLProposedModel, "MTL-Proposed", torch.device("cuda:0"))
        f_cv1 = executor.submit(run_cross_variety, "en-UK", "en-AU", MTLProposedModel, "MTL-Proposed", torch.device("cuda:1"))
        cross_variety_results.append(f_cv0.result())
        cross_variety_results.append(f_cv1.result())
else:
    cross_variety_results.append(
        run_cross_variety("en-AU", "en-UK", MTLProposedModel, "MTL-Proposed", device=DEVICE)
    )
    cross_variety_results.append(
        run_cross_variety("en-UK", "en-AU", MTLProposedModel, "MTL-Proposed", device=DEVICE)
    )

# Combined training
for test_variety in VARIETIES:
    result = run_mtl_experiment(
        variety=None,
        model_class=MTLProposedModel,
        model_name_str=f"MTL-Proposed-Combined",
        use_dynamic_loss=True,
        device=DEVICE
    )
    cross_variety_results.append({
        "train_variety": "Combined", "test_variety": test_variety,
        "model_type": "MTL-Proposed",
        "sent_f1": result["sent_f1"], "sarc_f1": result["sarc_f1"],
    })

print("\nCross-Variety Evaluation complete!")

# %% [markdown]
# ## Ablation Study

# %%
# ====================================================================
# ABLATION STUDY (HỖ TRỢ ĐA GPU TỰ ĐỘNG)
# ====================================================================

print("\n" + "="*80)
print("  EXPERIMENT 5: ABLATION STUDY")
print("="*80)

if "ablation_results" not in globals():
    ablation_results = []

def run_ablation_step(model_class, model_name_str):
    """Chạy ablation cho các variety (song song 2 GPU nếu có)."""
    if NUM_GPUS >= 2 and len(VARIETIES) >= 2:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            f0 = executor.submit(run_mtl_experiment, VARIETIES[0], model_class, model_name_str, NUM_EPOCHS, LAMBDA_SARCASM, False, None, torch.device("cuda:0"))
            f1 = executor.submit(run_mtl_experiment, VARIETIES[1], model_class, model_name_str, NUM_EPOCHS, LAMBDA_SARCASM, False, None, torch.device("cuda:1"))
            return [f0.result(), f1.result()]
    else:
        step_res = []
        for v in VARIETIES:
            step_res.append(run_mtl_experiment(variety=v, model_class=model_class, model_name_str=model_name_str, device=DEVICE))
        return step_res

# Ablation 1: CTAI only
print("\n--- Ablation 1: CTAI only ---")
ablation_results.extend(run_ablation_step(MTLCTAIOnlyModel, "MTL-CTAI-only"))

# Ablation 2: NTN only
print("\n--- Ablation 2: NTN only ---")
ablation_results.extend(run_ablation_step(MTLNTNOnlyModel, "MTL-NTN-only"))

# %%
# Ablation 3: Lambda sensitivity
print("\nLambda sensitivity analysis...")
if "lambda_results" not in globals():
    lambda_results = []

for lam in [0.3, 0.5, 0.7, 1.0]:
    print(f"\n--- Testing Lambda = {lam} ---")
    if NUM_GPUS >= 2 and len(VARIETIES) >= 2:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            f0 = executor.submit(run_mtl_experiment, VARIETIES[0], MTLBaselineModel, f"MTL-Baseline-lam={lam}", NUM_EPOCHS, lam, False, None, torch.device("cuda:0"))
            f1 = executor.submit(run_mtl_experiment, VARIETIES[1], MTLBaselineModel, f"MTL-Baseline-lam={lam}", NUM_EPOCHS, lam, False, None, torch.device("cuda:1"))
            r0 = f0.result()
            r0["lambda"] = lam
            r1 = f1.result()
            r1["lambda"] = lam
            lambda_results.extend([r0, r1])
    else:
        for v in VARIETIES:
            r = run_mtl_experiment(variety=v, model_class=MTLBaselineModel, model_name_str=f"MTL-Baseline-lam={lam}", lambda_sarc=lam, device=DEVICE)
            r["lambda"] = lam
            lambda_results.append(r)

print("\nAblation Study complete!")

# %% [markdown]
# ## Results Visualization

# %%
# ====================================================================
# RESULTS VISUALIZATION
# ====================================================================

print("\n" + "="*80)
print("  MAIN RESULTS: STL vs MTL-Baseline vs MTL-Proposed")
print("="*80)

if "all_results" not in globals():
    all_results = []

results_table = []
for r in all_results:
    if r["model_type"] == "STL":
        results_table.append({
            "Model": f"STL-{r['task'].capitalize()}",
            "Variety": r["variety"],
            "Sentiment F1": f"{r['f1']:.4f}" if r["task"] == "sentiment" else "-",
            "Sarcasm F1": f"{r['f1']:.4f}" if r["task"] == "sarcasm" else "-",
        })
    else:
        results_table.append({
            "Model": r["model_type"],
            "Variety": r["variety"],
            "Sentiment F1": f"{r['sent_f1']:.4f}",
            "Sarcasm F1": f"{r['sarc_f1']:.4f}",
        })

df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False))

# %%
# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("STL vs MTL-Baseline vs MTL-Proposed", fontsize=14, fontweight="bold")

for i, variety in enumerate(VARIETIES):
    ax = axes[i]

    stl_sent_f1 = [r["f1"] for r in all_results
                   if r["model_type"] == "STL" and r["variety"] == variety and r.get("task") == "sentiment"]
    stl_sarc_f1 = [r["f1"] for r in all_results
                   if r["model_type"] == "STL" and r["variety"] == variety and r.get("task") == "sarcasm"]
    mtl_base = [r for r in all_results
                if r["model_type"] == "MTL-Baseline" and r["variety"] == variety]
    mtl_prop = [r for r in all_results
                if r["model_type"] == "MTL-Proposed" and r["variety"] == variety]

    models = ["STL", "MTL-Baseline", "MTL-Proposed"]
    sent_f1s = [
        stl_sent_f1[0] if stl_sent_f1 else 0,
        mtl_base[0]["sent_f1"] if mtl_base else 0,
        mtl_prop[0]["sent_f1"] if mtl_prop else 0,
    ]
    sarc_f1s = [
        stl_sarc_f1[0] if stl_sarc_f1 else 0,
        mtl_base[0]["sarc_f1"] if mtl_base else 0,
        mtl_prop[0]["sarc_f1"] if mtl_prop else 0,
    ]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, sent_f1s, width, label="Sentiment F1", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + width/2, sarc_f1s, width, label="Sarcasm F1", color="#e67e22", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title(f"{variety}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("stl_vs_mtl_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: stl_vs_mtl_comparison.png")

# %%
# Cross-variety results
print("\nCross-Variety Evaluation Results:")
cv_df = pd.DataFrame(cross_variety_results)
print(cv_df.to_string(index=False))

# %%
# Ablation results
print("\nAblation Study Results:")
abl_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ["sent_preds", "sent_labels", "sarc_preds", "sarc_labels"]}
                        for r in ablation_results])
if len(abl_df) > 0:
    print(abl_df.to_string(index=False))

# %%
# Lambda sensitivity plot
if lambda_results:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Lambda Sensitivity Analysis", fontsize=14, fontweight="bold")

    for i, variety in enumerate(VARIETIES):
        ax = axes[i]
        variety_data = [r for r in lambda_results if r["variety"] == variety]
        lambdas = [r["lambda"] for r in variety_data]
        sent_f1s = [r["sent_f1"] for r in variety_data]
        sarc_f1s = [r["sarc_f1"] for r in variety_data]

        ax.plot(lambdas, sent_f1s, "o-", color="#2ecc71", label="Sentiment F1", linewidth=2)
        ax.plot(lambdas, sarc_f1s, "s-", color="#e67e22", label="Sarcasm F1", linewidth=2)
        ax.set_xlabel("Lambda (sarcasm loss weight)")
        ax.set_ylabel("Macro F1-Score")
        ax.set_title(f"{variety}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lambda_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: lambda_sensitivity.png")

# %% [markdown]
# ## Confusion Matrices & Error Analysis

# %%
# ====================================================================
# CONFUSION MATRICES
# ====================================================================

proposed_results = [r for r in all_results if r["model_type"] == "MTL-Proposed"]

if proposed_results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Confusion Matrices - MTL-Proposed", fontsize=14, fontweight="bold")

    for i, result in enumerate(proposed_results):
        variety = result["variety"]

        cm_sent = confusion_matrix(result["sent_labels"], result["sent_preds"])
        sns.heatmap(cm_sent, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    ax=axes[0][i])
        axes[0][i].set_title(f"{variety} - Sentiment")
        axes[0][i].set_xlabel("Predicted")
        axes[0][i].set_ylabel("Actual")

        cm_sarc = confusion_matrix(result["sarc_labels"], result["sarc_preds"])
        sns.heatmap(cm_sarc, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["Not Sarcastic", "Sarcastic"],
                    yticklabels=["Not Sarcastic", "Sarcastic"],
                    ax=axes[1][i])
        axes[1][i].set_title(f"{variety} - Sarcasm")
        axes[1][i].set_xlabel("Predicted")
        axes[1][i].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrices.png")

# %%
# Classification reports
print("\nDetailed Classification Reports:")
for result in proposed_results:
    variety = result["variety"]
    print(f"\n{'='*40}")
    print(f"  {variety} - Sentiment")
    print(f"{'='*40}")
    print(classification_report(
        result["sent_labels"], result["sent_preds"],
        target_names=["Negative", "Positive"]
    ))
    print(f"\n  {variety} - Sarcasm")
    print(f"{'='*40}")
    print(classification_report(
        result["sarc_labels"], result["sarc_preds"],
        target_names=["Not Sarcastic", "Sarcastic"]
    ))

# %%
# Error examples
print("\nError Analysis - Misclassified Examples:")

for variety in VARIETIES:
    test_subset = df_test_filtered[
        (df_test_filtered["variety"] == variety) &
        (df_test_filtered["domain"] == "REDDIT")
    ].reset_index(drop=True)

    proposed = [r for r in all_results
                if r["model_type"] == "MTL-Proposed" and r["variety"] == variety]

    if not proposed or len(test_subset) == 0:
        continue

    result = proposed[0]
    preds_sarc = np.array(result["sarc_preds"])
    labels_sarc = np.array(result["sarc_labels"])

    sarc_errors = np.where(preds_sarc != labels_sarc)[0]

    print(f"\n{'='*60}")
    print(f"  {variety} - Sarcasm Misclassifications ({len(sarc_errors)} total)")
    print(f"{'='*60}")

    for idx in sarc_errors[:5]:
        if idx < len(test_subset):
            text = test_subset.iloc[idx]["text_clean"]
            print(f"\n  Text: {text[:150]}...")
            print(f"  True: {'Sarcastic' if labels_sarc[idx]==1 else 'Not Sarcastic'} | "
                  f"Pred: {'Sarcastic' if preds_sarc[idx]==1 else 'Not Sarcastic'}")

# %% [markdown]
# ## Final Summary

# %%
# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("  FINAL SUMMARY")
print("="*80)

print("\nAll experiment results saved to:")
print("   - stl_vs_mtl_comparison.png")
print("   - confusion_matrices.png")
print("   - lambda_sensitivity.png")
print("   - label_distribution.png")

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
        "main_results": all_results_clean,
        "cross_variety": [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                           for k, v in r.items()} for r in cross_variety_results],
        "ablation": [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                      for k, v in r.items()
                      if k not in ["sent_preds", "sent_labels", "sarc_preds", "sarc_labels"]}
                     for r in ablation_results],
    }, f, indent=2)

print("Results saved to: experiment_results.json")
print(f"\nAll experiments completed!")
print(f"   Total configurations tested: {len(all_results)}")
print(f"   Cross-variety experiments: {len(cross_variety_results)}")
print(f"   Ablation experiments: {len(ablation_results)}")
