# %% [markdown]
# # 🔬 Đánh giá mô hình MTL-Proposed (Qwen2.5-14B) trên tập Validation (valid.csv)
#
# Script này:
# 1. Đọc file validation: `valid.csv`
# 2. Tải các checkpoint: `/mnt/en-au/checkpoints/best_MTL-Proposed_{variety}.pt`
# 3. Sử dụng `weights_only=False` để tương thích
# 4. Đánh giá và in kết quả dự đoán chuẩn CodaLab
# 5. KHÔNG tạo hay nén file ZIP rác.

# %%
import os
import re
import warnings
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {DEVICE}")

# --- Hyperparameters & Architecture Config ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B"
MAX_LEN = 384
BATCH_SIZE = 16  # Quá trình inference tốn ít VRAM hơn nên có thể nâng BS lên 16

# LoRA Params (Phải khớp hoàn toàn với lúc train)
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DROPOUT = 0.1
NTN_SLICES = 8
CTAI_HEADS = 16
USE_AMP = torch.cuda.is_available()

# Lấy hidden size tự động
_llm_config = AutoConfig.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
HIDDEN_DIM = _llm_config.hidden_size  # 5120

print(f"\n📋 Config:")
print(f"   Model: {LLM_MODEL_NAME}")
print(f"   Hidden Dim: {HIDDEN_DIM}")
print(f"   Max Length: {MAX_LEN}")
print(f"   Batch Size: {BATCH_SIZE}")

# %%
# ====================================================================
# 2. ĐƯỜNG DẪN DỮ LIỆU & CHECKPOINTS
# ====================================================================
VALID_PATH = "valid.csv"
if not os.path.exists(VALID_PATH) and os.path.exists("/mnt/en-au/valid.csv"):
    VALID_PATH = "/mnt/en-au/valid.csv"

# Đường dẫn 2 Checkpoint mặc định trên Modal Volume
CHECKPOINT_PATHS = {
    "en-AU": "/mnt/en-au/checkpoints/best_MTL-Proposed_en-AU.pt",
    "en-UK": "/mnt/en-au/checkpoints/best_MTL-Proposed_en-UK.pt",
}

# Fallback nếu ở local
for v, p in CHECKPOINT_PATHS.items():
    if not os.path.exists(p) and os.path.exists(f"checkpoints/best_MTL-Proposed_{v}.pt"):
        CHECKPOINT_PATHS[v] = f"checkpoints/best_MTL-Proposed_{v}.pt"

OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📂 File Validation: {VALID_PATH}")
for v, p in CHECKPOINT_PATHS.items():
    print(f"   Checkpoint {v}: {p} (Tồn tại: {os.path.exists(p)})")


# %%
# ====================================================================
# 3. KIẾN TRÚC MÔ HÌNH QWEN (GIỐNG MTL_QWEN_QLORA.PY)
# ====================================================================
class CrossTaskAttention(nn.Module):
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

class NTNFusion(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, k=NTN_SLICES):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.randn(k, hidden_dim, hidden_dim) * 0.01)
        self.V = nn.Linear(hidden_dim * 2, k, bias=True)

    def forward(self, sent_repr, sarc_repr):
        batch_size = sent_repr.size(0)
        bilinear = torch.zeros(batch_size, self.k, device=sent_repr.device, dtype=sent_repr.dtype)
        for i in range(self.k):
            bilinear[:, i] = torch.sum(
                sent_repr * torch.mm(sarc_repr, self.W[i].t()), dim=1
            )
        concat = torch.cat([sent_repr, sarc_repr], dim=1)
        linear = self.V(concat)
        fused = torch.tanh(bilinear + linear)
        return fused

class MTLProposedModel(nn.Module):
    def __init__(self, dropout=DROPOUT, ntn_slices=NTN_SLICES, ctai_heads=CTAI_HEADS):
        super().__init__()
        # --- Load Qwen2.5-14B ---
        self.encoder = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
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

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Dùng hidden_states của lớp cuối cùng, không phải logits
            hidden_states = outputs.hidden_states[-1]
            pooled_output = self._get_last_token_repr(hidden_states, attention_mask)
        return self.forward_heads(pooled_output)

print("✅ MTLProposedModel (Qwen2.5-14B) architecture defined")


# %%
# ====================================================================
# 4. TIỀN XỬ LÝ & DATASET
# ====================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_csv_auto(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(path, sep=sep)

def prepare_df(df):
    if "sentiment" in df.columns and "sentiment_label" not in df.columns:
        df["sentiment_label"] = df["sentiment"].astype(int)
    if "sarcasm" in df.columns and "sarcasm_label" not in df.columns:
        df["sarcasm_label"] = df["sarcasm"].astype(int)
    if "source" in df.columns and "domain" not in df.columns:
        df["domain"] = df["source"].str.upper()
    elif "domain" not in df.columns:
        df["domain"] = "UNKNOWN"
    return df

class BESSTIEValidationDataset(Dataset):
    def __init__(self, texts, sent_labels, sarc_labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.sent_labels = sent_labels
        self.sarc_labels = sarc_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            "sentiment_label": torch.tensor(self.sent_labels[idx], dtype=torch.long),
            "sarcasm_label": torch.tensor(self.sarc_labels[idx], dtype=torch.long),
        }

# %%
# ====================================================================
# 5. HÀM ĐÁNH GIÁ
# ====================================================================
@torch.no_grad()
def evaluate_mtl_validation(model, dataloader, device=DEVICE):
    model.eval()
    all_sent_preds, all_sent_labels = [], []
    all_sarc_preds, all_sarc_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_labels = batch["sentiment_label"].to(device)
        sarc_labels = batch["sarcasm_label"].to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16):
            sent_logits, sarc_logits = model(input_ids, attention_mask)

        all_sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
        all_sent_labels.extend(sent_labels.cpu().numpy())
        all_sarc_preds.extend(torch.argmax(sarc_logits, dim=1).cpu().numpy())
        all_sarc_labels.extend(sarc_labels.cpu().numpy())

    return {
        "sent_f1": f1_score(all_sent_labels, all_sent_preds, average="macro"),
        "sent_precision": precision_score(all_sent_labels, all_sent_preds, average="macro", zero_division=0),
        "sent_recall": recall_score(all_sent_labels, all_sent_preds, average="macro", zero_division=0),
        "sarc_f1": f1_score(all_sarc_labels, all_sarc_preds, average="macro"),
        "sarc_precision": precision_score(all_sarc_labels, all_sarc_preds, average="macro", zero_division=0),
        "sarc_recall": recall_score(all_sarc_labels, all_sarc_preds, average="macro", zero_division=0),
        "sent_preds": all_sent_preds, "sent_labels": all_sent_labels,
        "sarc_preds": all_sarc_preds, "sarc_labels": all_sarc_labels,
    }


# %%
# ====================================================================
# 6. ĐỌC DỮ LIỆU & CHẠY ĐÁNH GIÁ
# ====================================================================
print("\n📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Bắt buộc cho decoder-only models (Qwen) khi sinh input đồng đều
tokenizer.padding_side = "left"

print(f"\n📂 Đọc tập validation: {VALID_PATH}")
if not os.path.exists(VALID_PATH):
    print("❌ LỖI: Không tìm thấy file validation. Hãy chắc chắn đường dẫn đúng!")
else:
    df_val_raw = load_csv_auto(VALID_PATH)
    df_val = prepare_df(df_val_raw)
    df_val["text_clean"] = df_val["text"].apply(clean_text)

    print(f"   Tổng số dòng: {len(df_val):,}")
    
    validation_results = []
    df_val["pred_sentiment"] = -1
    df_val["pred_sarcasm"] = -1

    for variety in ["en-AU", "en-UK"]:
        idx = df_val["variety"].astype(str).str.strip().str.lower() == variety.lower()
        subset = df_val[idx].reset_index(drop=True)

        if len(subset) == 0:
            print(f"⚠️ Không tìm thấy dữ liệu cho {variety}. Bỏ qua.")
            continue

        ckpt_path = CHECKPOINT_PATHS[variety]
        print(f"\n{'='*60}")
        print(f"  MTL-Proposed | {variety} | Device: {DEVICE}")
        print(f"{'='*60}")
        print(f"  📥 Load checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"  ❌ Không tìm thấy file checkpoint: {ckpt_path}")
            continue

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        val_f1_display = checkpoint.get('best_avg_f1', 'N/A')
        if isinstance(val_f1_display, (int, float, np.floating)):
            val_f1_str = f"{float(val_f1_display):.4f}"
        else:
            val_f1_str = str(val_f1_display)
        print(f"     Epoch: {checkpoint.get('epoch', 'N/A')} | Best Train Val F1: {val_f1_str}")

        state_dict = checkpoint["model_state_dict"]
        detected_slices = state_dict["ntn.W"].shape[0] if "ntn.W" in state_dict else NTN_SLICES

        # Khởi tạo lại model & chuyển head lên GPU
        model = MTLProposedModel(ntn_slices=detected_slices, ctai_heads=CTAI_HEADS)
        model.to(DEVICE)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"     ℹ️ Bỏ qua các key phụ: {len(unexpected)} keys")
        model.eval()

        val_ds = BESSTIEValidationDataset(
            subset["text_clean"].values,
            subset["sentiment_label"].values,
            subset["sarcasm_label"].values,
            tokenizer
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        metrics = evaluate_mtl_validation(model, val_loader, device=DEVICE)
        metrics["variety"] = variety
        validation_results.append(metrics)

        df_val.loc[idx, "pred_sentiment"] = metrics["sent_preds"]
        df_val.loc[idx, "pred_sarcasm"] = metrics["sarc_preds"]

        avg_f1_current = (metrics['sent_f1'] + metrics['sarc_f1']) / 2
        print(f"\n  TEST Results ({variety} | MTL-Proposed):")
        print(f"     Sentiment - F1: {metrics['sent_f1']:.4f} | P: {metrics['sent_precision']:.4f} | R: {metrics['sent_recall']:.4f}")
        print(f"     Sarcasm   - F1: {metrics['sarc_f1']:.4f} | P: {metrics['sarc_precision']:.4f} | R: {metrics['sarc_recall']:.4f}")
        print(f"     Average F1: {avg_f1_current:.4f}")

        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%
# ====================================================================
# 7. IN BẢNG TỔNG KẾT & CONFUSION MATRIX
# ====================================================================
if 'validation_results' in locals() and validation_results:
    print("\n" + "="*75)
    print("  🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION - MTL PROPOSED MODEL")
    print("="*75)
    for r in validation_results:
        avg_f1 = (r['sent_f1'] + r['sarc_f1']) / 2
        print(f"  * Phương ngữ: {r['variety']:<6} | Sent F1: {r['sent_f1']:.4f} | Sarc F1: {r['sarc_f1']:.4f} | Avg F1: {avg_f1:.4f}")
    
    print("\nDetailed Classification Reports:")
    for r in validation_results:
        variety = r["variety"]
        print(f"\n{'='*40}\n  {variety} - Sentiment\n{'='*40}")
        print(classification_report(r["sent_labels"], r["sent_preds"], target_names=["Negative", "Positive"], digits=4))
        print(f"\n{'='*40}\n  {variety} - Sarcasm\n{'='*40}")
        print(classification_report(r["sarc_labels"], r["sarc_preds"], target_names=["Not Sarcastic", "Sarcastic"], digits=4))

    # Vẽ và lưu Confusion Matrix
    try:
        n_vars = len(validation_results)
        fig, axes = plt.subplots(2, n_vars, figsize=(6 * n_vars, 10))
        if n_vars == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle("Confusion Matrices - MTL-Proposed (Validation Set)", fontsize=14, fontweight="bold")
        for i, result in enumerate(validation_results):
            variety = result["variety"]
            cm_sent = confusion_matrix(result["sent_labels"], result["sent_preds"])
            sns.heatmap(cm_sent, annot=True, fmt="d", cmap="Greens", ax=axes[0][i])
            axes[0][i].set_title(f"{variety} - Sentiment")
            
            cm_sarc = confusion_matrix(result["sarc_labels"], result["sarc_preds"])
            sns.heatmap(cm_sarc, annot=True, fmt="d", cmap="Oranges", ax=axes[1][i])
            axes[1][i].set_title(f"{variety} - Sarcasm")

        plt.tight_layout()
        cm_save_path = os.path.join(OUTPUT_DIR, "confusion_matrices_validation.png")
        plt.savefig(cm_save_path, dpi=150, bbox_inches="tight")
        print(f"\n🖼️ Đã lưu biểu đồ ma trận nhầm lẫn: {cm_save_path}")
    except Exception as e:
        print(f"⚠️ Không thể vẽ Confusion Matrix: {e}")

# %%
# ====================================================================
# 8. LƯU KẾT QUẢ DỰ ĐOÁN RA CSV & ZIP (CODABENCH / CODALAB)
# ====================================================================
if 'df_val' in locals() and not df_val.empty:
    df_out = df_val.copy()
    df_out["sentiment"] = df_out["pred_sentiment"].astype(int)
    df_out["sarcasm"] = df_out["pred_sarcasm"].astype(int)

    cols_to_drop = [c for c in ["text_clean", "pred_sentiment", "pred_sarcasm", "sentiment_label", "sarcasm_label"] if c in df_out.columns]
    df_out = df_out.drop(columns=cols_to_drop)

    standard_order = ["source", "variety", "text", "sentiment", "sarcasm"]
    ordered_cols = [c for c in standard_order if c in df_out.columns]
    remaining_cols = [c for c in df_out.columns if c not in ordered_cols]
    df_out = df_out[ordered_cols + remaining_cols]

    csv_filename = "answer.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    df_out.to_csv(csv_path, sep=",", index=False, encoding="utf-8")

    zip_filename = "answer.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=csv_filename)

    print("\n" + "="*70)
    print(f"  💾 ĐÃ LƯU FILE KẾT QUẢ VÀ TẠO ZIP THÀNH CÔNG:")
    print(f"     CSV : {csv_path}")
    print(f"     ZIP : {zip_path}")
    print("="*70)
