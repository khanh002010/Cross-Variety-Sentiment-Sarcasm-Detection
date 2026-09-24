# B2. Mục tiêu, nội dung, kế hoạch nghiên cứu

## B2.1. Mục tiêu

### Mục tiêu:

**Mục tiêu 1: Xây dựng và đánh giá hệ thống Multi-Task Learning (MTL) cho bài toán đồng thời phân loại Sentiment và nhận diện Sarcasm trên hai phương ngữ Tiếng Anh**

- Thiết kế kiến trúc MTL dựa trên các mô hình Transformer pre-trained (BERT [1], RoBERTa [2], DeBERTa [3]), trong đó sentiment classification và sarcasm detection được mô hình hóa như hai tác vụ liên quan (related tasks) và huấn luyện đồng thời trong cùng một mạng nơ-ron.
- Nghiên cứu và triển khai các cơ chế chia sẻ thông tin giữa hai tác vụ (inter-task interaction), bao gồm:
  * Shared Encoder: Sử dụng chung một Transformer encoder cho cả hai tác vụ [4][5].
  * Task-specific Heads: Các lớp phân loại riêng biệt cho từng tác vụ [4][5].
  * Cross-Task Attention Interaction (CTAI): Cho phép thông tin sentiment hỗ trợ nhận diện sarcasm và ngược lại [6].
  * Neural Tensor Network (NTN) Fusion: Kết hợp biểu diễn hai tác vụ thông qua mạng tensor [4].
- Tiêu chí đánh giá:
  * Macro-averaged F1-Score, Precision, Recall cho từng tác vụ (sentiment, sarcasm) và từng phương ngữ (en-AU, en-UK).
  * So sánh hiệu suất MTL vs. Single-Task Learning (STL) để đánh giá mức độ cải thiện nhờ multi-task learning.
  * Cross-variety evaluation: đánh giá khả năng tổng quát hóa giữa hai phương ngữ.

**Mục tiêu 2: Nghiên cứu tác động của phương ngữ (Language Variety) đến hiệu suất phân loại Sentiment và nhận diện Sarcasm**

- Đánh giá sự khác biệt về hiệu suất mô hình giữa hai phương ngữ tiếng Anh Úc (Australian English, en-AU) và tiếng Anh Anh (British English, en-UK).
- Phân tích các yếu tố ngôn ngữ đặc trưng (slang, idioms, cultural pragmatics) ảnh hưởng đến khả năng nhận diện sarcasm trong từng phương ngữ, dựa trên các phát hiện từ nghiên cứu về slang trong các biến thể tiếng Anh [7][8].
- Thực hiện cross-variety evaluation: huấn luyện trên một phương ngữ, đánh giá trên phương ngữ còn lại, để kiểm tra tính tổng quát hóa [5].

**Mục tiêu 3: So sánh, phân tích và đề xuất kiến trúc MTL tối ưu cho bài toán variety-aware sentiment-sarcasm classification**

- So sánh các phương pháp fine-tuning: Standard Fine-tuning, LoRA [9], Adapter Layers, QLoRA.
- Đề xuất kiến trúc phù hợp nhất và hướng nghiên cứu tương lai.
- Hướng đến việc viết và submit bài báo khoa học lên tạp chí/hội nghị quốc tế (mục tiêu Q4).

---

### Đối tượng nghiên cứu:

**Bộ dữ liệu sử dụng cho nghiên cứu:**

- **BESSTIE** (BEnchmark for Sentiment and Sarcasm for varieTIes of English) [5] — Dataset chính, công khai trên HuggingFace (https://huggingface.co/datasets/unswnlporg/BESSTIE).
  - Đặc điểm chính:
    * Đa phương ngữ: Australian English (en-AU), Indian English (en-IN), British English (en-UK).
    * Phạm vi: Nghiên cứu này tập trung vào 2 phương ngữ en-AU và en-UK.
    * Hai domain: Google Places reviews (location-based) và Reddit comments (topic-based).
    * Hai nhãn phân loại: Sentiment (positive/negative) và Sarcasm (sarcastic/not sarcastic).
    * Kích thước: ~2,700–3,300 samples/variety/domain (chi tiết trong Bảng 1).

| Phương ngữ | Domain   | Train | Valid | Test | % Positive Sent. | % Sarcastic |
|-----------|----------|-------|-------|------|-------------------|-------------|
| en-AU     | GOOGLE   | 946   | 130   | 270  | 73%               | 7%          |
| en-AU     | REDDIT   | 1763  | 241   | 501  | 32%               | 42%         |
| en-UK     | GOOGLE   | 1817  | 248   | 517  | 75%               | 0%          |
| en-UK     | REDDIT   | 1007  | 138   | 287  | 12%               | 22%         |

*Bảng 1: Thống kê dữ liệu BESSTIE cho en-AU và en-UK [5]*

  - Tiền xử lý dữ liệu:
    * Lọc ngôn ngữ bằng fastText word vectors (ngưỡng 0.98 cho English).
    * Xử lý mất cân bằng nhãn: stratified sampling, weighted loss function.
    * Chuẩn hóa văn bản: loại bỏ URLs, ký tự đặc biệt, chuẩn hóa emoji/emoticon.
    * Phân chia: train/validation/test theo tỉ lệ stratified.
  - Thách thức:
    * Mất cân bằng nhãn nghiêm trọng đối với sarcasm (đặc biệt GOOGLE subset gần như không có sarcasm).
    * Sự khác biệt về style giữa Google reviews (formal) và Reddit comments (informal, nhiều slang).
    * Sarcasm trong en-AU có thể mang sắc thái văn hóa khác en-UK (ví dụ: dry humour, Australian irony).

**Mô tả bài toán:**

- Đầu vào: một đoạn văn bản (review hoặc comment) thuộc phương ngữ en-AU hoặc en-UK.
- Đầu ra (đồng thời):
  * Tác vụ 1 — Sentiment Classification: nhãn nhị phân positive (1) hoặc negative (0).
  * Tác vụ 2 — Sarcasm Detection: nhãn nhị phân sarcastic (1) hoặc not sarcastic (0).

**Các mô hình dùng trong nghiên cứu:**

- **Mô hình encoder monolingual:** BERT-Large [1], RoBERTa-Large [2], DeBERTa-v3-Large [3] — được fine-tune riêng cho từng phương ngữ.
- **Mô hình encoder multilingual:** mBERT [10], XLM-RoBERTa-Large [11] — đánh giá khả năng xử lý đa phương ngữ.
- **Mô hình decoder (so sánh):** Mistral-Small [12] — mô hình đạt hiệu suất tốt nhất trên BESSTIE trong [5].
- **Kiến trúc MTL:** Dựa trên shared encoder + task-specific heads + cross-task interaction module, lấy cảm hứng từ [4][6][13].

---

### Phạm vi nghiên cứu:

- **Đánh giá hiệu suất:** So sánh các mô hình dựa trên Macro-averaged F1-Score, Precision, Recall. Phân tích hiệu suất trên từng tác vụ (sentiment vs. sarcasm) và từng phương ngữ (en-AU vs. en-UK). Áp dụng weighted F1-score để giảm ảnh hưởng của mất cân bằng nhãn.
- **Phương ngữ và domain:** Phân tích hiệu suất riêng trên en-AU và en-UK, trên GOOGLE và REDDIT subsets. Đánh giá cross-variety generalisation (train en-AU → test en-UK và ngược lại).
- **MTL vs. STL:** So sánh có hệ thống giữa multi-task learning và single-task learning baseline, trên cùng kiến trúc và dữ liệu.
- **Fine-tuning và kỹ thuật:** Khảo sát Standard Fine-tuning, LoRA [9], Adapter Layers, QLoRA, và Prompt-based fine-tuning.
- **Môi trường thực nghiệm:** Hugging Face Transformers, PyTorch, NVIDIA GPU (A100/T4).

---

## B2.2. Nội dung và phương pháp nghiên cứu

### Nội dung 1: Tổng quan tài liệu và phân tích các nghiên cứu liên quan

**Mục tiêu:** Khảo sát toàn diện các công trình nghiên cứu trước đó liên quan đến 3 trục chính của đề tài: (1) Multi-task Learning cho NLP, (2) Sentiment-Sarcasm Classification, và (3) Language Varieties of English, nhằm xác định khoảng trống nghiên cứu (research gap).

**Phương pháp thực hiện:**
- Tra cứu trên các cơ sở dữ liệu khoa học (Google Scholar, ACL Anthology, IEEE Xplore, Semantic Scholar) bằng các từ khóa: "multi-task learning sentiment sarcasm", "language variety English NLP", "Australian English sarcasm", "British English sentiment".
- Phân tích có hệ thống các nghiên cứu chính:
  * **MTL cho Sentiment & Sarcasm:** Nghiên cứu của Majumder et al. [4] chứng minh shared GRU + NTN fusion cải thiện 3-4% F1; AMTF-Net [6] đề xuất adaptive transformer fusion + cross-task attention đạt ~98% accuracy; GAN-BERT MTL [13] kết hợp semi-supervised + MTL đạt 93.88% accuracy.
  * **Ensemble features cho Sentiment:** Nghiên cứu của Badlani et al. [14] cho thấy sarcasm/humor/hate embeddings cải thiện sentiment classification 0.2-0.5%.
  * **Multimodal MTL:** Chauhan et al. [15] dùng sentiment + emotion auxiliary tasks cho sarcasm detection trong multi-modal setting, cải thiện 1-5%.
  * **Language Variety và NLP:** BESSTIE [5] cung cấp benchmark đầu tiên cho sentiment/sarcasm trên varieties of English; Dilsiz et al. [7] phát hiện LLMs yếu với slang en-AU hơn en-IN.
  * **Sarcasm Detection techniques:** Dadu & Pant [16] cho thấy context separation tokens cải thiện F1 +5.13%; Qiu et al. [17] dùng commonsense reasoning + adversarial contrastive learning.
- Xác định research gap: **Chưa có nghiên cứu nào kết hợp MTL (sentiment + sarcasm) với variety-awareness (en-AU vs. en-UK) trên kiến trúc Transformer.**

**Kết quả dự kiến:** Báo cáo tổng quan tài liệu chi tiết (Literature Review), xác định rõ research gap và đóng góp tiềm năng (contribution) của đề tài.

---

### Nội dung 2: Nghiên cứu và tiền xử lý dữ liệu BESSTIE

**Mục tiêu:** Chuẩn bị bộ dữ liệu sạch, đồng nhất và sẵn sàng cho huấn luyện mô hình MTL.

**Phương pháp thực hiện:**
- Tải và phân tích dataset BESSTIE từ HuggingFace, tập trung vào subsets en-AU và en-UK.
- Phân tích thống kê dữ liệu: phân bố nhãn sentiment/sarcasm, độ dài trung bình, phân bố từ vựng.
- Tiền xử lý:
  * Chuẩn hóa văn bản: lowercase, loại bỏ URLs, xử lý emoji, chuẩn hóa ký tự đặc biệt.
  * Xử lý mất cân bằng nhãn:
    - Sarcasm labels trong GOOGLE subset rất ít (en-AU: 7%, en-UK: 0%) → chỉ sử dụng REDDIT subset cho sarcasm task, hoặc áp dụng oversampling.
    - Weighted cross-entropy loss theo phân bố nhãn [5].
  * Phân tích variety-specific features: xác định slang, idioms, cultural references đặc trưng cho en-AU và en-UK dựa trên [7][8].
- Thiết kế data splits cho MTL:
  * In-variety: train/val/test trên cùng phương ngữ.
  * Cross-variety: train trên en-AU → test trên en-UK (và ngược lại).
  * Combined: gộp cả 2 phương ngữ để huấn luyện.
- (Tùy chọn) Tăng cường dữ liệu:
  * Back-translation (dịch sang ngôn ngữ khác rồi dịch ngược).
  * Contextual augmentation sử dụng LLM.
  * Synonym replacement cho non-slang tokens.

**Kết quả dự kiến:** Bộ dữ liệu đã tiền xử lý với 3 chiến lược phân chia (in-variety, cross-variety, combined), kèm báo cáo phân tích thống kê chi tiết.

---

### Nội dung 3: Thiết kế và cài đặt kiến trúc Multi-Task Learning

**Mục tiêu:** Xây dựng kiến trúc MTL tối ưu cho bài toán đồng thời phân loại sentiment và nhận diện sarcasm trên hai phương ngữ.

**Phương pháp thực hiện:**

**(a) Thiết kế kiến trúc MTL cơ bản (Baseline MTL):**
- Shared Transformer Encoder: BERT-Large hoặc RoBERTa-Large dùng chung cho cả 2 tác vụ.
- Task-Specific Classification Heads:
  * Sentiment Head: `[CLS] → FC Layer → Softmax → {positive, negative}`
  * Sarcasm Head: `[CLS] → FC Layer → Softmax → {sarcastic, not sarcastic}`
- Joint Loss Function: `L_total = L_sentiment + λ · L_sarcasm`, trong đó λ là hyperparameter cân bằng hai tác vụ.

**(b) Thiết kế kiến trúc MTL nâng cao (Proposed Model):**
- Tích hợp Cross-Task Attention Interaction (CTAI) [6]: cho phép sentiment representation hỗ trợ sarcasm detection và ngược lại.
- Neural Tensor Network (NTN) Fusion [4]: kết hợp biểu diễn sentiment và sarcasm thành biểu diễn chung s⁺.
- Sarcasm head sử dụng `s_sar ⊕ s⁺` (có fusion), Sentiment head sử dụng `s_sen` (không cần fusion) — theo kết luận thực nghiệm từ [4].
- Dynamic Joint Loss [6]: `L_total = α(t) · L_sentiment + β(t) · L_sarcasm`, trong đó α(t), β(t) tự động cân bằng dựa trên tiến trình huấn luyện.

**(c) Cài đặt và huấn luyện:**
- Sử dụng Hugging Face Transformers + PyTorch.
- Hyperparameter tuning: learning rate (1e-5 đến 5e-5), batch size (8, 16, 32), epochs (10-30), λ (0.3 đến 1.0), dropout (0.1-0.3).
- Áp dụng các phương pháp fine-tuning:
  * Standard Fine-tuning: huấn luyện toàn bộ tham số.
  * LoRA [9]: chỉ tinh chỉnh các ma trận low-rank, giảm chi phí tính toán.
  * QLoRA: quantized LoRA cho các mô hình lớn.
  * Adapter Layers: chèn các lớp nhỏ vào giữa các transformer layers.
- (Tùy chọn nâng cao) Semi-supervised learning với GAN component [13]: nếu dữ liệu labeled ít, sử dụng discriminator phân biệt labeled/unlabeled data.

**Kết quả dự kiến:** Mã nguồn kiến trúc MTL hoàn chỉnh (Baseline MTL + Proposed MTL), sẵn sàng huấn luyện.

---

### Nội dung 4: Huấn luyện và thực nghiệm

**Mục tiêu:** Huấn luyện các mô hình theo nhiều cấu hình và thu thập kết quả thực nghiệm toàn diện.

**Phương pháp thực hiện:**

**(a) Thực nghiệm chính — So sánh MTL vs. STL:**

| Cấu hình        | Mô hình              | Tác vụ              | Phương ngữ |
|-----------------|----------------------|---------------------|-----------|
| STL-Sentiment   | BERT/RoBERTa/DeBERTa | Sentiment only      | en-AU, en-UK |
| STL-Sarcasm     | BERT/RoBERTa/DeBERTa | Sarcasm only        | en-AU, en-UK |
| MTL-Baseline    | Shared BERT + 2 heads | Sentiment + Sarcasm | en-AU, en-UK |
| MTL-Proposed    | Shared BERT + CTAI + NTN | Sentiment + Sarcasm | en-AU, en-UK |

**(b) Thực nghiệm phương ngữ:**
- In-variety: train/test trên cùng phương ngữ.
- Cross-variety: train en-AU → test en-UK (và ngược lại).
- Combined: train trên cả 2 phương ngữ → test riêng từng phương ngữ.

**(c) Ablation Study:**
- Ảnh hưởng của CTAI module: MTL có CTAI vs. không có CTAI.
- Ảnh hưởng của NTN Fusion: MTL có NTN vs. không có NTN.
- Ảnh hưởng của λ (loss weight): thay đổi λ từ 0.1 đến 2.0.
- Ảnh hưởng của fine-tuning method: Standard vs. LoRA vs. Adapter.

**(d) So sánh với baselines:**
- BESSTIE baselines [5]: BERT, RoBERTa, ALBERT, MBERT, MDISTIL, XLM-R, GEMMA, MISTRAL, QWEN (single-task).
- MTL baselines: Majumder et al. [4] (GRU-based MTL).

**Kết quả dự kiến:** Bảng tổng hợp chi tiết hiệu suất (F1-Score, Precision, Recall) cho tất cả cấu hình, trên từng tác vụ và từng phương ngữ.

---

### Nội dung 5: Phân tích kết quả và Error Analysis

**Mục tiêu:** Phân tích sâu kết quả thực nghiệm để hiểu rõ điểm mạnh, điểm yếu của mô hình MTL trên từng phương ngữ.

**Phương pháp thực hiện:**
- So sánh hiệu suất MTL vs. STL:
  * Xác định mức cải thiện (%) cho từng tác vụ khi sử dụng MTL.
  * Phân tích xem sentiment hay sarcasm được cải thiện nhiều hơn bởi MTL (theo [4], sentiment cải thiện nhiều hơn vì sarcasm detection là subtask của sentiment analysis).
- Phân tích theo phương ngữ:
  * So sánh en-AU vs. en-UK: phương ngữ nào khó hơn cho sentiment? cho sarcasm?
  * Phân tích cross-variety: mô hình train trên en-AU có generalise tốt sang en-UK không?
- Error Analysis chi tiết:
  * Phân loại lỗi theo taxonomy: literalisation, generic substitution, semantic drift, contextual misinterpretation [7].
  * Phân tích mẫu sai (misclassified examples): xác định pattern lỗi phổ biến liên quan đến slang, cultural references, irony style.
  * Attention visualization: trực quan hóa attention weights để hiểu mô hình tập trung vào đâu khi phân loại [4][15].
- Statistical significance test: paired t-test hoặc McNemar's test để xác nhận sự cải thiện có ý nghĩa thống kê [15].

**Kết quả dự kiến:** Báo cáo phân tích chi tiết với confusion matrices, attention heatmaps, error examples, và statistical tests.

---

### Nội dung 6: Viết bài báo khoa học

**Mục tiêu:** Tổng hợp kết quả nghiên cứu thành một bài báo khoa học và submit lên tạp chí/hội nghị quốc tế Q4.

**Phương pháp thực hiện:**
- Cấu trúc bài báo:
  1. Introduction: Bài toán, motivation, research gap, contribution.
  2. Related Work: MTL for NLP, Sentiment & Sarcasm, Language Varieties.
  3. Methodology: Kiến trúc MTL Proposed, Dataset, Training.
  4. Experiments: Setup, Results, Ablation Study.
  5. Analysis: Cross-variety evaluation, Error Analysis, Attention Visualization.
  6. Conclusion & Future Work.
- Xác định venue phù hợp (Q4):
  * **Hội nghị:** VarDial Workshop (collocated with COLING/EACL), FigLang Workshop (collocated with ACL/NAACL), ALTA (Australasian Language Technology Association).
  * **Tạp chí:** Journal of Natural Language Processing (JNLP), Applied Sciences, Information (MDPI).
- Viết, review nội bộ, chỉnh sửa theo phản hồi của giảng viên hướng dẫn.

**Kết quả dự kiến:** Bản thảo bài báo hoàn chỉnh, sẵn sàng submit.

---

### Nội dung 7: Kết luận, hạn chế và hướng phát triển

**Mục tiêu:** Tổng kết những đóng góp, xác định hạn chế và đề xuất hướng nghiên cứu tương lai.

**Phương pháp thực hiện:**
- Tổng kết đóng góp:
  * Kiến trúc MTL mới cho variety-aware sentiment-sarcasm classification.
  * Kết quả benchmark trên BESSTIE cho en-AU và en-UK với MTL.
  * Phân tích cross-variety generalisation.
- Xác định hạn chế:
  * Dữ liệu BESSTIE có thể không đủ lớn (đặc biệt en-UK REDDIT chỉ có ~1,400 samples).
  * Chỉ xét 2 phương ngữ (en-AU, en-UK); có thể mở rộng sang en-IN, en-US, etc.
  * Binary classification (positive/negative, sarcastic/not) — chưa xét fine-grained sentiment hoặc sarcasm intensity.
  * Tài nguyên tính toán hạn chế khi fine-tune các mô hình lớn.
- Đề xuất hướng nghiên cứu tương lai:
  * Mở rộng sang nhiều phương ngữ hơn (en-IN, African-American English, Nigerian English).
  * Tích hợp thêm auxiliary tasks: emotion detection, hate speech detection [14].
  * Kết hợp variety-specific pre-training hoặc continual pre-training trên dữ liệu en-AU/en-UK.
  * Few-shot / Zero-shot MTL cho các phương ngữ ít tài nguyên.
  * Multi-modal MTL: kết hợp text + audio/visual [15].

**Kết quả dự kiến:** Báo cáo cuối cùng hoàn chỉnh, danh sách hướng nghiên cứu tương lai khả thi.

---

## B2.3. Kế hoạch nghiên cứu

| Thời gian    | Nội dung                                                                                               | Kết quả dự kiến                                                                                                 |
|-------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Tuần 1–4**   | • Tìm hiểu, phân tích các nghiên cứu trước đó về MTL, sentiment/sarcasm, language varieties (Nội dung 1). | • Báo cáo Literature Review hoàn chỉnh.                                                                        |
|             | • Tải, khám phá và tiền xử lý dataset BESSTIE (Nội dung 2).                                            | • Hiểu rõ research gap và xác định đóng góp (contribution).                                                    |
|             |                                                                                                        | • Bộ dữ liệu en-AU + en-UK đã tiền xử lý, sẵn sàng huấn luyện.                                               |
| **Tuần 5–6**   | • Nghiên cứu chi tiết các kiến trúc MTL (shared encoder, CTAI, NTN Fusion, Dynamic Loss).               | • Tài liệu thiết kế kiến trúc chi tiết.                                                                        |
|             | • Thiết kế kiến trúc MTL Baseline và MTL Proposed (Nội dung 3).                                         | • Mã nguồn kiến trúc MTL cơ bản hoàn chỉnh.                                                                   |
|             | • Chuẩn bị cấu hình thực nghiệm và giả thuyết nghiên cứu.                                              | • Bảng thiết kế thực nghiệm (experimental design matrix).                                                       |
| **Tuần 7–10**  | • Cài đặt STL baselines (BERT, RoBERTa, DeBERTa) cho sentiment và sarcasm riêng (Nội dung 4).           | • Kết quả baseline STL cho tất cả mô hình và phương ngữ.                                                       |
|             | • Cài đặt và huấn luyện MTL Baseline (shared encoder + 2 heads).                                        | • Kết quả MTL Baseline.                                                                                         |
|             | • Cài đặt và huấn luyện MTL Proposed (+ CTAI + NTN Fusion).                                             | • Bảng so sánh STL vs. MTL Baseline vs. MTL Proposed.                                                           |
|             | • Áp dụng các phương pháp fine-tuning (Standard, LoRA, Adapter).                                        | • So sánh các phương pháp fine-tuning.                                                                          |
| **Tuần 11–12** | • Thực hiện cross-variety evaluation (train en-AU → test en-UK, ngược lại) (Nội dung 4).                | • Ma trận cross-variety performance.                                                                            |
|             | • Thực hiện ablation study (CTAI, NTN, λ, fine-tuning methods) (Nội dung 4).                            | • Bảng ablation study đầy đủ.                                                                                   |
| **Tuần 13–16** | • Phân tích kết quả chi tiết: per-variety, per-task, per-domain (Nội dung 5).                            | • Báo cáo phân tích hiệu suất chi tiết.                                                                        |
|             | • Error analysis: phân loại lỗi, attention visualization (Nội dung 5).                                   | • Error examples + Attention heatmaps.                                                                          |
|             | • Statistical significance tests (Nội dung 5).                                                          | • Kết quả t-test / McNemar's test.                                                                              |
| **Tuần 17–20** | • Viết bản thảo bài báo (Introduction, Related Work, Methodology, Experiments, Analysis) (Nội dung 6).   | • Bản thảo v1 hoàn chỉnh.                                                                                      |
|             | • Review nội bộ với giảng viên hướng dẫn.                                                                | • Danh sách phản hồi và chỉnh sửa.                                                                             |
| **Tuần 21–22** | • Chỉnh sửa bài báo theo phản hồi (Nội dung 6).                                                        | • Bản thảo v2 (final draft).                                                                                    |
|             | • Phân tích hạn chế và đề xuất hướng nghiên cứu tương lai (Nội dung 7).                                 | • Phần Conclusion & Future Work hoàn chỉnh.                                                                    |
| **Tuần 23–24** | • Hoàn thiện bài báo và submit (Nội dung 6).                                                            | • Bài báo đã submit lên venue mục tiêu (VarDial/FigLang/ALTA Workshop hoặc tạp chí Q4).                        |
|             | • Tổng kết nghiên cứu, hoàn thiện báo cáo NCKH.                                                         | • Báo cáo NCKH hoàn chỉnh.                                                                                     |
|             | • Xây dựng ứng dụng demo minh họa.                                                                      | • Demo web app (Streamlit/Gradio) nhận input text và dự đoán sentiment + sarcasm.                               |

---

## Tài liệu tham khảo (cho phần B2)

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". In: NAACL-HLT. 2019.

[2] Yinhan Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach". In: ICLR. 2020.

[3] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". In: ICLR. 2021.

[4] Navonil Majumder, Soujanya Poria, Haiyun Peng, Niyati Chhaya, Erik Cambria, and Alexander Gelbukh. "Sentiment and Sarcasm Classification with Multitask Learning". In: IEEE Intelligent Systems 34.3 (2019), pp. 38–43.

[5] Dipankar Srirag, Aditya Joshi, Jordan Painter, and Diptesh Kanojia. "BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English". In: arXiv preprint arXiv:2412.04726v3. 2025.

[6] M.L.S.N.S Lakshmi et al. "AMTF-Net: An Adaptive Multi-Task Transformer Fusion Network for Joint Sentiment Analysis and Sarcasm Detection". In: International Journal of Computer Information Systems and Industrial Management Applications 18.10s (2026), pp. 606–626.

[7] Deniz Kaya Dilsiz, Dipankar Srirag, and Aditya Joshi. "Far Out: Evaluating Language Models on Slang in Australian and Indian English". In: Proceedings of the Thirteenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial). 2026, pp. 18–31.

[8] Sidney Greenbaum and Gerald Nelson. "The International Corpus of English (ICE) Project". In: World Englishes 15.1 (1996), pp. 3–15.

[9] Edward J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models". In: ICLR. 2022.

[10] Abhilash Pathak et al. "Aspect-Based Sentiment Analysis in Hindi Language by Ensembling Pre-Trained mBERT Models". In: Electronics (2021).

[11] Alexis Conneau et al. "Unsupervised Cross-lingual Representation Learning at Scale". In: ACL. 2020.

[12] Albert Q. Jiang et al. "Mistral 7B". In: arXiv preprint arXiv:2310.06825. 2023.

[13] Afif Hossain Irfan, Shrabani Das, Jannatul Ferdaus, and Md. Tabil Ahammed. "Enhancing Sarcasm Detection using GAN-BERT with Multi-Task Learning". In: 2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN). 2025.

[14] Rohan Badlani, Nishit Asnani, and Manan Rai. "Disambiguating Sentiment: An Ensemble of Humour, Sarcasm, and Hate Speech Features for Sentiment Classification". In: Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT). EMNLP. 2019, pp. 337–345.

[15] Dushyant Singh Chauhan et al. "Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis". In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL). 2020, pp. 4351–4360.

[16] Tanvi Dadu and Kartikey Pant. "Sarcasm Detection using Context Separators in Online Discourse". In: Proceedings of the Second Workshop on Figurative Language Processing (FigLang). 2020, pp. 51–55.

[17] Ziqi Qiu et al. "Detecting Emotional Incongruity of Sarcasm by Commonsense Reasoning". In: Proceedings of the 31st International Conference on Computational Linguistics (COLING). 2025, pp. 9062–9073.
