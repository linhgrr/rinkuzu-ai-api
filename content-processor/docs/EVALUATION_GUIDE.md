# Hướng dẫn Đánh giá Prerequisite Prediction

## Tổng quan

Pipeline hỗ trợ đánh giá khả năng dự đoán quan hệ tiên quyết (prerequisite) giữa các khái niệm học thuật sử dụng **Lecture Bank dataset** với 208 topics và ground truth annotations.

## Dataset

### 1. 208 Topics với Definitions
- **File**: `dataset/lecture_bank/208topics_with_definitions.csv`
- **Format**: `topic_id,topic_name,wiki_url,definition`
- **Mô tả**: 208 khái niệm học thuật trong lĩnh vực NLP/ML với định nghĩa từ Wikipedia/Gemini

### 2. Ground Truth Annotations
- **File**: `dataset/lecture_bank/prerequisite_annotation.csv`
- **Format**: `source_id,target_id,label`
  - `label=1`: `source_id` là prerequisite của `target_id` (quan hệ có hướng)
  - `label=0`: không có quan hệ prerequisite
- **Tổng số**: ~42,751 cặp được annotate

### 3. Thống kê Dataset
```python
from dataset.lecture_bank import LectureBankDataset

dataset = LectureBankDataset("dataset/lecture_bank")
dataset.load_topics(with_definitions=True)
dataset.load_ground_truth()
stats = dataset.get_stats()
# => {
#   "num_topics": 208,
#   "num_annotated_pairs": 42751,
#   "num_positive_edges": ~XXX,
#   "num_negative_edges": ~XXX,
#   "positive_ratio": 0.XX
# }
```

## Phương pháp Đánh giá

### Context Similarity Ranking (CSR)

Pipeline sử dụng phương pháp **CSR** để tính điểm prerequisite:

```
CSR(A→B) = cosine_similarity(name_embedding(B), definition_embedding(A))
PRS(A,B) = max(CSR(A→B), CSR(B→A))
```

**Giải thích:**
- Nếu concept B xuất hiện (name) trong định nghĩa của A → A có thể là prerequisite của B
- PRS (Prerequisite Relation Score) đo mức độ liên quan prerequisite giữa 2 concepts (bất kể hướng)

### Quy trình Evaluation

1. **Load dataset** với definitions
2. **Generate embeddings** cho name và definition của từng topic
3. **Compute PRS matrix** (208x208)
4. **Predict edges** với các ngưỡng threshold khác nhau
5. **So sánh với ground truth** và tính metrics

### Metrics

- **Precision**: Tỷ lệ dự đoán đúng trong tổng số dự đoán
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall**: Tỷ lệ phát hiện được trong tổng số ground truth
  ```
  Recall = TP / (TP + FN)
  ```

- **F1 Score**: Trung bình điều hòa của Precision và Recall
  ```
  F1 = 2 * Precision * Recall / (Precision + Recall)
  ```

- **Accuracy**: Tỷ lệ dự đoán đúng (cả positive và negative)
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

## Cách sử dụng

### 1. Chuẩn bị

Đảm bảo đã cài đặt dependencies:
```bash
pip install -r requirements_api.txt
```

Cấu hình model embedding trong `.env`:
```env
EMBEDDING_MODEL=VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
# hoặc model khác tùy chọn
```

### 2. Chạy Evaluation Script

```bash
python scripts/evaluate_prerequisite_prediction.py \
  --dataset-dir dataset/lecture_bank \
  --output eval_results.json \
  --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 \
  --batch-size 50
```

**Tham số:**
- `--dataset-dir`: Thư mục chứa dataset (mặc định: `dataset/lecture_bank`)
- `--output`: File JSON lưu kết quả (mặc định: `eval_results.json`)
- `--thresholds`: Danh sách ngưỡng PRS để đánh giá (mặc định: `0.3 0.4 0.5 0.6 0.7 0.8`)
- `--batch-size`: Batch size cho embedding generation (mặc định: 50)
- `--save-embeddings`: Lưu embeddings để tái sử dụng
- `--load-embeddings PATH`: Load embeddings từ file thay vì tính lại

### 3. Kết quả

Script sẽ in ra:
```
============================================================
Evaluating at threshold: 0.60
============================================================
Evaluation Results:
  True Positives (TP): 120
  False Positives (FP): 45
  False Negatives (FN): 80
  True Negatives (TN): 42506
  Precision: 0.7273
  Recall: 0.6000
  F1 Score: 0.6576
  Accuracy: 0.9970
============================================================
BEST THRESHOLD: 0.60
============================================================
  Precision: 0.7273
  Recall: 0.6000
  F1 Score: 0.6576
  Accuracy: 0.9970
============================================================
```

File JSON output:
```json
{
  "0.30": {
    "precision": 0.65,
    "recall": 0.75,
    "f1": 0.70,
    ...
  },
  "0.60": {
    "precision": 0.73,
    "recall": 0.60,
    "f1": 0.66,
    ...
  }
}
```

## Lưu/Tái sử dụng Embeddings

Để tránh tính toán lại embeddings (tốn thời gian):

### Lưu embeddings lần đầu:
```bash
python scripts/evaluate_prerequisite_prediction.py \
  --save-embeddings \
  --output eval_results_v1.json
```
→ Tạo file `embeddings_208topics.npz`

### Tái sử dụng embeddings:
```bash
python scripts/evaluate_prerequisite_prediction.py \
  --load-embeddings embeddings_208topics.npz \
  --thresholds 0.55 0.65 0.75 \
  --output eval_results_v2.json
```

## Tích hợp với Pipeline

### Sử dụng trong Code

```python
from dataset.lecture_bank import LectureBankDataset
from src.eval import (
    compute_prs_scores_from_embeddings,
    evaluate_prerequisite_prediction
)
from src.embed import compute_embeddings_batch

# Load dataset
dataset = LectureBankDataset("dataset/lecture_bank")
topics = dataset.load_topics(with_definitions=True)
ground_truth, all_pairs = dataset.load_ground_truth()

# Generate embeddings
topic_ids = sorted(topics.keys())
names = [topics[tid].name for tid in topic_ids]
definitions = [topics[tid].definition for tid in topic_ids]

name_embeds = compute_embeddings_batch(names, batch_size=50)
def_embeds = compute_embeddings_batch(definitions, batch_size=50)

# Compute PRS matrix
import numpy as np
prs_matrix = compute_prs_scores_from_embeddings(
    name_embeddings=np.array(name_embeds),
    definition_embeddings=np.array(def_embeds)
)

# Predict at threshold
from src.eval import predict_prerequisites_from_prs
predicted_edges = predict_prerequisites_from_prs(
    prs_matrix=prs_matrix,
    threshold=0.6,
    topic_ids=topic_ids
)

# Evaluate
metrics = evaluate_prerequisite_prediction(
    predicted_edges=predicted_edges,
    ground_truth_edges=ground_truth,
    all_pairs=all_pairs
)

print(f"F1 Score: {metrics.f1:.4f}")
```

## Xác định Hướng Prerequisite

### Phương pháp hiện tại

Pipeline sử dụng **2 bước**:

1. **PRS Ranking**: Tìm các cặp có khả năng có quan hệ (PRS >= threshold)
   - Output: danh sách các cặp `(concept_id_1, concept_id_2)` **chưa xác định hướng**

2. **LLM Agent Verification**: Xác định hướng và loại quan hệ
   - Input: `(concept_a, concept_b)`
   - Agent sử dụng tools (Wikipedia, RAG) để research
   - Output: `EvidenceVerification` với:
     - `has_relation`: True/False
     - `relation_type`: "PREREQUISITE" hoặc "SAME_CONCEPT"
     - `direction`: "A_to_B", "B_to_A", hoặc "same_concept"
     - `confidence`: 0.0-1.0
     - `evidences`: danh sách evidences
     - `reasoning`: giải thích

### Nguyên tắc xác định hướng

**Concept A là prerequisite của Concept B** nếu:
- B được định nghĩa dựa trên A
- B sử dụng A trong công thức/giải thích
- B không thể hiểu được nếu không biết A trước
- Nguồn học thuật nêu rõ "học A trước B"

Xem chi tiết tại: `src/prompts/evidence_verification_prompt.txt`

### Ví dụ

```python
from src.llm.extract_chain import ExtractionChain

chain = ExtractionChain(...)
verification = chain.verify_relation(
    concept_a="Linear Regression",
    concept_b="Logistic Regression"
)

# => EvidenceVerification(
#   has_relation=True,
#   relation_type="PREREQUISITE",
#   direction="A_to_B",  # Linear Regression → Logistic Regression
#   confidence=0.85,
#   evidences=["Logistic regression extends linear regression..."],
#   reasoning="Linear regression is a foundation for logistic regression..."
# )
```

## Đánh giá với Direction

Để đánh giá **cả hướng**, cần:

1. Sử dụng LLM agent để xác định direction cho từng cặp predicted
2. Chuyển đổi predictions thành directed edges: `(source, target)`
3. So sánh trực tiếp với ground truth (có hướng)

Script hiện tại đánh giá **undirected pairs** (không xét hướng) vì:
- Nhanh hơn (không cần gọi LLM cho mọi cặp)
- Đo lường khả năng phát hiện quan hệ của PRS method
- Direction determination là bước riêng biệt (LLM-based)

## Troubleshooting

### Lỗi thiếu dependencies
```bash
pip install scikit-learn numpy tqdm loguru sentence-transformers
```

### Lỗi CUDA out of memory
- Giảm `--batch-size`: `--batch-size 16`
- Hoặc chuyển sang CPU trong `.env`:
  ```env
  EMBEDDING_DEVICE=cpu
  ```

### Dataset không tìm thấy
Đảm bảo cấu trúc:
```
dataset/lecture_bank/
├── 208topics.csv
├── 208topics_with_definitions.csv
└── prerequisite_annotation.csv
```

## Tham khảo

- **Paper gốc**: [Lecture Bank dataset paper](https://aclanthology.org/...)
- **Context Similarity Ranking**: Phương pháp CSR được mô tả trong `src/embed/prereq_ranking.py`
- **Full pipeline**: Xem `src/api/services.py` và `full_pipeline_app.py`

