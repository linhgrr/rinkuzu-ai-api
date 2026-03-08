# Evaluation on Lecture Bank Dataset

Hướng dẫn đánh giá pipeline prerequisite ranking trên dataset Lecture Bank.

## 📊 Dataset

**Lecture Bank** là một dataset benchmark cho việc xác định quan hệ tiên quyết giữa các khái niệm trong lĩnh vực NLP/ML.

- **208 concepts** với định nghĩa từ Wikipedia
- **42,751 cặp** được annotate (positive và negative relations)
- File dataset:
  - `208topics_with_definitions.csv`: Concepts và definitions
  - `prerequisite_annotation.csv`: Ground truth (concept_id_1, concept_id_2, label)

## 🚀 Cách chạy Evaluation

### Option 1: Quick Evaluation (Chỉ PRS Ranking - Nhanh)

Chỉ đánh giá bước ranking sử dụng embeddings, không có LLM verification:

```powershell
# Sử dụng threshold mặc định
python eval_lecture_bank_quick.py

# Custom threshold
python eval_lecture_bank_quick.py --prs-threshold 0.75

# Custom output file
python eval_lecture_bank_quick.py --prs-threshold 0.8 --output results_0.8.json
```

**Thời gian:** ~2-3 phút (chỉ tính embeddings)

### Option 2: Full Evaluation (PRS + LLM Verification - Chậm)

Đánh giá cả 2 bước: ranking + verification với LLM:

```powershell
# Full pipeline
python eval_lecture_bank.py

# Custom thresholds
python eval_lecture_bank.py --prs-threshold 0.75 --min-confidence 0.6

# Chỉ ranking, skip verification
python eval_lecture_bank.py --no-verification
```

**Thời gian:** ~30-60 phút (tùy số lượng pairs cần verify)

## 📈 Metrics được đo

### 1. PRS Ranking Stage

- **Precision**: Tỷ lệ predicted pairs đúng
- **Recall**: Tỷ lệ ground truth được tìm thấy
- **F1 Score**: Harmonic mean của Precision và Recall
- **Accuracy**: Tỷ lệ dự đoán đúng (cả positive và negative)

**Lưu ý:** Ranking không xác định hướng, chỉ xác định có quan hệ hay không.

### 2. LLM Verification Stage (nếu có)

- **Precision**: Tỷ lệ verified relations đúng (với hướng)
- **Recall**: Tỷ lệ ground truth được verify đúng
- **F1 Score**: Harmonic mean

**Lưu ý:** Verification xác định cả hướng của quan hệ.

### 3. Combined Pipeline

- **Ranking Recall**: Bao nhiêu % ground truth được tìm thấy bởi ranking
- **Verification Recall**: Bao nhiêu % ground truth được verify đúng
- **Pipeline Recall**: Overall recall của toàn pipeline

## 📋 Cấu trúc Output

### Quick Evaluation Output

```json
{
  "config": {
    "prs_threshold": 0.75,
    "num_concepts": 208,
    "num_total_pairs": 42751,
    "num_positive_relations": 1435
  },
  "metrics": {
    "num_predicted": 500,
    "total_possible_pairs": 21528,
    "true_positives": 120,
    "false_positives": 380,
    "false_negatives": 1315,
    "true_negatives": 19713,
    "precision": 0.24,
    "recall": 0.0836,
    "f1": 0.1244,
    "accuracy": 0.9213
  }
}
```

### Full Evaluation Output

```json
{
  "config": {...},
  "stages": {
    "ranking": {
      "num_predicted": 500,
      "precision": 0.24,
      "recall": 0.0836,
      "f1": 0.1244
    },
    "verification": {
      "num_verified": 200,
      "precision": 0.55,
      "recall": 0.0768,
      "f1": 0.1351
    }
  },
  "combined": {
    "ground_truth_total": 1435,
    "found_by_ranking": 120,
    "correctly_verified": 110,
    "ranking_recall": 0.0836,
    "verification_recall": 0.0768,
    "pipeline_recall": 0.0768
  }
}
```

## 🎯 Tối ưu Threshold

Để tìm threshold tốt nhất, chạy với nhiều giá trị khác nhau:

```powershell
# Thử nhiều threshold
python eval_lecture_bank_quick.py --prs-threshold 0.5 --output results_0.5.json
python eval_lecture_bank_quick.py --prs-threshold 0.6 --output results_0.6.json
python eval_lecture_bank_quick.py --prs-threshold 0.7 --output results_0.7.json
python eval_lecture_bank_quick.py --prs-threshold 0.75 --output results_0.75.json
python eval_lecture_bank_quick.py --prs-threshold 0.8 --output results_0.8.json
python eval_lecture_bank_quick.py --prs-threshold 0.85 --output results_0.85.json
python eval_lecture_bank_quick.py --prs-threshold 0.9 --output results_0.9.json
```

Sau đó so sánh F1 score để chọn threshold tốt nhất.

## 📊 Phân tích kết quả

### Confusion Matrix

```
                    Predicted
                 Yes        No
Actual  Yes      TP        FN
        No       FP        TN
```

- **TP (True Positive)**: Đúng dự đoán có quan hệ
- **FP (False Positive)**: Sai dự đoán có quan hệ (thực tế không có)
- **FN (False Negative)**: Bỏ sót quan hệ thực sự
- **TN (True Negative)**: Đúng dự đoán không có quan hệ

### Trade-off

- **Threshold cao** (0.85-0.95):

  - ✅ Precision cao (ít FP)
  - ❌ Recall thấp (nhiều FN)
  - **Use case**: Khi cần độ chính xác cao, chấp nhận bỏ sót

- **Threshold thấp** (0.5-0.7):

  - ✅ Recall cao (ít FN)
  - ❌ Precision thấp (nhiều FP)
  - **Use case**: Khi cần tìm được nhiều relations, chấp nhận noise

- **Threshold balanced** (0.7-0.8):
  - Cân bằng giữa Precision và Recall
  - F1 score thường cao nhất

## 🔧 Troubleshooting

### Import errors

Nếu gặp lỗi import, chạy:

```powershell
$env:PYTHONPATH = "."
```

### Embedding model issues

Kiểm tra embedding model đã được download:

```python
from embed.embedding_client import EmbeddingClient
client = EmbeddingClient()
print(f"Model: {client.model_name}")
print(f"Device: {client.device}")
```

### Memory issues

Nếu thiếu RAM, giảm batch size trong verification:

```python
# Trong eval_lecture_bank.py, line 221
max_workers=2  # Giảm từ 4 xuống 2
```

## 📚 References

- **Lecture Bank Dataset**: [RefD Paper](https://arxiv.org/abs/1906.05226)
- **PRS Method**: Context Similarity Ranking approach
- **Evaluation**: Standard binary classification metrics
