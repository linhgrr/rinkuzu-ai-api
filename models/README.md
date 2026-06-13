# Model Weights

| File | Size | Used by | Source |
|---|---|---|---|
| `saint_best.pt` | 4.8 MB | SAINT knowledge tracing | trained in module2 |
| `dqn_best.pt` | 662 KB | RL curriculum sequencer | trained in module3 |
| `prereq_mlp.pth` | 4.7 MB | MLP prerequisite ranker (Module 1) | `bc_with_code/module1_update/kaggle/prerequisite_pipeline_outputs/results/best_model.pth` |

## prereq_mlp.pth — provenance

Trained on **LectureBank** (208 NLP/ML/AI concepts, 913 positive prerequisite pairs and 41,008 labeled negative pairs) with **BAAI/bge-m3** embeddings (SentenceTransformer, L2-normalized):

- **Architecture:** MLP `2048 → 512 → 256 → 1`, LayerNorm + ReLU + Dropout 0.3, He init.
- **Encoder:** `BAAI/bge-m3` (1024-d, XLM-RoBERTa-large backbone); name-only; L2-normalized via `SentenceTransformer.encode(normalize_embeddings=True)`.
- **Training:** BCEWithLogitsLoss + AdamW (lr=3e-4, wd=1e-4) + ReduceLROnPlateau, batch=128, early stopping patience=10. Best checkpoint val_f1≈0.852.
- **Split:** group-aware approximately 80/10/10 after balanced negative undersampling; unordered concept pairs are kept within one split to avoid direct/reverse-pair leakage.
- **Test set (LectureBank, 184 balanced pairs, 5-seed):** F1=0.834±0.035, AUC=0.914±0.017, AP=0.929.
- **Seed:** 42.

Reproduction notebook: `bc_with_code/module1_update/kaggle/prerequisite_pipeline.ipynb`.

## BGE-M3 encoder

The MLP ranker pairs each concept name with embeddings from `BAAI/bge-m3` (~2.3 GB). Hugging Face downloads it on first run via `SentenceTransformer("BAAI/bge-m3")`.

For air-gapped or repeated CI:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('BAAI/bge-m3')"
```

Override the cache directory with `HF_HOME=/path/to/cache`.
