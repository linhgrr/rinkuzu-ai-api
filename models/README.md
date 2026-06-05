# Model Weights

| File | Size | Used by | Source |
|---|---|---|---|
| `saint_best.pt` | 4.8 MB | SAINT knowledge tracing | trained in module2 |
| `dqn_best.pt` | 662 KB | RL curriculum sequencer | trained in module3 |
| `prereq_mlp.pth` | 3.5 MB | MLP prerequisite ranker (Module 1) | `bc_with_code/results/results/best_model.pth` |

## prereq_mlp.pth — provenance

Trained on **LectureBank** (208 NLP/ML/AI concepts, 921 prerequisite labels) with XLM-RoBERTa-base mean-pool embeddings:

- **Architecture:** MLP `1536 → 512 → 256 → 1`, LayerNorm + ReLU + Dropout 0.3, He init, ~920K params.
- **Training:** BCEWithLogitsLoss + AdamW (lr=3e-4, wd=1e-4) + ReduceLROnPlateau, batch=128, early stopping patience=10. Best checkpoint at epoch 29 (val_f1=0.874).
- **Test set (LectureBank, 183 balanced pairs):** F1=0.825, AUC=0.908, Acc=0.814.
- **Seed:** 42.

Reproduction notebook: `bc_with_code/module1_update/kaggle/prerequisite_pipeline.ipynb`.

## XLM-RoBERTa encoder

The MLP ranker pairs each concept name with embeddings from `xlm-roberta-base` (~1.1 GB). Hugging Face downloads it on first run via `AutoTokenizer.from_pretrained` / `AutoModel.from_pretrained`.

For air-gapped or repeated CI:

```bash
python -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('xlm-roberta-base'); \
  AutoModel.from_pretrained('xlm-roberta-base')"
```

Override the cache directory with `TRANSFORMERS_CACHE=/path/to/cache`.
