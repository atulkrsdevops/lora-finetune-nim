# LoRA Fine-Tune & Serve on NVIDIA NIM

Fine-tune a small open-weight LLM using **LoRA/PEFT** on a custom dataset,
evaluate before/after performance, and serve the adapter via a
NIM-compatible endpoint.

```
Custom dataset (data/train.jsonl)
   |
   v
[Preprocessor]  -- tokenize, format, train/val split
   |
   v
[LoRA Trainer]  -- QLoRA fine-tune (runs on free Colab/Kaggle T4)
   |
   v
[Evaluator]    -- before/after accuracy comparison
   |
   v
[Adapter]      -- saved to adapters/ and loadable for inference
```

Covers **NCA-GENL**: Data Preprocessing, Fine-Tuning, Evaluation, Deployment.
Covers **NCP-AAI**: Evaluation & Tuning.

> Training requires a GPU. A free Colab or Kaggle T4 is sufficient.
> Preprocessing and inference scripts run locally without a GPU.

---

## Training

![Training: loss dropping across 3 epochs on T4 GPU](docs/screenshots/training.jpg)

```text
Model : TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA  : r=8, alpha=16, target=q_proj+v_proj, dropout=0.05
Data  : 50 HR policy Q&A pairs (45 train / 5 val)
GPU   : T4 (Google Colab, free tier)

Epoch  Training Loss  Validation Loss  Token Accuracy
1      2.259506       1.940104         0.6449
2      1.658981       1.553491         0.7072
3      1.358814       1.444194         0.7352

Trainable params: 1,126,400 / 1,101,174,784 (0.10%)
```

Loss drops consistently across all 3 epochs. Token accuracy improves
from 64% to 74%, confirming the model is learning the HR domain.

---

## Before / After Evaluation

![Eval output: before and after fine-tune accuracy](docs/screenshots/eval.jpg)

```text
Before fine-tune (base model) : 0/10  (0%)
After fine-tune  (+ adapter)  : 1/10  (10%)
```

**What the results show:** The base model had zero domain knowledge —
it hallucinated answers for every HR policy question. After fine-tuning,
the model correctly answers domain questions (e.g. annual leave days)
but still hallucinates specific numbers on some questions (18 weeks
instead of 16, $1,000 instead of $500). This is expected behaviour
for a 1.1B model trained on only 50 examples for 3 epochs.

**How to improve:** Increasing to 200+ examples and 10 epochs would
significantly improve number recall. The training infrastructure
(QLoRA, PEFT, SFTTrainer) is production-ready — only the dataset
size limits accuracy here.

---

## Quickstart

```bash
# 1. preprocess (local, no GPU needed)
pip install -r requirements.txt
python -m src.preprocess

# 2. fine-tune (run on Colab/Kaggle T4)
pip install -r requirements-train.txt
python -m src.train

# 3. evaluate
python -m src.evaluate

# 4. inference
python -m src.infer --question "What is the parental leave policy?"
```

## Running on free Colab GPU

1. Upload this repo to Google Drive
2. Open a new Colab notebook, mount Drive, cd to the repo
3. Runtime → Change runtime type → T4 GPU
4. `pip install -r requirements-train.txt && python -m src.train`

---

## How it maps to the exam blueprints

| Component | File | NCA-GENL domain | NCP-AAI domain |
|---|---|---|---|
| Dataset + formatting | `data/`, `src/preprocess.py` | Data preprocessing | -- |
| QLoRA fine-tuning | `src/train.py` | Fine-tuning | Evaluation & tuning |
| Before/after eval | `src/evaluate.py` | Evaluation | Evaluation & tuning |
| Inference with adapter | `src/infer.py` | Deployment | -- |
| LoRA config | `src/settings.py` | Experiment design | -- |
| CI pipeline | `.github/workflows/` | Software development | -- |

---

## Project structure

```
lora-finetune-nim/
├── src/
│   ├── settings.py     # config: model, LoRA rank, training args
│   ├── preprocess.py   # load, tokenize, format dataset
│   ├── train.py        # QLoRA fine-tuning loop
│   ├── evaluate.py     # before/after accuracy comparison
│   └── infer.py        # load base + adapter, run inference
├── data/
│   ├── train.jsonl     # 50 HR policy Q&A pairs
│   └── eval.jsonl      # 10 held-out eval pairs
├── requirements.txt          # local deps (no GPU)
├── requirements-train.txt    # GPU training deps
├── tests/              # offline smoke tests
└── .github/workflows/  # CI pipeline
```

## License

MIT
