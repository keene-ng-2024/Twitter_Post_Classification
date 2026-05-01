# Twitter Hate Speech Classification

Fine-tuning **Mistral-7B-Instruct-v0.2** with **QLoRA** to classify tweets into three categories: *hate speech*, *offensive language*, and *neither*.

## Motivation

The boundary between hate speech and offensive language is often ambiguous, even for humans. This project investigates whether a fine-tuned LLM can reliably make this nuanced distinction, supporting safer content moderation without over-censoring benign expression.

## Method

| Component | Details |
|---|---|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Technique | QLoRA (4-bit NF4 quantization + LoRA adapters) |
| Dataset | [Davidson et al. (2017)](https://arxiv.org/abs/1703.04009) — ~24,800 labeled tweets |
| Classes | `0` hate speech, `1` offensive language, `2` neither |
| Hardware | NVIDIA RTX 3090 (24 GB), CUDA 11.7, PyTorch 2.0.1 |

### Training Pipeline

1. **Data preprocessing** — cleaning, stratified train/val/test split, random undersampling to balance training classes
2. **Hyperparameter search** — 5 LoRA configurations evaluated over 1 epoch each; best config selected by validation loss
3. **Full training** — 5 epochs with class-weighted cross-entropy loss, cosine LR schedule, best checkpoint selection
4. **Evaluation** — base vs. fine-tuned comparison on a 99-sample equally stratified eval set, plus full test set report

### Selected Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank / alpha | 32 / 64 |
| Learning rate | 1e-4 |
| Batch size | 4 x 4 grad accum = 16 effective |
| Max sequence length | 256 tokens |
| Dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |

## Results

### Eval Set (99 samples, equally stratified)

| Model | Accuracy | F1-Macro | F1 Hate Speech | F1 Offensive | F1 Neither |
|---|---|---|---|---|---|
| Base Mistral-7B | 0.65 | 0.64 | 0.49 | 0.65 | 0.78 |
| **Fine-tuned QLoRA** | **0.83** | **0.82** | **0.82** | **0.75** | **0.91** |

Fine-tuning improved hate speech F1 from 0.49 to 0.82 — the primary goal of this project.

### Full Test Set (3,718 samples, natural distribution)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Hate Speech | 0.22 | 0.84 | 0.35 |
| Offensive Language | 0.99 | 0.73 | 0.84 |
| Neither | 0.76 | 0.96 | 0.85 |

The model achieves high hate speech recall (84%) at the cost of lower precision — a deliberate tradeoff favoring detection in a content moderation context.

## Repository Structure

```
├── Twitter_hate_speech_classification.ipynb   # Full training notebook
├── labeled_data.csv                           # Dataset (Davidson et al. 2017)
└── README.md
```

## Dataset

The [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) from:

> Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* Proceedings of ICWSM, 11(1).

**Content warning:** The dataset contains harsh, offensive, and potentially disturbing language, including hate speech. It is used solely for research and model training purposes.

## License

The dataset is released under the MIT License. Mistral-7B is released under the Apache 2.0 License.
