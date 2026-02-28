# CAST-ACF

**Robust Generation and Evaluation for Multi-Granularity Timeline Summarization**

[![Paper](https://img.shields.io/badge/ICASSP-2026-blue)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of our ICASSP 2026 paper:

> **CAST-ACF: Robust Generation and Evaluation for Multi-Granularity Timeline Summarization**
> 
> Yuming Ai, Fang Kong

## Overview

We propose **CAST** (Chronology-windowed Abstractive Summarization for multi-granularity Timelines), the first end-to-end method for Multi-Granularity Timeline Summarization (MGTLS), along with **ACF** (Alignment/Coverage/Factuality), a robust evaluation framework that addresses overestimation issues in existing metrics.

### Key Contributions

1. **CAST Framework**: A two-stage pipeline for MGTLS
   - Evidence Selector: Binary classifier for selecting relevant articles
   - Timeline Summarizer: LoRA fine-tuned LLM for headline generation

2. **ACF Metrics**: Robust evaluation without Hungarian algorithm bias
   - **Alignment (A)**: Overall quality with semantic + temporal scoring
   - **Coverage (C)**: Fraction of reference events adequately covered
   - **Factuality (F)**: Faithfulness to reference content

## Installation

```bash
git clone https://github.com/AmyIvan/CAST-ACF.git
cd CAST-ACF
pip install -r requirements.txt
```

### Optional Dependencies

For 4-bit quantization (reduces GPU memory):
```bash
pip install bitsandbytes>=0.41.0
```

## Quick Start

### 1. Evidence Pair Mining

Mine evidence-summary pairs from the dataset using the scoring function (Equation 1):

```bash
python scripts/build_pairs.py \
    --cand data/cand.tar.gz \
    --gold data/gold_reference.jsonl \
    --outdir outputs/pairs \
    --embed_model moka-ai/m3e-large
```

### 2. Train Evidence Selector

Train the binary classifier for evidence selection:

```bash
python scripts/train_selector.py \
    --cand data/cand.tar.gz \
    --pairs outputs/pairs/pairs_N.jsonl \
    --K N \
    --out_model outputs/selector/selector_N.pkl
```

### 3. Train Summarizer

Fine-tune the LLM with LoRA for headline generation:

```bash
python scripts/train_summarizer.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --pairs_train outputs/pairs/pairs_N.train.jsonl \
    --pairs_val outputs/pairs/pairs_N.val.jsonl \
    --K N \
    --outdir outputs/summarizer/N \
    --gradient_checkpointing
```

### 4. Generate Timelines

Run inference to generate timelines:

```bash
python scripts/run_inference.py \
    --cand data/cand.tar.gz \
    --gold data/gold_reference.jsonl \
    --selector outputs/selector/selector_N.pkl \
    --summarizer outputs/summarizer/N/checkpoint-final \
    --K N \
    --outfile outputs/predictions.jsonl
```

### 5. Evaluate with ACF Metrics

Evaluate generated timelines:

```bash
python scripts/evaluate.py \
    --pred outputs/predictions.jsonl \
    --gold data/gold_reference.jsonl \
    --K N \
    --outfile outputs/eval_results.json
```

## Project Structure

```
CAST-ACF/
├── cast/                       # Core package
│   ├── data/                   # Data processing
│   │   ├── mining.py          # Evidence pair mining (Eq. 1)
│   │   ├── prompts.py         # Prompt templates
│   │   └── dataset.py         # PyTorch datasets
│   ├── models/                 # Model components
│   │   ├── selector.py        # Evidence selector training
│   │   └── selector_feats.py  # 9-dimensional features
│   ├── evaluation/             # ACF metrics
│   │   └── acf.py             # A/C/F implementation
│   └── utils/                  # Utilities
│       ├── io.py              # JSONL/tar handling
│       ├── timebin.py         # Time window construction
│       ├── text.py            # Text processing
│       ├── embedding.py       # Embedding encoder
│       ├── bm25.py            # BM25 scorer
│       └── aligner.py         # MMR deduplication
├── scripts/                    # Training & inference
│   ├── build_pairs.py         # Evidence mining
│   ├── train_selector.py      # Selector training
│   ├── train_summarizer.py    # Summarizer LoRA training
│   ├── run_inference.py       # Timeline generation
│   └── evaluate.py            # ACF evaluation
├── configs/                    # Configuration files
│   └── default.yaml           # Default hyperparameters
└── requirements.txt
```

## Data Format

### Input: Candidate Articles (`cand.tar.gz`)

A tar.gz archive containing per-topic JSONL files (`{topic_id}.jsonl`):

```json
{"time": "2024-01-15T10:30:00", "title": "...", "content": "..."}
{"time": "2024-01-15T14:00:00", "title": "...", "content": "..."}
```

### Reference: Gold Timelines (`gold_reference.jsonl`)

```json
{
  "id": "topic_001",
  "timeline": [
    {"time": "2024-01-15", "summary": "Event A happened..."},
    {"time": "2024-01-16", "summary": "Event B followed..."}
  ],
  "meta_timeline": {
    "10": {"timeline": [...]},
    "5": {"timeline": [...]}
  }
}
```

### Output: Generated Timelines

```json
{
  "id": "topic_001",
  "K": "N",
  "timeline": [
    {"id": "topic_001_N_0", "time": "2024-01-15T00:00:00", "summary": "..."}
  ]
}
```

## Hyperparameters

Key hyperparameters from the paper (see `configs/default.yaml`):

| Component | Parameter | Value |
|-----------|-----------|-------|
| Evidence Mining | α (semantic) | 1.0 |
| | β (BM25) | 0.5 |
| | γ (time penalty) | 0.05 |
| | δ (same-day bonus) | 0.5 |
| | Dedup threshold | 0.85 |
| Selector | Negative downsample | 0.5 |
| Summarizer | LoRA rank | 16 |
| | LoRA alpha | 32 |
| | Learning rate | 2e-5 |
| | Epochs | 2 |
| ACF Evaluation | α_τ | 0.1 |
| | [L_φ, H_φ] | [0.15, 0.85] |
| | θ | 0.50 |
| | (θ_c, θ_t) | (0.38, 0.60) |

## Citation

```bibtex
@inproceedings{ai2026cast,
  title={CAST-ACF: Robust Generation and Evaluation for Multi-Granularity Timeline Summarization},
  author={Ai, Yuming and Kong, Fang},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The DTELS benchmark dataset is from [Zhang et al., NAACL 2025]
- We use [m3e-large](https://huggingface.co/moka-ai/m3e-large) for Chinese embeddings
- Base LLM: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
