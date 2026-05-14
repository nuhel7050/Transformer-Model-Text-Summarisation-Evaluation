<div align="center">

# Detoxification in Text Summarisation — Evaluation Framework

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](#)

**Evaluating whether LLM-based summarizers (BART, T5) can act as detoxifiers — reducing harmful content while preserving semantic accuracy.**

*NLP Course Project · SoSe 2025 · Hochschule Bonn-Rhein-Sieg*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deliverables](#deliverables)

---

## Overview

This research investigates whether large language models can suppress toxicity during text summarisation. Rather than focusing on whether they _introduce_ new toxicity, the emphasis is on whether they can act as **detoxifiers** — reducing or removing toxicity present in source documents while preserving meaning.

This is critical for deploying LLMs in environments such as:
- Media monitoring and content moderation
- Social platform content aggregation
- Customer feedback analysis

---

## Key Findings

| Finding | Detail |
|---------|--------|
| ✅ Natural detoxification | LLM summarizers demonstrate inherent detoxification capabilities |
| 🏆 BART wins balance | BART achieves optimal balance between content safety and semantic accuracy |
| 📏 Length correlation | Shorter texts exhibit higher toxicity levels in both sources and references |
| 📊 Surpassing humans | Model-generated summaries sometimes surpass human-written references in quality metrics |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline                         │
│                                                          │
│  Reddit TL;DR (3M docs)                                  │
│       → Subsample (20K docs)                             │
│           → Toxicity Filter (1K most toxic)              │
├─────────────────────────────────────────────────────────┤
│                  Summarization                           │
│                                                          │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │  BART-Large-CNN  │    │    T5-Base      │             │
│  │  (1024 tokens)   │    │  (512 tokens)   │             │
│  └────────┬────────┘    └────────┬────────┘             │
│           │    Hierarchical       │                      │
│           │    Chunking for       │                      │
│           │    long documents     │                      │
│           └──────────┬───────────┘                      │
├──────────────────────┼──────────────────────────────────┤
│                  Evaluation                              │
│                                                          │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │BERTScore │  ROUGE   │ METEOR   │  Toxicity        │  │
│  │(Semantic)│(N-gram)  │(Alignment)│ (Detoxify + API) │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                  Visualization                           │
│          (Matplotlib / Seaborn plots)                    │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

| Stage | Size | Description |
|-------|------|-------------|
| Initial | 3M documents | Reddit TL;DR corpus |
| Subsampled | 20,000 documents | Random subsample |
| Refined | 1,000 documents | Most toxic by Detoxify score |
| Document length | 300–6,000 chars | Filtered range |

**Toxicity Distribution:**

| Dataset | Metric | Doc Toxicity | Summary Toxicity | Doc Length | Summary Length |
|---------|--------|-------------|-----------------|-----------|---------------|
| Initial | Median | 0.004 | 0.003 | 1,025 | 98 |
| | Mean | 0.101 | 0.160 | 1,477 | 144 |
| Refined | Median | 0.850 | 0.402 | 900 | 83 |
| | Mean | 0.846 | 0.470 | 1,276 | 117 |

*Toxicity scores on 0–1 scale. Lengths in characters.*

---

## Models

| Model | Token Limit | Architecture | Long-Text Strategy |
|-------|------------|-------------|-------------------|
| **BART-Large-CNN** | 1,024 | Encoder-Decoder | Hierarchical chunking with overlap (50 tokens) |
| **T5-Base** | 512 | Encoder-Decoder | Hierarchical chunking with overlap (50 tokens) |

**Chunking Strategy:**
1. Split document into token-limited chunks with overlap
2. Summarize each chunk independently
3. Group chunk summaries and re-summarize
4. Produce final consolidated summary

---

## Evaluation Metrics

| Metric | Category | What It Measures |
|--------|----------|-----------------|
| **BERTScore** | Semantic | Contextual embedding similarity (F1) |
| **ROUGE-1 to ROUGE-5** | N-gram | Unigram through 5-gram overlap |
| **ROUGE Weighted Avg** | N-gram | Weighted combination (0.3, 0.25, 0.2, 0.15, 0.1) |
| **METEOR** | Alignment | Unigram matching with stemming and synonyms |
| **Detoxify** | Toxicity | Neural toxicity classification (local) |
| **Perspective API** | Toxicity | Google's toxicity scoring (API, rate-limited) |

---

## Project Structure

```
Transformer-Model-Text-Summarisation-Evaluation/
├── project/
│   ├── main.py                          # Entry point: dataset loading
│   ├── config.py                        # Configuration (model, API keys, dataset)
│   │
│   ├── summarizers/
│   │   ├── base.py                      # Abstract base class
│   │   ├── bart_summarizer.py           # BART-Large-CNN with hierarchical chunking
│   │   ├── t5_summarizer.py             # T5-Base with hierarchical chunking
│   │   ├── factory.py                   # Summarizer factory pattern
│   │   └── test_summarizer.py           # Unit tests
│   │
│   ├── evaluation/
│   │   ├── bertscore.py                 # BERTScore computation
│   │   ├── rouge.py                     # ROUGE-1 to ROUGE-5 + weighted average
│   │   ├── meteor_.py                   # METEOR score computation
│   │   └── toxicity.py                  # Detoxify + Perspective API scoring
│   │
│   ├── visualization/
│   │   ├── toxicity_plot.py             # Toxicity distribution visualizations
│   │   ├── rouge_plots.py              # ROUGE score plots
│   │   ├── bertscore_plots_.py          # BERTScore plots
│   │   └── meteor_plots_.py             # METEOR score plots
│   │
│   ├── data/
│   │   ├── dataset_loader.py            # HuggingFace dataset loading & filtering
│   │   ├── final_results_*.csv          # Evaluation results
│   │   ├── metric_*_values.csv          # Per-metric score files
│   │   └── plots/                       # Generated plot images
│   │
│   ├── requirements.txt
│   └── report.ipynb                     # Full analysis notebook
│
├── proposal/
│   └── NLP_Project_Proposal.pdf
│
├── Toxicity_Summary_eval_Poster.pdf     # Conference-style poster
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/nuhel7050/Transformer-Model-Text-Summarisation-Evaluation.git
cd Transformer-Model-Text-Summarisation-Evaluation

# Install dependencies
pip install -r project/requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | BART and T5 model pipelines |
| `datasets` | HuggingFace dataset loading |
| `torch` | Deep learning backend |
| `detoxify` | Local toxicity classification |
| `evaluate` | BERTScore and METEOR computation |
| `rouge-score` | ROUGE metric implementation |
| `requests` + `backoff` + `ratelimit` | Perspective API with retry logic |
| `matplotlib` + `seaborn` | Visualization |

---

## Usage

### Load Dataset

```bash
cd project
python main.py
```

### Run Summarization

```python
from summarizers.factory import SummarizerFactory

factory = SummarizerFactory()

# Generate BART summaries
bart_summaries = factory.summarize("bart", texts)

# Generate T5 summaries
t5_summaries = factory.summarize("t5", texts)
```

### Run Evaluation

```python
from evaluation.toxicity import ToxicityScorer
from evaluation.rouge import evaluate_rouge_scores
from evaluation.bertscore import evaluate_bertscore_scores
from evaluation.meteor_ import evaluate_meteor_scores

# Toxicity scoring
scorer = ToxicityScorer()
scores = scorer.score_detoxify(texts)

# ROUGE evaluation
df = evaluate_rouge_scores(df)

# BERTScore evaluation
df = evaluate_bertscore_scores(df)

# METEOR evaluation
df = evaluate_meteor_scores(df)
```

### Configuration

Edit `project/config.py`:

```python
class Config:
    model = "bart"              # "bart" or "t5"
    detoxify_model = 'original'
    dataset = 'reddit_tldr'     # 'reddit_tldr' or 'multi_news'
    api_keys = {"perspective": "YOUR_API_KEY"}
```

---

## Deliverables

| Deliverable | Location |
|------------|----------|
| Project Proposal | `proposal/NLP_Project_Proposal.pdf` |
| Research Poster | `Toxicity_Summary_eval_Poster.pdf` |
| Full Analysis | `project/report.ipynb` |
| Evaluation Results | `project/data/*.csv` |

---

## Acknowledgments

- Original repository: [arunimaCh29/Text-Summarisation-Eval](https://github.com/arunimaCh29/Text-Summarisation-Eval)
- NLP Course, SoSe 2025, Hochschule Bonn-Rhein-Sieg
