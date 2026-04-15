# Multilingual Sentiment Analysis with QLoRA Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A QLoRA-based fine-tuning pipeline for multilingual sentiment classification across **13 Indian languages**, built on top of Google's **Gemma-3-1B-IT** instruction-tuned model. Developed for the **NPPE DLP 2026 Term 1** Kaggle competition.

---

## Table of Contents

- [Overview](#overview)
- [Supported Languages](#supported-languages)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Prompt Engineering](#prompt-engineering)
  - [Quantization](#quantization)
  - [LoRA Configuration](#lora-configuration)
  - [Training Pipeline](#training-pipeline)
- [Results](#results)
  - [Overall Performance](#overall-performance)
  - [Per-Language Breakdown](#per-language-breakdown)
- [Installation](#installation)
- [Usage](#usage)
  - [Running on Kaggle](#running-on-kaggle)
  - [Running Locally](#running-locally)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)
- [Hardware Requirements](#hardware-requirements)
- [Key Dependencies](#key-dependencies)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project fine-tunes a large language model to perform binary sentiment classification (Positive / Negative) on text written in 13 Indian languages. Instead of training separate models per language, a single multilingual model is fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation), which enables efficient training on consumer-grade GPUs by quantizing the base model to 4-bit precision and training lightweight adapter layers on top.

The pipeline achieves a **Macro F1-Score of 0.8529** on the validation set, with perfect scores on Hindi, Marathi, and Telugu.

---

## Supported Languages

| Code | Language   | Script     | Train Samples | Test Samples |
|------|------------|------------|:-------------:|:------------:|
| `as` | Assamese   | Bengali    | 71            | 6            |
| `bd` | Bodo       | Devanagari | 71            | 6            |
| `bn` | Bengali    | Bengali    | 65            | 12           |
| `gu` | Gujarati   | Gujarati   | 69            | 8            |
| `hi` | Hindi      | Devanagari | 74            | 3            |
| `kn` | Kannada    | Kannada    | 66            | 11           |
| `ml` | Malayalam  | Malayalam  | 68            | 8            |
| `mr` | Marathi    | Devanagari | 67            | 10           |
| `or` | Odia       | Odia       | 72            | 5            |
| `pa` | Punjabi    | Gurmukhi   | 72            | 5            |
| `ta` | Tamil      | Tamil      | 76            | 1            |
| `te` | Telugu     | Telugu     | 63            | 14           |
| `ur` | Urdu       | Perso-Arabic | 66          | 11           |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gemma-3-1B-IT (Base)                      │
│                   703M total parameters                      │
│               Frozen in NF4 4-bit precision                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              QLoRA Adapter Layers                     │   │
│  │           52M trainable parameters (7.42%)            │   │
│  │                                                      │   │
│  │   Target Modules:                                    │   │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │   │ q_proj  │ │ k_proj  │ │ v_proj  │ │ o_proj  │  │   │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘  │   │
│  │   ┌──────────┐ ┌──────────┐ ┌───────────┐          │   │
│  │   │ gate_proj│ │ up_proj  │ │ down_proj │          │   │
│  │   └──────────┘ └──────────┘ └───────────┘          │   │
│  │                                                      │   │
│  │   Rank: 64  |  Alpha: 128  |  Scaling: 2.0          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

- **Competition:** NPPE DLP 2026 Term 1 (Kaggle)
- **Task:** Binary sentiment classification (Positive / Negative)
- **Train set:** 900 samples across 13 languages
- **Test set:** 100 samples (labels withheld)
- **Label distribution:** Positive: 456 (50.7%) | Negative: 444 (49.3%) — well balanced
- **Missing values:** None

The dataset is split into **750 training** and **150 validation** samples using stratified sampling to preserve class balance.

---

## Methodology

### Prompt Engineering

Each input is wrapped in a language-aware instruction prompt using the Gemma chat template format:

```
<start_of_turn>user
Classify the sentiment of the following {Language} text as exactly
'Positive' or 'Negative'. Respond with only one word.

Text: {sentence}
<end_of_turn>
<start_of_turn>model
{label}<end_of_turn>
```

The language name (e.g., "Punjabi", "Tamil") is dynamically inserted based on a language code mapping, providing the model with linguistic context for each sample.

### Quantization

The base model is loaded in **4-bit NF4 quantization** using `bitsandbytes`:

| Parameter              | Value     |
|------------------------|-----------|
| Quantization type      | NF4       |
| Compute dtype          | bfloat16  |
| Double quantization    | Enabled   |
| Memory reduction       | ~4 GB → ~1.5 GB |

### LoRA Configuration

| Parameter        | Value                                                        |
|------------------|--------------------------------------------------------------|
| Rank (r)         | 64                                                           |
| Alpha            | 128                                                          |
| Effective scaling| 2.0 (alpha / r)                                              |
| Dropout          | 0.05                                                         |
| Target modules   | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Bias             | None                                                         |
| Task type        | CAUSAL_LM                                                    |

All 7 linear projection layers in the transformer blocks receive LoRA adapters, resulting in **52.18M trainable parameters** out of 703.19M total (7.42%).

### Training Pipeline

| Parameter                  | Value            |
|----------------------------|------------------|
| Epochs                     | 5                |
| Batch size (per device)    | 2                |
| Gradient accumulation steps| 8                |
| Effective batch size       | 16               |
| Learning rate              | 2e-4             |
| LR scheduler               | Cosine           |
| Optimizer                  | Paged AdamW 8-bit|
| Weight decay               | 0.01             |
| Warmup ratio               | 10%              |
| Max sequence length        | 512 tokens       |
| Precision                  | bf16             |
| Gradient checkpointing     | Enabled          |
| Max gradient norm          | 0.3              |
| Best model selection       | Lowest eval loss |

**Training time:** ~2015 seconds (~34 minutes) on a single Tesla T4 GPU.

---

## Results

### Overall Performance

| Metric          | Score  |
|-----------------|--------|
| **Macro F1**    | **0.8529** |
| Accuracy        | 0.8533 |
| Positive F1     | 0.8608 |
| Negative F1     | 0.8451 |
| Positive Precision | 0.8293 |
| Negative Precision | 0.8824 |
| Positive Recall | 0.8947 |
| Negative Recall | 0.8108 |

### Per-Language Breakdown

| Language   | Code | Val Samples | Macro F1 |
|------------|------|:-----------:|:--------:|
| Hindi      | hi   | 11          | **1.0000** |
| Marathi    | mr   | 9           | **1.0000** |
| Telugu     | te   | 9           | **1.0000** |
| Bengali    | bn   | 18          | 0.9443   |
| Gujarati   | gu   | 13          | 0.9150   |
| Urdu       | ur   | 10          | 0.8990   |
| Assamese   | as   | 11          | 0.8706   |
| Tamil      | ta   | 9           | 0.8615   |
| Kannada    | kn   | 14          | 0.8542   |
| Malayalam  | ml   | 12          | 0.8333   |
| Punjabi    | pa   | 8           | 0.7333   |
| Bodo       | bd   | 12          | 0.6571   |
| Odia       | or   | 14          | 0.4269   |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 16 GB VRAM (Tesla T4 or better)
- CUDA 12.x toolkit

### Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate bitsandbytes peft trl
pip install scikit-learn pandas numpy
```

---

## Usage

### Running on Kaggle

1. Open the notebook on Kaggle with GPU (T4) accelerator enabled.
2. Add the **NPPE DLP 2026 Term 1** competition dataset as input.
3. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`.
4. Run all cells — the pipeline will:
   - Install dependencies
   - Load and explore the dataset
   - Fine-tune Gemma-3-1B-IT with QLoRA
   - Evaluate on the validation split
   - Generate `submission.csv` for the competition

### Running Locally

```bash
# Clone the repository
git clone https://github.com/anujdevsingh/multilingual-sentiment-analysis-qlora.git
cd multilingual-sentiment-analysis-qlora

# Install dependencies
pip install -r requirements.txt  # or install manually as listed above

# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Launch the notebook
jupyter notebook "Multilingual Sentiment Analysis.ipynb"
```

> **Note:** You will need to update the data paths in the `Config` class to point to your local copies of `train.csv` and `test.csv`.

---

## Project Structure

```
multilingual-sentiment-analysis-qlora/
├── Multilingual Sentiment Analysis.ipynb   # Main notebook (end-to-end pipeline)
├── README.md                               # This file
└── .gitattributes                          # Git configuration
```

### Notebook Sections

| Section | Title                        | Description                                              |
|---------|------------------------------|----------------------------------------------------------|
| 1       | Install Dependencies         | Installs bitsandbytes, peft, trl, accelerate, datasets   |
| 2       | Imports                      | All library imports                                      |
| 3       | Configuration                | Hyperparameters, paths, and reproducibility setup        |
| 4       | Data Loading & Exploration   | Load CSVs, inspect distributions and missing values      |
| 5       | Prompt Engineering           | Language-aware Gemma chat template construction          |
| 6       | Train/Validation Split       | Stratified split (85/15) preserving class balance        |
| 7       | Dataset Preparation          | Convert DataFrames to HuggingFace Datasets               |
| 8       | Load Model                   | Load Gemma-3-1B-IT in 4-bit NF4 quantization            |
| 9       | LoRA Adapter                 | Configure and attach LoRA adapters to all linear layers  |
| 10      | Training                     | SFTTrainer with cosine schedule and gradient checkpointing|
| 11      | Inference Helper             | Greedy decoding with keyword-based label extraction      |
| 12      | Validation Evaluation        | Compute Macro F1 and full classification report          |
| 13      | Per-Language F1 Breakdown    | F1 scores for each of the 13 languages                   |
| 14      | Test Set Predictions         | Generate predictions on the held-out test set            |
| 15      | Create Submission            | Export `submission.csv` in competition format             |

---

## Hyperparameters

All hyperparameters are centralized in the `Config` class for easy experimentation:

```python
class Config:
    MODEL_NAME     = "google/gemma-3-1b-it"
    LORA_R         = 64
    LORA_ALPHA     = 128
    LORA_DROPOUT   = 0.05
    EPOCHS         = 5
    BATCH_SIZE     = 2
    GRAD_ACCUM     = 8
    LEARNING_RATE  = 2e-4
    WEIGHT_DECAY   = 0.01
    WARMUP_RATIO   = 0.1
    MAX_SEQ_LEN    = 512
    VAL_SPLIT      = 0.15
    SEED           = 42
    MAX_NEW_TOKENS = 5
```

---

## Hardware Requirements

| Component | Minimum            | Recommended        |
|-----------|--------------------|--------------------|
| GPU       | Tesla T4 (16 GB)   | A100 (40 GB)       |
| RAM       | 16 GB              | 32 GB              |
| Disk      | 10 GB free         | 20 GB free         |
| CUDA      | 12.1+              | 12.6+              |

Training completes in approximately **34 minutes** on a Tesla T4 with the default configuration.

---

## Key Dependencies

| Package        | Purpose                                  |
|----------------|------------------------------------------|
| `transformers` | Model loading, tokenization, training    |
| `peft`         | LoRA / QLoRA adapter implementation      |
| `trl`          | SFTTrainer for supervised fine-tuning    |
| `bitsandbytes` | 4-bit NF4 quantization                  |
| `accelerate`   | Distributed training and device mapping  |
| `datasets`     | HuggingFace Dataset abstraction          |
| `scikit-learn` | Metrics (F1 score, classification report)|
| `pandas`       | Data loading and manipulation            |
| `torch`        | Deep learning framework                  |

---

## Acknowledgements

- [Google Gemma](https://ai.google.dev/gemma) for the base instruction-tuned model
- [Hugging Face](https://huggingface.co/) for the Transformers, PEFT, and TRL libraries
- [QLoRA paper](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) for the quantized fine-tuning methodology
- [Kaggle](https://www.kaggle.com/) for the competition platform and GPU compute
- NPPE DLP 2026 for organizing the competition
