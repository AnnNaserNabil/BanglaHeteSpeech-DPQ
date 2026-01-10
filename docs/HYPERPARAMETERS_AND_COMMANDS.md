# Hyperparameters and Execution Guide

This document details the default hyperparameters used in the `Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data` project and provides ready-to-run commands for various scenarios.

## 1. Default Hyperparameters

These values are defined in `config.py` and are used unless overridden by command-line arguments.

| Category | Parameter | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Training** | `batch` | 32 | Batch size for training and evaluation |
| | `lr` | 2e-5 | Learning rate for AdamW optimizer |
| | `epochs` | 15 | Maximum number of training epochs |
| | `max_length` | 128 | Max sequence length for tokenization |
| | `num_folds` | 5 | Number of folds for K-Fold cross-validation |
| | `dropout` | 0.1 | Dropout rate for classification head |
| | `weight_decay` | 0.01 | Weight decay for optimizer |
| | `warmup_ratio` | 0.1 | Ratio of steps for linear warmup |
| | `early_stopping_patience` | 5 | Epochs to wait before early stopping |
| **Distillation** | `kd_alpha` | 0.7 | Weight for soft target loss (0.7 soft, 0.3 hard) |
| | `kd_temperature` | 4.0 | Temperature for softening probability distributions |
| | `kd_method` | `logit` | Distillation method (logit, hidden, attention) |
| **Pruning** | `prune_method` | `magnitude` | Pruning strategy (magnitude, wanda, gradual) |
| | `prune_sparsity` | 0.5 | Target sparsity (50% of weights removed) |
| | `fine_tune_after_prune` | `True` | Whether to fine-tune after pruning |
| | `fine_tune_epochs` | 3 | Epochs for post-pruning fine-tuning |
| **Quantization** | `quant_method` | `dynamic` | Quantization method (dynamic, static, fp16, int4) |
| | `quant_dtype` | `int8` | Data type for quantization |

## 2. Run Commands

Run these commands from the `Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data` directory.

### 🚀 Scenario 1: Fast Verification (Debug Mode)
Use this to quickly verify that the code works without waiting for full training.
- **Dataset**: Small subset (`data/data_small.csv`)
- **Epochs**: 1
- **Folds**: 2
- **Pruning**: No fine-tuning (for speed)

```bash
python main.py \
  --pipeline baseline_kd_prune_quant \
  --dataset_path data/data_small.csv \
  --epochs 1 \
  --num_folds 2 \
  --batch 2 \
  --author_name debug_run \
  --no_fine_tune_after_prune \
  --save_huggingface
```

### 🏆 Scenario 2: Full Production Run (Standard)
This runs the complete pipeline with default rigorous settings.
- **Dataset**: Full dataset (`data/HateSpeech.csv`)
- **Epochs**: 15 (with early stopping)
- **Folds**: 5

```bash
python main.py \
  --pipeline baseline_kd_prune_quant \
  --dataset_path data/HateSpeech.csv \
  --epochs 15 \
  --num_folds 5 \
  --batch 32 \
  --author_name production_run \
  --save_huggingface
```

### 📉 Scenario 3: Aggressive Compression (Max Size Reduction)
Targeting maximum model shrinkage for mobile/edge deployment.
- **Pruning**: 70% sparsity using WANDA (advanced pruning)
- **Quantization**: INT4 (4-bit quantization)

```bash
python main.py \
  --pipeline baseline_kd_prune_quant \
  --dataset_path data/HateSpeech.csv \
  --prune_method wanda \
  --prune_sparsity 0.7 \
  --quant_method int4 \
  --epochs 10 \
  --author_name aggressive_compression \
  --save_huggingface
```

### 🎯 Scenario 4: High Accuracy Focus
Prioritizing performance over compression ratio.
- **Pruning**: Conservative 30% sparsity
- **Quantization**: FP16 (GPU inference) instead of INT8
- **Training**: Lower learning rate

```bash
python main.py \
  --pipeline baseline_kd_prune_quant \
  --dataset_path data/HateSpeech.csv \
  --prune_sparsity 0.3 \
  --quant_method fp16 \
  --lr 1e-5 \
  --epochs 20 \
  --author_name high_accuracy \
  --save_huggingface
```

### 🔬 Scenario 5: Baseline Only (No Compression)
Just run the rigorous K-Fold cross-validation baseline.

```bash
python main.py \
  --pipeline baseline \
  --dataset_path data/HateSpeech.csv \
  --epochs 15 \
  --num_folds 5 \
  --author_name baseline_only
```
