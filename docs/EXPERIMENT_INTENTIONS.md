# Experiment Intentions and Configurations

This document outlines the planned experiments for the Bangla Hate Speech Compression framework. The goal is to explore the trade-offs between model size, inference speed, and accuracy (F1 score).

## 1. Baseline Experiments (Teacher Training)

**Goal**: Establish a strong baseline performance with the `sagorsarker/bangla-bert-base` model.

| Experiment ID | Batch Size | Learning Rate | Epochs | Description |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_v1` | 32 | 2e-5 | 15 | Standard fine-tuning setup. |
| `baseline_v2` | 32 | 1e-5 | 20 | Lower LR for potentially better convergence on noisy data. |
| `baseline_v3` | 64 | 3e-5 | 15 | Larger batch size for faster training. |
| `baseline_v4` | 32 | 2e-5 | 15 | **Feature Extraction**: Freeze base layers, train only classifier. |

## 2. Knowledge Distillation (KD) Experiments

**Goal**: Transfer knowledge to `distilbert-base-multilingual-cased`.

| Experiment ID | Alpha | Temperature | Description |
| :--- | :--- | :--- | :--- |
| `kd_v1` | 0.5 | 4.0 | Balanced loss. |
| `kd_v2` | 0.7 | 4.0 | Higher weight on soft targets (teacher guidance). |
| `kd_v3` | 0.2 | 2.0 | Higher weight on hard labels (ground truth). |

## 3. Pruning Experiments

**Goal**: Reduce parameter count while maintaining accuracy.

| Experiment ID | Method | Sparsity | Fine-tune Epochs | Description |
| :--- | :--- | :--- | :--- | :--- |
| `prune_v1` | Magnitude | 0.3 (30%) | 3 | Conservative pruning. |
| `prune_v2` | Magnitude | 0.5 (50%) | 5 | Aggressive pruning. |
| `prune_v3` | Wanda | 0.5 (50%) | 5 | Advanced pruning using activation data. |

## 4. Quantization Experiments

**Goal**: Reduce model size and memory footprint.

| Experiment ID | Method | Target Device | Description |
| :--- | :--- | :--- | :--- |
| `quant_v1` | Dynamic INT8 | CPU | Best for CPU inference speed. |
| `quant_v2` | Static INT8 | CPU/Edge | Requires calibration, faster on supported hardware. |
| `quant_v3` | INT4 (NF4) | GPU | Maximum size reduction (requires `bitsandbytes`). |
| `quant_v4` | FP16 | GPU | Native GPU acceleration. |

## 5. Full Pipeline Goals

**Target Metric**: Maintain >95% of Baseline F1 Score.
**Target Compression**: >10x reduction in size (MB).

**Recommended Full Pipeline Config**:
```bash
python src/main.py \
    --pipeline baseline_kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --batch 32 \
    --epochs 15 \
    --kd_alpha 0.7 \
    --prune_sparsity 0.5 \
    --quant_method dynamic
```
