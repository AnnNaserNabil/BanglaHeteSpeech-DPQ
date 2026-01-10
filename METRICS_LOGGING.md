# Metrics Logging Documentation

This document explains how metrics are calculated, logged, and stored at every step of the Bangla Hate Speech Detection pipeline. It covers the logic behind the metrics for Baseline Training, Knowledge Distillation, Pruning, and Quantization.

## 1. Overview of Metrics

The pipeline uses a comprehensive set of metrics optimized for binary classification (Hate Speech vs. Non-Hate Speech).

| Metric | Description | Key Logic |
| :--- | :--- | :--- |
| **Accuracy** | Overall correctness. | `(TP + TN) / Total` |
| **Macro F1** | **Primary Metric**. Average of F1 scores for both classes. | `(F1_Hate + F1_NonHate) / 2` |
| **F1 (Hate)** | F1 score specifically for the Hate Speech class. | Harmonic mean of Precision and Recall for class 1. |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve. | Measures ability to distinguish between classes across all thresholds. |
| **Best Threshold** | The probability threshold (e.g., 0.45) that maximizes Macro F1. | Explored dynamically: `[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]` |

---

## 2. Baseline Training (Stage 1)

**Goal:** Train a high-performance "Teacher" model using rigorous K-Fold Cross-Validation.

### Logging Logic
1.  **Per-Epoch Logging**:
    *   At the end of every epoch, the model is evaluated on the validation set.
    *   **Threshold Exploration**: The system calculates Macro F1 at multiple thresholds (0.3 to 0.6) and selects the best one.
    *   **MLflow**: Logs `val_loss`, `val_accuracy`, `val_macro_f1`, `val_f1` (Hate), and `val_best_threshold`.
    *   **Console**: Prints a summary table for the epoch.

2.  **Per-Fold Summary**:
    *   After all epochs for a fold, the **best epoch** (highest Macro F1) is selected.
    *   All metrics from this best epoch are saved as the result for that fold.
    *   **Artifacts**: `fold_summary_*.csv` is saved, listing results for each fold.

3.  **Experiment Summary (Aggregated)**:
    *   After all K folds are complete, metrics are averaged (Mean ± Std Dev).
    *   **Artifacts**: `best_metrics_*.csv` contains the final averaged performance.
    *   **Model Saving**: The model from the *best single fold* is saved as `best_model_*.pt`.

---

## 3. Knowledge Distillation (Stage 2)

**Goal:** Train a smaller "Student" model to mimic the Teacher.

### Logging Logic
1.  **Loss Components**:
    *   **Soft Loss**: How well the student matches the teacher's probability distribution (Temperature-scaled).
    *   **Hard Loss**: How well the student matches the actual ground truth labels.
    *   **Total Loss**: Weighted sum (`alpha * Soft + (1-alpha) * Hard`).
    *   **MLflow**: Logs `soft_loss`, `hard_loss`, and `total_loss` per step/epoch.

2.  **Student Performance**:
    *   The student is evaluated on the validation set using the same metrics as the baseline (Accuracy, Macro F1, etc.).
    *   **Comparison**: You can compare `student_macro_f1` vs. `teacher_macro_f1` to see the "cost" of compression.

3.  **Model Size**:
    *   Logs `student_params` and `compression_ratio` (Teacher Params / Student Params).

---

## 4. Pruning (Stage 3)

**Goal:** Remove unimportant weights from the Student model to make it sparse.

### Logging Logic
1.  **Sparsity**:
    *   Logs the percentage of weights set to zero (e.g., `sparsity: 0.5` means 50% zeros).
    *   **Method**: Logs whether `magnitude` or `wanda` pruning was used.

2.  **Performance Recovery**:
    *   If `fine_tune_after_prune` is enabled, logs the metrics *after* fine-tuning.
    *   This shows how well the model recovered its accuracy after losing weights.

---

## 5. Quantization (Stage 4)

**Goal:** Reduce precision (e.g., FP32 -> INT8) for faster inference and smaller size.

### Logging Logic
1.  **Size Reduction**:
    *   Logs `original_size_mb` vs. `quantized_size_mb`.
    *   Logs `size_reduction_pct` (e.g., "75% reduction" for INT8).

2.  **Inference Speed (Latency)**:
    *   Benchmarks the model to measure `ms/sample` (milliseconds per sample).
    *   Logs `throughput` (samples per second).

3.  **Accuracy Check**:
    *   Evaluates the quantized model to ensure accuracy hasn't dropped significantly.
    *   **Critical**: For INT8, this evaluation often happens on CPU (since PyTorch INT8 is CPU-optimized).

---

## 6. Final Pipeline Summary

At the very end of the `baseline_kd_prune_quant` pipeline, a **Master Summary** is generated.

### `pipeline_summary_*.csv`
This file (saved in `outputs/`) contains a row for each stage, allowing easy comparison:

| Stage | Accuracy | Macro F1 | F1 (Hate) | Size (MB) | Speed (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.852 | 0.841 | 0.810 | 420 | 12.5 |
| **Student (KD)** | 0.845 | 0.835 | 0.805 | 260 | 8.2 |
| **Pruned** | 0.840 | 0.830 | 0.798 | 260 (sparse) | 8.2 |
| **Quantized** | 0.838 | 0.828 | 0.795 | 65 | 4.1 |

*Note: Actual values will vary based on your experiment.*

## How to Check Your Configuration

To verify your configuration based on logs:
1.  **Open MLflow**: Run `mlflow ui` and check the "Parameters" tab for your run.
2.  **Check `pipeline` param**: It will say `baseline`, `baseline_kd`, etc.
3.  **Check `pipeline_description`**: A human-readable string explaining the active stages.
