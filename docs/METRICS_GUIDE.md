# Metrics Tracking and Saving Guide

## Overview

The enhanced modular pipeline now has **comprehensive MLflow integration** across all modules, ensuring consistent metrics tracking from baseline training through all compression stages.

---

## Metrics Flow Architecture

### 1. Baseline Training (`train.py`)

**MLflow Integration:** ✅ **Full**

```python
# In run_kfold_training()
with mlflow.start_run(run_name=f"{config.author_name}_..."):
    # Log all hyperparameters
    mlflow.log_params({
        'model_path': config.model_path,
        'batch_size': config.batch,
        'learning_rate': config.lr,
        # ... all config parameters
    })
    
    # For each fold and epoch
    for fold in range(num_folds):
        for epoch in range(epochs):
            # Calculate metrics with threshold exploration
            metrics = calculate_metrics_with_threshold_exploration(y_true, y_pred)
            
            # Log to MLflow
            mlflow.log_metrics({
                f'fold_{fold+1}_epoch_{epoch+1}_val_accuracy': metrics['accuracy'],
                f'fold_{fold+1}_epoch_{epoch+1}_val_macro_f1': metrics['macro_f1'],
                # ... all metrics
            })
    
    # Log aggregate metrics
    mlflow.log_metrics(aggregate_metrics)
    
    # Log CSV artifacts
    mlflow.log_artifact('fold_summary.csv')
    mlflow.log_artifact('best_metrics.csv')
```

**Saved Files:**
- `./outputs/fold_summary_{model}_{params}_{timestamp}.csv`
- `./outputs/best_metrics_{model}_{params}_{timestamp}.csv`
- `./mlruns/{experiment_id}/{run_id}/artifacts/`

---

### 2. Pipeline Mode (`pipeline.py`)

**MLflow Integration:** ✅ **Full** (newly added!)

The pipeline now wraps all stages in a single MLflow run:

```python
# In run_compression_pipeline()
with mlflow.start_run(run_name=f"{config.author_name}_{config.pipeline}"):
    run_id = mlflow.active_run().info.run_id
    
    # Log pipeline configuration
    mlflow.log_params({
        'pipeline': config.pipeline,
        'enable_kd': config.enable_kd,
        'enable_pruning': config.enable_pruning,
        'enable_quantization': config.enable_quantization,
        # ... all pipeline params
    })
    
    # STAGE 1: Baseline (uses existing MLflow from train.py)
    baseline_results = run_kfold_training(...)
    
    # STAGE 2: Knowledge Distillation
    if config.enable_kd:
        kd_results = _run_knowledge_distillation_stage(...)
        
        # Log KD metrics
        mlflow.log_metrics({
            'kd_accuracy': kd_results['best_metrics']['accuracy'],
            'kd_macro_f1': kd_results['best_metrics']['macro_f1'],
            'kd_f1_hate': kd_results['best_metrics']['f1'],
            'kd_f1_non_hate': kd_results['best_metrics']['f1_negative'],
            'kd_roc_auc': kd_results['best_metrics']['roc_auc'],
            'kd_best_threshold': kd_results['best_metrics']['best_threshold'],
            'student_total_params': kd_results['student_params'],
            'compression_ratio': teacher_params / student_params
        })
    
    # STAGE 3: Pruning
    if config.enable_pruning:
        pruning_results = _run_pruning_stage(...)
        
        # Log pruning metrics
        mlflow.log_metrics({
            'pruning_accuracy': pruning_results['best_metrics']['accuracy'],
            'pruning_macro_f1': pruning_results['best_metrics']['macro_f1'],
            'pruning_sparsity': pruning_results['sparsity'],
            # ... all metrics
        })
    
    # STAGE 4: Quantization
    if config.enable_quantization:
        quant_results = _run_quantization_stage(...)
        
        # Log quantization metrics
        mlflow.log_metrics({
            'quantization_accuracy': quant_results['best_metrics']['accuracy'],
            'quantization_macro_f1': quant_results['best_metrics']['macro_f1'],
            # ... all metrics
        })
    
    # Save pipeline summary CSV
    summary_df = _create_pipeline_summary_df(pipeline_results)
    summary_df.to_csv('pipeline_summary.csv')
    mlflow.log_artifact('pipeline_summary.csv')
```

**Saved Files:**
- `./outputs/pipeline_summary_{pipeline_mode}.csv`
- `./mlruns/{experiment_id}/{run_id}/artifacts/pipeline_summary.csv`
- All metrics logged to MLflow UI

---

## Complete Metrics Tracking Matrix

| Stage | Metrics Calculated | MLflow Logged | CSV Saved | JSON Saved |
|-------|-------------------|---------------|-----------|------------|
| **Baseline** | ✅ Threshold exploration<br>✅ Per-class metrics<br>✅ ROC-AUC | ✅ Per-fold per-epoch<br>✅ Aggregate stats | ✅ fold_summary.csv<br>✅ best_metrics.csv | ❌ |
| **KD** | ✅ Threshold exploration<br>✅ Per-class metrics<br>✅ ROC-AUC | ✅ Best metrics<br>✅ Model size<br>✅ Compression ratio | ✅ pipeline_summary.csv | ❌ |
| **Pruning** | ✅ Threshold exploration<br>✅ Per-class metrics<br>✅ ROC-AUC | ✅ Best metrics<br>✅ Sparsity | ✅ pipeline_summary.csv | ❌ |
| **Quantization** | ✅ Threshold exploration<br>✅ Per-class metrics<br>✅ ROC-AUC | ✅ Best metrics<br>✅ Method | ✅ pipeline_summary.csv | ❌ |

---

## Metrics Logged to MLflow

### Baseline Stage
- `teacher_total_params`
- `teacher_size_mb`
- Per-fold per-epoch metrics (from `train.py`)

### KD Stage
- `kd_accuracy`
- `kd_macro_f1`
- `kd_f1_hate`
- `kd_f1_non_hate`
- `kd_roc_auc`
- `kd_best_threshold`
- `student_total_params`
- `student_size_mb`
- `compression_ratio`

### Pruning Stage
- `pruning_accuracy`
- `pruning_macro_f1`
- `pruning_f1_hate`
- `pruning_f1_non_hate`
- `pruning_roc_auc`
- `pruning_best_threshold`
- `pruning_sparsity`

### Quantization Stage
- `quantization_accuracy`
- `quantization_macro_f1`
- `quantization_f1_hate`
- `quantization_f1_non_hate`
- `quantization_roc_auc`
- `quantization_best_threshold`

### Pipeline Parameters
- `pipeline`
- `pipeline_description`
- `enable_kd`, `enable_pruning`, `enable_quantization`
- `model_path`, `student_path` (if KD)
- `batch_size`, `learning_rate`, `epochs`
- `kd_alpha`, `kd_temperature` (if KD)
- `prune_method`, `prune_sparsity` (if pruning)
- `quant_method`, `quant_dtype` (if quantization)
- `final_model_path` (if saved)

---

## Viewing Metrics

### MLflow UI

```bash
# Start MLflow UI
cd Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data
mlflow ui

# Open browser
http://localhost:5000
```

**What you'll see:**
- All experiments organized by `mlflow_experiment_name`
- Each run with unique ID
- Parameters tab: All configuration
- Metrics tab: All logged metrics with charts
- Artifacts tab: CSV files, model cards

### CSV Files

**Baseline Mode:**
```
outputs/
├── fold_summary_sagorsarker-bangla-bert-base_batch32_lr2e-05_20260110_195532.csv
└── best_metrics_sagorsarker-bangla-bert-base_batch32_lr2e-05_20260110_195532.csv
```

**Pipeline Mode:**
```
outputs/
├── pipeline_summary_baseline_kd_prune_quant.csv
└── final_quantized_student/
    ├── config.json
    ├── pytorch_model.bin
    ├── classifier.pt
    ├── README.md
    └── how_to_load.py
```

---

## Example: Full Pipeline Metrics

When running the full pipeline (`baseline_kd_prune_quant`), you get:

### MLflow Run
- **Run Name:** `researcher_baseline_kd_prune_quant`
- **Parameters:** 25+ parameters logged
- **Metrics:** 30+ metrics logged
- **Artifacts:** 
  - `pipeline_summary_baseline_kd_prune_quant.csv`
  - Baseline fold summaries (from nested run)

### CSV Summary
```csv
Stage,Accuracy,Macro F1,F1 (Hate),F1 (Non-Hate),ROC-AUC
Baseline,0.8523,0.8456,0.8234,0.8678,0.9123
KD Student,0.8401,0.8312,0.8145,0.8479,0.8956
Pruned,0.8267,0.8189,0.7998,0.8380,0.8834
Quantized,0.8198,0.8123,0.7912,0.8334,0.8789
```

---

## Consistency Across Modules

All modules now use the same metrics calculation:

```python
from metrics import calculate_metrics_with_threshold_exploration

# In any stage (baseline, KD, pruning, quantization)
metrics = calculate_metrics_with_threshold_exploration(y_true, y_pred)

# Returns:
{
    'accuracy': 0.8523,
    'macro_f1': 0.8456,
    'f1': 0.8234,              # Hate class
    'f1_negative': 0.8678,     # Non-hate class
    'precision': 0.8312,
    'recall': 0.8156,
    'roc_auc': 0.9123,
    'best_threshold': 0.45,
    'macro_f1_th_0.3': 0.8234,  # For each threshold
    'macro_f1_th_0.35': 0.8312,
    # ... more thresholds
}
```

---

## Summary

✅ **Baseline Training:** Full MLflow + CSV export  
✅ **KD Stage:** Full MLflow + included in pipeline summary  
✅ **Pruning Stage:** Full MLflow + included in pipeline summary  
✅ **Quantization Stage:** Full MLflow + included in pipeline summary  
✅ **Pipeline Summary:** CSV export + MLflow artifact  
✅ **Consistent Metrics:** Same calculation across all stages  
✅ **Threshold Exploration:** Applied at every stage  

**All metrics are now tracked consistently across all modules!** 🎉
