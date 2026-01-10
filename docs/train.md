# train.py - Baseline Training Module

## Overview

The `train.py` module implements rigorous baseline training with K-fold cross-validation, threshold exploration, and comprehensive metrics tracking. This is the foundation of the entire pipeline.

## Key Features

- ✅ K-fold cross-validation with stratified splitting
- ✅ Threshold exploration for optimal macro F1
- ✅ Early stopping with patience
- ✅ Mixed precision training (AMP)
- ✅ MLflow experiment tracking
- ✅ CSV export (29 columns)
- ✅ Comprehensive metrics (train + validation)

---

## Main Function

### `run_kfold_training(config, comments, labels, tokenizer, device)`

**Purpose:** Execute K-fold cross-validation training for hate speech detection

**Arguments:**
- `config`: Configuration object with all hyperparameters
- `comments`: Array of text comments (strings)
- `labels`: Array of binary labels (0/1)
- `tokenizer`: HuggingFace tokenizer
- `device`: PyTorch device (cuda/cpu)

**Returns:**
- `list`: List of dictionaries, one per fold, containing best metrics

**Process:**
1. Setup MLflow experiment
2. Calculate class weights
3. Create K-fold splits (stratified)
4. For each fold:
   - Initialize model
   - Setup optimizer (AdamW) and scheduler (linear warmup)
   - Train for N epochs with early stopping
   - Track best metrics
5. Aggregate results across folds
6. Export CSVs (fold_summary, best_metrics)
7. Log to MLflow

---

## Training Functions

### `train_epoch(model, dataloader, optimizer, scheduler, device, class_weights, max_norm)`

**Purpose:** Train model for one epoch

**Features:**
- Mixed precision training (AMP)
- Gradient clipping
- Class-weighted loss (for imbalanced data)
- Progress bar with tqdm

**Returns:**
- `dict`: Training metrics (loss, accuracy, macro_f1, etc.)

---

### `evaluate_model(model, dataloader, device, class_weights)`

**Purpose:** Evaluate model on validation/test set

**Features:**
- No gradient computation
- Threshold exploration
- Comprehensive metrics calculation

**Returns:**
- `dict`: Validation metrics with best threshold

---

## Metrics Calculation

### `calculate_metrics(y_true, y_pred)`

**Purpose:** Calculate comprehensive metrics with threshold exploration

**Thresholds Tested:** [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

**Metrics Returned:**
- `accuracy`: Accuracy at best threshold
- `precision`: Precision for hate class
- `recall`: Recall for hate class
- `f1`: F1 for hate class
- `precision_negative`: Precision for non-hate class
- `recall_negative`: Recall for non-hate class
- `f1_negative`: F1 for non-hate class
- `macro_f1`: Average F1 (both classes)
- `roc_auc`: ROC-AUC score
- `best_threshold`: Optimal threshold
- `macro_f1_th_{thresh}`: Macro F1 at each threshold

**Optimization:** Selects threshold that maximizes macro F1

---

## Utility Functions

### `cache_dataset(comments, labels, tokenizer, max_length, cache_file)`

**Purpose:** Cache preprocessed dataset to disk

**Note:** Currently not used in main training loop (on-the-fly tokenization preferred)

---

### `print_epoch_metrics(epoch, num_epochs, fold, num_folds, train_metrics, val_metrics, best_macro_f1, best_epoch)`

**Purpose:** Print formatted metrics for current epoch

**Output:**
```
Fold 1/5 | Epoch 12/15
  Train: Acc=0.8712 | Macro F1=0.8700 | F1(Hate)=0.8478 | Loss=0.2987
  Val:   Acc=0.8523 | Macro F1=0.8509* | F1(Hate)=0.8234 | Loss=0.3456
```

---

## MLflow Integration

### Experiment Setup
```python
mlflow.set_experiment(config.mlflow_experiment_name)
with mlflow.start_run(run_name=f"{config.author_name}_{timestamp}"):
    # Training code
```

### Parameters Logged
- Model configuration (model_path, batch_size, lr, epochs, etc.)
- Training hyperparameters (dropout, weight_decay, warmup_ratio, etc.)
- K-fold configuration (num_folds, stratification_type, seed)

### Metrics Logged
**Per-fold, per-epoch:**
- `fold_{N}_epoch_{M}_val_loss`
- `fold_{N}_epoch_{M}_val_accuracy`
- `fold_{N}_epoch_{M}_val_macro_f1`
- `fold_{N}_epoch_{M}_val_f1`
- `fold_{N}_epoch_{M}_val_roc_auc`

**Aggregate (across folds):**
- `mean_val_accuracy`, `std_val_accuracy`
- `mean_val_macro_f1`, `std_val_macro_f1`
- `mean_val_f1`, `std_val_f1`
- ... (all metrics)

**Best metrics:**
- `best_fold_index`
- `best_epoch`
- `best_accuracy`, `best_macro_f1`, etc.

### Artifacts Logged
- `fold_summary_{params}_{timestamp}.csv`
- `best_metrics_{params}_{timestamp}.csv`

---

## CSV Export

### fold_summary.csv

**Columns (29 total):**
1. Model
2. Batch Size
3. Learning Rate
4. Epochs
5. Fold
6. Best Epoch
7-17. Validation metrics (11 columns)
18-27. Training metrics (10 columns)
28. total_parameters
29. trainable_parameters

**Additional Rows:**
- Mean (across folds)
- Std (across folds)

### best_metrics.csv

Same 29 columns, single row with best fold's metrics.

---

## Training Configuration

### Optimizer
- **Type:** AdamW
- **Learning Rate:** Configurable (default: 2e-5)
- **Weight Decay:** Configurable (default: 0.01)

### Scheduler
- **Type:** Linear warmup with linear decay
- **Warmup Steps:** `warmup_ratio * total_steps`
- **Total Steps:** `num_epochs * steps_per_epoch`

### Loss Function
- **Type:** BCEWithLogitsLoss
- **Class Weights:** Optional (calculated from training data)

### Early Stopping
- **Metric:** Validation macro F1
- **Patience:** Configurable (default: 5 epochs)
- **Mode:** Maximize macro F1

---

## Usage Example

```python
from train import run_kfold_training
from transformers import AutoTokenizer
import torch

# Load data
comments, labels = load_and_preprocess_data('data.csv')

# Setup
tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run training
results = run_kfold_training(config, comments, labels, tokenizer, device)

# Results is a list of dicts, one per fold
print(f"Best fold: {results[0]['best_fold']}")
print(f"Best macro F1: {results[0]['macro_f1']:.4f}")
```

---

## Output Files

**Location:** `./outputs/`

**Files:**
- `fold_summary_{model}_{batch}_lr{lr}_epochs{epochs}_{timestamp}.csv`
- `best_metrics_{model}_{batch}_lr{lr}_epochs{epochs}_{timestamp}.csv`

**MLflow:**
- `./mlruns/{experiment_id}/{run_id}/`
  - `params/` - All parameters
  - `metrics/` - All metrics
  - `artifacts/` - CSV files

---

## Performance Considerations

### Memory Optimization
- Mixed precision training (AMP) reduces memory by ~40%
- Batch size can be adjusted based on GPU memory
- Gradient accumulation not yet implemented

### Speed Optimization
- DataLoader with `num_workers=2`
- Pin memory for faster GPU transfer
- Early stopping prevents unnecessary epochs

### Accuracy Optimization
- Threshold exploration ensures optimal F1
- Stratified k-fold maintains class balance
- Class weights handle imbalance

---

## Dependencies

- PyTorch 1.9+
- Transformers 4.0+
- scikit-learn
- MLflow
- tqdm
- pandas
- numpy

---

## See Also

- [metrics.py](metrics.md) - Enhanced metrics calculation
- [config.py](config.md) - Configuration options
- [data.py](data.md) - Data loading and preprocessing
- [model.py](model.md) - Model architecture
