# CSV Export Format Reference

## Overview

The baseline training now exports CSV files with **exactly the same columns** as your previous experiments, ensuring compatibility and consistency.

---

## Fold Summary CSV

**Filename:** `fold_summary_{model}_{batch}_lr{lr}_epochs{epochs}_{timestamp}.csv`

**Location:** `./outputs/`

### Columns (in order):

1. **Model** - Model path (e.g., `sagorsarker/bangla-bert-base`)
2. **Batch Size** - Batch size used for training
3. **Learning Rate** - Learning rate used
4. **Epochs** - Maximum epochs configured
5. **Fold** - Fold identifier (e.g., `Fold 1`, `Fold 2`, ...)
6. **Best Epoch** - Epoch where best validation macro F1 was achieved
7. **Val Accuracy** - Validation accuracy at best threshold
8. **Val Precision (Hate)** - Validation precision for hate speech class
9. **Val Recall (Hate)** - Validation recall for hate speech class
10. **Val F1 (Hate)** - Validation F1 for hate speech class
11. **Val Precision (Non-Hate)** - Validation precision for non-hate class
12. **Val Recall (Non-Hate)** - Validation recall for non-hate class
13. **Val F1 (Non-Hate)** - Validation F1 for non-hate class
14. **Val Macro F1** - Validation macro F1 (average of both classes)
15. **Val ROC-AUC** - Validation ROC-AUC score
16. **Val Loss** - Validation loss at best epoch
17. **Best Threshold** - Optimal classification threshold (from threshold exploration)
18. **Train Accuracy** - Training accuracy at best epoch
19. **Train Precision (Hate)** - Training precision for hate speech class
20. **Train Recall (Hate)** - Training recall for hate speech class
21. **Train F1 (Hate)** - Training F1 for hate speech class
22. **Train Precision (Non-Hate)** - Training precision for non-hate class
23. **Train Recall (Non-Hate)** - Training recall for non-hate class
24. **Train F1 (Non-Hate)** - Training F1 for non-hate class
25. **Train Macro F1** - Training macro F1
26. **Train ROC-AUC** - Training ROC-AUC score
27. **Train Loss** - Training loss at best epoch
28. **total_parameters** - Total model parameters
29. **trainable_parameters** - Trainable model parameters

**Additional Rows:**
- **Mean** - Mean of all numeric columns across folds
- **Std** - Standard deviation of all numeric columns across folds

---

## Best Metrics CSV

**Filename:** `best_metrics_{model}_{batch}_lr{lr}_epochs{epochs}_{timestamp}.csv`

**Location:** `./outputs/`

### Columns (in order):

Same as Fold Summary, but contains only **one row** with the best performing fold's metrics:

1. **Model**
2. **Batch Size**
3. **Learning Rate**
4. **Epochs**
5. **Best Fold** - Which fold achieved the best results
6. **Best Epoch** - Epoch where best validation macro F1 was achieved
7-29. *(Same validation and training metrics as above)*

---

## Example CSV Output

### fold_summary.csv

```csv
Model,Batch Size,Learning Rate,Epochs,Fold,Best Epoch,Val Accuracy,Val Precision (Hate),Val Recall (Hate),Val F1 (Hate),Val Precision (Non-Hate),Val Recall (Non-Hate),Val F1 (Non-Hate),Val Macro F1,Val ROC-AUC,Val Loss,Best Threshold,Train Accuracy,Train Precision (Hate),Train Recall (Hate),Train F1 (Hate),Train Precision (Non-Hate),Train Recall (Non-Hate),Train F1 (Non-Hate),Train Macro F1,Train ROC-AUC,Train Loss,total_parameters,trainable_parameters
sagorsarker/bangla-bert-base,32,2e-05,15,Fold 1,12,0.8523,0.8312,0.8156,0.8234,0.8678,0.8890,0.8783,0.8509,0.9123,0.3456,0.45,0.8712,0.8534,0.8423,0.8478,0.8845,0.9001,0.8922,0.8700,0.9234,0.2987,110000000,110000000
sagorsarker/bangla-bert-base,32,2e-05,15,Fold 2,10,0.8456,0.8245,0.8089,0.8167,0.8612,0.8823,0.8716,0.8442,0.9056,0.3523,0.45,0.8645,0.8467,0.8356,0.8411,0.8778,0.8934,0.8855,0.8633,0.9167,0.3054,110000000,110000000
...
Mean,,,,,11.2,0.8489,0.8278,0.8122,0.8200,0.8645,0.8856,0.8749,0.8475,0.9089,0.3489,0.45,0.8678,0.8500,0.8389,0.8444,0.8811,0.8967,0.8888,0.8666,0.9200,0.3020,110000000,110000000
Std,,,,,0.8,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0,0
```

### best_metrics.csv

```csv
Model,Batch Size,Learning Rate,Epochs,Best Fold,Best Epoch,Val Accuracy,Val Precision (Hate),Val Recall (Hate),Val F1 (Hate),Val Precision (Non-Hate),Val Recall (Non-Hate),Val F1 (Non-Hate),Val Macro F1,Val ROC-AUC,Val Loss,Best Threshold,Train Accuracy,Train Precision (Hate),Train Recall (Hate),Train F1 (Hate),Train Precision (Non-Hate),Train Recall (Non-Hate),Train F1 (Non-Hate),Train Macro F1,Train ROC-AUC,Train Loss,total_parameters,trainable_parameters
sagorsarker/bangla-bert-base,32,2e-05,15,Fold 1,12,0.8523,0.8312,0.8156,0.8234,0.8678,0.8890,0.8783,0.8509,0.9123,0.3456,0.45,0.8712,0.8534,0.8423,0.8478,0.8845,0.9001,0.8922,0.8700,0.9234,0.2987,110000000,110000000
```

---

## MLflow Logging

All these metrics are also logged to MLflow:

### Parameters Logged:
- `model_path`
- `batch_size`
- `learning_rate`
- `epochs`
- `num_folds`
- `seed`
- `dropout`
- `weight_decay`
- `warmup_ratio`
- `gradient_clip_norm`
- `early_stopping_patience`
- `stratification_type`
- `max_length`
- `freeze_base`

### Metrics Logged (per fold, per epoch):
- `fold_{N}_epoch_{M}_val_loss`
- `fold_{N}_epoch_{M}_val_accuracy`
- `fold_{N}_epoch_{M}_val_macro_f1`
- `fold_{N}_epoch_{M}_val_f1`
- `fold_{N}_epoch_{M}_val_roc_auc`
- ... (all validation metrics)

### Aggregate Metrics Logged:
- `mean_val_accuracy`
- `std_val_accuracy`
- `mean_val_macro_f1`
- `std_val_macro_f1`
- ... (mean and std for all metrics)

### Best Metrics Logged:
- `best_fold_index`
- `best_epoch`
- `best_accuracy`
- `best_macro_f1`
- ... (all best metrics)

### Artifacts Logged:
- `fold_summary_{...}.csv`
- `best_metrics_{...}.csv`

---

## Compatibility

✅ **100% Compatible** with your previous experiment CSV format  
✅ **Same column order** as before  
✅ **Same column names** as before  
✅ **Additional MLflow tracking** for better experiment management  
✅ **Enhanced metrics** with threshold exploration  

You can directly compare results with your previous experiments!
