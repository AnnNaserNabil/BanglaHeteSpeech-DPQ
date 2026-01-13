# Hyperparameter Selection Guide

This document explains the rationale behind the selected hyperparameters for various models used in this project. The parameters were chosen based on extensive experimentation across multiple folds, focusing on **Macro F1 Score**, **Validation Loss**, and **ROC-AUC**.

## Quick Reference: Best Configurations (Option 1)

If you are looking for the best performing and most stable settings, use these:

| Option | Model | Batch Size | Learning Rate | Epochs | Expected Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Option 1** | **XLM-RoBERTa** | 32 | 3e-5 | 30 | ~0.8515 |
| **Option 1** | **mBERT** | 64 | 3e-5 | 30 | ~0.8439 |
| **Option 1** | **Bangla-BERT** | 64 | 3e-5 | 30 | ~0.8436 |
| **Option 1** | **Mixed-Distil-BERT** | 16 | 1e-5 | 10 | ~0.8470 |
| **Option 1** | **XtremeDistil** | 16 | 3e-5 | 30 | ~0.8243 |
| **Option 1** | **sahajBERT** | 16 | 1e-5 | 10 | ~0.8456 |

---

## Selection Criteria
1. **Macro F1 Score**: Primary metric for balanced performance across Hate and Non-Hate classes.
2. **Validation Loss**: Used to identify stability and potential overfitting.
3. **ROC-AUC**: Measures the model's ability to distinguish between classes.
4. **Best Epoch**: Indicates how quickly the model converges and where early stopping should ideally trigger.

---

## 1. FacebookAI/xlm-roberta-base

| Option | Batch Size | Learning Rate | Epochs | Val Macro F1 | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option 1** | **32** | **3e-5** | **30** | **0.8515** | **0.3988** | **Recommended** |
| Option 2 | 64 | 3e-5 | 30 | 0.8533 | 0.5279 | High Performance |
| Option 3 | 64 | 2e-5 | 20 | 0.8510 | 0.4249 | Stable |

**Rationale**: Option 1 is recommended because it provides a near-peak Macro F1 score while maintaining a significantly lower validation loss (0.3988) compared to Option 2. This suggests better generalization and less overfitting. [See detailed rationale](docss/xlm%20roberta.md).

---

## 2. google-bert/bert-base-multilingual-cased

| Option | Batch Size | Learning Rate | Epochs | Val Macro F1 | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option 1** | **64** | **3e-5** | **30** | **0.8439** | **0.4328** | **Recommended** |
| Option 2 | 64 | 2e-5 | 20 | 0.8436 | 0.4315 | Stable |
| Option 3 | 16 | 1e-5 | 10 | 0.8414 | 0.3733 | Low Resource |

**Rationale**: Option 1 achieves the highest Macro F1 for this model. While Option 3 has a lower loss, the performance gain in F1 with Option 1 makes it the preferred choice for accuracy. [See detailed rationale](docss/mbert.md).

---

## 3. sagorsarker/bangla-bert-base

| Option | Batch Size | Learning Rate | Epochs | Val Macro F1 | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option 1** | **64** | **3e-5** | **30** | **0.8436** | **0.3663** | **Recommended** |
| Option 2 | 32 | 3e-5 | 30 | 0.8427 | 0.3912 | Alternative |
| Option 3 | 32 | 1e-5 | 10 | 0.8398 | 0.4078 | Stable |

**Rationale**: Option 1 is the clear winner with the highest Macro F1 and the lowest validation loss in its group, indicating excellent stability for this Bangla-specific model. [See detailed rationale](docss/bangla%20bert.md).

---

## 4. md-nishat-008/Mixed-Distil-BERT

| Option | Batch Size | Learning Rate | Epochs | Val Macro F1 | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option 1** | **16** | **1e-5** | **10** | **0.8470** | **0.4465** | **Recommended** |
| Option 2 | 64 | 2e-5 | 20 | 0.8460 | 0.3763 | High Stability |
| Option 3 | 32 | 2e-5 | 20 | 0.8443 | 0.3609 | Lowest Loss |

**Rationale**: Option 1 is recommended for peak performance. However, if stability is a priority, Option 2 offers nearly identical performance with a much lower validation loss. [See detailed rationale](docss/mixed%20distil%20bert.md).

---

## 5. microsoft/xtremedistil-l12-h384-uncased

| Option | Batch Size | Learning Rate | Epochs | Val Macro F1 | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option 1** | **16** | **3e-5** | **30** | **0.8243** | **0.4476** | **Recommended** |
| Option 2 | 32 | 3e-5 | 30 | 0.8207 | 0.5118 | Alternative |
| Option 3 | 16 | 2e-5 | 20 | 0.8008 | 0.4832 | Stable |

**Rationale**: Option 1 provides the best Macro F1 score and the lowest loss for this distilled architecture, making it the most efficient choice. [See detailed rationale](docss/xtreme%20distil.md).

---

## 6. neuropark/sahajBERT

| Option | Batch Size | Learning Rate | Epochs | Val Macro F1 | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option 1** | **16** | **1e-5** | **10** | **0.8456** | **0.3433** | **Recommended** |
| Option 2 | 32 | 1e-5 | 10 | 0.8408 | 0.3577 | Stable |
| Option 3 | 32 | 3e-5 | 30 | 0.8506 | 0.8136 | High Risk |

**Rationale**: While Option 3 reaches a higher F1, its validation loss is extremely high (0.8136), indicating severe instability. Option 1 is recommended as it provides high performance with the best stability (lowest loss). [See detailed rationale](docss/shahaj%20bert.md).

---

## Summary of Recommendations

For most models, a **Learning Rate of 3e-5** and **Batch Size of 32 or 64** proved most effective. However, for distilled or smaller models, a lower **Learning Rate of 1e-5** often provided better stability without sacrificing significant performance.
