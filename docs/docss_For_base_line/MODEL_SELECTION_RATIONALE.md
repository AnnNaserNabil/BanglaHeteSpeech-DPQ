# Model Selection Rationale

This document outlines the reasoning behind selecting specific baseline models for further experimentation and deployment in the Bangla Hate Speech detection project. The models are categorized into **Big** (>= 100M parameters) and **Small** (< 100M parameters) to balance performance with computational efficiency.

## 📊 Performance Comparison Summary

| Category | Model | Val Macro F1 | Val ROC-AUC | Val Loss | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Big** | **XLM-RoBERTa Base** | **0.8515** | **0.9296** | 0.1563 | 278M |
| **Big** | **mBERT Base** | 0.8351 | 0.9175 | 0.1109 | 178M |
| **Small** | **sahajBERT** | **0.8445** | 0.9057 | 0.3140 | 18M |
| **Small** | **Mixed-Distil-BERT** | 0.8330 | 0.9187 | **0.0956** | 66M |

---

## 🚀 Big Models Selection

### 1. FacebookAI/xlm-roberta-base
*   **Performance:** Achieved the highest **Val Macro F1 (0.8515)** and **Val ROC-AUC (0.9296)** among all baseline models.
*   **Rationale:** As the top-performing model, it serves as the primary "Teacher" model for knowledge distillation and the gold standard for accuracy in this project. Its cross-lingual pre-training proves highly effective for Bangla hate speech detection.

### 2. google-bert/bert-base-multilingual-cased (mBERT)
*   **Performance:** Strong **Val Macro F1 (0.8351)** and competitive ROC-AUC.
*   **Rationale:** mBERT outperformed the Bangla-specific BERT base model (`sagorsarker/bangla-bert-base`) in Macro F1. It provides a robust, well-supported alternative to XLM-R with slightly lower parameter counts.

---

## ⚡ Small Models Selection

### 1. neuropark/sahajBERT
*   **Performance:** Exceptional **Val Macro F1 (0.8445)** despite having only **18M parameters**.
*   **Rationale:** sahajBERT is the most efficient model in the lineup. It outperforms several much larger models (including mBERT and Bangla-BERT) while being nearly 10x-15x smaller. This makes it an ideal candidate for edge deployment or as a highly efficient student model.

### 2. md-nishat-008/Mixed-Distil-BERT
*   **Performance:** Solid **Val Macro F1 (0.8330)** and the **lowest Validation Loss (0.0956)**.
*   **Rationale:** The low validation loss indicates excellent generalization and stability. At 66M parameters, it offers a middle ground between the ultra-light sahajBERT and the larger base models, maintaining high precision and recall across both classes.

---

## 📉 Models Not Selected for Primary Focus

*   **sagorsarker/bangla-bert-base (164M):** While performing well (0.8302 F1), it was slightly outperformed by mBERT and significantly by sahajBERT (which is much smaller).
*   **microsoft/xtremedistil-l12-h384-uncased (33M):** Although small, its performance (0.8006 F1) was notably lower than sahajBERT and Mixed-Distil-BERT, making it less favorable for the accuracy-efficiency trade-off.
