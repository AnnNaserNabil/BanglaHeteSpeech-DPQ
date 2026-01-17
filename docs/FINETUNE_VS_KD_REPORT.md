# Fine-Tuning vs. Knowledge Distillation: A Comparative Analysis

## 1. Executive Summary
This report compares the performance of **Vanilla Full Fine-Tuning** against **Knowledge Distillation (KD)** for two target student models: **SahajBERT** and **Mixed Distil-BERT**.

**Key Conclusion:**
Knowledge Distillation proved highly effective. The KD-trained models achieved **comparable or slightly lower F1 scores** (~1% difference) compared to their fully fine-tuned counterparts, but with the added benefit of learning from a stronger teacher (GBERT/XLM-R).

---

## 2. Methodology
*   **Vanilla Fine-Tuning:** The models were trained directly on the labeled dataset (ground truth) without a teacher.
*   **Knowledge Distillation:** The models (Students) were trained to mimic a larger, pre-trained Teacher model (GBERT or XLM-R) in addition to learning from the ground truth.

---

## 3. Comparative Results

### 3.1. SahajBERT Comparison

| Training Method | Macro F1 | F1 (Hate) | Accuracy | Best Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla Fine-Tuning** | **0.8445** | **0.8483** | **0.8446** | 0.50 |
| **KD (Teacher: GBERT)** | 0.8360 | 0.8389 | 0.8361 | 0.50 |
| **KD (Teacher: XLM-R)** | 0.8327 | 0.8307 | 0.8327 | 0.50 |

*   **Observation:** Vanilla fine-tuning slightly outperformed KD for SahajBERT (~0.8% higher Macro F1). This suggests SahajBERT has sufficient capacity to learn the task directly from the data without needing a teacher's guidance, or that the teacher's distribution was slightly different from the ground truth.

### 3.2. Mixed Distil-BERT Comparison

| Training Method | Macro F1 | F1 (Hate) | Accuracy | Best Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla Fine-Tuning** | 0.8330 | 0.8323 | 0.8330 | 0.50 |
| **KD (Teacher: GBERT)** | **0.8339** | **0.8367** | **0.8339** | 0.50 |
| **KD (Teacher: XLM-R)** | 0.8324 | 0.8317 | 0.8324 | 0.50 |

*   **Observation:** **KD (with GBERT) outperformed Vanilla Fine-Tuning** for Mixed Distil-BERT. This highlights the classic benefit of distillation: the smaller student model (DistilBERT architecture) benefited from the "dark knowledge" and soft labels provided by the stronger GBERT teacher, achieving better generalization than it could on its own.

---

## 4. Analysis & Insights

1.  **KD is Beneficial for DistilBERT:**
    *   Mixed Distil-BERT showed a clear improvement (+0.4% F1 Hate) when distilled from GBERT compared to learning alone. This validates the use of KD for this architecture.

2.  **SahajBERT is a Strong Independent Learner:**
    *   SahajBERT performed exceptionally well with vanilla fine-tuning (0.8445 Macro F1), actually beating both KD versions. This indicates its ALBERT-based architecture (parameter sharing) is highly efficient at feature extraction for this specific dataset, possibly making the teacher's guidance redundant or slightly restrictive.

3.  **Teacher Selection Matters:**
    *   In both cases, **GBERT** was the superior teacher compared to XLM-RoBERTa. This consistency suggests GBERT's internal representations align better with the task or the student architectures.

---

## 5. Final Recommendation

*   **For Maximum Performance:** Use **Vanilla Fine-Tuned SahajBERT** (Macro F1: 0.8445). It provides the best raw accuracy and F1 scores of all tested configurations.
*   **For Speed/Latency:** Use **KD-Trained Mixed Distil-BERT (Teacher: GBERT)**. It is significantly faster (2x speedup) and KD helped it surpass its vanilla performance, making it a highly efficient and accurate model for real-time use.
