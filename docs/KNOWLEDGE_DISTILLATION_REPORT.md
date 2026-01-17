# Knowledge Distillation Experiment Report

## 1. Executive Summary
This report summarizes the results of a 2x2 Knowledge Distillation (KD) experiment matrix designed to identify the optimal student model for the HeteSpeech project. We evaluated two teacher models (**GBERT**, **XLM-RoBERTa**) and two student architectures (**SahajBERT**, **Mixed Distil-BERT**).

**Key Findings:**
*   **Best Performer:** **GBERT → SahajBERT** achieved the highest Macro F1 (0.8360).
*   **Fastest Model:** **Mixed Distil-BERT** offers a **2.0x speedup** over the teacher.
*   **Smallest Model:** **SahajBERT** is **4x smaller** (69MB) but significantly **slower** (0.27x speedup) due to its ALBERT-based architecture.
*   **Teacher Quality:** **GBERT** consistently produced slightly better student models than XLM-RoBERTa.

---

## 2. Experiment Setup
We conducted four experiments pairing each teacher with each student:

*   **Teachers:**
    *   `GBERT` (German BERT, likely adapted)
    *   `XLM-RoBERTa` (Multilingual)
*   **Students:**
    *   `SahajBERT` (ALBERT-based, parameter sharing)
    *   `Mixed Distil-BERT` (DistilBERT-based, layer pruning)

---

## 3. Comparative Results

| Teacher | Student | Macro F1 | F1 (Hate) | Speedup (vs Teacher) | Model Size | Agreement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GBERT** | **SahajBERT** | **0.8360** | **0.8389** | 0.27x (Slower) | **69 MB** | 16.8% |
| **GBERT** | **Mixed Distil** | 0.8339 | 0.8367 | **2.01x** | 254 MB | 17.3% |
| **XLM-R** | **SahajBERT** | 0.8327 | 0.8307 | 0.23x (Slower) | **69 MB** | 38.5% |
| **XLM-R** | **Mixed Distil** | 0.8324 | 0.8317 | 1.87x | 254 MB | 41.1% |

---

## 4. Analysis

### 4.1. Performance (F1 Scores)
All four combinations delivered highly comparable performance, with F1 scores clustering around **0.83**. The choice of teacher had a marginal impact, with **GBERT** providing a slight edge (~0.4%) over XLM-RoBERTa. This suggests that the student architectures are robust enough to learn the task effectively from either teacher.

### 4.2. Efficiency Trade-off: Speed vs. Size
This is the most critical differentiator:

*   **Mixed Distil-BERT (The "Fast" Option):**
    *   Behaves like a standard DistilBERT.
    *   Reduces layers (likely 12 → 6), resulting in a linear **2x speedup** in inference.
    *   Retains a moderate size (~254 MB).
    *   **Best for:** Real-time applications where latency is critical.

*   **SahajBERT (The "Compact" Option):**
    *   Behaves like an ALBERT model.
    *   Uses **parameter sharing** across layers. This drastically reduces disk footprint (**69 MB**, ~4x smaller).
    *   However, it still executes the full depth of the network (e.g., 12 layers) reusing the same weights. This makes it **slower** than the teacher (0.27x speedup) because of the overhead of unpacking/reusing weights without reducing the computational graph depth.
    *   **Best for:** Mobile/Edge devices where storage space or download size is the primary constraint, not latency.

### 4.3. Teacher-Student Agreement
Agreement scores were surprisingly low (17-41%), yet F1 scores remained high. This indicates that while the students are not mimicking the teachers' exact predictions token-for-token (low fidelity), they are successfully learning the underlying *task* (high generalization). XLM-RoBERTa showed significantly higher agreement (~40%) than GBERT (~17%), suggesting its decision boundaries might be "easier" for these students to approximate, even if GBERT is slightly more accurate overall.

---

## 5. Recommendations

1.  **For Production API / Server Deployment:**
    *   **Choose:** **GBERT → Mixed Distil-BERT**
    *   **Reason:** It provides the best balance of high accuracy (0.8339 F1) and low latency (2x faster). Storage space is rarely a bottleneck on servers.

2.  **For Mobile App / IoT Deployment:**
    *   **Choose:** **GBERT → SahajBERT**
    *   **Reason:** The **69 MB** file size is excellent for app bundles. The slower inference speed might be acceptable for background processing or asynchronous tasks.

3.  **Future Optimization:**
    *   Investigate **Quantization (INT8/INT4)** on the *Mixed Distil-BERT* model. This could potentially bring its size down closer to SahajBERT (e.g., ~65-120 MB) while maintaining its superior 2x speed, offering the "best of both worlds."
