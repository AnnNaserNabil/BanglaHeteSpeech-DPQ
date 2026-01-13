I chose the configuration of **Batch Size: 64, Learning Rate: 3e-5, and 30 Epochs** for the `mBERT` (Multilingual BERT) model because it provided the peak performance while maintaining high training stability.

Here is the rationale based on your data:

### 1. Absolute Peak Performance
*   **Selected (64, 3e-5, 30)**: Val Macro F1 = **0.8439**
*   **Runner up (64, 2e-5, 20)**: Val Macro F1 = **0.8436**
*   **Others**: Most other configurations (especially with batch size 32 or 16) hovered around 0.839 or lower.

This configuration reached the highest F1 score recorded for this model, ensuring the best possible detection of hate speech across both classes.

### 2. Stability of Large Batch Sizes
Multilingual BERT (mBERT) has a very large vocabulary and handles many languages simultaneously. Using a **Batch Size of 64** provides a much smoother and more stable gradient during training compared to smaller batches. 
*   Notice that the **Batch Size 32** run with 2e-5 LR had a massive **Val Loss of 0.9790** (indicating instability).
*   In contrast, the **Batch Size 64** runs consistently kept the loss around **0.43**, showing that the larger batch size is much safer for this specific architecture.

### 3. Optimal Learning Rate (3e-5)
For BERT-base models, 3e-5 is often the "Goldilocks" learning rate—it is high enough to converge quickly but low enough to avoid overshooting. 
*   In this run, the model reached its "Best Epoch" at **Epoch 5**. 
*   This shows that the model learns efficiently and reaches its peak early, allowing for effective early stopping to save time and compute.

### 4. Better ROC-AUC
This configuration achieved a **Val ROC-AUC of 0.9232**, which is one of the highest for mBERT in your dataset. This confirms that the model has a very strong ability to rank hate speech higher than non-hate speech, giving you more flexibility when choosing a classification threshold.

### Summary
I chose this as **Option 1** because it represents the **maximum potential** of the mBERT model. It combines the highest accuracy with the most stable loss profile, making it the most reliable choice for a multilingual baseline.