# Running the Hate Speech Detection Pipeline on Kaggle

This guide provides step-by-step instructions for running the Bangla BERT Hate Speech Detection pipeline on Kaggle.

## Prerequisites

- A Kaggle account.
- The dataset file (`data.csv`) containing the hate speech data.
- The source code from this repository.

## Step 1: Notebook Setup

1.  **Create a New Notebook**:
    - Go to Kaggle and click "Create" -> "New Notebook".
    - Select "Python" as the language.
    - Select "GPU P100" (or T4 x2) as the accelerator for faster training.

2.  **Upload Data**:
    - In the "Data" pane (right side), click "Add Data".
    - Upload your `data.csv` file.
    - Note the path to the uploaded file (usually `/kaggle/input/<dataset-name>/data.csv`).

3.  **Upload Code**:
    - You have two options:
        - **Option A (Upload)**: Upload the entire `src` folder and `data` folder (if needed) as a dataset and add it to your notebook.
        - **Option B (Clone)**: Clone the repository directly in the notebook cell (if public) or copy-paste the code.
    - *Recommended*: Upload the `src` folder as a dataset so you can import modules easily.

## Step 2: Environment Setup

Kaggle kernels come with most deep learning libraries pre-installed (PyTorch, Transformers, Pandas, NumPy). You might need to install `mlflow` if it's not available, or update `transformers`.

Run this in the first cell:

```python
!pip install -q mlflow
# !pip install -q transformers --upgrade  # Uncomment if you need the latest version
```

## Step 3: Directory Structure

Ensure your directory structure in the Kaggle working directory looks like this (you may need to move files if you uploaded them as a dataset):

```
/kaggle/working/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── pipeline.py
│   └── ... (other modules)
└── data.csv (or link to input)
```

If you uploaded code as a dataset, you might need to add it to the system path:

```python
import sys
sys.path.append('/kaggle/input/<your-code-dataset-name>/src')
```

## Step 4: Running the Pipeline

You can run the pipeline using the `!python` command. Below are examples with **all available arguments** explicitly listed. You can remove or modify them as needed.

### 1. Baseline Training (Full Arguments)

This runs the standard rigorous K-Fold training.

```bash
!python src/main.py \
    --pipeline baseline \
    --dataset_path /kaggle/input/<your-dataset>/data.csv \
    --author_name "YourName" \
    --mlflow_experiment_name "Bangla-HateSpeech-Baseline" \
    --model_path "sagorsarker/bangla-bert-base" \
    --num_folds 5 \
    --stratification_type binary \
    --epochs 15 \
    --batch 32 \
    --lr 2e-5 \
    --max_length 128 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --gradient_clip_norm 1.0 \
    --early_stopping_patience 5 \
    --seed 42 \
    --output_dir "./outputs/baseline" \
    --save_huggingface
```

### 2. Knowledge Distillation (Full Arguments)

Trains a smaller student model to mimic the teacher.

```bash
!python src/main.py \
    --pipeline baseline_kd \
    --dataset_path /kaggle/input/<your-dataset>/data.csv \
    --author_name "YourName" \
    --mlflow_experiment_name "Bangla-HateSpeech-KD" \
    --model_path "sagorsarker/bangla-bert-base" \
    --student_path "distilbert-base-multilingual-cased" \
    --student_hidden_size 256 \
    --kd_method logit \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --hidden_loss_weight 0.3 \
    --attention_loss_weight 0.2 \
    --num_folds 5 \
    --epochs 15 \
    --batch 32 \
    --lr 2e-5 \
    --max_length 128 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --output_dir "./outputs/kd" \
    --save_huggingface
```

### 3. Pruning (Full Arguments)

Prunes the model to reduce size/complexity.

```bash
!python src/main.py \
    --pipeline baseline_prune \
    --dataset_path /kaggle/input/<your-dataset>/data.csv \
    --author_name "YourName" \
    --mlflow_experiment_name "Bangla-HateSpeech-Pruning" \
    --model_path "sagorsarker/bangla-bert-base" \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --prune_layers all \
    --prune_schedule cubic \
    --prune_start_epoch 0 \
    --prune_end_epoch 10 \
    --prune_frequency 100 \
    --calib_samples 512 \
    --fine_tune_after_prune \
    --fine_tune_epochs 3 \
    --num_folds 5 \
    --epochs 15 \
    --batch 32 \
    --lr 2e-5 \
    --output_dir "./outputs/pruning" \
    --save_huggingface
```

### 4. Quantization (Full Arguments)

Quantizes the model to lower precision (e.g., INT8).

```bash
!python src/main.py \
    --pipeline baseline_quant \
    --dataset_path /kaggle/input/<your-dataset>/data.csv \
    --author_name "YourName" \
    --mlflow_experiment_name "Bangla-HateSpeech-Quantization" \
    --model_path "sagorsarker/bangla-bert-base" \
    --quant_method static \
    --quant_dtype int8 \
    --quant_calibration_batches 100 \
    --num_folds 5 \
    --epochs 15 \
    --batch 32 \
    --lr 2e-5 \
    --output_dir "./outputs/quantization" \
    --save_huggingface
```

### 5. Full Pipeline: KD + Pruning + Quantization (Full Arguments)

Runs the complete compression pipeline.

```bash
!python src/main.py \
    --pipeline baseline_kd_prune_quant \
    --dataset_path /kaggle/input/<your-dataset>/data.csv \
    --author_name "YourName" \
    --mlflow_experiment_name "Bangla-HateSpeech-FullPipeline" \
    --model_path "sagorsarker/bangla-bert-base" \
    --student_path "distilbert-base-multilingual-cased" \
    --student_hidden_size 256 \
    --kd_method logit \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --fine_tune_after_prune \
    --fine_tune_epochs 3 \
    --quant_method static \
    --quant_dtype int8 \
    --quant_calibration_batches 100 \
    --num_folds 5 \
    --epochs 15 \
    --batch 32 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --output_dir "./outputs/full_pipeline" \
    --save_huggingface
```

## Step 5: Handling Outputs

Results are saved in the directory specified by `--output_dir` (default: `./outputs`).

To download results from Kaggle:
1.  After the script finishes, check the "Output" section of your notebook.
2.  You should see the `outputs` folder.
3.  You can zip it and download it:

```python
!zip -r results.zip ./outputs
from IPython.display import FileLink
FileLink(r'results.zip')
```

## Troubleshooting

-   **CUDA Out of Memory**: Reduce `--batch` (e.g., to 16 or 8).
-   **Path Errors**: Double-check the path to `data.csv` and `src/main.py`. Use `!ls -R` to verify file locations.
