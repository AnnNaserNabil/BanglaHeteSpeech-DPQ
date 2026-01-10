# Quick Reference Guide

## Installation

```bash
cd Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data
pip install -r requirements.txt
```

## Common Commands

### 1. Baseline Training Only
```bash
python main.py \
    --dataset_path data/hate_speech.csv \
    --author_name researcher \
    --pipeline baseline
```

### 2. Baseline + Knowledge Distillation
```bash
python main.py \
    --dataset_path data/hate_speech.csv \
    --author_name researcher \
    --pipeline baseline_kd \
    --student_path distilbert-base-multilingual-cased
```

### 3. Baseline + Pruning
```bash
python main.py \
    --dataset_path data/hate_speech.csv \
    --author_name researcher \
    --pipeline baseline_prune \
    --prune_sparsity 0.5
```

### 4. Full Pipeline
```bash
python main.py \
    --dataset_path data/hate_speech.csv \
    --author_name researcher \
    --pipeline baseline_kd_prune_quant \
    --student_path distilbert-base-multilingual-cased \
    --prune_sparsity 0.5 \
    --quant_method dynamic \
    --save_huggingface \
    --output_dir ./models/compressed
```

## Pipeline Modes

- `baseline` - Original training only
- `baseline_kd` - + Knowledge Distillation
- `baseline_prune` - + Pruning
- `baseline_quant` - + Quantization
- `baseline_kd_prune` - + KD + Pruning
- `baseline_kd_quant` - + KD + Quantization
- `baseline_prune_quant` - + Pruning + Quantization
- `baseline_kd_prune_quant` - Full pipeline

## Key Parameters

- `--batch 32` - Batch size
- `--lr 2e-5` - Learning rate
- `--epochs 15` - Training epochs
- `--num_folds 5` - K-fold splits
- `--kd_alpha 0.7` - KD loss weight
- `--prune_sparsity 0.5` - Target sparsity
- `--quant_method dynamic` - Quantization method
- `--save_huggingface` - Save in HF format

## View Results

```bash
# MLflow UI
mlflow ui
# Open http://localhost:5000

# Check output directory
ls -lh ./outputs/
```

## Load Saved Model

```python
from transformers import AutoModel, AutoTokenizer
import torch, torch.nn as nn, json

encoder = AutoModel.from_pretrained("path/to/model")
with open("path/to/model/classifier_config.json") as f:
    cfg = json.load(f)
classifier = nn.Sequential(
    nn.Linear(cfg['hidden_size'], 256),
    nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, cfg['num_labels'])
)
classifier.load_state_dict(torch.load("path/to/model/classifier.pt"))
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = encoder(**inputs)
        logits = classifier(outputs.last_hidden_state[:, 0, :])
        return torch.sigmoid(logits).item()
```
