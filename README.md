# Enhanced Bangla BERT Hate Speech Detection with Compression Pipeline

A modular framework for training and compressing transformer models for Bangla hate speech detection. Combines rigorous baseline training methodology with state-of-the-art compression techniques (Knowledge Distillation, Pruning, Quantization).

## Features

✨ **Rigorous Training Methodology**
- K-fold cross-validation with stratification
- Threshold exploration for macro F1 optimization
- Per-class metrics (hate/non-hate)
- Comprehensive MLflow tracking
- Early stopping with patience
- Mixed precision training (AMP)

🔄 **Modular Compression Pipeline**
- Knowledge Distillation (logit, hidden, attention, multi-level)
- Pruning (magnitude, WANDA, gradual, structured)
- Quantization (dynamic INT8, static INT8, FP16, INT4)
- HuggingFace-compatible model saving

📊 **Enhanced Metrics**
- Threshold exploration at every stage
- Detailed per-class performance
- ROC-AUC calculation
- Confusion matrix metrics
- K-fold aggregation with mean/std

## Installation

```bash
# Clone the repository
cd Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Baseline Training Only

```bash
python src/main.py \
    --dataset_path data/hate_speech.csv \
    --model_path sagorsarker/bangla-bert-base \
    --author_name researcher \
    --pipeline baseline \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15 \
    --num_folds 5
```

### Baseline + Knowledge Distillation

```bash
python src/main.py \
    --dataset_path data/hate_speech.csv \
    --model_path sagorsarker/bangla-bert-base \
    --student_path distilbert-base-multilingual-cased \
    --author_name researcher \
    --pipeline baseline_kd \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --save_huggingface \
    --output_dir ./models/kd_model
```

### Full Compression Pipeline

```bash
python src/main.py \
    --dataset_path data/hate_speech.csv \
    --model_path sagorsarker/bangla-bert-base \
    --student_path distilbert-base-multilingual-cased \
    --author_name researcher \
    --pipeline baseline_kd_prune_quant \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --prune_method magnitude \
    --prune_sparsity 0.5 \
    --quant_method dynamic \
    --save_huggingface \
    --output_dir ./models/compressed_model
```

## Pipeline Modes

| Mode | Description | Stages |
|------|-------------|--------|
| `baseline` | Original rigorous training | Baseline only |
| `baseline_kd` | Baseline + KD | Baseline → KD |
| `baseline_prune` | Baseline + Pruning | Baseline → Prune |
| `baseline_quant` | Baseline + Quantization | Baseline → Quant |
| `baseline_kd_prune` | Baseline + KD + Pruning | Baseline → KD → Prune |
| `baseline_kd_quant` | Baseline + KD + Quantization | Baseline → KD → Quant |
| `baseline_prune_quant` | Baseline + Pruning + Quantization | Baseline → Prune → Quant |
| `baseline_kd_prune_quant` | Full pipeline | Baseline → KD → Prune → Quant |

## Configuration Options

### Training Parameters

- `--batch`: Batch size (default: 32)
- `--lr`: Learning rate (default: 2e-5)
- `--epochs`: Training epochs (default: 15)
- `--num_folds`: K-fold splits (default: 5)
- `--early_stopping_patience`: Patience for early stopping (default: 5)
- `--dropout`: Dropout rate (default: 0.1)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_ratio`: Warmup ratio (default: 0.1)
- `--gradient_clip_norm`: Gradient clipping (default: 1.0)

### Knowledge Distillation

- `--student_path`: Student model (default: distilbert-base-multilingual-cased)
- `--student_hidden_size`: Student classifier hidden size (default: 256)
- `--kd_alpha`: KD loss weight (default: 0.7)
- `--kd_temperature`: Temperature (default: 4.0)
- `--kd_method`: KD method [logit, hidden, attention, multi_level] (default: logit)

### Pruning

- `--prune_method`: Pruning method [magnitude, wanda, gradual, structured] (default: magnitude)
- `--prune_sparsity`: Target sparsity (default: 0.5)
- `--fine_tune_after_prune`: Fine-tune after pruning (default: True)
- `--fine_tune_epochs`: Fine-tuning epochs (default: 3)

### Quantization

- `--quant_method`: Quantization method [dynamic, static, fp16, int4] (default: dynamic)
- `--quant_dtype`: Data type [int8, int4, fp16] (default: int8)

### Output

- `--output_dir`: Output directory (default: ./outputs)
- `--save_huggingface`: Save in HuggingFace format
- `--cache_dir`: Cache directory (default: ./cache)

## Project Structure

```
Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data/
├── src/                     # Source code
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration
│   ├── data.py              # Data loading
│   ├── model.py             # Model definitions
│   ├── train.py             # Baseline training
│   ├── metrics.py           # Metrics
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── distillation.py      # Knowledge distillation
│   ├── pruning.py           # Pruning
│   ├── quantization.py      # Quantization
│   ├── evaluation.py        # Evaluation
│   └── utils.py             # Utilities
├── docs/                    # Documentation
│   ├── DETAILED_FRAMEWORK_GUIDE.md
│   ├── METRICS_GUIDE.md
│   └── ...
├── data/                    # Data directory
├── outputs/                 # Output directory
└── README.md                # This file
```

## Metrics Calculation

The framework uses enhanced metrics calculation with threshold exploration:

1. **Threshold Exploration**: Tests multiple thresholds (0.3-0.6) to find optimal macro F1
2. **Per-Class Metrics**: Separate precision/recall/F1 for hate and non-hate classes
3. **Macro F1**: Average F1 across both classes (primary metric)
4. **ROC-AUC**: Area under ROC curve
5. **Best Threshold**: Threshold that maximizes macro F1

This methodology is applied at **every pipeline stage** for consistent evaluation.

## Output Files

### Baseline Training

- `fold_summary_{model}_{params}_{timestamp}.csv`: Metrics for each fold
- `best_metrics_{model}_{params}_{timestamp}.csv`: Best fold metrics
- MLflow artifacts in `./mlruns/`

### Pipeline Modes

- `results_{stage}.csv`: Metrics for each stage
- `results_all.csv`: Cumulative results
- HuggingFace model (if `--save_huggingface`):
  - `config.json`
  - `pytorch_model.bin`
  - `classifier.pt`
  - `tokenizer files`
  - `README.md` (model card)
  - `how_to_load.py` (loading script)

## Loading Saved Models

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import json

# Load encoder
encoder = AutoModel.from_pretrained("path/to/model")

# Load classifier
with open("path/to/model/classifier_config.json", 'r') as f:
    c_config = json.load(f)

classifier = nn.Sequential(
    nn.Linear(c_config['hidden_size'], 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, c_config['num_labels'])
)
classifier.load_state_dict(torch.load("path/to/model/classifier.pt"))

tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Predict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding)
        prob = torch.sigmoid(logits).item()
    return prob

text = "আপনার বাংলা টেক্সট এখানে"
prob = predict(text)
print(f"Hate Speech Probability: {prob:.4f}")
```

## MLflow Tracking

View experiment results:

```bash
mlflow ui
# Open http://localhost:5000
```

## Advanced Usage

### Custom Pipeline

You can create custom pipelines by modifying `config.py`:

```python
PIPELINE_CONFIGS['my_custom'] = {
    'enable_kd': True,
    'enable_pruning': True,
    'enable_quantization': False,
    'description': 'My custom pipeline'
}
```

### Programmatic Usage

```python
from config import parse_arguments
from data import load_and_preprocess_data
from pipeline import run_compression_pipeline
from transformers import AutoTokenizer
import torch

# Create config
config = parse_arguments([
    '--dataset_path', 'data.csv',
    '--author_name', 'researcher',
    '--pipeline', 'baseline_kd_prune',
    '--batch', '32'
])

# Load data
comments, labels = load_and_preprocess_data(config.dataset_path)
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run pipeline
results = run_compression_pipeline(config, comments, labels, tokenizer, device)
```

## Citation

If you use this framework, please cite:

```bibtex
@misc{bangla-hate-speech-compression,
  title={Enhanced Bangla BERT Hate Speech Detection with Compression Pipeline},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/your-repo}
}
```

## License

MIT License

## Acknowledgments

- Original baseline training methodology from Finetune-Bangla-BERT project
- Compression techniques adapted from root folder pipeline
- Enhanced metrics and modular architecture designed for research flexibility
