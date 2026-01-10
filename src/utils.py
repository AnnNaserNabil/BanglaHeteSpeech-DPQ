"""
Enhanced Utility Functions for Bangla BERT Hate Speech Detection
=================================================================

Includes:
- Seed setting for reproducibility
- Model metrics calculation
- HuggingFace model saving and deployment
- Model card generation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import json
from typing import Optional, Dict


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_model_metrics(model):
    """
    Calculate model size and parameter counts.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary containing total parameters, trainable parameters, and model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(size_mb, 2)
    }


def print_experiment_header(config):
    """
    Print formatted experiment header with configuration details.

    Args:
        config: Configuration object with experiment parameters
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT: {config.author_name} (Hate Speech Detection)")
    print("="*60)
    print(f"Model: {config.model_path}")
    print(f"Batch Size: {config.batch}")
    print(f"Learning Rate: {config.lr}")
    print(f"Max Epochs: {config.epochs}")
    print(f"Max Length: {config.max_length}")
    print(f"Freeze Base: {config.freeze_base}")
    print(f"Stratification: {config.stratification_type}")
    print(f"K-Folds: {config.num_folds}")
    print(f"Dropout: {config.dropout}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Warmup Ratio: {config.warmup_ratio}")
    print(f"Gradient Clip Norm: {config.gradient_clip_norm}")
    print(f"MLflow Experiment: {config.mlflow_experiment_name}")
    print("="*60 + "\n")


def print_fold_summary(fold_num, best_metrics, best_epoch):
    """
    Print summary of fold performance.

    Args:
        fold_num (int): Fold number (0-indexed)
        best_metrics (dict): Best metrics achieved in this fold
        best_epoch (int): Epoch number where best performance was achieved
    """
    print("\n" + "-"*60)
    print(f"FOLD {fold_num + 1} SUMMARY")
    print("-"*60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best F1: {best_metrics['f1']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best Precision: {best_metrics['precision']:.4f}")
    print(f"Best Recall: {best_metrics['recall']:.4f}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print("-"*60 + "\n")


def print_experiment_summary(best_fold_idx, best_fold_metrics, model_metrics):
    """
    Print final experiment summary.

    Args:
        best_fold_idx (int): Index of best performing fold
        best_fold_metrics (dict): Metrics from the best fold
        model_metrics (dict): Model size and parameter information
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Best performing fold: Fold {best_fold_idx + 1}")
    print(f"  Accuracy: {best_fold_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_fold_metrics['precision']:.4f}")
    print(f"  Recall: {best_fold_metrics['recall']:.4f}")
    print(f"  F1: {best_fold_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {best_fold_metrics['roc_auc']:.4f}")
    print("\nModel Information:")
    print(f"  Model size: {model_metrics['model_size_mb']} MB")
    print(f"  Total parameters: {model_metrics['total_parameters']:,}")
    print(f"  Trainable parameters: {model_metrics['trainable_parameters']:,}")
    print("="*60)


def save_model_for_huggingface(model, tokenizer, save_path: str, config: Optional[Dict] = None):
    """
    Save model in HuggingFace format for deployment.
    
    Creates a folder with:
    - config.json (model configuration)
    - pytorch_model.bin (weights)
    - tokenizer files (if tokenizer provided)
    - classifier_config.json (classifier head info)
    - how_to_load.py (loading script)
    - README.md (model card)
    
    Args:
        model: Model to save (TeacherModel, StudentModel, or TransformerBinaryClassifier)
        tokenizer: Tokenizer to save
        save_path: Directory to save to
        config: Optional configuration dict for model card
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n💾 Saving model in HuggingFace format...")
    
    # Save encoder
    if hasattr(model, 'encoder'):
        try:
            model.encoder.save_pretrained(save_path)
            print(f"   ✓ Encoder saved")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not save encoder using save_pretrained: {e}")
            print("   Falling back to standard torch.save...")
            torch.save(model.encoder.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
            if hasattr(model.encoder, 'config'):
                model.encoder.config.save_pretrained(save_path)
    
    # Save classifier separately
    if hasattr(model, 'classifier'):
        classifier_path = os.path.join(save_path, 'classifier.pt')
        torch.save(model.classifier.state_dict(), classifier_path)
        
        # Save classifier config
        classifier_config = {
            'type': 'sequential',
            'layers': str(model.classifier),
            'num_labels': model.num_labels if hasattr(model, 'num_labels') else 1,
            'hidden_size': model.hidden_size if hasattr(model, 'hidden_size') else 768
        }
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(classifier_config, f, indent=2)
        print(f"   ✓ Classifier saved")
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        print(f"   ✓ Tokenizer saved")
    
    # Create loading script
    loading_script = f'''# How to load this model:

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import json

# Load encoder
encoder = AutoModel.from_pretrained("{save_path}")

# Load classifier config
with open("{save_path}/classifier_config.json", 'r') as f:
    c_config = json.load(f)

num_labels = c_config.get('num_labels', 1)
hidden_size = c_config.get('hidden_size', 768)

# Reconstruct classifier
classifier = nn.Sequential(
    nn.Linear(hidden_size, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, num_labels)
)
classifier.load_state_dict(torch.load("{save_path}/classifier.pt"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{save_path}")

# Inference function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding)
        probs = torch.sigmoid(logits)
    return probs.item()

# Example
text = "আপনার বাংলা টেক্সট এখানে"
prob = predict(text)
print(f"Hate Speech Probability: {{prob:.4f}}")
print(f"Prediction: {{'Hate Speech' if prob > 0.5 else 'Non-Hate Speech'}}")
'''
    
    with open(os.path.join(save_path, 'how_to_load.py'), 'w') as f:
        f.write(loading_script)
    print(f"   ✓ Loading script created")
    
    # Create model card
    if config:
        create_model_card(save_path, config)
        print(f"   ✓ Model card created")
    
    print(f"\n✅ Model saved successfully to: {save_path}")
    print(f"\n📄 Files created:")
    for f in sorted(os.listdir(save_path)):
        print(f"   - {f}")


def create_model_card(save_path: str, config: Dict):
    """
    Create a README.md model card for HuggingFace Hub.
    
    Args:
        save_path: Directory where model is saved
        config: Configuration dict with training info
    """
    model_card = f'''---
language: bn
tags:
- hate-speech-detection
- bangla
- bert
- binary-classification
license: mit
---

# Bangla Hate Speech Detection Model

This model is fine-tuned for binary hate speech detection in Bangla text.

## Model Description

- **Base Model**: {config.get('model_path', 'N/A')}
- **Task**: Binary Classification (Hate Speech vs Non-Hate Speech)
- **Language**: Bangla (Bengali)
- **Training Method**: {config.get('pipeline_description', 'Baseline fine-tuning')}

## Training Details

### Training Hyperparameters

- **Batch Size**: {config.get('batch', 32)}
- **Learning Rate**: {config.get('lr', 2e-5)}
- **Epochs**: {config.get('epochs', 15)}
- **Max Sequence Length**: {config.get('max_length', 128)}
- **Dropout**: {config.get('dropout', 0.1)}
- **Weight Decay**: {config.get('weight_decay', 0.01)}
- **Warmup Ratio**: {config.get('warmup_ratio', 0.1)}

### Training Data

- **K-Fold Cross-Validation**: {config.get('num_folds', 5)} folds
- **Stratification**: {config.get('stratification_type', 'binary')}

## Performance

*Add your metrics here after training*

## Usage

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import json

# Load model components
encoder = AutoModel.from_pretrained("path/to/model")

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
print(f"Hate Speech Probability: {{prob:.4f}}")
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{bangla-hate-speech-model,
  author = {{{config.get('author_name', 'Anonymous')}}},
  title = {{Bangla Hate Speech Detection Model}},
  year = {{2026}},
  publisher = {{HuggingFace}},
}}
```

## License

MIT License
'''
    
    with open(os.path.join(save_path, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(model_card)
