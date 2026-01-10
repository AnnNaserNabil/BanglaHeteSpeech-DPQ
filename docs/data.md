# 📘 Script 2: `data.py`

## Overview

This script handles **everything related to data**: loading, preprocessing, tokenization, caching, and creating data loaders. It's the bridge between your raw CSV file and the tensors that PyTorch needs.

**Why this script is critical:**
- Tokenization is SLOW (90+ seconds per run)
- Caching saves you 10-72 minutes per experiment
- Proper data handling prevents subtle bugs

---

## Section 1: Imports and Environment Setup

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # MUST be before transformers import!

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
import pickle
import hashlib
from typing import List, Tuple, Optional, Dict
```

### What This Does:

1. **`TOKENIZERS_PARALLELISM = "false"`**: Prevents a warning about forking processes
2. **Import order matters**: Environment variables must be set BEFORE importing transformers

### Why This Specific Order:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE TOKENIZER PARALLELISM PROBLEM                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ What happens without the fix:                                              │
│                                                                             │
│ 1. HuggingFace tokenizers use multiple threads by default                  │
│ 2. PyTorch DataLoader ALSO uses multiple workers                           │
│ 3. When DataLoader forks processes, each gets a copy of tokenizer          │
│ 4. Multiple tokenizers × multiple threads = DEADLOCK RISK                  │
│                                                                             │
│ Warning message you'd see:                                                  │
│ "huggingface/tokenizers: The current process just got forked..."           │
│ "Disabling parallelism to avoid deadlocks..."                              │
│                                                                             │
│ The fix:                                                                    │
│ os.environ["TOKENIZERS_PARALLELISM"] = "false"                             │
│ Sets tokenizer to single-threaded mode BEFORE it loads                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | When | Effect |
|--------------|------|--------|
| Set to "true" | Single-threaded DataLoader (num_workers=0) | Faster tokenization |
| Keep "false" | Multi-worker DataLoader (num_workers>0) | Prevents deadlocks |

**Recommendation:** Keep it "false" unless you have specific performance issues.

---

## Section 2: Label Configuration

In the current setup, **labels are no longer hardcoded**. Instead, they are passed dynamically via command-line arguments.

### How it works:
- You specify labels using `--label_columns` (e.g., `--label_columns HateSpeech`).
- The framework supports both single-label (binary) and multi-label classification.

### Your Dataset Structure (`HateSpeech.csv`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPECTED CSV FORMAT (Single-Label)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Comments,HateSpeech                                                         │
│ "যদি দেখো আকাশ ভরা তারা...",1                                               │
│ "ওর পুরা গুষ্টি শ্বশুরের গুষ্টি...",0                                       │
│                                                                             │
│ Important:                                                                  │
│ - Labels are 0 or 1 (binary)                                               │
│ - Column names are flexible (looks for 'Comments', 'comments', 'text', etc.)│
│ - The framework renames the text column to 'comment' internally             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 3: Data Loading and Preprocessing

```python
def load_and_preprocess_data(dataset_path: str, label_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset_path)
    
    # Column name flexibility
    comment_col = None
    for col in ['Comments', 'comments', 'Comment', 'comment', 'text', 'Text', 'content']:
        if col in df.columns:
            comment_col = col
            break
            
    if comment_col != 'comment':
        df = df.rename(columns={comment_col: 'comment'})
        
    df = df.dropna(subset=['comment'] + label_columns)
    
    comments = df['comment'].values
    labels = df[label_columns].values
    
    return comments, labels
```

### Key Features:
1. **Column Flexibility**: Automatically detects common text column names.
2. **Robustness**: Drops rows with missing text or labels.
3. **Internal Consistency**: Standardizes the text column to `comment` for the rest of the pipeline.

### What This Does:

1. Reads your CSV file
2. Extracts text column
3. Extracts label columns as numpy array
4. Optionally cleans text

### Step-by-Step Walkthrough:

```python
def load_and_preprocess_data(data_path, text_column='comment_text', ...):
    
    # Step 1: Load CSV
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} samples")
    
    # Step 2: Validate columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found!")
    
    for col in label_columns:
        if col not in df.columns:
            raise ValueError(f"Label column '{col}' not found!")
    
    # Step 3: Extract text
    comments = df[text_column].astype(str).tolist()
    
    # Step 4: Extract labels as numpy array
    labels = df[label_columns].values.astype(np.float32)
    # Shape: (num_samples, num_labels) = (44000, 5)
    
    # Step 5: Optional text cleaning
    if clean_text:
        comments = [clean_bangla_text(text) for text in comments]
    
    # Step 6: Print statistics
    print(f"\n📊 Label Distribution:")
    for i, col in enumerate(label_columns):
        positive = labels[:, i].sum()
        print(f"   {col}: {positive:.0f} ({positive/len(labels)*100:.1f}%)")
    
    return comments, labels
```

### Visual Data Flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA LOADING FLOW                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CSV File                                                                  │
│   ┌────────────────────────────────────────────────────────────┐           │
│   │ comment_text          │ bully │ sexual │ religious │ ... │           │
│   │ "তুমি বোকা"            │   1   │   0    │     0     │ ... │           │
│   │ "এটা স্প্যাম"          │   0   │   0    │     0     │ ... │           │
│   │ ...                   │  ...  │  ...   │    ...    │ ... │           │
│   └────────────────────────────────────────────────────────────┘           │
│                          ↓                                                  │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                    load_and_preprocess_data()                        │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                          ↓                                                  │
│   ┌─────────────────────┐    ┌──────────────────────────────────┐         │
│   │ comments (List)     │    │ labels (numpy array)             │         │
│   │ ["তুমি বোকা",        │    │ [[1, 0, 0, 0, 0],                │         │
│   │  "এটা স্প্যাম",      │    │  [0, 0, 0, 0, 1],                │         │
│   │  ...]               │    │  ...]                            │         │
│   │ Length: 44000       │    │ Shape: (44000, 5)                │         │
│   └─────────────────────┘    └──────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Code Change | When to Use |
|--------------|-------------|-------------|
| Different text column | `text_column='content'` | Your CSV uses different name |
| Skip cleaning | `clean_text=False` | Text is already preprocessed |
| Add custom cleaning | Modify `clean_bangla_text()` | Need specific preprocessing |

**Example - Custom Text Cleaning:**
```python
def clean_bangla_text(text: str) -> str:
    """Clean Bangla text for processing."""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove English characters (keep Bangla only)
    # text = re.sub(r'[a-zA-Z]', '', text)  # Uncomment if needed
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # ADD YOUR CUSTOM CLEANING HERE
    # Example: Remove specific words
    # text = text.replace('spam_word', '')
    
    return text.strip()
```

---

## Section 4: Tokenization Caching (THE MOST IMPORTANT SECTION!)

```python
def get_cache_filename(tokenizer_name: str, max_length: int, cache_dir: str) -> str:
    """Generate unique cache filename based on tokenizer and settings."""
    # Create hash of settings for unique filename
    settings_str = f"{tokenizer_name}_{max_length}"
    settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:8]
    
    safe_name = tokenizer_name.replace('/', '_')
    filename = f"{safe_name}_maxlen{max_length}_{settings_hash}_tokenized.pkl"
    
    return os.path.join(cache_dir, filename)
```

### What This Does:

Creates a unique filename for cached tokenized data based on:
- Tokenizer name (e.g., "csebuetnlp/banglabert")
- Max sequence length (e.g., 128)

### Why Caching is Critical:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY TOKENIZATION CACHING SAVES HOURS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ WITHOUT CACHING:                                                            │
│ ─────────────────                                                           │
│                                                                             │
│ Experiment 1: python main.py --pipeline baseline                           │
│               → Tokenize 44,000 texts... 90 seconds                        │
│                                                                             │
│ Experiment 2: python main.py --pipeline kd_only                            │
│               → Tokenize 44,000 texts... 90 seconds (AGAIN!)               │
│                                                                             │
│ Experiment 3: python main.py --pipeline kd_prune                           │
│               → Tokenize 44,000 texts... 90 seconds (AGAIN!)               │
│                                                                             │
│ 8 experiments × 90 seconds = 12 MINUTES wasted on tokenization             │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ WITH CACHING:                                                               │
│ ──────────────                                                              │
│                                                                             │
│ Experiment 1: python main.py --pipeline baseline                           │
│               → Cache miss! Tokenize... 90 seconds                         │
│               → Save to cache/banglabert_maxlen128_tokenized.pkl           │
│                                                                             │
│ Experiment 2: python main.py --pipeline kd_only                            │
│               → Cache hit! Load from cache... 2 seconds                    │
│                                                                             │
│ Experiment 3-8: All cache hits... 2 seconds each                           │
│                                                                             │
│ Total: 90 + (7 × 2) = 104 seconds vs 720 seconds = 7× FASTER!              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Section 4: Tokenization Caching (The Performance Booster)

```python
def get_or_create_tokenized_dataset(
    comments: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_length: int,
    cache_dir: str = './cache',
    student_tokenizer = None
) -> Dict[str, torch.Tensor]:
```

### What This Does:
1. **Generates a Unique Key**: Based on the model name and max length.
2. **Dual Tokenization Support**: If a `student_tokenizer` is provided (e.g., when the student uses a different vocabulary than the teacher), it tokenizes for BOTH models and caches them together.
3. **Saves as Torch Tensors**: Uses `torch.save` for efficient storage and loading.

### Why this is a "Game Changer":
Imagine you are running a 5-fold cross-validation with 10 different hyperparameter combinations.
```

### Cache File Structure:

```
cache/
├── csebuetnlp_banglabert_maxlen128_a1b2c3d4_tokenized.pkl  (420 MB)
│   └── Contains: {
│         'input_ids': tensor of shape (44000, 128),
│         'attention_mask': tensor of shape (44000, 128),
│         'labels': tensor of shape (44000, 5)
│       }
│
├── distilbert-base-multilingual-cased_maxlen128_e5f6g7h8_tokenized.pkl
│   └── Different tokenizer = different cache file!
│
└── csebuetnlp_banglabert_maxlen256_i9j0k1l2_tokenized.pkl
    └── Different max_length = different cache file!
```

### What You Can Modify:

| Modification | Effect | When to Use |
|--------------|--------|-------------|
| Clear cache | Delete files in `cache/` | Dataset changed, force re-tokenization |
| Change cache_dir | `--cache_dir /fast_ssd/cache` | Faster disk for cache files |
| Disable caching | Remove cache logic | Debugging tokenization issues |

**When Cache is Invalidated Automatically:**
- Different tokenizer model
- Different max_length
- Different dataset (detected by filename hash)

**When You MUST Manually Clear Cache:**
- Same filename but different content (rare)
- Tokenizer was updated on HuggingFace

---

## Section 5: PyTorch Dataset Class

```python
class IndexedDataset(Dataset):
    """
    PyTorch Dataset that uses indices to access cached data.
    
    Why indices instead of copying data?
    - Memory efficient: One copy of tokenized data shared across train/val
    - Fast: No data copying when creating train/val splits
    """
    
    def __init__(self, tokenized_data: Dict[str, torch.Tensor], indices: np.ndarray):
        self.tokenized_data = tokenized_data
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our subset
        real_idx = self.indices[idx]
        
        return {
            'input_ids': self.tokenized_data['input_ids'][real_idx],
            'attention_mask': self.tokenized_data['attention_mask'][real_idx],
            'labels': self.tokenized_data['labels'][real_idx]
        }
```

### Why This Design:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MEMORY-EFFICIENT DATA SPLITTING                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ BAD APPROACH (copies data):                                                │
│ ────────────────────────────                                                │
│                                                                             │
│   Full tokenized data: 420 MB                                              │
│         ↓                                                                   │
│   train_data = full_data[train_indices]  → Copy: 340 MB                    │
│   val_data = full_data[val_indices]      → Copy: 80 MB                     │
│                                                                             │
│   Total memory: 420 + 340 + 80 = 840 MB ❌                                 │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ GOOD APPROACH (uses indices):                                              │
│ ──────────────────────────────                                              │
│                                                                             │
│   Full tokenized data: 420 MB (stored once)                                │
│         ↓                                                                   │
│   train_dataset = IndexedDataset(full_data, train_indices)                 │
│                   └── Just stores: [0, 2, 5, 7, ...]  (~0.3 MB)            │
│                                                                             │
│   val_dataset = IndexedDataset(full_data, val_indices)                     │
│                 └── Just stores: [1, 3, 4, 6, ...]  (~0.1 MB)              │
│                                                                             │
│   Total memory: 420 + 0.3 + 0.1 = 420.4 MB ✅                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

**Add data augmentation:**
```python
def __getitem__(self, idx):
    real_idx = self.indices[idx]
    
    input_ids = self.tokenized_data['input_ids'][real_idx].clone()
    
    # AUGMENTATION: Randomly mask some tokens during training
    if self.training and random.random() < 0.1:
        mask_idx = random.randint(1, len(input_ids) - 2)
        input_ids[mask_idx] = self.mask_token_id
    
    return {
        'input_ids': input_ids,
        'attention_mask': self.tokenized_data['attention_mask'][real_idx],
        'labels': self.tokenized_data['labels'][real_idx]
    }
```

**Add sample weighting:**
```python
def __getitem__(self, idx):
    real_idx = self.indices[idx]
    
    labels = self.tokenized_data['labels'][real_idx]
    
    # Higher weight for threat samples (rare but important)
    weight = 5.0 if labels[3] == 1 else 1.0  # labels[3] = threat
    
    return {
        'input_ids': self.tokenized_data['input_ids'][real_idx],
        'attention_mask': self.tokenized_data['attention_mask'][real_idx],
        'labels': labels,
        'weight': weight  # Use in loss calculation
    }
```

---

## Section 6: K-Fold Cross-Validation

```python
def prepare_kfold_splits(
    comments: List[str],
    labels: np.ndarray,
    num_folds: int = 5,
    stratification_type: str = 'multiclass',
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare K-fold cross-validation splits with stratification.
    """
```

### What This Does:

Splits your data into K folds for cross-validation, ensuring each fold has similar label distribution.

### Why Stratification Matters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STRATIFICATION EXPLAINED                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Your dataset label distribution:                                           │
│   bully: 40%                                                               │
│   threat: 5%    ← Rare class!                                              │
│   spam: 30%                                                                │
│   sexual: 15%                                                              │
│   religious: 10%                                                           │
│                                                                             │
│ WITHOUT STRATIFICATION:                                                     │
│ ────────────────────────                                                    │
│                                                                             │
│   Fold 1: threat = 8%   (lucky, got more)                                  │
│   Fold 2: threat = 2%   (unlucky, got fewer)                               │
│   Fold 3: threat = 7%                                                      │
│   Fold 4: threat = 3%                                                      │
│   Fold 5: threat = 5%                                                      │
│                                                                             │
│   Problem: Model trained on Fold 2 never learns threats well!              │
│            Results vary wildly between folds.                              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ WITH STRATIFICATION:                                                        │
│ ─────────────────────                                                       │
│                                                                             │
│   Fold 1: threat = 5%   (same as original)                                 │
│   Fold 2: threat = 5%   (same as original)                                 │
│   Fold 3: threat = 5%   (same as original)                                 │
│   Fold 4: threat = 5%   (same as original)                                 │
│   Fold 5: threat = 5%   (same as original)                                 │
│                                                                             │
│   Each fold is representative of the whole dataset!                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-label Stratification Challenge:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE MULTI-LABEL PROBLEM                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ In multi-label classification, a sample can have MULTIPLE labels:          │
│                                                                             │
│   Sample 1: [bully=1, threat=1, spam=0, ...]   ← bully AND threat          │
│   Sample 2: [bully=1, threat=0, spam=0, ...]   ← only bully                │
│   Sample 3: [bully=0, threat=1, spam=0, ...]   ← only threat               │
│                                                                             │
│ Problem: How do you stratify when labels aren't mutually exclusive?        │
│                                                                             │
│ Solutions implemented:                                                      │
│                                                                             │
│ 1. 'multiclass' (default):                                                 │
│    Convert multi-label to single label by finding most common combination  │
│    Good enough for most cases                                              │
│                                                                             │
│ 2. 'multilabel' (advanced):                                                │
│    Use iterative-stratification library                                    │
│    pip install iterative-stratification                                    │
│    Better but slower, requires extra dependency                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation:

```python
## Section 6: K-Fold Cross-Validation

```python
def prepare_kfold_splits(
    comments: np.ndarray,
    labels: np.ndarray,
    num_folds: int = 5,
    stratification_type: str = 'binary',
    seed: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
```

### What This Does:
1. **Stratified Splitting**: For binary classification, it uses `StratifiedKFold` to ensure each fold has the same ratio of HateSpeech vs. Non-HateSpeech as the original dataset.
2. **Reproducibility**: Uses a fixed `seed` so your folds are the same every time you run the experiment.
3. **Memory Efficient**: Uses a `Generator` to yield indices, not the actual data.

### Why use Stratified K-Fold?
If your dataset is imbalanced (e.g., only 10% HateSpeech), a random split might accidentally put all HateSpeech examples in the training set and none in the validation set. Stratification prevents this.
```

### What You Can Modify:

| Modification | Effect | Research Use |
|--------------|--------|--------------|
| `num_folds=10` | More folds, smaller validation sets | More robust estimates |
| `num_folds=3` | Fewer folds, larger validation sets | Faster experiments |
| `stratification_type='multilabel'` | Better stratification | More balanced folds |
| Use single fold | `splits[0]` only | Quick experiments |

**Example - Using all 5 folds for robust evaluation:**
```python
# In main.py, modify to use all folds:
all_fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(splits):
    print(f"\n=== FOLD {fold_idx + 1}/{num_folds} ===")
    
    # Train on this fold
    metrics = train_and_evaluate(train_idx, val_idx)
    all_fold_metrics.append(metrics)

# Report mean ± std across folds
mean_f1 = np.mean([m.f1_macro for m in all_fold_metrics])
std_f1 = np.std([m.f1_macro for m in all_fold_metrics])
print(f"F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")
```

---

## Section 7: Class Weight Calculation

```python
def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    labels = labels.ravel()
    pos_counts = np.sum(labels)
    neg_counts = len(labels) - pos_counts
    
    weight = neg_counts / pos_counts if pos_counts > 0 else 1.0
    return torch.FloatTensor([weight])
```

### What This Does:
Calculates a weight for the positive class to handle **class imbalance**. If you have 9,000 non-hate comments and 1,000 hate comments, the weight for "HateSpeech" will be `9.0`.

### How it's used:
This weight is passed to the loss function (`BCEWithLogitsLoss`), telling the model to "pay 9x more attention" when it misses a HateSpeech example.

### Visual Explanation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLASS WEIGHTING FOR IMBALANCED DATA                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Your dataset (44,000 samples):                                             │
│                                                                             │
│ Label      │ Positive │ Negative │ Ratio        │ Weight                   │
│ ───────────┼──────────┼──────────┼──────────────┼───────────               │
│ bully      │  17,600  │  26,400  │ 40% positive │ 26400/17600 = 1.5        │
│ sexual     │   6,600  │  37,400  │ 15% positive │ 37400/6600  = 5.7        │
│ religious  │   4,400  │  39,600  │ 10% positive │ 39600/4400  = 9.0        │
│ threat     │   2,200  │  41,800  │  5% positive │ 41800/2200  = 19.0 ←!!   │
│ spam       │  13,200  │  30,800  │ 30% positive │ 30800/13200 = 2.3        │
│                                                                             │
│ Effect in loss function:                                                    │
│                                                                             │
│   Without weights:                                                          │
│   - Missing a threat: Loss = 1.0                                           │
│   - Missing a bully:  Loss = 1.0                                           │
│   - Model optimizes for common classes (bully, spam)                       │
│                                                                             │
│   With weights:                                                             │
│   - Missing a threat: Loss = 19.0  (19× more important!)                   │
│   - Missing a bully:  Loss = 1.5                                           │
│   - Model pays attention to rare classes!                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

**Cap maximum weight (prevent instability):**
```python
def calculate_class_weights(labels, max_weight=10.0):
    # ... calculate weights ...
    weights = [min(w, max_weight) for w in weights]  # Cap at 10
    return torch.tensor(weights)
```

**Use different weighting scheme:**
```python
# Option 1: Square root weighting (less aggressive)
weight = np.sqrt(num_negative / num_positive)

# Option 2: Logarithmic weighting (even less aggressive)
weight = np.log1p(num_negative / num_positive)

# Option 3: Effective number weighting (from "Class-Balanced Loss" paper)
beta = 0.9999
effective_num = 1.0 - np.power(beta, num_positive)
weight = (1.0 - beta) / effective_num
```

---

## Section 8: DataLoader Creation

```python
def create_data_loaders(
    tokenized_data: Dict[str, torch.Tensor],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    """
    
    train_dataset = IndexedDataset(tokenized_data, train_indices)
    val_dataset = IndexedDataset(tokenized_data, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle for training
        num_workers=num_workers,
        pin_memory=True         # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

### Understanding DataLoader Parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATALOADER PARAMETERS EXPLAINED                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ batch_size=32:                                                              │
│ ──────────────                                                              │
│   How many samples per forward/backward pass                               │
│                                                                             │
│   Smaller (8-16):  ✓ Uses less GPU memory                                  │
│                    ✗ Noisier gradients, slower convergence                 │
│                    Use when: GPU memory limited                            │
│                                                                             │
│   Larger (64-128): ✓ More stable gradients, faster training                │
│                    ✗ Uses more GPU memory                                  │
│                    Use when: Plenty of GPU memory                          │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ shuffle=True (training only):                                              │
│ ──────────────────────────────                                              │
│   Randomizes sample order each epoch                                       │
│                                                                             │
│   Why: Prevents model from memorizing order                                │
│        Forces model to generalize                                          │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ num_workers=2:                                                              │
│ ───────────────                                                             │
│   Number of parallel processes for data loading                            │
│                                                                             │
│   0: Load data in main process (slowest, but safe)                         │
│   2-4: Parallel loading (faster)                                           │
│   >4: Diminishing returns, may cause issues                                │
│                                                                             │
│   Set to 0 if you see deadlocks or memory issues                           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ pin_memory=True:                                                            │
│ ─────────────────                                                           │
│   Pre-loads data into GPU-compatible memory                                │
│   Faster CPU → GPU transfer                                                 │
│   Only helps with CUDA, disable for CPU-only                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Parameter | Effect | When to Modify |
|-----------|--------|----------------|
| `batch_size` | Memory usage, training speed | GPU out of memory? Reduce it |
| `num_workers` | Data loading speed | Deadlocks? Set to 0 |
| `pin_memory` | GPU transfer speed | CPU only? Set to False |

**Adding gradient accumulation for large effective batch size:**
```python
# In training loop, instead of:
for batch in train_loader:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Use accumulation (effective batch = batch_size × accumulation_steps):
accumulation_steps = 4  # Effective batch = 32 × 4 = 128

for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1: Load CSV                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ load_and_preprocess_data("HateSpeech.csv", ["HateSpeech"])          │  │
│   │   → comments: np.ndarray (Bengali texts)                            │  │
│   │   → labels: np.ndarray (Binary 0/1)                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                ↓                                            │
│   Step 2: Tokenize (with dual caching!)                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ get_or_create_tokenized_dataset(comments, labels, teacher_tok, ...) │  │
│   │   → Check cache: cache/model_maxlen128_tokenized.pkl               │  │
│   │   → If exists: Load in ~2 seconds ✓                                 │  │
│   │   → If not: Tokenize for Teacher & Student, save to cache           │  │
│   │   → Returns: Dict with input_ids, student_input_ids, etc.           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                ↓                                            │
│   Step 3: Split into folds                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ prepare_kfold_splits(comments, labels, num_folds=5)                │  │
│   │   → Yields: (train_indices, val_indices) for each fold             │  │
│   │   → Stratified: Preserves HateSpeech ratio in each fold            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                ↓                                            │
│   Step 4: Create DataLoaders                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ create_data_loaders(tokenized_data, train_idx, val_idx)            │  │
│   │   → train_loader: Shuffled, for training                           │  │
│   │   → val_loader: Not shuffled, for evaluation                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                ↓                                            │
│   Step 5: Calculate class weights                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ calculate_class_weights(labels[train_idx])                         │  │
│   │   → weight: [9.0] (example for 10% hate speech)                     │  │
│   │   → Used in BCEWithLogitsLoss(pos_weight=weight)                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                ↓                                            │
│   Ready for training!                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: What You Can Modify in This Script

| Category | Modification | Research Impact |
|----------|--------------|-----------------|
| **Labels** | `--label_columns` argument | Change classification task |
| **Dataset** | `--dataset_path` argument | Use different CSV files |
| **Caching** | `--cache_dir` argument | Faster storage |
| **Folds** | `num_folds`, stratification | Validation robustness |
| **Weighting** | Weight calculation method | Handle imbalance differently |
| **Batching** | `batch_size`, `num_workers` | Memory/speed tradeoff |

---

## Practice Exercise

Before moving to the next script:

1. **Check your CSV**: Open `data/HateSpeech.csv` and verify the column names.
2. **Run with custom labels**: Try running `main.py` with a different label column if you have one:
   ```bash
   python main.py --dataset_path data/HateSpeech.csv --label_columns HateSpeech
   ```
3. **Inspect the cache**: After running, look into the `cache/` directory. Can you identify which file belongs to which model?

---

**Ready for the next script? The next one is `distillation.py` which implements all Knowledge Distillation methods (logit, hidden, attention, multi_level).**

Would you like me to continue?