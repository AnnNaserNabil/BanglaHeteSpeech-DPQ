"""
Enhanced Main Script for Bangla BERT Hate Speech Detection
===========================================================

Supports both baseline training and full compression pipeline:
- Baseline mode: Original rigorous k-fold training
- Pipeline modes: baseline + KD/pruning/quantization combinations

Usage:
    # Baseline only (original behavior)
    python main.py --dataset_path data.csv --author_name researcher --pipeline baseline
    
    # Baseline + Knowledge Distillation
    python main.py --dataset_path data.csv --author_name researcher --pipeline baseline_kd
    
    # Full pipeline
    python main.py --dataset_path data.csv --author_name researcher --pipeline baseline_kd_prune_quant
"""

from transformers import AutoTokenizer
import torch
import os
from config import parse_arguments, print_config
from data import load_and_preprocess_data
from train import run_kfold_training
from pipeline import run_compression_pipeline
from utils import set_seed, save_model_for_huggingface


def main():
    """
    Main entry point for training and compression pipeline.
    """
    # Parse arguments
    config = parse_arguments()
    print_config(config)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    print(f"\n✓ Tokenizer loaded: {config.model_path}")

    # Load and preprocess data
    comments, labels = load_and_preprocess_data(config.dataset_path)

    # Apply data fraction if specified
    if config.data_fraction < 1.0:
        import numpy as np
        num_samples = len(comments)
        num_to_keep = int(num_samples * config.data_fraction)
        print(f"\n✂️  Applying data fraction: {config.data_fraction} ({num_to_keep}/{num_samples} samples)")
        
        # Use a fixed seed for slicing if provided in config to ensure reproducibility
        indices = np.arange(num_samples)
        np.random.seed(config.seed)
        np.random.shuffle(indices)
        
        selected_indices = indices[:num_to_keep]
        comments = comments[selected_indices]
        labels = labels[selected_indices]
        print(f"✅ Data sliced successfully. New total: {len(comments)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"\n🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n🖥️  Using CPU (GPU not available)")

    # Run appropriate mode
    if config.pipeline == 'baseline':
        # Original behavior - just baseline training
        print("\n" + "="*70)
        print("📚 RUNNING BASELINE TRAINING ONLY")
        print("="*70)
        print("This is the original rigorous k-fold training methodology.")
        
        if config.data_fraction < 1.0:
            print(f"\n" + "!"*70)
            print(f"⚠️  CRITICAL WARNING: DATA FRACTION IS SET TO {config.data_fraction}")
            print(f"   This will use only {len(comments)} samples out of the full dataset.")
            print(f"   This is likely why your accuracy is lower (~0.83) than expected (>0.87).")
            print(f"   For full performance comparison, set --data_fraction 1.0")
            print("!"*70 + "\n")
            
        print(f"\n💡 TIP: If performance is still unexpectedly low, the cache might be stale.")
        print(f"   Run: rm -rf {config.cache_dir}")
        print(f"   This ensures the dataset is re-tokenized for the current model.")
        print("="*70 + "\n")
        
        results = run_kfold_training(config, comments, labels, tokenizer, device)
        
        # Save for HuggingFace if requested
        if config.save_huggingface:
            print("\n🤗 Saving best baseline model for HuggingFace...")
            from model import TransformerBinaryClassifier
            best_model = TransformerBinaryClassifier(
                config.model_path, 
                dropout=config.dropout,
                pooling_type=config.pooling_type,
                use_multi_layers=(config.pooling_type == 'multi')
            )
            best_model.load_state_dict(torch.load(results['best_model_path']))
            
            output_path = os.path.join(config.output_dir, "final_baseline_best")
            save_model_for_huggingface(best_model, tokenizer, output_path, vars(config))
        
        print("\n" + "="*70)
        print("✅ BASELINE TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    else:
        # New behavior - full pipeline
        print("\n" + "="*70)
        print("🚀 RUNNING COMPRESSION PIPELINE")
        print("="*70)
        print(f"Pipeline: {config.pipeline}")
        print(f"Description: {config.pipeline_description}")
        print("="*70 + "\n")
        
        results = run_compression_pipeline(config, comments, labels, tokenizer, device)
        
        print("\n" + "="*70)
        print("✅ COMPRESSION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
    
    print("\n📊 Results saved to:", config.output_dir)
    if config.save_huggingface:
        print("🤗 HuggingFace-compatible model saved!")
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
