"""
Enhanced Configuration Module for Bangla BERT Hate Speech Detection
====================================================================

Supports both baseline training and full compression pipeline modes:
- baseline: Original rigorous training only
- baseline_kd: Baseline + Knowledge Distillation
- baseline_prune: Baseline + Pruning
- baseline_kd_prune: Baseline + KD + Pruning
- baseline_kd_prune_quant: Full pipeline (KD + Pruning + Quantization)
"""

import argparse


# Pipeline configuration presets
PIPELINE_CONFIGS = {
    'baseline': {
        'enable_kd': False,
        'enable_pruning': False,
        'enable_quantization': False,
        'description': 'Baseline training only (original behavior)'
    },
    'baseline_kd': {
        'enable_kd': True,
        'enable_pruning': False,
        'enable_quantization': False,
        'description': 'Baseline → Knowledge Distillation'
    },
    'baseline_prune': {
        'enable_kd': False,
        'enable_pruning': True,
        'enable_quantization': False,
        'description': 'Baseline → Pruning'
    },
    'baseline_quant': {
        'enable_kd': False,
        'enable_pruning': False,
        'enable_quantization': True,
        'description': 'Baseline → Quantization'
    },
    'baseline_kd_prune': {
        'enable_kd': True,
        'enable_pruning': True,
        'enable_quantization': False,
        'description': 'Baseline → KD → Pruning'
    },
    'baseline_kd_quant': {
        'enable_kd': True,
        'enable_pruning': False,
        'enable_quantization': True,
        'description': 'Baseline → KD → Quantization'
    },
    'baseline_prune_quant': {
        'enable_kd': False,
        'enable_pruning': True,
        'enable_quantization': True,
        'description': 'Baseline → Pruning → Quantization'
    },
    'baseline_kd_prune_quant': {
        'enable_kd': True,
        'enable_pruning': True,
        'enable_quantization': True,
        'description': 'Full Pipeline: Baseline → KD → Pruning → Quantization'
    }
}


def parse_arguments():
    """
    Parse command-line arguments for experiment configuration.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Bangla BERT Hate Speech Detection with Compression Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =========================================================================
    # PIPELINE CONFIGURATION
    # =========================================================================
    parser.add_argument('--pipeline', type=str, default='baseline',
                       choices=list(PIPELINE_CONFIGS.keys()),
                       help='Pipeline mode: baseline, baseline_kd, baseline_prune, etc.')

    # =========================================================================
    # BASIC TRAINING PARAMETERS
    # =========================================================================
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Maximum number of training epochs.')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the CSV dataset file.')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for tokenization.')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of folds for K-Fold cross-validation.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility.')
    parser.add_argument('--stratification_type', type=str, default='binary',
                       choices=['binary', 'none'],
                       help='Type of stratification for K-fold splitting.')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                       help='Fraction of data to use for training (0.0-1.0).')

    # =========================================================================
    # MODEL PARAMETERS
    # =========================================================================
    parser.add_argument('--model_path', type=str, default='sagorsarker/bangla-bert-base',
                       help='Pre-trained model name or path (teacher/baseline model).')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze the base transformer layers during fine-tuning.')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for the classification head.')

    # =========================================================================
    # OPTIMIZER PARAMETERS
    # =========================================================================
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of total steps for learning rate warmup.')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                       help='Maximum norm for gradient clipping.')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Number of epochs without improvement before early stopping.')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights to handle imbalance.')
    parser.add_argument('--scheduler_type', type=str, default='linear',
                       choices=['linear', 'cosine'],
                       help='Type of learning rate scheduler.')
    parser.add_argument('--loss_type', type=str, default='bce',
                       choices=['bce', 'focal'],
                       help='Type of loss function.')
    parser.add_argument('--pooling_type', type=str, default='cls',
                       choices=['cls', 'mean', 'max'],
                       help='Type of pooling for transformer outputs.')

    # =========================================================================
    # MLFLOW PARAMETERS
    # =========================================================================
    parser.add_argument('--author_name', type=str, required=True,
                       help='Author name for MLflow run tagging.')
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla-HateSpeech-Detection',
                       help='MLflow experiment name for tracking.')

    # =========================================================================
    # KNOWLEDGE DISTILLATION PARAMETERS
    # =========================================================================
    parser.add_argument('--student_path', type=str, default='distilbert-base-multilingual-cased',
                       help='Student model for knowledge distillation.')
    parser.add_argument('--student_hidden_size', type=int, default=256,
                       help='Hidden size for student classifier head.')
    parser.add_argument('--kd_alpha', type=float, default=0.7,
                       help='KD loss weight (0=hard labels only, 1=soft labels only).')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                       help='Temperature for softening teacher predictions.')
    parser.add_argument('--kd_method', type=str, default='logit',
                       choices=['logit', 'hidden', 'attention', 'multi_level'],
                       help='Knowledge distillation method.')
    parser.add_argument('--hidden_loss_weight', type=float, default=0.3,
                       help='Weight for hidden state matching loss.')
    parser.add_argument('--attention_loss_weight', type=float, default=0.2,
                       help='Weight for attention matching loss.')

    # =========================================================================
    # PRUNING PARAMETERS
    # =========================================================================
    parser.add_argument('--prune_method', type=str, default='magnitude',
                       choices=['magnitude', 'wanda', 'gradual', 'structured'],
                       help='Pruning method.')
    parser.add_argument('--prune_sparsity', type=float, default=0.5,
                       help='Target sparsity for pruning (0.0-1.0).')
    parser.add_argument('--prune_schedule', type=str, default='cubic',
                       choices=['linear', 'cubic', 'exponential'],
                       help='Pruning schedule for gradual pruning.')
    parser.add_argument('--prune_start_epoch', type=int, default=0,
                       help='Start epoch for gradual pruning.')
    parser.add_argument('--prune_end_epoch', type=int, default=10,
                       help='End epoch for gradual pruning.')
    parser.add_argument('--prune_frequency', type=int, default=100,
                       help='Pruning frequency (steps) for gradual pruning.')
    parser.add_argument('--prune_layers', type=str, default='all',
                       choices=['all', 'attention', 'ffn', 'encoder'],
                       help='Which layers to prune.')
    parser.add_argument('--calib_samples', type=int, default=512,
                       help='Number of calibration samples for WANDA pruning.')
    parser.add_argument('--fine_tune_after_prune', action='store_true', default=True,
                       help='Fine-tune after pruning (default: True).')
    parser.add_argument('--no_fine_tune_after_prune', action='store_false',
                       dest='fine_tune_after_prune',
                       help='Skip fine-tuning after pruning.')
    parser.add_argument('--fine_tune_epochs', type=int, default=3,
                       help='Number of epochs for fine-tuning after pruning.')

    # =========================================================================
    # QUANTIZATION PARAMETERS
    # =========================================================================
    parser.add_argument('--quant_method', type=str, default='dynamic',
                       choices=['dynamic', 'static', 'qat', 'fp16', 'int4'],
                       help='Quantization method.')
    parser.add_argument('--quant_dtype', type=str, default='int8',
                       choices=['int8', 'int4', 'fp16'],
                       help='Quantization data type.')
    parser.add_argument('--quant_calibration_batches', type=int, default=100,
                       help='Number of batches for static quantization calibration.')

    # =========================================================================
    # OUTPUT PARAMETERS
    # =========================================================================
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and results.')
    parser.add_argument('--save_huggingface', action='store_true', default=True,
                       help='Save final model in HuggingFace format.')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory for tokenized data.')

    args = parser.parse_args()

    # Validation
    if args.batch <= 0:
        raise ValueError("Batch size must be positive")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.num_folds < 1:
        raise ValueError("Number of folds must be at least 1")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("Dropout must be between 0 and 1")
    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        raise ValueError("Warmup ratio must be between 0 and 1")
    if args.kd_alpha < 0 or args.kd_alpha > 1:
        raise ValueError("KD alpha must be between 0 and 1")
    if args.prune_sparsity < 0 or args.prune_sparsity >= 1:
        raise ValueError("Prune sparsity must be between 0 and 1")
    if args.data_fraction <= 0 or args.data_fraction > 1:
        raise ValueError("Data fraction must be between 0 and 1")

    # Apply pipeline configuration
    _apply_pipeline_config(args)

    return args


def _apply_pipeline_config(args):
    """
    Apply pipeline-specific settings based on selected pipeline mode.
    
    Args:
        args: Argument namespace to modify
    """
    pipeline_config = PIPELINE_CONFIGS.get(args.pipeline, {})
    args.enable_kd = pipeline_config.get('enable_kd', False)
    args.enable_pruning = pipeline_config.get('enable_pruning', False)
    args.enable_quantization = pipeline_config.get('enable_quantization', False)
    args.pipeline_description = pipeline_config.get('description', '')


def print_config(config):
    """
    Print configuration in a formatted way.

    Args:
        config: Configuration namespace
    """
    print("\n" + "="*70)
    print("ENHANCED CONFIGURATION (Hate Speech Detection + Compression)")
    print("="*70)
    
    # Pipeline Information
    print(f"\n📊 Pipeline Mode: {config.pipeline.upper()}")
    print(f"   {config.pipeline_description}")
    print(f"\n   Stages Enabled:")
    print(f"      Baseline Training:    ✓ (always)")
    print(f"      Knowledge Distillation: {'✓' if config.enable_kd else '✗'}")
    print(f"      Pruning:               {'✓' if config.enable_pruning else '✗'}")
    print(f"      Quantization:          {'✓' if config.enable_quantization else '✗'}")
    
    # Training Parameters
    print("\n🎓 Training Parameters:")
    print(f"   Batch Size:              {config.batch}")
    print(f"   Learning Rate:           {config.lr}")
    print(f"   Max Epochs:              {config.epochs}")
    print(f"   Early Stopping Patience: {config.early_stopping_patience}")
    print(f"   Use Class Weights:       {'✓' if config.use_class_weights else '✗'}")
    print(f"   Scheduler Type:          {config.scheduler_type}")
    print(f"   Loss Type:               {config.loss_type}")
    print(f"   Pooling Type:            {config.pooling_type}")
    
    # Model Parameters
    print("\n🤖 Model Parameters:")
    print(f"   Teacher/Baseline Model:  {config.model_path}")
    if config.enable_kd:
        print(f"   Student Model:           {config.student_path}")
        print(f"   Student Hidden Size:     {config.student_hidden_size}")
    print(f"   Max Sequence Length:     {config.max_length}")
    print(f"   Freeze Base:             {config.freeze_base}")
    print(f"   Dropout:                 {config.dropout}")
    
    # Optimizer Parameters
    print("\n⚙️  Optimizer Parameters:")
    print(f"   Weight Decay:            {config.weight_decay}")
    print(f"   Warmup Ratio:            {config.warmup_ratio}")
    print(f"   Gradient Clip Norm:      {config.gradient_clip_norm}")
    
    # KD Parameters (if enabled)
    if config.enable_kd:
        print("\n📚 Knowledge Distillation:")
        print(f"   Method:                  {config.kd_method}")
        print(f"   Alpha:                   {config.kd_alpha}")
        print(f"   Temperature:             {config.kd_temperature}")
        if config.kd_method in ['hidden', 'multi_level']:
            print(f"   Hidden Loss Weight:      {config.hidden_loss_weight}")
        if config.kd_method in ['attention', 'multi_level']:
            print(f"   Attention Loss Weight:   {config.attention_loss_weight}")
    
    # Pruning Parameters (if enabled)
    if config.enable_pruning:
        print("\n✂️  Pruning:")
        print(f"   Method:                  {config.prune_method}")
        print(f"   Target Sparsity:         {config.prune_sparsity*100:.0f}%")
        print(f"   Layers:                  {config.prune_layers}")
        print(f"   Fine-tune After:         {'Yes' if config.fine_tune_after_prune else 'No'}")
        if config.fine_tune_after_prune:
            print(f"   Fine-tune Epochs:        {config.fine_tune_epochs}")
        if config.prune_method == 'gradual':
            print(f"   Schedule:                {config.prune_schedule}")
            print(f"   Start/End Epoch:         {config.prune_start_epoch}/{config.prune_end_epoch}")
    
    # Quantization Parameters (if enabled)
    if config.enable_quantization:
        print("\n📉 Quantization:")
        print(f"   Method:                  {config.quant_method}")
        print(f"   Data Type:               {config.quant_dtype}")
        if config.quant_method == 'static':
            print(f"   Calibration Batches:     {config.quant_calibration_batches}")
    
    # Experiment Parameters
    print("\n🔬 Experiment Parameters:")
    print(f"   Author:                  {config.author_name}")
    print(f"   K-Folds:                 {config.num_folds}")
    print(f"   Stratification:          {config.stratification_type}")
    print(f"   Random Seed:             {config.seed}")
    print(f"   MLflow Experiment:       {config.mlflow_experiment_name}")
    
    # Data & Output Parameters
    print("\n📁 Data & Output:")
    print(f"   Dataset Path:            {config.dataset_path}")
    print(f"   Output Directory:        {config.output_dir}")
    print(f"   Cache Directory:         {config.cache_dir}")
    print(f"   Save HuggingFace Format: {'Yes' if config.save_huggingface else 'No'}")
    
    print("="*70 + "\n")
