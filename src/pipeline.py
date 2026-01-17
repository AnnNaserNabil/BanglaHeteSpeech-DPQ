"""
Pipeline Orchestrator for Enhanced Modular Compression
=======================================================

This module orchestrates the full compression pipeline:
1. Baseline Training (always)
2. Knowledge Distillation (optional)
3. Pruning (optional)
4. Quantization (optional)
5. Final Evaluation & Saving

Uses the rigorous training methodology from the original train.py
with enhanced metrics calculation at every stage.
"""

import os
import torch
import torch.nn as nn
import copy
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import mlflow
import pandas as pd
import gc
from typing import Dict, Tuple, Optional

# Local imports
from data import load_and_preprocess_data, prepare_kfold_splits, calculate_class_weights, HateSpeechDataset
from torch.utils.data import DataLoader, random_split
from model import TransformerBinaryClassifier
from metrics import (
    calculate_metrics_with_threshold_exploration,
    log_metrics_to_mlflow,
    aggregate_fold_metrics,
    print_metrics_summary,
    print_fold_summary,
    print_experiment_summary,
    compare_stage_metrics
)
from distillation import TeacherModel, StudentModel, DistillationTrainer, verify_teacher_performance
from pruning import PruningManager, GradualPruner, WandaPruner, fine_tune_after_pruning
from quantization import (
    QuantizationManager, 
    apply_int4_quantization,
    quantize_model
)
from evaluation import CompressionEvaluator, CompressionStageMetrics
from utils import get_model_metrics, save_model_for_huggingface
from train import run_kfold_training  # Original baseline training


def run_compression_pipeline(config, comments, labels, tokenizer, device):
    """
    Execute the full compression pipeline based on configuration with MLflow tracking.
    
    Pipeline stages:
    1. Baseline Training (Teacher) - Always executed
    2. Knowledge Distillation - If config.enable_kd
    3. Pruning - If config.enable_pruning
    4. Quantization - If config.enable_quantization
    
    All stages are tracked in MLflow for comprehensive experiment tracking.
    
    Args:
        config: Configuration object
        comments: Array of text comments
        labels: Array of binary labels
        tokenizer: Tokenizer for text encoding
        device: Device to run on
    
    Returns:
        dict: Results from all pipeline stages
    """
    print("\n" + "="*70)
    print("🚀 STARTING COMPRESSION PIPELINE")
    print("="*70)
    print(f"Pipeline Mode: {config.pipeline}")
    print(f"Description: {config.pipeline_description}")
    print("="*70)
    
    # =========================================================================
    # MLFLOW SETUP
    # =========================================================================
    # Set MLflow tracking
    mlflow_dir = os.path.abspath('./mlruns')
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    print(f"\n📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"📁 MLflow logs directory: {mlflow_dir}")
    
    # Start MLflow run for the entire pipeline
    with mlflow.start_run(run_name=f"{config.author_name}_{config.pipeline}"):
        run_id = mlflow.active_run().info.run_id
        print(f"🔖 MLflow Run ID: {run_id}\n")
        
        # Log pipeline configuration
        mlflow.log_params({
            'pipeline': config.pipeline,
            'pipeline_description': config.pipeline_description,
            'enable_kd': config.enable_kd,
            'enable_pruning': config.enable_pruning,
            'enable_quantization': config.enable_quantization,
            'model_path': config.model_path,
            'batch_size': config.batch,
            'learning_rate': config.lr,
            'epochs': config.epochs,
            'num_folds': config.num_folds,
            'seed': config.seed
        })
        
        # Log KD parameters if enabled
        if config.enable_kd:
            mlflow.log_params({
                'student_path': config.student_path,
                'kd_alpha': config.kd_alpha,
                'kd_temperature': config.kd_temperature,
                'kd_method': config.kd_method
            })
        
        # Log pruning parameters if enabled
        if config.enable_pruning:
            mlflow.log_params({
                'prune_method': config.prune_method,
                'prune_sparsity': config.prune_sparsity,
                'fine_tune_after_prune': config.fine_tune_after_prune
            })
        
        # Log quantization parameters if enabled
        if config.enable_quantization:
            mlflow.log_params({
                'quant_method': config.quant_method,
                'quant_dtype': config.quant_dtype
            })
        
        # Initialize results dictionary
        pipeline_results = {
            'baseline': None,
            'kd': None,
            'pruning': None,
            'quantization': None
        }
        
        # Track current model through pipeline
        current_model = None
        current_model_name = "baseline"
        
        # =====================================================================
        # STAGE 1: BASELINE TRAINING (Always unless skipped)
        # =====================================================================
        print("\n" + "="*70)
        print("📚 STAGE 1: BASELINE TRAINING")
        print("="*70)
        
        if config.skip_baseline:
            print("⏭️  Skipping baseline training as requested (--skip_baseline)")
            print(f"🔍 Using pre-trained teacher: {config.model_path}")
            
            # Create teacher model
            teacher = TeacherModel(
                model_name=config.model_path,
                num_labels=1,
                dropout=config.dropout
            )
            
            # Load checkpoint if provided
            if config.teacher_checkpoint and os.path.exists(config.teacher_checkpoint):
                print(f"📥 Loading teacher checkpoint: {config.teacher_checkpoint}")
                teacher.load_state_dict(torch.load(config.teacher_checkpoint, map_location=device))
            elif config.teacher_checkpoint:
                print(f"⚠️  Warning: Teacher checkpoint not found at {config.teacher_checkpoint}")
                print("   Continuing with base model weights.")
            
            teacher.to(device)
            baseline_results = {'best_model_path': 'pre-trained'}
        else:
            print("Using rigorous training methodology with threshold exploration...")
            # Run original k-fold training (already has MLflow tracking)
            baseline_results = run_kfold_training(config, comments, labels, tokenizer, device)
            
            # Load best model from training
            print(f"\n📥 Loading best model from rigorous training: {baseline_results['best_model_path']}")
            
            teacher = TeacherModel(
                model_name=config.model_path,
                num_labels=1,
                dropout=config.dropout
            )
            teacher.load_state_dict(torch.load(baseline_results['best_model_path']))
            teacher.to(device)
            
        pipeline_results['baseline'] = baseline_results
        current_model = teacher
        current_model_name = "teacher"
        
        # Log teacher model info
        teacher_params = sum(p.numel() for p in teacher.parameters())
        mlflow.log_metrics({
            'teacher_total_params': teacher_params,
            'teacher_size_mb': teacher_params * 4 / (1024**2)  # Assuming FP32
        })
        
        print("\n✅ Baseline stage complete! Teacher model ready for pipeline.")
        
        # =====================================================================
        # STAGE 2: KNOWLEDGE DISTILLATION (Optional)
        # =====================================================================
        if config.enable_kd:
            print("\n" + "="*70)
            print("🔄 STAGE 2: KNOWLEDGE DISTILLATION")
            print("="*70)
            
            kd_results = _run_knowledge_distillation_stage(
                teacher, comments, labels, tokenizer, config, device
            )
            pipeline_results['kd'] = kd_results
            
            # Log KD metrics to MLflow
            mlflow.log_metrics({
                'kd_accuracy': kd_results['best_metrics'].get('acc_exact', 0),
                'kd_macro_f1': kd_results['best_metrics'].get('f1_macro', 0),
                'kd_f1_hate': kd_results['best_metrics'].get('f1_hate', 0),
                'kd_roc_auc': kd_results['best_metrics'].get('roc_auc_macro', 0),
                'kd_best_threshold': kd_results['best_metrics'].get('best_threshold', 0.5),
                'kd_agreement': kd_results.get('agreement', 0),
                'kd_speedup': kd_results.get('speedup', 1.0),
                'kd_compression_ratio': kd_results.get('compression_ratio', 1.0),
                'student_size_mb': kd_results['best_metrics'].get('size_mb', 0)
            })
            
            # Update current model to student
            current_model = kd_results['student_model']
            current_model_name = "student"
            
            print("\n✅ Knowledge distillation complete!")
            
            # Clean up teacher to save memory
            del teacher
            torch.cuda.empty_cache()
            gc.collect()
        
        # =====================================================================
        # STAGE 3: PRUNING (Optional)
        # =====================================================================
        if config.enable_pruning:
            print("\n" + "="*70)
            print("✂️  STAGE 3: PRUNING")
            print("="*70)
            
            pruning_results = _run_pruning_stage(
                current_model, comments, labels, tokenizer, config, device, current_model_name
            )
            pipeline_results['pruning'] = pruning_results
            
            # Log pruning metrics to MLflow
            mlflow.log_metrics({
                'pruning_accuracy': pruning_results['best_metrics']['accuracy'],
                'pruning_macro_f1': pruning_results['best_metrics']['macro_f1'],
                'pruning_f1_hate': pruning_results['best_metrics']['f1'],
                'pruning_f1_non_hate': pruning_results['best_metrics']['f1_negative'],
                'pruning_roc_auc': pruning_results['best_metrics']['roc_auc'],
                'pruning_best_threshold': pruning_results['best_metrics']['best_threshold'],
                'pruning_sparsity': pruning_results['sparsity']
            })
            
            # Update current model to pruned version
            current_model = pruning_results['pruned_model']
            current_model_name = f"pruned_{current_model_name}"
            
            print("\n✅ Pruning complete!")
            
            torch.cuda.empty_cache()
            gc.collect()
        
        # =====================================================================
        # STAGE 4: QUANTIZATION (Optional)
        # =====================================================================
        if config.enable_quantization:
            print("\n" + "="*70)
            print("📉 STAGE 4: QUANTIZATION")
            print("="*70)
            
            quant_results = _run_quantization_stage(
                current_model, comments, labels, tokenizer, config, device, current_model_name
            )
            pipeline_results['quantization'] = quant_results
            
            # Log quantization metrics to MLflow
            mlflow.log_metrics({
                'quantization_accuracy': quant_results['best_metrics']['accuracy'],
                'quantization_macro_f1': quant_results['best_metrics']['macro_f1'],
                'quantization_f1_hate': quant_results['best_metrics']['f1'],
                'quantization_f1_non_hate': quant_results['best_metrics']['f1_negative'],
                'quantization_roc_auc': quant_results['best_metrics']['roc_auc'],
                'quantization_best_threshold': quant_results['best_metrics']['best_threshold']
            })
            
            mlflow.log_param('quantization_method', quant_results['method'])
            
            # Update current model to quantized version
            current_model = quant_results['quantized_model']
            current_model_name = f"quantized_{current_model_name}"
            
            print("\n✅ Quantization complete!")
        
        # =====================================================================
        # STAGE 5: FINAL EVALUATION & SAVING
        # =====================================================================
        print("\n" + "="*70)
        print("💾 STAGE 5: FINAL EVALUATION & SAVING")
        print("="*70)
        
        # Save final model if requested
        if config.save_huggingface:
            output_path = os.path.join(config.output_dir, f"final_{current_model_name}")
            save_model_for_huggingface(current_model, tokenizer, output_path, vars(config))
            print(f"✅ Model saved to: {output_path}")
            
            # Log model path to MLflow
            mlflow.log_param('final_model_path', output_path)
        
        # Save pipeline summary as CSV and log to MLflow
        summary_df = _create_pipeline_summary_df(pipeline_results)
        summary_path = os.path.join(config.output_dir, f'pipeline_summary_{config.pipeline}.csv')
        os.makedirs(config.output_dir, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)
        print(f"📊 Pipeline summary saved: {summary_path}")
        
        # Print final summary
        _print_pipeline_summary(pipeline_results, config)
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 MLflow Run ID: {run_id}")
        print(f"💡 View results: mlflow ui")
        print("="*70)
        
        return pipeline_results





def _run_knowledge_distillation_stage(teacher, comments, labels, tokenizer, config, device):
    """
    Run knowledge distillation stage with enhanced metrics.
    """
    print(f"Teacher: {config.model_path}")
    print(f"Student: {config.student_path}")
    print(f"KD Method: {config.kd_method}")
    print(f"Alpha: {config.kd_alpha}, Temperature: {config.kd_temperature}")
    
    # Create student model
    student = StudentModel(
        model_name=config.student_path,
        num_labels=1,
        dropout=config.dropout,
        classifier_hidden_size=config.student_hidden_size
    ).to(device)
    
    # Create datasets
    
    # Load student tokenizer if different from teacher
    student_tokenizer = None
    if config.student_path != config.model_path:
        print(f"Loading separate tokenizer for student: {config.student_path}")
        student_tokenizer = AutoTokenizer.from_pretrained(config.student_path)
    
    dataset = HateSpeechDataset(
        comments, 
        labels, 
        tokenizer, 
        config.max_length,
        student_tokenizer=student_tokenizer
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False, num_workers=2)
    
    # Setup distillation trainer
    trainer = DistillationTrainer(teacher, student, config, device)
    optimizer = AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop with enhanced metrics
    best_macro_f1 = 0
    best_metrics = {}
    best_train_metrics = {}
    best_student_state = None
    best_epoch = 0
    
    # Early Stopping
    patience = config.early_stopping_patience
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Train
        student.train()
        epoch_losses = []
        all_train_preds = []
        all_train_labels = []
        
        pbar = tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            outputs = trainer.train_step(batch, optimizer)
            scheduler.step()
            
            loss = outputs['loss']
            epoch_losses.append(loss)
            pbar.set_postfix({'loss': f"{loss:.4f}"})
            
            # Accumulate predictions for training metrics
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels)
        
        # Calculate Training Metrics
        train_metrics = calculate_metrics_with_threshold_exploration(
            np.array(all_train_labels),
            np.array(all_train_preds)
        )
        train_metrics['loss'] = np.mean(epoch_losses)
        
        # Evaluate with enhanced metrics
        eval_results = trainer.evaluate(val_loader)
        
        # Calculate Validation Metrics
        val_metrics = calculate_metrics_with_threshold_exploration(
            eval_results['labels'],
            eval_results['predictions']
        )
        val_metrics['loss'] = eval_results['loss']
        
        print_metrics_summary(val_metrics, f"KD Epoch {epoch+1} Validation")
        
        # Check for improvement
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_metrics = val_metrics.copy()
            best_train_metrics = train_metrics.copy()
            best_student_state = copy.deepcopy(student.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best state before returning
    if best_student_state is not None:
        student.load_state_dict(best_student_state)
        print(f"\n✅ Loaded best model from Epoch {best_epoch} (Macro F1: {best_macro_f1:.4f})")
    
    # Comprehensive Evaluation
    print("\n🔍 Performing comprehensive evaluation of Student model...")
    evaluator = CompressionEvaluator(
        label_columns=['hate'],
        label_priority={'hate': 1}
    )
    
    # Evaluate Teacher as baseline for KD stage
    teacher_metrics = evaluator.evaluate_model(
        teacher, val_loader, device, stage='kd_teacher',
        measure_latency=True, config=vars(config)
    )
    
    # Evaluate Student
    student_metrics = evaluator.evaluate_model(
        student, val_loader, device, stage='kd_student',
        measure_latency=True, use_student_input_ids=(student_tokenizer is not None),
        config=vars(config)
    )
    
    # Calculate Fidelity (Agreement)
    agreement = trainer.calculate_agreement(val_loader)
    print(f"🤝 Teacher-Student Agreement: {agreement:.4f}")
    
    # Calculate Ratios
    compression_ratio = teacher_metrics.model_size_mb / max(student_metrics.model_size_mb, 0.01)
    speedup = teacher_metrics.inference_latency_mean_ms / max(student_metrics.inference_latency_mean_ms, 0.01)
    
    print(f"📦 Compression Ratio: {compression_ratio:.2f}x")
    print(f"⚡ Speedup Ratio: {speedup:.2f}x")
    
    # Log to MLflow
    mlflow.log_metrics({
        'kd_agreement': agreement,
        'kd_compression_ratio': compression_ratio,
        'kd_speedup': speedup,
        'kd_student_latency_ms': student_metrics.inference_latency_mean_ms,
        'kd_student_throughput': student_metrics.throughput_samples_per_sec,
        'kd_best_epoch': best_epoch
    })
    
    return {
        'student_model': student,
        'best_metrics': best_metrics,
        'best_train_metrics': best_train_metrics,
        'teacher_metrics': teacher_metrics.to_flat_dict(),
        'agreement': agreement,
        'compression_ratio': compression_ratio,
        'speedup': speedup
    }


def _run_pruning_stage(model, comments, labels, tokenizer, config, device, model_name):
    """
    Run pruning stage with enhanced metrics.
    """
    print(f"Pruning {model_name}")
    print(f"Method: {config.prune_method}")
    print(f"Target Sparsity: {config.prune_sparsity*100:.0f}%")
    
    # Load student tokenizer if different from teacher
    student_tokenizer = None
    if config.student_path != config.model_path:
        print(f"Loading separate tokenizer for student: {config.student_path}")
        student_tokenizer = AutoTokenizer.from_pretrained(config.student_path)
    
    dataset = HateSpeechDataset(
        comments, 
        labels, 
        tokenizer, 
        config.max_length,
        student_tokenizer=student_tokenizer
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False, num_workers=2)
    
    # Apply pruning
    if config.prune_method == 'magnitude':
        pruner = PruningManager(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers,
            global_pruning=True
        )
        pruner.apply_magnitude_pruning()
    elif config.prune_method == 'wanda':
        pruner = WandaPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers
        )
        pruner.collect_activations(train_loader, device, num_samples=config.calib_samples)
        pruner.apply_wanda_pruning()
    
    # Fine-tune if enabled
    if config.fine_tune_after_prune:
        print(f"\n🔧 Fine-tuning for {config.fine_tune_epochs} epochs...")
        fine_tune_metrics = fine_tune_after_pruning(
            model, train_loader, val_loader, config, device
        )
    
    
    # Make pruning permanent (remove masks/hooks)
    if 'pruner' in locals():
        pruner.make_pruning_permanent()
    
    # Evaluate with enhanced metrics
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            # Handle dual tokenization
            if 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(device)
                attention_mask = batch['student_attention_mask'].to(device)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())
    
    metrics = calculate_metrics_with_threshold_exploration(
        np.array(all_labels),
        np.array(all_preds)
    )
    
    print_metrics_summary(metrics, "Pruned Model")
    
    return {
        'pruned_model': model,
        'best_metrics': metrics,
        'sparsity': config.prune_sparsity
    }


def _run_quantization_stage(model, comments, labels, tokenizer, config, device, model_name):
    """
    Run quantization stage with enhanced metrics.
    """
    print(f"Quantizing {model_name}")
    print(f"Method: {config.quant_method}")
    
    # Create dataset for evaluation
    
    # Load student tokenizer if different from teacher
    student_tokenizer = None
    if config.student_path != config.model_path:
        print(f"Loading separate tokenizer for student: {config.student_path}")
        student_tokenizer = AutoTokenizer.from_pretrained(config.student_path)
        
    dataset = HateSpeechDataset(
        comments, 
        labels, 
        tokenizer, 
        config.max_length,
        student_tokenizer=student_tokenizer
    )
    loader = DataLoader(dataset, batch_size=config.batch, shuffle=False, num_workers=2)
    
    # Apply quantization
    print(f"\n📉 Applying {config.quant_method} quantization...")
    
    quantized_model = quantize_model(
        model,
        config.quant_method,
        config,
        calibration_loader=loader if config.quant_method == 'static' else None,
        device=device,
        use_student_input_ids=True
    )
    
    # Evaluate (on CPU for quantized models)
    eval_device = 'cpu' if config.quant_method in ['dynamic', 'static'] else device
    quantized_model = quantized_model.to(eval_device)
    quantized_model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            # Handle dual tokenization
            if 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(eval_device)
                attention_mask = batch['student_attention_mask'].to(eval_device)
            else:
                input_ids = batch['input_ids'].to(eval_device)
                attention_mask = batch['attention_mask'].to(eval_device)
                
            outputs = quantized_model(input_ids, attention_mask)
            
            # Handle different output formats (dict, tuple, or raw tensor)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('last_hidden_state'))
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs
                
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())
    
    metrics = calculate_metrics_with_threshold_exploration(
        np.array(all_labels),
        np.array(all_preds)
    )
    
    print_metrics_summary(metrics, "Quantized Model")
    
    return {
        'quantized_model': quantized_model,
        'best_metrics': metrics,
        'method': config.quant_method
    }


def _print_pipeline_summary(results, config):
    """Print summary of all pipeline stages."""
    print("\n" + "="*70)
    print("📊 PIPELINE SUMMARY")
    print("="*70)
    
    stages = []
    if results['baseline']:
        stages.append(('Baseline', results['baseline']))
    if results['kd']:
        stages.append(('KD Student', results['kd']))
    if results['pruning']:
        stages.append(('Pruned', results['pruning']))
    if results['quantization']:
        stages.append(('Quantized', results['quantization']))
    
    # Print comparison table
    print("\nStage Comparison:")
    print("-" * 90)
    print(f"{'Stage':<20} {'Accuracy':<12} {'Macro F1':<12} {'Speedup':<12} {'Agreement':<12}")
    print("-" * 90)
    
    for stage_name, stage_results in stages:
        if isinstance(stage_results, list):  # Old format compatibility
            acc = np.mean([r.get('accuracy', 0) for r in stage_results])
            macro_f1 = np.mean([r.get('macro_f1', 0) for r in stage_results])
            speedup = 1.0
            agreement = 1.0
        elif isinstance(stage_results, dict):
            # Check for new format (CompressionStageMetrics.to_flat_dict)
            metrics = stage_results.get('best_metrics', stage_results)
            acc = metrics.get('acc_exact', metrics.get('accuracy', 0))
            macro_f1 = metrics.get('f1_macro', metrics.get('macro_f1', 0))
            
            speedup = stage_results.get('speedup', 1.0)
            agreement = stage_results.get('agreement', 1.0)
        else:
            acc = getattr(stage_results, 'accuracy', 0)
            macro_f1 = getattr(stage_results, 'macro_f1', 0)
            speedup = 1.0
            agreement = 1.0
        
        print(f"{stage_name:<20} {acc:<12.4f} {macro_f1:<12.4f} {speedup:<12.2f} {agreement:<12.4f}")
    
    print("-" * 90)


def _create_pipeline_summary_df(results):
    """Create a DataFrame summarizing all pipeline stages."""
    summary_data = []
    
    # Baseline
    if results['baseline']:
        if isinstance(results['baseline'], list):
            baseline_metrics = {
                'accuracy': np.mean([r.get('accuracy', 0) for r in results['baseline']]),
                'macro_f1': np.mean([r.get('macro_f1', 0) for r in results['baseline']]),
                'f1': np.mean([r.get('f1', 0) for r in results['baseline']]),
                'f1_negative': np.mean([r.get('f1_negative', 0) for r in results['baseline']]),
                'roc_auc': np.mean([r.get('roc_auc', 0) for r in results['baseline']])
            }
        elif isinstance(results['baseline'], dict) and 'best_metrics' in results['baseline']:
            baseline_metrics = results['baseline']['best_metrics']
        else:
            baseline_metrics = results['baseline']
        
        summary_data.append({
            'Stage': 'Baseline',
            'Accuracy': baseline_metrics.get('accuracy', 0),
            'Macro F1': baseline_metrics.get('macro_f1', 0),
            'F1 (Hate)': baseline_metrics.get('f1', 0),
            'F1 (Non-Hate)': baseline_metrics.get('f1_negative', 0),
            'ROC-AUC': baseline_metrics.get('roc_auc', 0)
        })
    
    # KD
    if results['kd']:
        metrics = results['kd']['best_metrics']
        train_metrics = results['kd'].get('best_train_metrics', {})
        
        summary_data.append({
            'Stage': 'KD Student',
            'Accuracy': metrics.get('acc_exact', metrics.get('accuracy', 0)),
            'Macro F1': metrics.get('f1_macro', metrics.get('macro_f1', 0)),
            'F1 (Hate)': metrics.get('f1_hate', metrics.get('f1', 0)),
            'F1 (Non-Hate)': metrics.get('f1_negative', 0),
            'Precision (Hate)': metrics.get('precision', 0),
            'Recall (Hate)': metrics.get('recall', 0),
            'Precision (Non-Hate)': metrics.get('precision_negative', 0),
            'Recall (Non-Hate)': metrics.get('recall_negative', 0),
            'ROC-AUC': metrics.get('roc_auc_macro', metrics.get('roc_auc', 0)),
            'Val Loss': metrics.get('loss', 0),
            'Best Threshold': metrics.get('best_threshold', 0.5),
            'Train Accuracy': train_metrics.get('accuracy', 0),
            'Train Macro F1': train_metrics.get('macro_f1', 0),
            'Train F1 (Hate)': train_metrics.get('f1', 0),
            'Train Loss': train_metrics.get('loss', 0),
            'Speedup': results['kd'].get('speedup', 1.0),
            'Agreement': results['kd'].get('agreement', 1.0),
            'Size (MB)': metrics.get('size_mb', 0)
        })
    
    # Pruning
    if results['pruning']:
        metrics = results['pruning']['best_metrics']
        summary_data.append({
            'Stage': 'Pruned',
            'Accuracy': metrics.get('acc_exact', metrics.get('accuracy', 0)),
            'Macro F1': metrics.get('f1_macro', metrics.get('macro_f1', 0)),
            'F1 (Hate)': metrics.get('f1_hate', metrics.get('f1', 0)),
            'ROC-AUC': metrics.get('roc_auc_macro', metrics.get('roc_auc', 0)),
            'Speedup': results['pruning'].get('speedup', 1.0),
            'Size (MB)': metrics.get('size_mb', 0)
        })
    
    # Quantization
    if results['quantization']:
        metrics = results['quantization']['best_metrics']
        summary_data.append({
            'Stage': 'Quantized',
            'Accuracy': metrics.get('acc_exact', metrics.get('accuracy', 0)),
            'Macro F1': metrics.get('f1_macro', metrics.get('macro_f1', 0)),
            'F1 (Hate)': metrics.get('f1_hate', metrics.get('f1', 0)),
            'ROC-AUC': metrics.get('roc_auc_macro', metrics.get('roc_auc', 0)),
            'Speedup': results['quantization'].get('speedup', 1.0),
            'Size (MB)': metrics.get('size_mb', 0)
        })
    
    return pd.DataFrame(summary_data)
