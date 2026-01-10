"""
Enhanced Metrics Module for Hate Speech Detection
==================================================

This module provides comprehensive metrics calculation with threshold exploration,
optimized for binary classification tasks. It supports:
- Threshold exploration for macro F1 optimization
- Per-class metrics (hate/non-hate)
- ROC-AUC calculation
- MLflow integration
- Aggregation across k-folds

Author: Enhanced from original train.py metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from typing import Dict, List, Optional, Tuple
import mlflow


def calculate_metrics_with_threshold_exploration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics with threshold exploration.
    
    Explores multiple classification thresholds to find the one that
    maximizes macro F1 score (average of F1 for both classes).
    
    Args:
        y_true: Ground truth labels (0 or 1), shape (n_samples,)
        y_pred: Predicted probabilities, shape (n_samples,)
        thresholds: List of thresholds to explore (default: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    
    Returns:
        dict: Dictionary containing:
            - accuracy: Accuracy at best threshold
            - precision: Precision for positive class (hate speech)
            - recall: Recall for positive class
            - f1: F1 for positive class
            - precision_negative: Precision for negative class (non-hate)
            - recall_negative: Recall for negative class
            - f1_negative: F1 for negative class
            - macro_f1: Macro F1 (average of both classes)
            - roc_auc: ROC-AUC score
            - best_threshold: Threshold that maximizes macro F1
            - macro_f1_th_{thresh}: Macro F1 at each threshold (for analysis)
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate ROC-AUC from probabilities (before thresholding)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle case where only one class is present
        auc = 0.0
    
    # Explore thresholds
    best_macro_f1 = -1
    best_threshold = None
    best_threshold_metrics = {}
    metrics = {}
    
    for thresh in thresholds:
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > thresh).astype(int)
        
        # Calculate precision, recall, F1 for both classes
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, labels=[0, 1], average=None, zero_division=0
        )
        
        # Calculate macro F1 (average of both classes)
        macro_f1 = (f1[0] + f1[1]) / 2 if len(f1) == 2 else f1[0]
        
        # Store macro F1 for this threshold
        metrics[f'macro_f1_th_{thresh}'] = macro_f1
        
        # Update best threshold if this is better
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = thresh
            best_threshold_metrics = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision[1] if len(precision) > 1 else 0.0,  # Positive class
                'recall': recall[1] if len(recall) > 1 else 0.0,
                'f1': f1[1] if len(f1) > 1 else 0.0,
                'macro_f1': macro_f1,
                'precision_negative': precision[0],  # Negative class
                'recall_negative': recall[0],
                'f1_negative': f1[0]
            }
    
    # Combine best metrics with threshold exploration results
    metrics.update(best_threshold_metrics)
    metrics['roc_auc'] = auc
    metrics['best_threshold'] = best_threshold
    
    return metrics


def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calculate confusion matrix metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary, not probabilities)
    
    Returns:
        dict: Dictionary with tn, fp, fn, tp
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        tn = fp = fn = tp = 0
    
    return {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp)
    }


def log_metrics_to_mlflow(
    metrics: Dict[str, float],
    prefix: str = "",
    fold: Optional[int] = None,
    epoch: Optional[int] = None
) -> None:
    """
    Log metrics to MLflow with appropriate prefixes.
    
    Args:
        metrics: Dictionary of metrics to log
        prefix: Prefix for metric names (e.g., 'train', 'val')
        fold: Fold number (optional)
        epoch: Epoch number (optional)
    """
    metric_dict = {}
    
    for key, value in metrics.items():
        # Build metric name
        parts = []
        if fold is not None:
            parts.append(f"fold_{fold+1}")
        if epoch is not None:
            parts.append(f"epoch_{epoch+1}")
        if prefix:
            parts.append(prefix)
        parts.append(key)
        
        metric_name = "_".join(parts)
        metric_dict[metric_name] = value
    
    # Log to MLflow
    mlflow.log_metrics(metric_dict)


def aggregate_fold_metrics(fold_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across k-folds.
    
    Calculates mean and standard deviation for each metric.
    
    Args:
        fold_results: List of metric dictionaries, one per fold
    
    Returns:
        dict: Aggregated metrics with mean and std for each metric
    """
    if not fold_results:
        return {}
    
    # Get all metric keys (excluding threshold-specific ones for aggregation)
    metric_keys = [k for k in fold_results[0].keys() 
                   if not k.startswith('macro_f1_th_') and k != 'best_threshold']
    
    aggregated = {}
    
    for key in metric_keys:
        values = [fr[key] for fr in fold_results]
        aggregated[f'mean_{key}'] = np.mean(values)
        aggregated[f'std_{key}'] = np.std(values)
    
    # Also aggregate best threshold
    if 'best_threshold' in fold_results[0]:
        thresholds = [fr['best_threshold'] for fr in fold_results]
        aggregated['mean_best_threshold'] = np.mean(thresholds)
        aggregated['std_best_threshold'] = np.std(thresholds)
    
    return aggregated


def print_metrics_summary(
    metrics: Dict[str, float],
    stage_name: str = "Model",
    show_threshold_exploration: bool = False
) -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        stage_name: Name of the stage (e.g., 'Training', 'Validation', 'Teacher', 'Student')
        show_threshold_exploration: Whether to show all threshold exploration results
    """
    print(f"\n{'='*60}")
    print(f"{stage_name.upper()} METRICS")
    print('='*60)
    
    # Core metrics
    if 'accuracy' in metrics:
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
    if 'macro_f1' in metrics:
        print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    if 'best_threshold' in metrics:
        print(f"Best Threshold:  {metrics['best_threshold']:.2f}")
    
    print(f"\nPositive Class (Hate Speech):")
    if 'precision' in metrics:
        print(f"  Precision:     {metrics['precision']:.4f}")
    if 'recall' in metrics:
        print(f"  Recall:        {metrics['recall']:.4f}")
    if 'f1' in metrics:
        print(f"  F1 Score:      {metrics['f1']:.4f}")
    
    print(f"\nNegative Class (Non-Hate):")
    if 'precision_negative' in metrics:
        print(f"  Precision:     {metrics['precision_negative']:.4f}")
    if 'recall_negative' in metrics:
        print(f"  Recall:        {metrics['recall_negative']:.4f}")
    if 'f1_negative' in metrics:
        print(f"  F1 Score:      {metrics['f1_negative']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"\nROC-AUC:         {metrics['roc_auc']:.4f}")
    
    if 'loss' in metrics:
        print(f"Loss:            {metrics['loss']:.4f}")
    
    # Threshold exploration (optional)
    if show_threshold_exploration:
        print(f"\nThreshold Exploration:")
        threshold_keys = [k for k in metrics.keys() if k.startswith('macro_f1_th_')]
        for key in sorted(threshold_keys):
            thresh = key.split('_')[-1]
            print(f"  Threshold {thresh}: Macro F1 = {metrics[key]:.4f}")
    
    print('='*60)


def print_fold_summary(
    fold: int,
    metrics: Dict[str, float],
    best_epoch: int,
    num_folds: int = 5
) -> None:
    """
    Print summary for a single fold.
    
    Args:
        fold: Fold number (0-indexed)
        metrics: Best metrics for this fold
        best_epoch: Epoch where best metrics were achieved
        num_folds: Total number of folds
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{num_folds} SUMMARY")
    print('='*60)
    print(f"Best Epoch:      {best_epoch}")
    print(f"Val Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"Val Macro F1:    {metrics.get('macro_f1', 0):.4f}")
    print(f"Val F1 (Hate):   {metrics.get('f1', 0):.4f}")
    print(f"Val F1 (Non):    {metrics.get('f1_negative', 0):.4f}")
    print(f"Val ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
    print(f"Best Threshold:  {metrics.get('best_threshold', 0.5):.2f}")
    print('='*60)


def print_experiment_summary(
    best_fold: int,
    best_metrics: Dict[str, float],
    aggregated_metrics: Dict[str, float],
    model_info: Optional[Dict[str, any]] = None
) -> None:
    """
    Print final experiment summary across all folds.
    
    Args:
        best_fold: Index of best performing fold
        best_metrics: Metrics from best fold
        aggregated_metrics: Mean/std metrics across all folds
        model_info: Optional model information (parameters, size, etc.)
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print('='*60)
    
    if model_info:
        print(f"\nModel Information:")
        if 'total_parameters' in model_info:
            print(f"  Total Parameters:     {model_info['total_parameters']:,}")
        if 'trainable_parameters' in model_info:
            print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
        if 'model_size_mb' in model_info:
            print(f"  Model Size:           {model_info['model_size_mb']:.2f} MB")
    
    print(f"\nBest Fold: {best_fold+1}")
    print(f"  Accuracy:    {best_metrics.get('accuracy', 0):.4f}")
    print(f"  Macro F1:    {best_metrics.get('macro_f1', 0):.4f}")
    print(f"  F1 (Hate):   {best_metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC:     {best_metrics.get('roc_auc', 0):.4f}")
    
    print(f"\nAcross All Folds (Mean ± Std):")
    print(f"  Accuracy:    {aggregated_metrics.get('mean_accuracy', 0):.4f} ± {aggregated_metrics.get('std_accuracy', 0):.4f}")
    print(f"  Macro F1:    {aggregated_metrics.get('mean_macro_f1', 0):.4f} ± {aggregated_metrics.get('std_macro_f1', 0):.4f}")
    print(f"  F1 (Hate):   {aggregated_metrics.get('mean_f1', 0):.4f} ± {aggregated_metrics.get('std_f1', 0):.4f}")
    print(f"  F1 (Non):    {aggregated_metrics.get('mean_f1_negative', 0):.4f} ± {aggregated_metrics.get('std_f1_negative', 0):.4f}")
    print(f"  ROC-AUC:     {aggregated_metrics.get('mean_roc_auc', 0):.4f} ± {aggregated_metrics.get('std_roc_auc', 0):.4f}")
    
    print('='*60)


def compare_stage_metrics(
    baseline_metrics: Dict[str, float],
    compressed_metrics: Dict[str, float],
    stage_name: str = "Compressed"
) -> None:
    """
    Compare metrics between baseline and compressed model.
    
    Args:
        baseline_metrics: Metrics from baseline model
        compressed_metrics: Metrics from compressed model
        stage_name: Name of compression stage (e.g., 'Student', 'Pruned', 'Quantized')
    """
    print(f"\n{'='*60}")
    print(f"BASELINE vs {stage_name.upper()} COMPARISON")
    print('='*60)
    
    metrics_to_compare = ['accuracy', 'macro_f1', 'f1', 'f1_negative', 'roc_auc']
    
    for metric in metrics_to_compare:
        if metric in baseline_metrics and metric in compressed_metrics:
            baseline_val = baseline_metrics[metric]
            compressed_val = compressed_metrics[metric]
            diff = compressed_val - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            
            print(f"{metric:20s}: {baseline_val:.4f} → {compressed_val:.4f} "
                  f"({symbol} {abs(diff):.4f}, {diff_pct:+.2f}%)")
    
    print('='*60)


# Utility function for quick metric calculation
def quick_evaluate(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Tuple[float, float, float]:
    """
    Quick evaluation returning only the most important metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Predicted probabilities
    
    Returns:
        tuple: (accuracy, macro_f1, best_threshold)
    """
    metrics = calculate_metrics_with_threshold_exploration(y_true, y_pred_probs)
    return metrics['accuracy'], metrics['macro_f1'], metrics['best_threshold']
