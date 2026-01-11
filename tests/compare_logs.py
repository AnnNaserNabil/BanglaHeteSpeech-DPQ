"""
Log Comparison Script
=====================

Compares logs from root and subfolder baseline experiments to identify
differences in:
- Data loading statistics
- Training metrics per epoch
- Final evaluation results
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

def extract_data_stats(log_content: str) -> Dict:
    """Extract data loading statistics from log."""
    stats = {}
    
    # Extract total samples
    match = re.search(r'Total samples loaded:\s*(\d+)', log_content)
    if match:
        stats['total_samples'] = int(match.group(1))
    
    # Extract class distribution
    match = re.search(r'Hate speech:\s*(\d+)\s*\(([\d.]+)%\)', log_content)
    if match:
        stats['hate_count'] = int(match.group(1))
        stats['hate_percent'] = float(match.group(2))
    
    match = re.search(r'Non-hate speech:\s*(\d+)\s*\(([\d.]+)%\)', log_content)
    if match:
        stats['non_hate_count'] = int(match.group(1))
        stats['non_hate_percent'] = float(match.group(2))
    
    return stats

def extract_fold_metrics(log_content: str) -> List[Dict]:
    """Extract metrics for each fold."""
    folds = []
    
    # Find all fold sections
    fold_pattern = r'FOLD (\d+)/(\d+).*?(?=FOLD \d+/\d+|FINAL RESULTS|$)'
    fold_matches = re.finditer(fold_pattern, log_content, re.DOTALL)
    
    for match in fold_matches:
        fold_num = int(match.group(1))
        fold_content = match.group(0)
        
        fold_data = {
            'fold_number': fold_num,
            'epochs': []
        }
        
        # Extract epoch metrics
        epoch_pattern = r'Epoch (\d+)/(\d+).*?Val Loss: ([\d.]+).*?Val Acc: ([\d.]+).*?Val F1: ([\d.]+)'
        epoch_matches = re.finditer(epoch_pattern, fold_content, re.DOTALL)
        
        for epoch_match in epoch_matches:
            epoch_data = {
                'epoch': int(epoch_match.group(1)),
                'val_loss': float(epoch_match.group(3)),
                'val_acc': float(epoch_match.group(4)),
                'val_f1': float(epoch_match.group(5))
            }
            fold_data['epochs'].append(epoch_data)
        
        # Extract best epoch for this fold
        best_match = re.search(r'Best Epoch: (\d+)', fold_content)
        if best_match:
            fold_data['best_epoch'] = int(best_match.group(1))
        
        # Extract final fold metrics
        metrics_pattern = r'Accuracy: ([\d.]+).*?Precision: ([\d.]+).*?Recall: ([\d.]+).*?F1 Score: ([\d.]+)'
        metrics_match = re.search(metrics_pattern, fold_content, re.DOTALL)
        if metrics_match:
            fold_data['final_metrics'] = {
                'accuracy': float(metrics_match.group(1)),
                'precision': float(metrics_match.group(2)),
                'recall': float(metrics_match.group(3)),
                'f1': float(metrics_match.group(4))
            }
        
        folds.append(fold_data)
    
    return folds

def extract_final_results(log_content: str) -> Dict:
    """Extract final averaged results."""
    results = {}
    
    # Extract mean metrics
    patterns = {
        'mean_accuracy': r'Mean Accuracy:\s*([\d.]+)',
        'mean_precision': r'Mean Precision:\s*([\d.]+)',
        'mean_recall': r'Mean Recall:\s*([\d.]+)',
        'mean_f1': r'Mean F1 Score:\s*([\d.]+)',
        'std_accuracy': r'Std Accuracy:\s*([\d.]+)',
        'std_f1': r'Std F1 Score:\s*([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, log_content)
        if match:
            results[key] = float(match.group(1))
    
    return results

def compare_logs(root_log_path: str, subfolder_log_path: str) -> Dict:
    """Compare two experiment logs and return differences."""
    
    # Read log files
    with open(root_log_path, 'r') as f:
        root_content = f.read()
    
    with open(subfolder_log_path, 'r') as f:
        subfolder_content = f.read()
    
    comparison = {
        'root': {},
        'subfolder': {},
        'differences': []
    }
    
    # Extract data stats
    print("Extracting data statistics...")
    comparison['root']['data_stats'] = extract_data_stats(root_content)
    comparison['subfolder']['data_stats'] = extract_data_stats(subfolder_content)
    
    # Extract fold metrics
    print("Extracting fold metrics...")
    comparison['root']['folds'] = extract_fold_metrics(root_content)
    comparison['subfolder']['folds'] = extract_fold_metrics(subfolder_content)
    
    # Extract final results
    print("Extracting final results...")
    comparison['root']['final_results'] = extract_final_results(root_content)
    comparison['subfolder']['final_results'] = extract_final_results(subfolder_content)
    
    # Identify differences
    print("\nAnalyzing differences...")
    
    # Data loading differences
    root_data = comparison['root']['data_stats']
    sub_data = comparison['subfolder']['data_stats']
    
    if root_data.get('total_samples') != sub_data.get('total_samples'):
        comparison['differences'].append({
            'category': 'data_loading',
            'issue': 'Different number of samples',
            'root_value': root_data.get('total_samples'),
            'subfolder_value': sub_data.get('total_samples')
        })
    
    # Final metrics differences
    root_final = comparison['root']['final_results']
    sub_final = comparison['subfolder']['final_results']
    
    for metric in ['mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1']:
        if metric in root_final and metric in sub_final:
            diff = abs(root_final[metric] - sub_final[metric])
            if diff > 0.001:  # Threshold for significant difference
                comparison['differences'].append({
                    'category': 'final_metrics',
                    'metric': metric,
                    'root_value': root_final[metric],
                    'subfolder_value': sub_final[metric],
                    'difference': diff
                })
    
    return comparison

def print_comparison_report(comparison: Dict):
    """Print a formatted comparison report."""
    
    print("\n" + "="*70)
    print("BASELINE FINETUNING COMPARISON REPORT")
    print("="*70)
    
    # Data Statistics
    print("\n📊 DATA STATISTICS")
    print("-" * 70)
    
    root_data = comparison['root']['data_stats']
    sub_data = comparison['subfolder']['data_stats']
    
    print(f"\n{'Metric':<30} {'Root':<20} {'Subfolder':<20}")
    print("-" * 70)
    print(f"{'Total Samples':<30} {root_data.get('total_samples', 'N/A'):<20} {sub_data.get('total_samples', 'N/A'):<20}")
    print(f"{'Hate Count':<30} {root_data.get('hate_count', 'N/A'):<20} {sub_data.get('hate_count', 'N/A'):<20}")
    print(f"{'Non-Hate Count':<30} {root_data.get('non_hate_count', 'N/A'):<20} {sub_data.get('non_hate_count', 'N/A'):<20}")
    
    # Final Results
    print("\n📈 FINAL RESULTS")
    print("-" * 70)
    
    root_final = comparison['root']['final_results']
    sub_final = comparison['subfolder']['final_results']
    
    print(f"\n{'Metric':<30} {'Root':<20} {'Subfolder':<20} {'Difference':<15}")
    print("-" * 70)
    
    for metric in ['mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1']:
        root_val = root_final.get(metric, 0)
        sub_val = sub_final.get(metric, 0)
        diff = abs(root_val - sub_val)
        print(f"{metric:<30} {root_val:<20.4f} {sub_val:<20.4f} {diff:<15.4f}")
    
    # Differences Summary
    print("\n⚠️  IDENTIFIED DIFFERENCES")
    print("-" * 70)
    
    if comparison['differences']:
        for i, diff in enumerate(comparison['differences'], 1):
            print(f"\n{i}. {diff['category'].upper()}: {diff.get('issue', diff.get('metric', 'Unknown'))}")
            if 'root_value' in diff and 'subfolder_value' in diff:
                print(f"   Root: {diff['root_value']}")
                print(f"   Subfolder: {diff['subfolder_value']}")
                if 'difference' in diff:
                    print(f"   Difference: {diff['difference']:.4f}")
    else:
        print("\n✅ No significant differences found!")
    
    print("\n" + "="*70)

def save_comparison_json(comparison: Dict, output_path: str):
    """Save comparison results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n💾 Detailed comparison saved to: {output_path}")

def main():
    """Main comparison function."""
    
    root_log = 'root_baseline_experiment.log'
    subfolder_log = 'subfolder_baseline_experiment.log'
    output_json = 'baseline_comparison_results.json'
    
    # Check if log files exist
    if not Path(root_log).exists():
        print(f"❌ Error: {root_log} not found!")
        return
    
    if not Path(subfolder_log).exists():
        print(f"❌ Error: {subfolder_log} not found!")
        return
    
    print("Starting log comparison...")
    print(f"Root log: {root_log}")
    print(f"Subfolder log: {subfolder_log}")
    
    # Compare logs
    comparison = compare_logs(root_log, subfolder_log)
    
    # Print report
    print_comparison_report(comparison)
    
    # Save to JSON
    save_comparison_json(comparison, output_json)
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()
