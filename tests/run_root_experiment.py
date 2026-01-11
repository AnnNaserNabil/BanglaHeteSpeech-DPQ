"""
Experiment Runner for Root Folder Baseline Finetuning
======================================================

Runs baseline finetuning with controlled parameters:
- Model: sagorsarker/bangla-bert-base
- Epochs: 20
- Folds: 2
- Pipeline: baseline
- Data fraction: 1.0 (full data)

Logs all output to: root_baseline_experiment.log
"""

import subprocess
import sys
import os
from datetime import datetime

def run_experiment():
    """Run root folder baseline experiment with detailed logging."""
    
    # Experiment parameters
    params = {
        'dataset_path': 'data/HateSpeech_minimal.csv',
        'author_name': 'baseline_comparison_root',
        'pipeline': 'baseline',
        'epochs': 20,
        'num_folds': 2,
        'model_path': 'sagorsarker/bangla-bert-base',
        'data_fraction': 1.0,
        'seed': 42,
        'batch': 32,
        'lr': 2e-5
    }
    
    # Build command
    cmd = [
        sys.executable, 'src/main.py',
        '--dataset_path', params['dataset_path'],
        '--author_name', params['author_name'],
        '--pipeline', params['pipeline'],
        '--epochs', str(params['epochs']),
        '--num_folds', str(params['num_folds']),
        '--model_path', params['model_path'],
        '--data_fraction', str(params['data_fraction']),
        '--seed', str(params['seed']),
        '--batch', str(params['batch']),
        '--lr', str(params['lr'])
    ]
    
    # Log file
    log_file = 'root_baseline_experiment.log'
    
    print("="*70)
    print("ROOT FOLDER BASELINE EXPERIMENT")
    print("="*70)
    print(f"\nExperiment Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"\nLog file: {log_file}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\nStarting experiment...\n")
    
    # Run experiment and capture output
    with open(log_file, 'w') as f:
        # Write header
        f.write("="*70 + "\n")
        f.write("ROOT FOLDER BASELINE EXPERIMENT\n")
        f.write("="*70 + "\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nExperiment Parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.write("="*70 + "\n\n")
        f.flush()
        
        # Run command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output to both console and file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
        
        # Write footer
        f.write("\n" + "="*70 + "\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write("="*70 + "\n")
    
    print("\n" + "="*70)
    print(f"Experiment completed with exit code: {process.returncode}")
    print(f"Log saved to: {log_file}")
    print("="*70)
    
    return process.returncode

if __name__ == "__main__":
    exit_code = run_experiment()
    sys.exit(exit_code)
