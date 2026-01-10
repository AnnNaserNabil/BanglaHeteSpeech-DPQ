import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath('src'))

from data import HateSpeechDataset

def test_dual_tokenization():
    print("Testing Dual Tokenization...")
    
    # Mock data
    comments = ["This is a test comment.", "Another hate speech example."]
    labels = np.array([0, 1])
    
    # Load two different tokenizers
    print("Loading tokenizers...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    
    # Initialize dataset with both
    dataset = HateSpeechDataset(
        comments, 
        labels, 
        teacher_tokenizer, 
        max_length=32, 
        student_tokenizer=student_tokenizer
    )
    
    # Get an item
    item = dataset[0]
    
    # Verify keys
    print("\nChecking keys in dataset item:")
    required_keys = ['input_ids', 'attention_mask', 'labels', 'student_input_ids', 'student_attention_mask']
    for key in required_keys:
        if key in item:
            print(f"  [OK] {key} found")
        else:
            print(f"  [FAIL] {key} missing!")
            return False
            
    # Verify shapes
    print("\nChecking shapes:")
    print(f"  Teacher input_ids: {item['input_ids'].shape}")
    print(f"  Student input_ids: {item['student_input_ids'].shape}")
    
    if item['input_ids'].shape != (32,):
        print("  [FAIL] Teacher input_ids shape incorrect")
        return False
        
    if item['student_input_ids'].shape != (32,):
        print("  [FAIL] Student input_ids shape incorrect")
        return False
        
    # Verify content difference (mBERT vs DistilBERT might be similar but let's check)
    # Actually mBERT and DistilBERT-multilingual share the same vocab usually, 
    # so IDs might be identical. Let's use a different one for student to force difference.
    
    print("\nTesting with a clearly different tokenizer for student...")
    diff_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Different vocab
    
    dataset_diff = HateSpeechDataset(
        comments, 
        labels, 
        teacher_tokenizer, 
        max_length=32, 
        student_tokenizer=diff_tokenizer
    )
    
    item_diff = dataset_diff[0]
    
    t_ids = item_diff['input_ids']
    s_ids = item_diff['student_input_ids']
    
    print(f"  Teacher IDs (first 5): {t_ids[:5].tolist()}")
    print(f"  Student IDs (first 5): {s_ids[:5].tolist()}")
    
    if not torch.equal(t_ids, s_ids):
        print("  [OK] Token IDs are different as expected.")
    else:
        print("  [WARNING] Token IDs are identical (might be coincidence or same vocab).")
        
    print("\nDual tokenization test passed!")
    return True

if __name__ == "__main__":
    try:
        if test_dual_tokenization():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
