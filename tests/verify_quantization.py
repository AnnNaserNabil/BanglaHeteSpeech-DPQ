import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath('src'))

from model import TransformerBinaryClassifier
from quantization import QuantizationManager, quantize_model

class MockConfig:
    def __init__(self):
        self.quant_dtype = 'int8'
        self.quant_calibration_batches = 1

def test_hf_model_quantization():
    print("\n--- Testing Standard HF Model Quantization ---")
    model_name = "distilbert-base-multilingual-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    config = MockConfig()
    
    # Test Dynamic Quantization
    print("Testing Dynamic Quantization...")
    manager = QuantizationManager(model, method='dynamic')
    quantized = manager.apply_dynamic_quantization()
    manager.compare_sizes()
    
    # Test FP16
    print("Testing FP16 Conversion...")
    manager_fp16 = QuantizationManager(model, method='fp16')
    quantized_fp16 = manager_fp16.apply_fp16_quantization(device='cpu')
    manager_fp16.compare_sizes()
    
    print("✅ Standard HF Model Quantization tests passed!")

def test_custom_model_quantization():
    print("\n--- Testing Custom TransformerBinaryClassifier Quantization ---")
    model_name = "distilbert-base-multilingual-cased"
    model = TransformerBinaryClassifier(model_name)
    
    # Test save/load
    print("Testing save_pretrained/from_pretrained...")
    temp_path = "temp_test_model"
    model.save_pretrained(temp_path)
    loaded_model = TransformerBinaryClassifier.from_pretrained(temp_path)
    
    # Verify weights loaded (simple check)
    assert type(loaded_model) == TransformerBinaryClassifier
    print("✅ Save/Load passed!")
    
    # Test Quantization
    print("Testing Dynamic Quantization on Custom Model...")
    manager = QuantizationManager(model, method='dynamic')
    quantized = manager.apply_dynamic_quantization()
    manager.compare_sizes()
    
    # Clean up
    import shutil
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        
    print("✅ Custom Model Quantization tests passed!")

if __name__ == "__main__":
    try:
        test_hf_model_quantization()
        test_custom_model_quantization()
        print("\n🎉 ALL QUANTIZATION VERIFICATION TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
