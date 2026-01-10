# 📘 Script 5: `quantization.py`

## Overview

This script implements **quantization** - the technique of reducing the numerical precision of model weights and activations. Instead of using 32-bit floating-point numbers, we use smaller representations like 16-bit, 8-bit, or even 4-bit numbers.

**Why quantization is the "last mile" of compression:**
- Provides 2-8× size reduction with minimal code changes
- Often the easiest compression technique to apply
- Works well combined with KD and pruning for maximum compression
- Critical for deploying to mobile devices, CPUs, and edge hardware

---

## The Big Picture: What Quantization Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ QUANTIZATION: REDUCING NUMERICAL PRECISION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ORIGINAL WEIGHT (FP32 - 32 bits):                                          │
│                                                                             │
│   0.123456789012345678901234567890123                                      │
│   └──────────────── 32 bits ─────────────────┘                             │
│                                                                             │
│   Range: ±3.4 × 10³⁸                                                        │
│   Precision: ~7 decimal digits                                             │
│   Memory: 4 bytes per weight                                               │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QUANTIZED TO FP16 (16 bits):                                               │
│                                                                             │
│   0.1235                                                                    │
│   └── 16 bits ──┘                                                           │
│                                                                             │
│   Range: ±65,504                                                            │
│   Precision: ~3-4 decimal digits                                           │
│   Memory: 2 bytes per weight (2× compression)                              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QUANTIZED TO INT8 (8 bits):                                                │
│                                                                             │
│   31 (representing ~0.12 after scaling)                                    │
│   └ 8 bits ┘                                                                │
│                                                                             │
│   Range: -128 to 127 (mapped to original weight range)                     │
│   Memory: 1 byte per weight (4× compression)                               │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QUANTIZED TO INT4 (4 bits):                                                │
│                                                                             │
│   7 (representing ~0.12 after scaling)                                     │
│   └4b┘                                                                      │
│                                                                             │
│   Range: -8 to 7 (or 0 to 15 unsigned)                                     │
│   Memory: 0.5 bytes per weight (8× compression!)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 1: Imports and Setup

```python
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm
import copy
import warnings
import time
```

### Key Imports Explained:

| Import | Purpose |
|--------|---------|
| `quantize_dynamic` | Dynamic quantization (easiest method) |
| `prepare`, `convert` | Static quantization workflow |
| `QConfig` | Configuration for quantization observers |
| `default_observer` | Tracks activation statistics |
| `default_weight_observer` | Tracks weight statistics |

### Understanding PyTorch Quantization Architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PYTORCH QUANTIZATION SYSTEM                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ OBSERVERS: Track statistics to determine quantization parameters           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ MinMaxObserver                                                      │  │
│   │   Tracks: min and max values                                        │  │
│   │   Use: Simple, works for most cases                                 │  │
│   │   Formula: scale = (max - min) / 255                                │  │
│   │            zero_point = -min / scale                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ MovingAverageMinMaxObserver                                         │  │
│   │   Tracks: Exponential moving average of min/max                     │  │
│   │   Use: When values change over time (training)                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ HistogramObserver                                                   │  │
│   │   Tracks: Full histogram of values                                  │  │
│   │   Use: Most accurate, but slower                                    │  │
│   │   Finds optimal scale by minimizing quantization error              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ QCONFIG: Combines observers for weights and activations                    │
│                                                                             │
│   qconfig = QConfig(                                                        │
│       activation=default_observer,    # How to observe activations         │
│       weight=default_weight_observer  # How to observe weights             │
│   )                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 2: The Quantization Manager Class

```python
class QuantizationManager:
    """
    Manages quantization operations for transformer models.
    """
    
    def __init__(self, model: nn.Module, method: str = 'dynamic', dtype: str = 'int8'):
        """
        Initialize quantization manager.
        
        Args:
            model: Model to quantize
            method: 'dynamic', 'static', 'qat', or 'fp16'
            dtype: 'int8' or 'fp16'
        """
        self.original_model = model
        self.method = method
        self.dtype = dtype
        self.quantized_model = None
        self.calibrated = False
```

### Getting Model Size:

```python
def get_model_size(self, model: Optional[nn.Module] = None) -> Dict:
    """
    Calculate model size in memory.
    
    Returns:
        Dict with size information
    """
    model = model or self.quantized_model or self.original_model
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'param_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2),
        'total_size_mb': total_mb,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

def compare_sizes(self) -> Dict:
    """Compare original and quantized model sizes."""
    original_size = self.get_model_size(self.original_model)
    quantized_size = self.get_model_size(self.quantized_model)
    
    compression = original_size['total_size_mb'] / quantized_size['total_size_mb']
    
    print(f"\n📊 Quantization Results:")
    print(f"   Original size: {original_size['total_size_mb']:.2f} MB")
    print(f"   Quantized size: {quantized_size['total_size_mb']:.2f} MB")
    print(f"   Compression ratio: {compression:.2f}×")
    
    return {
        'original_size_mb': original_size['total_size_mb'],
        'quantized_size_mb': quantized_size['total_size_mb'],
        'compression_ratio': compression,
        'size_reduction_pct': (1 - 1/compression) * 100
    }
```

### Understanding `element_size()`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA TYPES AND THEIR SIZES                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Data Type      │ Bits │ Bytes │ element_size() │ Range                     │
│ ───────────────┼──────┼───────┼────────────────┼─────────────────────────  │
│ torch.float32  │  32  │   4   │       4        │ ±3.4 × 10³⁸              │
│ torch.float16  │  16  │   2   │       2        │ ±65,504                   │
│ torch.bfloat16 │  16  │   2   │       2        │ ±3.4 × 10³⁸ (less precise)│
│ torch.int8     │   8  │   1   │       1        │ -128 to 127               │
│ torch.qint8    │   8  │   1   │       1        │ -128 to 127 (quantized)   │
│ torch.quint8   │   8  │   1   │       1        │ 0 to 255 (unsigned)       │
│                                                                             │
│ Example calculation:                                                        │
│   BanglaBERT: 110M parameters × 4 bytes = 440 MB                           │
│   After INT8: 110M parameters × 1 byte = 110 MB (4× smaller!)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 3: Dynamic Quantization (The Easiest Method)

Dynamic quantization is the simplest approach. Weights are quantized ahead of time, but activations are quantized dynamically during inference.

```python
def apply_dynamic_quantization(self) -> nn.Module:
    """
    Apply dynamic quantization.
    
    WHAT: Quantizes weights ahead of time; activations quantized at runtime
    WHY: Simple to apply, no calibration needed
    HOW: One function call!
    """
    print("\n   Applying dynamic quantization...")
    
    # Create a copy to avoid modifying original
    model_copy = copy.deepcopy(self.original_model)
    
    # Apply dynamic quantization to Linear layers
    self.quantized_model = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8
    )
    
    return self.quantized_model
```

### Visual: How Dynamic Quantization Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DYNAMIC QUANTIZATION WORKFLOW                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ STEP 1: QUANTIZE WEIGHTS (done once, at model load time)                   │
│                                                                             │
│   Original weight (FP32):  [0.123, -0.456, 0.789, ...]                     │
│                                                                             │
│   Compute scale and zero_point:                                            │
│     min_val = -0.456, max_val = 0.789                                      │
│     scale = (0.789 - (-0.456)) / 255 = 0.00488                            │
│     zero_point = round(-(-0.456) / 0.00488) = 93                          │
│                                                                             │
│   Quantize:                                                                 │
│     q_weight = round(weight / scale) + zero_point                          │
│     q_weight = [118, 0, 255, ...]  (INT8 values)                          │
│                                                                             │
│   Store: q_weight (INT8) + scale (FP32) + zero_point (INT32)              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ STEP 2: INFERENCE (happens for each input)                                 │
│                                                                             │
│   Input activation (FP32):  [1.5, -0.3, 2.1, ...]                         │
│                                                                             │
│   Dynamically quantize activation:                                         │
│     Compute min/max of THIS batch                                          │
│     Compute scale and zero_point                                           │
│     q_activation = [...]  (INT8)                                           │
│                                                                             │
│   Matrix multiply in INT8:                                                  │
│     q_output = q_activation @ q_weight                                     │
│     (This is the fast part! INT8 ops are 2-4× faster than FP32)           │
│                                                                             │
│   Dequantize output back to FP32:                                          │
│     output = (q_output - zero_point) × scale                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Effect | When to Use |
|--------------|--------|-------------|
| `{nn.Linear, nn.Conv2d}` | Quantize Conv layers too | If model has convolutions |
| `dtype=torch.quint8` | Unsigned quantization | When values are always positive |

---

## Section 4: Static Quantization (Better Accuracy)

Static quantization pre-computes quantization parameters for both weights AND activations using calibration data. This gives better accuracy but requires representative data.

```python
def prepare_static_quantization(self, backend: str = 'fbgemm'):
    """
    Prepare model for static quantization.
    
    WHAT: Inserts quantization observers into the model
    WHY: Need to collect statistics for calibration
    HOW: Replace modules with quantization-aware versions
    
    Args:
        backend: 'fbgemm' for x86 CPU, 'qnnpack' for ARM
    """
    print(f"\n   Preparing static quantization (backend: {backend})...")
    
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Get quantization config
    self.original_model.qconfig = quant.get_default_qconfig(backend)
    
    # Prepare model (inserts observers)
    self.quantized_model = quant.prepare(
        copy.deepcopy(self.original_model),
        inplace=False
    )
    
    print("   ✅ Model prepared for static quantization")
```

### Calibration Step:

```python
def calibrate(self, dataloader, device: str, num_batches: int = 100, use_student_input_ids: bool = False):
    """
    Calibrate static quantization with representative data.
    
    WHAT: Run forward passes to collect activation statistics
    WHY: Need to know the range of activations for quantization
    HOW: Run inference on calibration data, observers record min/max values
    
    Args:
        dataloader: DataLoader with calibration data
        device: Must be 'cpu' for static quantization
        num_batches: Number of batches for calibration
        use_student_input_ids: Whether to use student tokenization
    """
    if self.quantized_model is None:
        raise ValueError("Must call prepare_static_quantization() first!")
    
    print(f"\n   Calibrating on {num_batches} batches...")
    
    # MUST be on CPU for calibration
    self.quantized_model.to('cpu')
    self.quantized_model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if i >= num_batches:
                break
            
            if use_student_input_ids and 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to('cpu')
                attention_mask = batch['student_attention_mask'].to('cpu')
            else:
                input_ids = batch['input_ids'].to('cpu')
                attention_mask = batch['attention_mask'].to('cpu')
            
            self.quantized_model(input_ids, attention_mask)
    
    self.calibrated = True
    print("   ✅ Calibration complete")
```

### Converting to Quantized Model:

```python
def convert_static_quantization(self) -> nn.Module:
    """
    Convert prepared model to quantized model.
    
    Must call after calibrate()!
    """
    if not self.calibrated:
        raise ValueError("Must call calibrate() first!")
    
    print("\n   Converting to quantized model...")
    
    self.quantized_model = quant.convert(
        self.quantized_model,
        inplace=False
    )
    
    print("   ✅ Static quantization complete")
    return self.quantized_model
```

### Visual: Static vs Dynamic Quantization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STATIC VS DYNAMIC QUANTIZATION                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ DYNAMIC QUANTIZATION:                                                       │
│                                                                             │
│   Preparation:  None needed                                                │
│   Calibration:  None needed                                                │
│                                                                             │
│   Inference (each batch):                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Input → [Compute activation stats] → [Quantize activation] →        │  │
│   │         [INT8 matmul with pre-quantized weights] → [Dequantize] → Out│  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                    ↑                                                        │
│                    Extra computation at runtime                             │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ STATIC QUANTIZATION:                                                        │
│                                                                             │
│   Preparation:  Run calibration data once                                  │
│   Calibration:  Observers track min/max of all activations                 │
│                                                                             │
│   Inference (each batch):                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Input → [Quantize with PRE-COMPUTED params] →                        │  │
│   │         [INT8 matmul] → [Dequantize] → Output                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│              ↑                                                              │
│              No runtime stat computation (faster!)                         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ COMPARISON:                                                                 │
│                                                                             │
│                    │ Dynamic        │ Static                               │
│   ─────────────────┼────────────────┼───────────────────────────────────── │
│   Calibration      │ Not needed     │ Required (100+ samples)              │
│   Accuracy         │ Good           │ Better (more accurate params)        │
│   Speed            │ Fast           │ Faster (no runtime stats)            │
│   Flexibility      │ High           │ Lower (fixed activation range)       │
│   Best for         │ Variable data  │ Consistent data distribution         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---


## Section 5: FP16 Quantization (GPU-Friendly)

Unlike INT8 which only works on CPU in PyTorch, FP16 works on GPU and is much simpler.

```python
def apply_fp16_quantization(self, device: str = 'cuda') -> nn.Module:
    """
    Convert model to FP16 (half precision).
    
    WHAT: Convert all parameters from 32-bit to 16-bit floats
    WHY: 2× smaller, 2× faster on GPU with Tensor Cores
    HOW: Just call model.half()!
    """
    print("\n   Applying FP16 (half precision) conversion...")
    
    # Simple! Just convert to half precision
    self.quantized_model = copy.deepcopy(self.original_model)
    self.quantized_model = self.quantized_model.half().to(device)
    
    print("   ✅ FP16 conversion complete")
    return self.quantized_model
```

### FP32 vs FP16 vs BF16:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FLOATING POINT FORMATS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ FP32 (Single Precision):                                                   │
│ ┌─────┬────────────────────┬───────────────────────────────────────────┐   │
│ │Sign │    Exponent (8)    │           Mantissa (23)                   │   │
│ │  1  │    8 bits          │           23 bits                         │   │
│ └─────┴────────────────────┴───────────────────────────────────────────┘   │
│   Range: ±3.4 × 10³⁸                                                        │
│   Precision: ~7 decimal digits                                             │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ FP16 (Half Precision):                                                     │
│ ┌─────┬───────────┬───────────────────┐                                    │
│ │Sign │ Exp (5)   │   Mantissa (10)   │                                    │
│ │  1  │  5 bits   │    10 bits        │                                    │
│ └─────┴───────────┴───────────────────┘                                    │
│   Range: ±65,504 (MUCH smaller!)                                           │
│   Precision: ~3 decimal digits                                             │
│   Good for: Most neural network computations                               │
│   Bad for: Loss computation, gradient accumulation                         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ BF16 (Brain Float):                                                         │
│ ┌─────┬────────────────────┬───────────┐                                   │
│ │Sign │    Exponent (8)    │ Mant (7)  │                                   │
│ │  1  │    8 bits          │  7 bits   │                                   │
│ └─────┴────────────────────┴───────────┘                                   │
│   Range: ±3.4 × 10³⁸ (same as FP32!)                                       │
│   Precision: ~2 decimal digits (less than FP16)                            │
│   Good for: Training (same range as FP32, less overflow risk)              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ PRACTICAL ADVICE:                                                           │
│                                                                             │
│   Training:   Use BF16 or mixed precision (FP16 + FP32 where needed)       │
│   Inference:  Use FP16 (best speed/accuracy tradeoff)                      │
│   Edge/Mobile: Use INT8 (maximum compression)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 6: INT4 Quantization (Maximum Compression)

INT4 provides 8× compression but requires special libraries (bitsandbytes) and careful implementation.

```python
def apply_int4_quantization(model: nn.Module, device: str) -> nn.Module:
    """
    Apply INT4 quantization using bitsandbytes.
    
    WHAT: Quantizes model to 4-bit precision (NF4)
    WHY: Extreme compression (smaller than INT8)
    HOW: Saves model to temp dir, reloads with load_in_4bit=True
    """
    print("\n   Applying INT4 quantization (via bitsandbytes)...")
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    from distillation import StudentModel, TeacherModel

    # 1. Save temporary model for reloading
    temp_dir = "temp_int4_conversion"
    model.save_pretrained(temp_dir)
    
    # 2. Reload with INT4 quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    if isinstance(model, StudentModel):
        quantized_model = StudentModel.from_pretrained(
            temp_dir,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        from transformers import AutoModel
        quantized_model = AutoModel.from_pretrained(
            temp_dir,
            quantization_config=quantization_config,
            device_map=device
        )
    
    return quantized_model
```

### Visual: NF4 Quantization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ NF4 (NORMALIZED FLOAT 4-BIT) QUANTIZATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PROBLEM WITH UNIFORM INT4:                                                  │
│                                                                             │
│   Weight distribution is typically GAUSSIAN, not uniform:                  │
│                                                                             │
│         │        ▄▄                                                         │
│   Count │       ████                                                        │
│         │      ██████                                                       │
│         │     ████████                                                      │
│         │    ██████████                                                     │
│         │   ████████████                                                    │
│         │  ██████████████                                                   │
│         │ ████████████████                                                  │
│         └──────────────────────                                             │
│           -1.0       0       +1.0                                           │
│                                                                             │
│   Uniform INT4: 16 evenly spaced values across range                       │
│   Problem: Most weights are near 0, wasting precision on extremes!         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ NF4 SOLUTION:                                                               │
│                                                                             │
│   Use quantization levels that match the distribution:                     │
│                                                                             │
│   NF4 code book (16 values, not evenly spaced):                            │
│   [-1.0, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0,                           │
│     0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]                                  │
│                                                                             │
│   More levels near 0 (where most weights are) = better accuracy!           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ DOUBLE QUANTIZATION:                                                        │
│                                                                             │
│   Normal quantization stores:                                              │
│     - Quantized weights (INT4)                                             │
│     - Scale factors (FP32) ← Still 32 bits!                                │
│                                                                             │
│   Double quantization:                                                      │
│     - Quantized weights (INT4)                                             │
│     - Quantized scale factors (INT8) ← Even smaller!                       │
│     - Second-level scale (FP32, shared across many weights)                │
│                                                                             │
│   Extra compression at minimal accuracy cost!                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 7: Benchmarking and Comparison

### Inference Benchmarking:

```python
def benchmark_inference_speed(
    model: nn.Module,
    dataloader,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10,
    use_student_input_ids: bool = False
) -> Dict:
    """
    Benchmark inference speed of a model.
    """
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask)
    
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'throughput_samples_per_sec': (batch_size / np.mean(latencies)) * 1000
    }
```

### Memory Profiling:

```python
def profile_memory(model: nn.Module, dataloader, device: str, use_student_input_ids: bool = False) -> Dict:
    """
    Profile peak memory usage on GPU.
    """
    if device != 'cuda':
        return {'peak_memory_mb': 0}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_ids, attention_mask)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {'peak_memory_mb': peak_memory}
```

---

## Section 8: The Main Quantize Function

```python
def quantize_model(
    model: nn.Module,
    method: str,
    config,
    calibration_loader=None,
    device: str = 'cpu',
    use_student_input_ids: bool = False
) -> nn.Module:
    """
    High-level function to quantize a model.
    """
    manager = QuantizationManager(model, method=method, dtype=config.quant_dtype)
    
    if method == 'dynamic':
        quantized_model = manager.apply_dynamic_quantization()
    elif method == 'static':
        manager.prepare_static_quantization()
        manager.calibrate(calibration_loader, 'cpu', config.quant_calibration_batches)
        quantized_model = manager.convert_static_quantization()
    elif method == 'fp16':
        quantized_model = manager.apply_fp16_quantization(device)
    
    manager.compare_sizes()
    return quantized_model
```

---

---

## Summary: What You Can Modify in This Script

| Category | What to Modify | Research Question |
|----------|----------------|-------------------|
| **Method** | dynamic, static, fp16, int4 | Which method gives best accuracy/compression? |
| **Calibration** | Number of batches | How much calibration data is needed? |
| **Backend** | fbgemm vs qnnpack | Best backend for your hardware? |

---

## Key Takeaways

1. **FP16 is the safest choice** - works on GPU, minimal accuracy loss.
2. **INT8 is for CPU deployment** - use dynamic for simplicity, static for speed.
3. **INT4 is for maximum compression** - requires bitsandbytes and GPU.
4. **Always benchmark** - speed gains vary significantly by hardware.

---

**Ready for the next script? The next one is `evaluation.md` which implements comprehensive metrics calculation and comparison across all compression stages.**