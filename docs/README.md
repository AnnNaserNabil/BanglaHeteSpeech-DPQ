# Documentation Index

Welcome to the comprehensive documentation for the Enhanced Bangla BERT Hate Speech Detection with Compression Pipeline.

## Quick Links

- [Main README](../README.md) - Project overview and quick start
- [Quick Start Guide](../QUICK_START.md) - Common commands and usage
- [Metrics Guide](../METRICS_GUIDE.md) - Metrics tracking and MLflow integration
- [CSV Format Reference](../CSV_FORMAT.md) - CSV export format specification

## Module Documentation

### Core Modules
- [config.md](config.md) - Configuration system and pipeline modes
- [data.md](data.md) - Data loading, preprocessing, and k-fold splitting
- [model.md](model.md) - Model architectures (Teacher, Student, Baseline)
- [train.md](train.md) - Baseline training with k-fold cross-validation
- [metrics.md](metrics.md) - Enhanced metrics calculation with threshold exploration

### Compression Modules
- [distillation.md](distillation.md) - Knowledge distillation implementation
- [pruning.md](pruning.md) - Pruning methods (magnitude, WANDA, gradual)
- [quantization.md](quantization.md) - Quantization methods (dynamic, static, FP16, INT4)
- [evaluation.md](evaluation.md) - Model evaluation and comparison

### Pipeline
- [pipeline.md](pipeline.md) - Pipeline orchestrator and stage integration
- [utils.md](utils.md) - Utility functions and HuggingFace model saving

## Comprehensive Guides

- [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) - Complete pipeline guide
- [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) - Technical implementation details
- [PIPELINE_MODES.md](PIPELINE_MODES.md) - All 8 pipeline mode configurations

## Architecture

```
Enhanced Modular Pipeline
├── Baseline Training (train.py)
│   ├── K-fold cross-validation
│   ├── Threshold exploration
│   ├── Enhanced metrics
│   └── MLflow tracking
│
├── Knowledge Distillation (distillation.py)
│   ├── Teacher model
│   ├── Student model
│   ├── Distillation loss
│   └── Multi-level KD
│
├── Pruning (pruning.py)
│   ├── Magnitude pruning
│   ├── WANDA pruning
│   ├── Gradual pruning
│   └── Fine-tuning
│
└── Quantization (quantization.py)
    ├── Dynamic INT8
    ├── Static INT8
    ├── FP16
    └── INT4
```

## Getting Started

1. **Installation**: See [README.md](../README.md#installation)
2. **Quick Start**: See [QUICK_START.md](../QUICK_START.md)
3. **Configuration**: See [config.md](config.md)
4. **Running Pipelines**: See [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)

## Key Features

- ✅ **8 Pipeline Modes**: From baseline-only to full compression
- ✅ **Threshold Exploration**: Automatic optimal threshold selection
- ✅ **MLflow Integration**: Comprehensive experiment tracking
- ✅ **HuggingFace Compatible**: Deploy models directly
- ✅ **Enhanced Metrics**: Per-class metrics at every stage
- ✅ **Modular Design**: Use stages independently or together

## Contributing

When adding new features:
1. Update relevant module documentation
2. Add examples to PIPELINE_DOCUMENTATION.md
3. Update TECHNICAL_DOCS.md with implementation details
4. Add tests and verification steps

## Support

For issues or questions:
1. Check module documentation
2. Review PIPELINE_DOCUMENTATION.md
3. Check TECHNICAL_DOCS.md for implementation details
4. Review example commands in QUICK_START.md
