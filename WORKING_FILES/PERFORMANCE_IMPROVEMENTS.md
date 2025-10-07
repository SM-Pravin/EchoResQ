# Emergency AI Performance & Scaling Improvements (Phase 2)

This document outlines the comprehensive performance and scaling improvements implemented in Phase 2 of the Emergency AI project.

## 🚀 Overview

All improvements focus on reducing latency, increasing throughput, and supporting concurrent processing while maintaining accuracy and reliability.

## 🔧 Implemented Features

### 1. Enhanced Parallel Chunk Analysis
**Location**: `analysis_pipeline.py`

- **Worker Initialization**: Models are now pre-loaded in worker processes using `ProcessPoolExecutor` initializers
- **Improved Error Handling**: Robust fallback to sequential processing if parallel fails
- **Memory Optimization**: Uses spawn context for better isolation
- **Configuration**: Control via `PARALLEL_MAX_WORKERS` environment variable

```python
# Example usage
os.environ["PARALLEL_MAX_WORKERS"] = "4"
result = process_audio_file("audio.wav")
```

### 2. Batch Audio Inference
**Location**: `modules/emotion_audio.py`

- **Batched WAV2VEC Processing**: Process multiple audio chunks simultaneously
- **Dynamic Batch Sizing**: Configurable batch sizes for optimal performance
- **Fallback Protection**: Graceful degradation to individual processing
- **Memory Efficient**: Prevents OOM errors with large batches

```python
# Batch processing multiple audio inputs
results = analyze_audio_emotion_batch(audio_list, max_batch_size=8)
```

**Configuration**:
```bash
export ENABLE_BATCH_PROCESSING=true
export AUDIO_BATCH_SIZE=8
```

### 3. Streaming Transcription with Early Keyword Detection
**Location**: `modules/speech_to_text.py`

- **Real-time Partial Results**: Emit transcription results as they become available
- **Early Keyword Detection**: Detect emergency keywords before full transcription
- **Priority-based Detection**: Critical, high, and medium priority keyword classification
- **Callback Support**: Custom callbacks for real-time processing

```python
def keyword_callback(keyword, timestamp, text):
    print(f"URGENT: {keyword} detected at {timestamp}s")

def partial_callback(text, timestamp, is_final):
    print(f"Partial: {text}")

result = transcribe_audio_streaming(
    "audio.wav", 
    partial_callback=partial_callback,
    keyword_callback=keyword_callback
)
```

### 4. Device-Aware GPU Support
**Location**: `modules/model_loader.py`

- **Intelligent Device Selection**: Automatically detects optimal device (GPU/CPU)
- **Memory Checking**: Ensures sufficient GPU memory before usage
- **Environment Control**: Force CPU mode when needed
- **Fallback Mechanisms**: Graceful degradation if GPU unavailable

```python
# Force CPU mode
os.environ["FORCE_CPU"] = "true"

# Models automatically use optimal device
```

### 5. Model Optimization & Export
**Location**: `modules/model_optimization.py`

- **ONNX Export**: Convert models to ONNX for faster, cross-platform inference
- **Quantization**: Dynamic quantization for smaller models and faster CPU inference
- **TorchScript**: Compile models for production deployment
- **Batch Operations**: Export multiple models with single command

```python
from modules.model_optimization import optimize_models_for_production

# Optimize all models
optimize_models_for_production()

# Use optimized models
os.environ["USE_OPTIMIZED_MODELS"] = "true"
os.environ["MODEL_OPTIMIZATION_TYPE"] = "onnx"  # or "quantized" or "torchscript"
```

### 6. Comprehensive Benchmarking Suite
**Location**: `benchmarks/`

#### Performance Profiler (`performance_profiler.py`)
- **Detailed Metrics**: Duration, memory usage, CPU utilization
- **Operation Breakdown**: Per-function performance analysis
- **Memory Monitoring**: Real-time memory tracking
- **Automated Reporting**: JSON reports with visualizations

#### Stress Testing (`stress_test.py`)
- **Concurrent Load Testing**: Test multiple simultaneous requests
- **Memory Stress Testing**: Detect memory leaks and usage patterns
- **Edge Case Testing**: Handle unusual inputs gracefully
- **Throughput Analysis**: Measure system capacity

#### Quick Benchmarks (`tests/bench_parallel.py`)
- **Configuration Comparison**: Test different parallelization settings
- **Performance Baseline**: Quick performance validation
- **Speedup Analysis**: Compare optimization effectiveness

## 📊 Performance Gains

Expected improvements with full optimization:

| Feature | Performance Gain |
|---------|------------------|
| Parallel Chunk Processing | 2-4x speedup (depends on CPU cores) |
| Batch Audio Inference | 1.5-2x speedup for multiple chunks |
| ONNX Models | 2-3x faster inference |
| Quantized Models | 1.5-2x speedup + 30-50% smaller |
| GPU Processing | 3-5x speedup (when available) |
| Streaming Keywords | 50-80% faster emergency detection |

## 🛠️ Usage Instructions

### Quick Start
```bash
# Run quick benchmark
python run_benchmarks.py --type quick --workers 4 --batch

# Run comprehensive benchmark
python run_benchmarks.py --type full

# Run stress test
python run_benchmarks.py --type stress
```

### Environment Configuration
```bash
# Core performance settings
export PARALLEL_MAX_WORKERS=4          # Number of worker processes
export ENABLE_BATCH_PROCESSING=true    # Enable batch inference
export AUDIO_BATCH_SIZE=8              # Batch size for audio processing

# Device settings
export FORCE_CPU=false                  # Force CPU mode
export CUDA_VISIBLE_DEVICES=0          # Specific GPU device

# Model optimization
export USE_OPTIMIZED_MODELS=true       # Use optimized models
export MODEL_OPTIMIZATION_TYPE=onnx    # onnx|quantized|torchscript

# Logging (reduce verbosity)
export TF_CPP_MIN_LOG_LEVEL=3
# Vosk removed; set logging for Whisper or other components as needed
```

### Production Deployment
1. **Optimize Models**: Run model optimization to create production-ready versions
2. **Configure Environment**: Set optimal worker count and enable batch processing
3. **Enable GPU**: Use GPU acceleration when available
4. **Monitor Performance**: Use benchmarking tools to validate performance

```bash
# Optimize models for production
python -c "from modules.model_optimization import optimize_models_for_production; optimize_models_for_production()"

# Configure for production
export PARALLEL_MAX_WORKERS=8
export ENABLE_BATCH_PROCESSING=true
export USE_OPTIMIZED_MODELS=true
export MODEL_OPTIMIZATION_TYPE=onnx

# Run with optimizations
python main.py
```

## 🔍 Monitoring & Profiling

### Real-time Performance Monitoring
```python
from benchmarks.performance_profiler import profile_operation

with profile_operation("audio_processing", {"file": "test.wav"}):
    result = process_audio_file("test.wav")

# Metrics automatically collected and reported
```

### Memory Usage Tracking
```python
from benchmarks.performance_profiler import MemoryMonitor

monitor = MemoryMonitor()
monitor.start()
# ... process audio ...
memory_info = monitor.stop()
print(f"Peak memory: {memory_info['peak_mb']:.1f}MB")
```

### Comprehensive Analysis
```bash
# Generate detailed performance report
python benchmarks/performance_profiler.py

# Run stress tests
python benchmarks/stress_test.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `AUDIO_BATCH_SIZE`
   - Decrease `PARALLEL_MAX_WORKERS`
   - Enable `FORCE_CPU=true`

2. **Slow Performance**
   - Enable batch processing: `ENABLE_BATCH_PROCESSING=true`
   - Increase worker count: `PARALLEL_MAX_WORKERS=8`
   - Use optimized models: `USE_OPTIMIZED_MODELS=true`

3. **GPU Not Working**
   - Check CUDA installation
   - Verify GPU memory availability
   - Set `CUDA_VISIBLE_DEVICES` correctly

### Performance Debugging
```bash
# Check current configuration
python -c "
from modules.model_loader import TORCH_DEVICE, USE_GPU;
print(f'Device: {TORCH_DEVICE}, GPU: {USE_GPU}')
"

# Test basic functionality
python tests/bench_parallel.py

# Detailed profiling
python benchmarks/performance_profiler.py
```

## 📈 Benchmarking Results

Run benchmarks to get performance baseline:

```bash
# Quick performance test
python run_benchmarks.py --type quick

# Expected output:
# Sequential (1 worker, no batch): 15.2s
# Parallel (4 workers, with batch): 4.8s (3.2x speedup)
```

## 🔄 Future Optimizations

Additional optimizations that could be implemented:

1. **Model Distillation**: Smaller, faster models with similar accuracy
2. **Caching**: Cache model outputs for repeated inputs
3. **Hardware-Specific Optimizations**: Intel MKL, ARM NEON optimizations
4. **Distributed Processing**: Multi-machine scaling
5. **Model Pruning**: Remove unnecessary model parameters

## 📝 Notes

- All optimizations maintain backward compatibility
- Default settings prioritize reliability over performance
- Performance gains vary by hardware and audio characteristics
- Memory usage scales with number of workers and batch size
- GPU performance depends on model size and available VRAM

For questions or issues, refer to the benchmarking results and adjust configuration accordingly.
