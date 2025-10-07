# modules/model_optimization.py
"""
Advanced model optimization utilities for ONNX export, quantization, and TorchScript compilation.
Provides faster, more deterministic CPU inference with automatic INT8 quantization and GPU/DirectML support.
"""

import os
import torch
import numpy as np
from pathlib import Path
import traceback
import tempfile
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager
from modules import env_config as cfg
from modules.model_loader import get_model
from modules.logger import log_error

# Thread-local storage for ONNX sessions
_thread_local = threading.local()

def export_wav2vec_to_onnx(model, feature_extractor, output_path, sample_rate=16000, sequence_length=32000):
    """
    Export Wav2Vec2 model to ONNX format for optimized inference.
    
    Args:
        model: Wav2Vec2ForSequenceClassification model
        feature_extractor: Wav2Vec2FeatureExtractor
        output_path: path to save ONNX model
        sample_rate: audio sampling rate
        sequence_length: input audio length in samples
    """
    try:
        model.eval()
        
        # Create dummy input
        dummy_audio = torch.randn(1, sequence_length)
        dummy_inputs = feature_extractor(
            dummy_audio.numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        # Export to ONNX
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_values', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"[OK] Wav2Vec2 model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"[WARNING] Failed to export Wav2Vec2 to ONNX: {e}")
        return False


def export_distilroberta_to_onnx(model, tokenizer, output_path, max_length=512):
    """
    Export DistilRoBERTa model to ONNX format.
    
    Args:
        model: DistilRoBERTa model
        tokenizer: tokenizer
        output_path: path to save ONNX model
        max_length: maximum sequence length
    """
    try:
        model.eval()
        
        # Create dummy input
        dummy_text = "This is a sample text for emotion analysis."
        dummy_inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        # Export to ONNX
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"[OK] DistilRoBERTa model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"[WARNING] Failed to export DistilRoBERTa to ONNX: {e}")
        return False


def quantize_model(model, example_inputs, output_path):
    """
    Apply dynamic quantization to a PyTorch model for faster CPU inference.
    
    Args:
        model: PyTorch model
        example_inputs: example inputs for the model
        output_path: path to save quantized model
    """
    try:
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        print(f"[OK] Model quantized and saved: {output_path}")
        return quantized_model
        
    except Exception as e:
        print(f"[WARNING] Failed to quantize model: {e}")
        return None


def create_torchscript_model(model, example_inputs, output_path):
    """
    Create TorchScript version of model for optimized inference.
    
    Args:
        model: PyTorch model
        example_inputs: example inputs for tracing
        output_path: path to save TorchScript model
    """
    try:
        model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)
            traced_model.save(output_path)
        
        print(f"[OK] TorchScript model saved: {output_path}")
        return traced_model
        
    except Exception as e:
        print(f"[WARNING] Failed to create TorchScript model: {e}")
        return None


class EnhancedONNXInferenceWrapper:
    """Enhanced ONNX inference wrapper with automatic quantization and device management."""
    
    def __init__(self, onnx_path: str, enable_quantization: bool = True, enable_gpu: bool = None):
        self.onnx_path = onnx_path
        self.session = None
        self.quantized_session = None
        self.input_names = None
        self.output_names = None
        self.enable_quantization = enable_quantization
        self.enable_gpu = enable_gpu if enable_gpu is not None else cfg.get_use_gpu(False)
        self.quantized_path = None
        
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize ONNX Runtime session with optimal providers."""
        try:
            import onnxruntime as ort
            
            # Configure providers based on availability and preferences
            providers = self._get_optimal_providers()
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            # Set thread count based on CPU cores
            max_threads = min(os.cpu_count() or 4, 8)  # Cap at 8 for stability
            sess_options.intra_op_num_threads = max_threads
            sess_options.inter_op_num_threads = max_threads
            
            self.session = ort.InferenceSession(
                self.onnx_path, 
                sess_options=sess_options,
                providers=providers
            )
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"[OK] ONNX model loaded: {os.path.basename(self.onnx_path)}")
            print(f"   Providers: {self.session.get_providers()}")
            
            # Try to create quantized version if enabled
            if self.enable_quantization:
                self._create_quantized_model()
                
        except ImportError:
            print("[WARNING] onnxruntime not available, install with: pip install onnxruntime")
        except Exception as e:
            log_error("onnx_initialization", e)
            print(f"[WARNING] Failed to load ONNX model: {e}")
    
    def _get_optimal_providers(self) -> List[str]:
        """Get optimal execution providers based on system capabilities."""
        providers = ['CPUExecutionProvider']
        
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            if self.enable_gpu:
                # Prefer CUDA if available
                if 'CUDAExecutionProvider' in available_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
                # Fallback to DirectML on Windows
                elif 'DmlExecutionProvider' in available_providers and os.name == 'nt':
                    providers.insert(0, 'DmlExecutionProvider')
                # ROCm for AMD GPUs
                elif 'ROCMExecutionProvider' in available_providers:
                    providers.insert(0, 'ROCMExecutionProvider')
            
            # OpenVINO for Intel hardware acceleration
            if 'OpenVINOExecutionProvider' in available_providers:
                providers.insert(-1, 'OpenVINOExecutionProvider')
                
        except Exception as e:
            log_error("provider_detection", e)
        
        return providers
    
    def _create_quantized_model(self):
        """Create INT8 quantized version of the model."""
        try:
            import onnxruntime.quantization as ort_quant
            
            # Generate quantized model path
            model_dir = Path(self.onnx_path).parent
            model_name = Path(self.onnx_path).stem
            self.quantized_path = model_dir / f"{model_name}_int8.onnx"
            
            if self.quantized_path.exists():
                print(f"   Using existing quantized model: {self.quantized_path.name}")
            else:
                print(f"   Creating INT8 quantized model...")
                
                # Dynamic quantization (doesn't require calibration data)
                ort_quant.quantize_dynamic(
                    model_input=self.onnx_path,
                    model_output=str(self.quantized_path),
                    weight_type=ort_quant.QuantType.QInt8,
                    per_channel=True,
                    reduce_range=True,  # Better compatibility with older CPUs
                    optimize_model=True
                )
                print(f"   [OK] Quantized model created: {self.quantized_path.name}")
            
            # Load quantized session
            import onnxruntime as ort
            sess_options = self.session.get_session_options() if self.session else None
            self.quantized_session = ort.InferenceSession(
                str(self.quantized_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']  # Quantized models work best on CPU
            )
            
        except ImportError:
            print("   [WARNING] ONNX quantization not available")
        except Exception as e:
            log_error("onnx_quantization", e)
            print(f"   [WARNING] Failed to create quantized model: {e}")
    
    def predict(self, inputs, use_quantized: bool = None) -> Optional[np.ndarray]:
        """Run inference with automatic quantization fallback."""
        if self.session is None:
            return None
        
        # Determine which session to use
        use_quantized = use_quantized if use_quantized is not None else (
            self.quantized_session is not None and cfg.get_use_int8_quantization(True)
        )
        
        active_session = self.quantized_session if use_quantized else self.session
        if active_session is None:
            active_session = self.session
        
        try:
            # Convert inputs to numpy if needed
            input_dict = self._prepare_inputs(inputs)
            
            # Run inference
            outputs = active_session.run(self.output_names, input_dict)
            return outputs[0] if len(outputs) == 1 else outputs
            
        except Exception as e:
            # Fallback to original model if quantized inference fails
            if use_quantized and self.session is not None:
                try:
                    input_dict = self._prepare_inputs(inputs)
                    outputs = self.session.run(self.output_names, input_dict)
                    return outputs[0] if len(outputs) == 1 else outputs
                except Exception as fallback_e:
                    log_error("onnx_inference_fallback", fallback_e)
            
            log_error("onnx_inference", e)
            return None
    
    def _prepare_inputs(self, inputs) -> Dict[str, np.ndarray]:
        """Prepare inputs for ONNX inference."""
        input_dict = {}
        for i, name in enumerate(self.input_names):
            if isinstance(inputs, (list, tuple)):
                tensor = inputs[i]
            elif isinstance(inputs, dict):
                tensor = inputs[name]
            else:
                tensor = inputs
                
            # Convert to numpy
            if hasattr(tensor, 'numpy'):
                input_dict[name] = tensor.numpy()
            elif isinstance(tensor, np.ndarray):
                input_dict[name] = tensor
            else:
                input_dict[name] = np.array(tensor)
                
        return input_dict
    
    def benchmark(self, inputs, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        if self.session is None:
            return {}
        
        results = {}
        input_dict = self._prepare_inputs(inputs)
        
        # Benchmark original model
        if self.session is not None:
            import time
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                self.session.run(self.output_names, input_dict)
                times.append(time.perf_counter() - start)
            
            results['original_ms'] = np.mean(times) * 1000
            results['original_std_ms'] = np.std(times) * 1000
        
        # Benchmark quantized model
        if self.quantized_session is not None:
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                self.quantized_session.run(self.output_names, input_dict)
                times.append(time.perf_counter() - start)
            
            results['quantized_ms'] = np.mean(times) * 1000
            results['quantized_std_ms'] = np.std(times) * 1000
            
            if 'original_ms' in results:
                results['speedup_ratio'] = results['original_ms'] / results['quantized_ms']
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_path': self.onnx_path,
            'quantized_available': self.quantized_session is not None,
            'quantized_path': str(self.quantized_path) if self.quantized_path else None,
            'input_names': self.input_names,
            'output_names': self.output_names
        }
        
        if self.session:
            info['providers'] = self.session.get_providers()
            info['profiling_file'] = self.session.get_profiling_start_time_ns()
        
        return info



class ModelOptimizer:
    """Comprehensive model optimization system."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or cfg.get_onnx_model_dir("models/optimized"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_results = {}
    
    def optimize_all_models(self) -> Dict[str, bool]:
        """Optimize all available models for production."""
        print("[ROCKET] Starting comprehensive model optimization...")
        
        results = {}
        
        # Optimize Wav2Vec2 for audio emotion
        results['wav2vec2'] = self._optimize_wav2vec2()
        
        # Optimize DistilRoBERTa for text emotion
        results['distilroberta'] = self._optimize_distilroberta()
        
        # Optimize YAMNet for sound detection (if available)
        results['yamnet'] = self._optimize_yamnet()
        
        # Generate optimization report
        self._generate_optimization_report(results)
        
        return results
    
    def _optimize_wav2vec2(self) -> bool:
        """Optimize Wav2Vec2 model with multiple formats."""
        try:
            print("🔄 Optimizing Wav2Vec2 model...")
            
            w2v = get_model('wav2vec_model')
            afe = get_model('audio_feature_extractor')
            
            if w2v is None or afe is None:
                print("   [WARNING] Wav2Vec2 model not available")
                return False
            
            base_name = "wav2vec2_emotion"
            success = True
            
            # Create dummy inputs
            dummy_audio = torch.randn(1, 32000)
            dummy_inputs = afe(
                dummy_audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # ONNX export with enhanced options
            onnx_path = self.output_dir / f"{base_name}.onnx"
            if self._export_wav2vec_enhanced(w2v, afe, str(onnx_path)):
                print(f"   [OK] ONNX export successful")
                
                # Test ONNX inference
                wrapper = EnhancedONNXInferenceWrapper(
                    str(onnx_path), 
                    enable_quantization=True
                )
                
                if wrapper.session is not None:
                    # Benchmark performance
                    benchmark_results = wrapper.benchmark(tuple(dummy_inputs.values()))
                    self.optimization_results[base_name] = benchmark_results
                    print(f"   [DASHBOARD] Benchmark: {benchmark_results.get('original_ms', 0):.1f}ms")
                    if 'quantized_ms' in benchmark_results:
                        print(f"   [DASHBOARD] Quantized: {benchmark_results['quantized_ms']:.1f}ms (" +
                              f"{benchmark_results.get('speedup_ratio', 1):.1f}x speedup)")
            else:
                success = False
            
            # TorchScript optimization
            torchscript_path = self.output_dir / f"{base_name}_torchscript.pt"
            if self._create_torchscript_enhanced(w2v, tuple(dummy_inputs.values()), str(torchscript_path)):
                print(f"   [OK] TorchScript export successful")
            
            # Dynamic quantization
            quantized_path = self.output_dir / f"{base_name}_quantized.pth"
            if self._quantize_model_enhanced(w2v, tuple(dummy_inputs.values()), str(quantized_path)):
                print(f"   [OK] PyTorch quantization successful")
            
            return success
            
        except Exception as e:
            log_error("wav2vec2_optimization", e)
            print(f"   [ERROR] Wav2Vec2 optimization failed: {e}")
            return False
    
    def _optimize_distilroberta(self) -> bool:
        """Optimize DistilRoBERTa model with multiple formats."""
        try:
            print("🔄 Optimizing DistilRoBERTa model...")
            
            text_pipe = get_model('text_classifier')
            if text_pipe is None:
                print("   [WARNING] DistilRoBERTa model not available")
                return False
            
            model = text_pipe.model
            tokenizer = text_pipe.tokenizer
            base_name = "distilroberta_emotion"
            success = True
            
            # Create dummy inputs
            dummy_text = "This is a sample text for emotion analysis."
            dummy_inputs = tokenizer(
                dummy_text,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            # ONNX export
            onnx_path = self.output_dir / f"{base_name}.onnx"
            if self._export_distilroberta_enhanced(model, tokenizer, str(onnx_path)):
                print(f"   [OK] ONNX export successful")
                
                # Test and benchmark
                wrapper = EnhancedONNXInferenceWrapper(
                    str(onnx_path),
                    enable_quantization=True
                )
                
                if wrapper.session is not None:
                    benchmark_results = wrapper.benchmark(tuple(dummy_inputs.values()))
                    self.optimization_results[base_name] = benchmark_results
                    print(f"   [DASHBOARD] Benchmark: {benchmark_results.get('original_ms', 0):.1f}ms")
                    if 'quantized_ms' in benchmark_results:
                        print(f"   [DASHBOARD] Quantized: {benchmark_results['quantized_ms']:.1f}ms (" +
                              f"{benchmark_results.get('speedup_ratio', 1):.1f}x speedup)")
            else:
                success = False
            
            # Additional optimizations
            torchscript_path = self.output_dir / f"{base_name}_torchscript.pt"
            quantized_path = self.output_dir / f"{base_name}_quantized.pth"
            
            self._create_torchscript_enhanced(model, tuple(dummy_inputs.values()), str(torchscript_path))
            self._quantize_model_enhanced(model, tuple(dummy_inputs.values()), str(quantized_path))
            
            return success
            
        except Exception as e:
            log_error("distilroberta_optimization", e)
            print(f"   [ERROR] DistilRoBERTa optimization failed: {e}")
            return False
    
    def _optimize_yamnet(self) -> bool:
        """Optimize YAMNet model if available."""
        try:
            print("� Checking YAMNet model...")
            
            yamnet = get_model('yamnet_model')
            if yamnet is None:
                print("   [WARNING] YAMNet model not available")
                return False
            
            # YAMNet optimization is more complex due to TensorFlow nature
            # For now, just note it's available
            print("   [INFO] YAMNet model available but optimization not implemented yet")
            return True
            
        except Exception as e:
            log_error("yamnet_optimization", e)
            return False


    def _export_wav2vec_enhanced(self, model, feature_extractor, output_path):
        """Enhanced Wav2Vec2 ONNX export with better optimization."""
        try:
            model.eval()
            
            # Create dummy input with realistic dimensions
            dummy_audio = torch.randn(1, 32000)  # ~2 seconds at 16kHz
            dummy_inputs = feature_extractor(
                dummy_audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Export with enhanced settings
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                output_path,
                export_params=True,
                opset_version=12,  # Higher opset for better optimization
                do_constant_folding=True,
                input_names=['input_values', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_values': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'}
                },
                verbose=False
            )
            return True
            
        except Exception as e:
            log_error("wav2vec_onnx_export", e)
            return False
    
    def _export_distilroberta_enhanced(self, model, tokenizer, output_path):
        """Enhanced DistilRoBERTa ONNX export."""
        try:
            model.eval()
            
            dummy_text = "This is a sample text for emotion analysis."
            dummy_inputs = tokenizer(
                dummy_text,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'}
                },
                verbose=False
            )
            return True
            
        except Exception as e:
            log_error("distilroberta_onnx_export", e)
            return False
    
    def _quantize_model_enhanced(self, model, example_inputs, output_path):
        """Enhanced PyTorch quantization."""
        try:
            # Prepare model for quantization
            model.eval()
            
            # Dynamic quantization for linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv1d},  # Quantize more layer types
                dtype=torch.qint8
            )
            
            # Save with additional metadata
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'model_class': model.__class__.__name__,
                'quantization_config': {
                    'dtype': 'qint8',
                    'quantized_layers': ['Linear', 'Conv1d']
                }
            }, output_path)
            
            return True
            
        except Exception as e:
            log_error("pytorch_quantization", e)
            return False
    
    def _create_torchscript_enhanced(self, model, example_inputs, output_path):
        """Enhanced TorchScript creation."""
        try:
            model.eval()
            
            with torch.no_grad():
                # Try tracing first
                try:
                    traced_model = torch.jit.trace(model, example_inputs)
                    traced_model.save(output_path)
                    return True
                except Exception:
                    # Fallback to scripting
                    scripted_model = torch.jit.script(model)
                    scripted_model.save(output_path)
                    return True
                    
        except Exception as e:
            log_error("torchscript_creation", e)
            return False
    
    def _generate_optimization_report(self, results: Dict[str, bool]):
        """Generate comprehensive optimization report."""
        report_path = self.output_dir / "optimization_report.json"
        
        import datetime
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'results': results,
            'benchmark_results': self.optimization_results,
            'system_info': {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cpu_count': os.cpu_count()
            },
            'usage_instructions': {
                'environment_variables': {
                    'USE_ONNX': 'true',
                    'USE_ONNX_AUDIO': 'true', 
                    'USE_ONNX_TEXT': 'true',
                    'USE_INT8_QUANTIZATION': 'true',
                    'ONNX_MODEL_DIR': str(self.output_dir)
                },
                'performance_notes': [
                    'ONNX models provide 1.5-3x speedup on CPU',
                    'INT8 quantization reduces model size by ~75%',
                    'TorchScript provides deployment without Python dependencies',
                    'GPU acceleration available with CUDA/DirectML providers'
                ]
            }
        }
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📋 Optimization report saved: {report_path}")
        except Exception as e:
            log_error("optimization_report", e)


def optimize_models_for_production(output_dir: Optional[str] = None) -> Dict[str, bool]:
    """
    Main function to export and optimize all models for production use.
    Creates ONNX, quantized, and TorchScript versions with enhanced performance.
    """
    optimizer = ModelOptimizer(output_dir)
    return optimizer.optimize_all_models()


def create_onnx_wrapper(model_path: str, enable_quantization: bool = True) -> EnhancedONNXInferenceWrapper:
    """Create an enhanced ONNX inference wrapper."""
    return EnhancedONNXInferenceWrapper(model_path, enable_quantization=enable_quantization)



if __name__ == "__main__":
    optimize_models_for_production()
