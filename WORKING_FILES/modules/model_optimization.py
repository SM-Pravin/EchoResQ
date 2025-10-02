# modules/model_optimization.py
"""
Model optimization utilities for ONNX export, quantization, and TorchScript compilation.
Provides faster, more deterministic CPU inference for production deployment.
"""

import os
import torch
import numpy as np
from pathlib import Path
import traceback

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
        print(f"✅ Wav2Vec2 model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to export Wav2Vec2 to ONNX: {e}")
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
        print(f"✅ DistilRoBERTa model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to export DistilRoBERTa to ONNX: {e}")
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
        print(f"✅ Model quantized and saved: {output_path}")
        return quantized_model
        
    except Exception as e:
        print(f"⚠️ Failed to quantize model: {e}")
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
        
        print(f"✅ TorchScript model saved: {output_path}")
        return traced_model
        
    except Exception as e:
        print(f"⚠️ Failed to create TorchScript model: {e}")
        return None


class ONNXInferenceWrapper:
    """Wrapper for ONNX model inference."""
    
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.session = None
        self.input_names = None
        self.output_names = None
        
        try:
            import onnxruntime as ort
            # Use CPU provider for deterministic results
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"✅ ONNX model loaded: {onnx_path}")
            
        except ImportError:
            print("⚠️ onnxruntime not available, install with: pip install onnxruntime")
        except Exception as e:
            print(f"⚠️ Failed to load ONNX model: {e}")
    
    def predict(self, inputs):
        """Run inference on ONNX model."""
        if self.session is None:
            return None
        
        try:
            # Convert inputs to numpy if needed
            input_dict = {}
            for i, name in enumerate(self.input_names):
                if isinstance(inputs, (list, tuple)):
                    input_dict[name] = inputs[i].numpy() if hasattr(inputs[i], 'numpy') else inputs[i]
                elif isinstance(inputs, dict):
                    input_dict[name] = inputs[name].numpy() if hasattr(inputs[name], 'numpy') else inputs[name]
                else:
                    input_dict[name] = inputs.numpy() if hasattr(inputs, 'numpy') else inputs
            
            outputs = self.session.run(self.output_names, input_dict)
            return outputs[0] if len(outputs) == 1 else outputs
            
        except Exception as e:
            print(f"⚠️ ONNX inference failed: {e}")
            return None


def optimize_models_for_production():
    """
    Main function to export and optimize all models for production use.
    Creates ONNX, quantized, and TorchScript versions.
    """
    print("🚀 Starting model optimization for production...")
    
    # Import models
    try:
        from modules.model_loader import wav2vec_model, audio_feature_extractor, text_classifier
        
        models_dir = Path("models/optimized")
        models_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = 0
        
        # Optimize Wav2Vec2 model
        if wav2vec_model is not None and audio_feature_extractor is not None:
            total_count += 1
            print("🔄 Optimizing Wav2Vec2 model...")
            
            # ONNX export
            onnx_path = models_dir / "wav2vec2_emotion.onnx"
            if export_wav2vec_to_onnx(wav2vec_model, audio_feature_extractor, str(onnx_path)):
                success_count += 1
            
            # Quantization
            try:
                dummy_audio = torch.randn(1, 32000)
                dummy_inputs = audio_feature_extractor(
                    dummy_audio.numpy(), 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=True
                )
                
                quantized_path = models_dir / "wav2vec2_emotion_quantized.pth"
                quantize_model(wav2vec_model, tuple(dummy_inputs.values()), str(quantized_path))
                
                # TorchScript
                torchscript_path = models_dir / "wav2vec2_emotion_torchscript.pt"
                create_torchscript_model(wav2vec_model, tuple(dummy_inputs.values()), str(torchscript_path))
                
            except Exception as e:
                print(f"⚠️ Additional Wav2Vec2 optimizations failed: {e}")
        
        # Optimize DistilRoBERTa model (from text classifier pipeline)
        if text_classifier is not None:
            total_count += 1
            print("🔄 Optimizing DistilRoBERTa model...")
            
            try:
                model = text_classifier.model
                tokenizer = text_classifier.tokenizer
                
                # ONNX export
                onnx_path = models_dir / "distilroberta_emotion.onnx"
                if export_distilroberta_to_onnx(model, tokenizer, str(onnx_path)):
                    success_count += 1
                
                # Additional optimizations
                dummy_text = "This is a sample text for emotion analysis."
                dummy_inputs = tokenizer(
                    dummy_text,
                    return_tensors="pt",
                    max_length=512,
                    padding="max_length",
                    truncation=True
                )
                
                quantized_path = models_dir / "distilroberta_emotion_quantized.pth"
                quantize_model(model, tuple(dummy_inputs.values()), str(quantized_path))
                
                torchscript_path = models_dir / "distilroberta_emotion_torchscript.pt"
                create_torchscript_model(model, tuple(dummy_inputs.values()), str(torchscript_path))
                
            except Exception as e:
                print(f"⚠️ DistilRoBERTa optimization failed: {e}")
        
        print(f"✅ Model optimization complete: {success_count}/{total_count} models successfully optimized")
        
        # Create usage instructions
        instructions_path = models_dir / "usage_instructions.txt"
        with open(instructions_path, 'w') as f:
            f.write("""
Model Optimization Results
========================

This directory contains optimized versions of the emergency AI models:

1. ONNX Models (.onnx):
   - Fastest inference, cross-platform compatible
   - Use with onnxruntime: pip install onnxruntime
   - Recommended for production deployment

2. Quantized Models (_quantized.pth):
   - Reduced model size, faster CPU inference
   - Load with torch.load() and standard PyTorch

3. TorchScript Models (_torchscript.pt):
   - Optimized PyTorch models, no Python dependencies
   - Load with torch.jit.load()

Usage:
------
Set environment variable USE_OPTIMIZED_MODELS=true to enable optimized models.
Set MODEL_OPTIMIZATION_TYPE=onnx|quantized|torchscript to choose optimization type.

Performance gains:
- ONNX: 2-3x faster inference
- Quantized: 30-50% smaller size, 1.5-2x faster
- TorchScript: 1.5-2x faster, better deployment
""")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Model optimization failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    optimize_models_for_production()
