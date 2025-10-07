# modules/model_manager.py
"""
Singleton model manager to ensure models are loaded only once across the application.
Prevents duplicate model loading when using Streamlit or other multi-import scenarios.
"""

import os
import threading
from typing import Optional, Dict, Any

class ModelManager:
    """Singleton class to manage model loading and access."""
    
    _instance = None
    _lock = threading.Lock()
    _models_loaded = False
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._models = {
                # Vosk removed
                'whisper_medium': None,
                'audio_feature_extractor': None,
                'wav2vec_model': None,
                'text_classifier': None,
                'yamnet_model': None,
                'yamnet_classes': [],
                'embedder': None,
                'torch_device': None,
                'use_gpu': False,
                'pipeline_device': -1
            }
    
    def load_models(self, force_reload=False):
        """Load all models once. Subsequent calls return immediately unless force_reload=True."""
        if self._models_loaded and not force_reload:
            return self._models
        
        with self._lock:
            if self._models_loaded and not force_reload:
                return self._models
            
            print("[MODEL_MANAGER] Loading models...")
            
            # Import and execute model loading logic
            try:
                # Set environment variables first
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
                # Vosk removed; no VOSK_LOG_LEVEL needed
                
                # Suppress warnings
                import warnings
                warnings.filterwarnings("ignore")
                
                try:
                    import tensorflow as tf
                    tf.get_logger().setLevel('ERROR')
                    from transformers.utils import logging as hf_logging
                    hf_logging.set_verbosity_error()
                except Exception:
                    pass
                
                # Import the actual model loading logic
                from modules import model_loader
                
                # Copy loaded models
                self._models.update({
                    'whisper_medium': getattr(model_loader, 'whisper_medium', None),
                    'audio_feature_extractor': model_loader.audio_feature_extractor,
                    'wav2vec_model': model_loader.wav2vec_model,
                    'text_classifier': model_loader.text_classifier,
                    'yamnet_model': model_loader.yamnet_model,
                    'yamnet_classes': model_loader.yamnet_classes,
                    'embedder': model_loader.embedder,
                    'torch_device': model_loader.TORCH_DEVICE,
                    'use_gpu': model_loader.USE_GPU,
                    'pipeline_device': model_loader.PIPELINE_DEVICE
                })
                
                self._models_loaded = True
                print("[MODEL_MANAGER] All models loaded successfully!")
                
            except Exception as e:
                print(f"[MODEL_MANAGER] Error loading models: {e}")
                self._models_loaded = False
        
        return self._models
    
    def get_model(self, model_name: str) -> Any:
        """Get a specific model by name."""
        if not self._models_loaded:
            self.load_models()
        return self._models.get(model_name)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all loaded models."""
        if not self._models_loaded:
            self.load_models()
        return self._models.copy()
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded
    
    def reload_models(self):
        """Force reload all models."""
        self._models_loaded = False
        return self.load_models(force_reload=True)

# Global instance
model_manager = ModelManager()

# Convenience functions for backward compatibility
def get_models():
    """Get all models (loads them if not already loaded)."""
    return model_manager.get_all_models()

def get_model(name):
    """Get a specific model by name."""
    return model_manager.get_model(name)

def ensure_models_loaded():
    """Ensure models are loaded."""
    if not model_manager.is_loaded():
        model_manager.load_models()
    return model_manager.is_loaded()
