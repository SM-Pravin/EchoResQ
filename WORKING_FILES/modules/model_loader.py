# modules/model_loader.py
"""
Optimized model loader that prevents duplicate loading in Streamlit and other scenarios.
Thread-safe singleton pattern ensures models are loaded only once.
"""

import os
import threading
import traceback

# Global state
_loading_lock = threading.Lock()
_models_initialized = False
_models = {}

# Environment setup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")

# Suppress warnings early
import warnings
warnings.filterwarnings("ignore")

try:
    from vosk import SetLogLevel
    SetLogLevel(-1)
except Exception:
    pass

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import torch
from modules import env_config as cfg

class DeviceManager:
    """Advanced device management with GPU/DirectML acceleration and fallback logic."""
    
    def __init__(self):
        self.device_info = self._detect_devices()
        self.optimal_device = self._get_optimal_device()
        self.directml_available = self._check_directml()
        self.memory_threshold = 2048  # MB minimum for GPU usage
    
    def _detect_devices(self) -> dict:
        """Detect available compute devices and their capabilities."""
        info = {
            'cpu_cores': os.cpu_count() or 4,
            'cuda_available': False,
            'cuda_devices': [],
            'directml_available': False,
            'gpu_memory_mb': 0,
            'force_cpu': os.environ.get("FORCE_CPU", "false").lower() == "true"
        }
        
        # CUDA detection
        if torch.cuda.is_available() and not info['force_cpu']:
            info['cuda_available'] = True
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': device_props.name,
                    'memory_mb': device_props.total_memory // (1024 * 1024),
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                }
                info['cuda_devices'].append(device_info)
                
                # Get current memory usage
                try:
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    device_info['free_memory_mb'] = free_memory // (1024 * 1024)
                    if i == 0:  # Use primary GPU memory for overall assessment
                        info['gpu_memory_mb'] = device_info['free_memory_mb']
                except Exception:
                    device_info['free_memory_mb'] = device_info['memory_mb'] // 2  # Estimate
        
        # DirectML detection (Windows only)
        if os.name == 'nt' and not info['force_cpu']:
            info['directml_available'] = self._check_directml()
        
        return info
    
    def _check_directml(self) -> bool:
        """Check if DirectML is available for hardware acceleration."""
        try:
            # Try importing DirectML-related packages
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            return 'DmlExecutionProvider' in available_providers
        except ImportError:
            return False
        except Exception:
            return False
    
    def _get_optimal_device(self) -> torch.device:
        """Select optimal device based on capabilities and preferences."""
        if self.device_info['force_cpu']:
            return torch.device("cpu")
        
        # Prefer CUDA if available with sufficient memory
        if (self.device_info['cuda_available'] and 
            self.device_info['gpu_memory_mb'] >= self.memory_threshold):
            
            # Select best CUDA device
            best_device = 0
            best_memory = 0
            for device in self.device_info['cuda_devices']:
                if device['free_memory_mb'] > best_memory:
                    best_memory = device['free_memory_mb']
                    best_device = device['index']
            
            return torch.device(f"cuda:{best_device}")
        
        # Fallback to CPU
        return torch.device("cpu")
    
    def get_onnx_providers(self, prefer_gpu: bool = None) -> list:
        """Get optimal ONNX Runtime execution providers."""
        if prefer_gpu is None:
            prefer_gpu = cfg.get_use_gpu(True)
        
        providers = ['CPUExecutionProvider']
        
        if prefer_gpu and not self.device_info['force_cpu']:
            # Add GPU providers in order of preference
            if self.device_info['cuda_available']:
                providers.insert(0, 'CUDAExecutionProvider')
            elif self.directml_available:
                providers.insert(0, 'DmlExecutionProvider')
        
        # Add other acceleration providers
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            
            # Intel OpenVINO
            if 'OpenVINOExecutionProvider' in available:
                providers.insert(-1, 'OpenVINOExecutionProvider')
            
            # AMD ROCm
            if 'ROCMExecutionProvider' in available and not self.device_info['cuda_available']:
                providers.insert(0, 'ROCMExecutionProvider')
                
        except Exception:
            pass
        
        return providers
    
    def optimize_for_inference(self, model):
        """Optimize model settings for inference on the target device."""
        if hasattr(model, 'eval'):
            model.eval()
        
        # Move to optimal device
        if hasattr(model, 'to'):
            model.to(self.optimal_device)
        
        # Enable inference optimizations
        if self.optimal_device.type == 'cuda':
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Try to enable mixed precision for newer GPUs
            try:
                from torch.cuda.amp import autocast
                # Check if GPU supports mixed precision
                if hasattr(torch.cuda.get_device_properties(self.optimal_device.index), 'major'):
                    major = torch.cuda.get_device_properties(self.optimal_device.index).major
                    if major >= 7:  # Volta and newer
                        model._use_amp = True
            except Exception:
                pass
        
        return model
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage information."""
        usage = {
            'cpu_percent': 0,
            'system_memory_mb': 0,
            'gpu_memory_mb': 0,
            'gpu_memory_cached_mb': 0
        }
        
        try:
            import psutil
            usage['cpu_percent'] = psutil.cpu_percent()
            usage['system_memory_mb'] = psutil.virtual_memory().used // (1024 * 1024)
        except ImportError:
            pass
        
        if self.device_info['cuda_available']:
            try:
                usage['gpu_memory_mb'] = torch.cuda.memory_allocated() // (1024 * 1024)
                usage['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() // (1024 * 1024)
            except Exception:
                pass
        
        return usage
    
    def clear_cache(self):
        """Clear GPU memory cache if available."""
        if self.device_info['cuda_available']:
            try:
                torch.cuda.empty_cache()
                print("🧹 GPU memory cache cleared")
            except Exception as e:
                print(f"⚠️ Failed to clear GPU cache: {e}")
    
    def get_device_summary(self) -> str:
        """Get human-readable device summary."""
        summary = f"Optimal device: {self.optimal_device}\n"
        
        if self.device_info['cuda_available']:
            summary += f"CUDA devices: {len(self.device_info['cuda_devices'])}\n"
            for device in self.device_info['cuda_devices']:
                summary += f"  - {device['name']}: {device['memory_mb']}MB\n"
        
        if self.directml_available:
            summary += "DirectML: Available\n"
        
        summary += f"CPU cores: {self.device_info['cpu_cores']}\n"
        
        return summary


# Global device manager instance
_device_manager = None


def get_device_manager() -> DeviceManager:
    """Get or create global device manager."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_optimal_device():
    """Get optimal device based on availability and environment settings."""
    return get_device_manager().optimal_device

# Device configuration with enhanced management
device_manager = get_device_manager()
TORCH_DEVICE = device_manager.optimal_device
USE_GPU = TORCH_DEVICE.type == "cuda"
PIPELINE_DEVICE = 0 if USE_GPU else -1

# Print device information
print("🔧 Device Configuration:")
print(device_manager.get_device_summary())

# Model paths - adjusted for actual structure
BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
VOSK_PATH = os.path.join(BASE, "vosk-model-large-en-us")
WAV2VEC2_PATH = os.path.join(BASE, "wav2vec2")
DISTILROBERTA_PATH = os.path.join(BASE, "distilroberta")
YAMNET_PATH = os.path.join(BASE, "yamnet")
ONNX_DIR = cfg.get_onnx_model_dir(os.path.join(BASE, "optimized"))

def _init_model_dict():
    global _models
    if not _models:
        _models = {
            'vosk_model': None,
            'audio_feature_extractor': None,
            'wav2vec_model': None,
            'wav2vec_onnx': None,
            'text_classifier': None,
            'text_onnx': None,
            'yamnet_model': None,
            'yamnet_classes': [],
            'embedder': None,
            'TORCH_DEVICE': TORCH_DEVICE,
            'USE_GPU': USE_GPU,
            'PIPELINE_DEVICE': PIPELINE_DEVICE,
        }


def _load_vosk():
    try:
        from vosk import Model as VoskModel
        if os.path.isdir(VOSK_PATH):
            _models['vosk_model'] = VoskModel(VOSK_PATH)
            print(" Vosk model loaded")
        else:
            print(f" WARNING: Vosk model not found at: {VOSK_PATH}")
    except Exception as e:
        print(f" WARNING: Failed to initialize Vosk model: {e}")


def _load_wav2vec2():
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        if os.path.isdir(WAV2VEC2_PATH):
            try:
                _models['audio_feature_extractor'] = Wav2Vec2FeatureExtractor.from_pretrained(
                    WAV2VEC2_PATH, local_files_only=True
                )
                _models['wav2vec_model'] = Wav2Vec2ForSequenceClassification.from_pretrained(
                    WAV2VEC2_PATH, local_files_only=True
                )
                
                # Optimize for inference with device manager
                _models['wav2vec_model'] = device_manager.optimize_for_inference(_models['wav2vec_model'])
                
                # Enable gradient checkpointing for memory efficiency if on GPU
                if USE_GPU and hasattr(_models['wav2vec_model'], 'gradient_checkpointing_enable'):
                    try:
                        _models['wav2vec_model'].gradient_checkpointing_enable()
                    except Exception:
                        pass  # Not all models support this
                
                print(f" ✅ Wav2Vec2 model loaded (device: {TORCH_DEVICE})")
                memory_usage = device_manager.get_memory_usage()
                if memory_usage['gpu_memory_mb'] > 0:
                    print(f"    GPU memory: {memory_usage['gpu_memory_mb']}MB")
                    
            except Exception as e:
                print(f" ⚠️ Failed to load Wav2Vec2 from {WAV2VEC2_PATH}: {e}")
                # Try CPU fallback
                if TORCH_DEVICE.type != 'cpu':
                    try:
                        _models['wav2vec_model'] = Wav2Vec2ForSequenceClassification.from_pretrained(
                            WAV2VEC2_PATH, local_files_only=True
                        ).to('cpu').eval()
                        print(f" ✅ Wav2Vec2 model loaded on CPU (fallback)")
                    except Exception as fallback_e:
                        print(f" ❌ CPU fallback also failed: {fallback_e}")
        else:
            print(f" ⚠️ Wav2Vec2 model folder not found at: {WAV2VEC2_PATH}")
    except Exception as e:
        print(f" ⚠️ Transformers not available or error loading wav2vec2: {e}")


def _load_wav2vec2_onnx():
    """Load ONNX runtime session for Wav2Vec2 emotion model with enhanced acceleration."""
    try:
        import onnxruntime as ort
        
        # Get optimal providers from device manager
        providers = device_manager.get_onnx_providers(prefer_gpu=cfg.get_use_onnx_cuda("auto") != "false")
        
        onnx_path = os.path.join(ONNX_DIR, "wav2vec2_emotion.onnx")
        if os.path.isfile(onnx_path):
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            # Thread configuration
            max_threads = min(device_manager.device_info['cpu_cores'], 8)
            sess_options.intra_op_num_threads = max_threads
            sess_options.inter_op_num_threads = max_threads
            
            _models['wav2vec_onnx'] = ort.InferenceSession(
                onnx_path, 
                sess_options=sess_options,
                providers=providers
            )
            print(f" ✅ ONNX Wav2Vec2 loaded (providers: {_models['wav2vec_onnx'].get_providers()})")
        else:
            print(f" ⚠️ ONNX Wav2Vec2 model not found at {onnx_path}")
    except Exception as e:
        print(f" ⚠️ Failed to load ONNX Wav2Vec2: {e}")


def _load_distilroberta():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        if os.path.isdir(DISTILROBERTA_PATH):
            try:
                tok = AutoTokenizer.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
                
                # Optimize for inference
                model = device_manager.optimize_for_inference(model)
                
                # Create pipeline with optimized settings
                _models['text_classifier'] = pipeline(
                    "text-classification", 
                    model=model, 
                    tokenizer=tok, 
                    device=PIPELINE_DEVICE,
                    return_all_scores=True,  # Get all emotion scores
                    truncation=True,
                    max_length=512
                )
                
                print(f" ✅ DistilRoBERTa loaded (device: {TORCH_DEVICE})")
                memory_usage = device_manager.get_memory_usage()
                if memory_usage['gpu_memory_mb'] > 0:
                    print(f"    GPU memory: {memory_usage['gpu_memory_mb']}MB")
                    
            except Exception as e:
                print(f" ⚠️ Failed to load DistilRoBERTa from {DISTILROBERTA_PATH}: {e}")
                # Try CPU fallback
                if TORCH_DEVICE.type != 'cpu':
                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            DISTILROBERTA_PATH, local_files_only=True
                        ).to('cpu').eval()
                        _models['text_classifier'] = pipeline(
                            "text-classification", model=model, tokenizer=tok, device=-1
                        )
                        print(f" ✅ DistilRoBERTa loaded on CPU (fallback)")
                    except Exception as fallback_e:
                        print(f" ❌ CPU fallback also failed: {fallback_e}")
        else:
            print(f" ⚠️ DistilRoBERTa model folder not found at: {DISTILROBERTA_PATH}")
    except Exception as e:
        print(f" ⚠️ Transformers not available or error loading DistilRoBERTa: {e}")


def _load_distilroberta_onnx():
    """Load ONNX runtime session for DistilRoBERTa text emotion model with enhanced acceleration."""
    try:
        import onnxruntime as ort
        
        # Get optimal providers from device manager
        providers = device_manager.get_onnx_providers(prefer_gpu=cfg.get_use_onnx_cuda("auto") != "false")
        
        onnx_path = os.path.join(ONNX_DIR, "distilroberta_emotion.onnx")
        if os.path.isfile(onnx_path):
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            # Thread configuration
            max_threads = min(device_manager.device_info['cpu_cores'], 8)
            sess_options.intra_op_num_threads = max_threads
            sess_options.inter_op_num_threads = max_threads
            
            _models['text_onnx'] = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options, 
                providers=providers
            )
            print(f" ✅ ONNX DistilRoBERTa loaded (providers: {_models['text_onnx'].get_providers()})")
        else:
            print(f" ⚠️ ONNX DistilRoBERTa model not found at {onnx_path}")
    except Exception as e:
        print(f" ⚠️ Failed to load ONNX DistilRoBERTa: {e}")


def _load_yamnet():
    try:
        import tensorflow as tf
        if os.path.isdir(YAMNET_PATH):
            try:
                _models['yamnet_model'] = tf.saved_model.load(YAMNET_PATH)
                class_map_path = os.path.join(YAMNET_PATH, "yamnet_class_map.csv")
                if os.path.isfile(class_map_path):
                    with open(class_map_path, "r") as f:
                        lines = f.readlines()[1:]
                        _models['yamnet_classes'] = [line.strip().split(",")[2] for line in lines]
                else:
                    print(f" WARNING: YAMNet class map not found at {class_map_path}")
                print(" YAMNet loaded")
            except Exception as e:
                print(f" WARNING: Failed to load YAMNet from {YAMNET_PATH}: {e}")
        else:
            print(f" WARNING: YAMNet folder not found at: {YAMNET_PATH}")
    except Exception as e:
        print(f" WARNING: TensorFlow not available or error loading YAMNet: {e}")


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        local_embed_path = os.path.join(BASE, "all-MiniLM-L6-v2")
        if os.path.isdir(local_embed_path):
            try:
                _models['embedder'] = SentenceTransformer(local_embed_path, device=TORCH_DEVICE.type)
                print(" SentenceTransformer loaded from local folder")
            except Exception:
                _models['embedder'] = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE.type)
                print(" SentenceTransformer loaded from cache/hub")
        else:
            try:
                _models['embedder'] = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE.type)
                print(" SentenceTransformer loaded")
            except Exception as e:
                print(f" WARNING: Failed to load SentenceTransformer: {e}")
    except Exception as e:
        print(f" WARNING: sentence-transformers not available: {e}")


_MODEL_LOADERS = {
    'vosk_model': _load_vosk,
    'audio_feature_extractor': _load_wav2vec2,
    'wav2vec_model': _load_wav2vec2,
    'wav2vec_onnx': _load_wav2vec2_onnx,
    'text_classifier': _load_distilroberta,
    'text_onnx': _load_distilroberta_onnx,
    'yamnet_model': _load_yamnet,
    'yamnet_classes': _load_yamnet,
    'embedder': _load_embedder,
}


def _ensure_model(name: str):
    global _models_initialized
    _init_model_dict()
    if name in _models and _models.get(name) not in (None, []):
        return
    loader = _MODEL_LOADERS.get(name)
    if loader is None:
        return
    with _loading_lock:
        # Load only if still missing
        if _models.get(name) in (None, []):
            loader()
    # Mark initialized when at least one model has been loaded
    _models_initialized = True


def _load_models():
    """Load all models with thread safety (eager full init)."""
    _init_model_dict()
    
    print("🚀 Initializing models with enhanced device management...")
    
    # Load core models
    for key in ['vosk_model', 'audio_feature_extractor', 'wav2vec_model', 'text_classifier', 'yamnet_model', 'yamnet_classes', 'embedder']:
        _ensure_model(key)
    
    # Load ONNX models if enabled
    if cfg.get_use_onnx_audio(cfg.get_use_onnx(False)):
        _ensure_model('wav2vec_onnx')
    
    if cfg.get_use_onnx_text(cfg.get_use_onnx(False)):
        _ensure_model('text_onnx')
    
    print("✅ All models initialized.")
    print_device_summary()
    
    return _models

def get_models():
    """Get all models (thread-safe, loads once)."""
    return _load_models()

def get_model(name):
    """Get a specific model by name; lazily load only what's requested."""
    _ensure_model(name)
    return _models.get(name)

def is_loaded():
    """Check if models are already loaded."""
    return _models_initialized

# Initialize compatibility variables (lazy)
vosk_model = None
audio_feature_extractor = None
wav2vec_model = None
text_classifier = None
yamnet_model = None
yamnet_classes = []
embedder = None

def _ensure_compatibility_lazy():
    global vosk_model, audio_feature_extractor, wav2vec_model, text_classifier
    global yamnet_model, yamnet_classes, embedder
    if vosk_model is None or audio_feature_extractor is None or wav2vec_model is None or text_classifier is None or yamnet_model is None or not yamnet_classes or embedder is None:
        m = get_models()
        vosk_model = m['vosk_model']
        audio_feature_extractor = m['audio_feature_extractor']
        wav2vec_model = m['wav2vec_model']
        text_classifier = m['text_classifier']
        yamnet_model = m['yamnet_model']
        yamnet_classes = m['yamnet_classes']
        embedder = m['embedder']


# Device management API
def get_device_info() -> dict:
    """Get comprehensive device information."""
    return device_manager.device_info


def get_memory_usage() -> dict:
    """Get current memory usage across all devices."""
    return device_manager.get_memory_usage()


def clear_gpu_cache():
    """Clear GPU memory cache if available."""
    device_manager.clear_cache()


def optimize_model_for_device(model):
    """Optimize any model for the current optimal device."""
    return device_manager.optimize_for_inference(model)


def get_onnx_providers(prefer_gpu: bool = None) -> list:
    """Get optimal ONNX Runtime execution providers."""
    return device_manager.get_onnx_providers(prefer_gpu)


def print_device_summary():
    """Print comprehensive device configuration summary."""
    print("\n" + "="*50)
    print("DEVICE CONFIGURATION SUMMARY")
    print("="*50)
    print(device_manager.get_device_summary())
    
    memory_usage = device_manager.get_memory_usage()
    print(f"Current Memory Usage:")
    print(f"  CPU: {memory_usage['cpu_percent']:.1f}%")
    print(f"  System RAM: {memory_usage['system_memory_mb']}MB")
    if memory_usage['gpu_memory_mb'] > 0:
        print(f"  GPU Memory: {memory_usage['gpu_memory_mb']}MB")
        print(f"  GPU Cached: {memory_usage['gpu_memory_cached_mb']}MB")
    
    onnx_providers = device_manager.get_onnx_providers()
    print(f"ONNX Providers: {', '.join(onnx_providers)}")
    print("="*50 + "\n")
