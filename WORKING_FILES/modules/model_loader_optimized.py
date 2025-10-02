# modules/model_loader_optimized.py
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

# Enhanced device selection
def get_optimal_device():
    """Get optimal device based on availability and environment settings."""
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                free_memory = int(result.stdout.strip().split('\n')[0])
                if free_memory > 2048:  # 2GB minimum
                    return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cuda")
    
    return torch.device("cpu")

# Device configuration
TORCH_DEVICE = get_optimal_device()
USE_GPU = TORCH_DEVICE.type == "cuda"
PIPELINE_DEVICE = 0 if USE_GPU else -1

# Model paths
BASE = os.path.join(os.getcwd(), "models")
VOSK_PATH = os.path.join(BASE, "vosk-model-large-en-us")
WAV2VEC2_PATH = os.path.join(BASE, "wav2vec2")
DISTILROBERTA_PATH = os.path.join(BASE, "distilroberta")
YAMNET_PATH = os.path.join(BASE, "yamnet")

def _load_models():
    """Load all models with thread safety."""
    global _models_initialized, _models
    
    if _models_initialized:
        return _models
    
    with _loading_lock:
        if _models_initialized:  # Double-check pattern
            return _models
        
        print(f"[DEVICE] Selected device: {TORCH_DEVICE} (GPU enabled: {USE_GPU})")
        print("[INIT] Loading models... device:", TORCH_DEVICE)
        
        # Initialize model dictionary
        _models = {
            'vosk_model': None,
            'audio_feature_extractor': None,
            'wav2vec_model': None,
            'text_classifier': None,
            'yamnet_model': None,
            'yamnet_classes': [],
            'embedder': None,
            'TORCH_DEVICE': TORCH_DEVICE,
            'USE_GPU': USE_GPU,
            'PIPELINE_DEVICE': PIPELINE_DEVICE
        }
        
        # Load VOSK model
        try:
            from vosk import Model as VoskModel
            if os.path.isdir(VOSK_PATH):
                _models['vosk_model'] = VoskModel(VOSK_PATH)
                print(" Vosk model loaded")
            else:
                print(f" WARNING: Vosk model not found at: {VOSK_PATH}")
        except Exception as e:
            print(f" WARNING: Failed to initialize Vosk model: {e}")
        
        # Load Wav2Vec2 model
        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
            if os.path.isdir(WAV2VEC2_PATH):
                try:
                    _models['audio_feature_extractor'] = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_PATH, local_files_only=True)
                    _models['wav2vec_model'] = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC2_PATH, local_files_only=True)
                    _models['wav2vec_model'].to(TORCH_DEVICE)
                    _models['wav2vec_model'].eval()
                    print(" Wav2Vec2 model loaded (device: {})".format(TORCH_DEVICE))
                except Exception as e:
                    print(f" WARNING: Failed to load Wav2Vec2 from {WAV2VEC2_PATH}: {e}")
            else:
                print(f" WARNING: Wav2Vec2 model folder not found at: {WAV2VEC2_PATH}")
        except Exception as e:
            print(f" WARNING: Transformers not available or error loading wav2vec2: {e}")
        
        # Load DistilRoBERTa model
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            if os.path.isdir(DISTILROBERTA_PATH):
                try:
                    tok = AutoTokenizer.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
                    model.to(TORCH_DEVICE)
                    _models['text_classifier'] = pipeline("text-classification", model=model, tokenizer=tok, device=PIPELINE_DEVICE)
                    print(" DistilRoBERTa loaded (device: {})".format(TORCH_DEVICE))
                except Exception as e:
                    print(f" WARNING: Failed to load DistilRoBERTa from {DISTILROBERTA_PATH}: {e}")
            else:
                print(f" WARNING: DistilRoBERTa model folder not found at: {DISTILROBERTA_PATH}")
        except Exception as e:
            print(f" WARNING: Transformers not available or error loading DistilRoBERTa: {e}")
        
        # Load YAMNet model
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
        
        # Load SentenceTransformer
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
        
        _models_initialized = True
        print("[INIT] All models initialized.\n")
        
    return _models

def get_models():
    """Get all models (thread-safe, loads once)."""
    return _load_models()

def get_model(name):
    """Get a specific model by name."""
    models = get_models()
    return models.get(name)

def is_loaded():
    """Check if models are already loaded."""
    return _models_initialized

# Initialize module-level variables for backward compatibility
def _ensure_compatibility():
    """Ensure backward compatibility with direct imports."""
    global vosk_model, audio_feature_extractor, wav2vec_model, text_classifier
    global yamnet_model, yamnet_classes, embedder
    
    if not _models_initialized:
        models = get_models()
        vosk_model = models['vosk_model']
        audio_feature_extractor = models['audio_feature_extractor']
        wav2vec_model = models['wav2vec_model']
        text_classifier = models['text_classifier']
        yamnet_model = models['yamnet_model']
        yamnet_classes = models['yamnet_classes']
        embedder = models['embedder']

# Initialize compatibility variables
vosk_model = None
audio_feature_extractor = None
wav2vec_model = None
text_classifier = None
yamnet_model = None
yamnet_classes = []
embedder = None

# Load models only if this is the first import
if not _models_initialized:
    _ensure_compatibility()
