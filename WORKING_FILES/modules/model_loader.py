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


def _load_wav2vec2_onnx():
    """Load ONNX runtime session for Wav2Vec2 emotion model."""
    try:
        import onnxruntime as ort
        # Determine providers
        use_cuda_pref = cfg.get_use_onnx_cuda("auto").lower()
        providers = ['CPUExecutionProvider']
        if use_cuda_pref in ("true", "auto") and torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        onnx_path = os.path.join(ONNX_DIR, "wav2vec2_emotion.onnx")
        if os.path.isfile(onnx_path):
            _models['wav2vec_onnx'] = ort.InferenceSession(onnx_path, providers=providers)
            print(f" ONNX Wav2Vec2 session loaded (providers: {providers})")
        else:
            print(f" WARNING: ONNX Wav2Vec2 model not found at {onnx_path}")
    except Exception as e:
        print(f" WARNING: Failed to load ONNX Wav2Vec2: {e}")


def _load_distilroberta():
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


def _load_distilroberta_onnx():
    """Load ONNX runtime session for DistilRoBERTa text emotion model."""
    try:
        import onnxruntime as ort
        use_cuda_pref = cfg.get_use_onnx_cuda("auto").lower()
        providers = ['CPUExecutionProvider']
        if use_cuda_pref in ("true", "auto") and torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        onnx_path = os.path.join(ONNX_DIR, "distilroberta_emotion.onnx")
        if os.path.isfile(onnx_path):
            _models['text_onnx'] = ort.InferenceSession(onnx_path, providers=providers)
            print(f" ONNX DistilRoBERTa session loaded (providers: {providers})")
        else:
            print(f" WARNING: ONNX DistilRoBERTa model not found at {onnx_path}")
    except Exception as e:
        print(f" WARNING: Failed to load ONNX DistilRoBERTa: {e}")


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
    for key in ['vosk_model', 'audio_feature_extractor', 'wav2vec_model', 'text_classifier', 'yamnet_model', 'yamnet_classes', 'embedder']:
        _ensure_model(key)
    print("[INIT] All models initialized.\n")
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
