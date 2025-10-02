# modules/model_loader.py
"""
Centralized model loader for the emergency_ai project.

Loads models once (offline/local-first). Exposes:
- vosk_model
- audio_feature_extractor
- wav2vec_model
- text_classifier
- yamnet_model
- yamnet_classes
- embedder

This file is defensive: if a model folder is missing or a model fails to load,
it sets the corresponding variable to None and prints a clear message.
"""

import os
import threading

# Global lock to prevent concurrent model loading
_loading_lock = threading.Lock()
_models_initialized = False

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF logs if TF present

# Suppress Vosk C++ logs before importing/initializing model
try:
    from vosk import SetLogLevel
    SetLogLevel(-1)
except Exception:
    # If Vosk not installed or SetLogLevel fails, continue silently
    pass

import traceback
import torch

# Enhanced device selection with user control
def get_optimal_device():
    """Get optimal device based on availability and environment settings."""
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        # Check if GPU has enough memory (at least 2GB free)
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
        
        # Fallback: just check if CUDA is available
        return torch.device("cuda")
    
    return torch.device("cpu")

# Device selection for PyTorch models
TORCH_DEVICE = get_optimal_device()
USE_GPU = TORCH_DEVICE.type == "cuda"
# For huggingface pipeline device argument: 0 for cuda:0, -1 for CPU
PIPELINE_DEVICE = 0 if USE_GPU else -1

print(f"[DEVICE] Selected device: {TORCH_DEVICE} (GPU enabled: {USE_GPU})")

# Initialize model refs
vosk_model = None
audio_feature_extractor = None
wav2vec_model = None
text_classifier = None
yamnet_model = None
yamnet_classes = []
embedder = None

# Paths (adjust these if you keep models in different structure)
BASE = os.path.join(os.getcwd(), "models")
VOSK_PATH = os.path.join(BASE, "vosk-model-large-en-us")
WAV2VEC2_PATH = os.path.join(BASE, "wav2vec2")
DISTILROBERTA_PATH = os.path.join(BASE, "distilroberta")
YAMNET_PATH = os.path.join(BASE, "yamnet")

def _initialize_models():
    """Initialize models with thread safety."""
    global _models_initialized, vosk_model, audio_feature_extractor, wav2vec_model
    global text_classifier, yamnet_model, yamnet_classes, embedder
    
    if _models_initialized:
        return
    
    with _loading_lock:
        if _models_initialized:  # Double-check pattern
            return
        
        print("[INIT] Loading models... device:", TORCH_DEVICE)

        # --------------------
        # VOSK (Speech-to-Text)
        # --------------------
        try:
            from vosk import Model as VoskModel
            if os.path.isdir(VOSK_PATH):
                vosk_model = VoskModel(VOSK_PATH)
                print(" Vosk model loaded")
            else:
                print(f" WARNING: Vosk model not found at: {VOSK_PATH}")
                vosk_model = None
        except Exception as e:
            print(f" WARNING: Failed to initialize Vosk model: {e}")
            vosk_model = None

# --------------------
# Wav2Vec2 (Audio Emotion)
# --------------------
try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    if os.path.isdir(WAV2VEC2_PATH):
        try:
            audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_PATH, local_files_only=True)
            wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC2_PATH, local_files_only=True)
            wav2vec_model.to(TORCH_DEVICE)
            wav2vec_model.eval()
            print(" Wav2Vec2 model loaded (device: {})".format(TORCH_DEVICE))
        except Exception as e:
            print(f" WARNING: Failed to load Wav2Vec2 from {WAV2VEC2_PATH}: {e}")
            audio_feature_extractor = None
            wav2vec_model = None
    else:
        print(f" WARNING: Wav2Vec2 model folder not found at: {WAV2VEC2_PATH}")
        audio_feature_extractor = None
        wav2vec_model = None
except Exception as e:
    print(f" WARNING: Transformers not available or error loading wav2vec2: {e}")
    audio_feature_extractor = None
    wav2vec_model = None

# --------------------
# DistilRoBERTa (Text Emotion)
# --------------------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    if os.path.isdir(DISTILROBERTA_PATH):
        try:
            tok = AutoTokenizer.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
            model.to(TORCH_DEVICE)
            text_classifier = pipeline("text-classification", model=model, tokenizer=tok, device=PIPELINE_DEVICE)
            print(" DistilRoBERTa loaded (device: {})".format(TORCH_DEVICE))
        except Exception as e:
            print(f" WARNING: Failed to load DistilRoBERTa from {DISTILROBERTA_PATH}: {e}")
            text_classifier = None
    else:
        print(f" WARNING: DistilRoBERTa model folder not found at: {DISTILROBERTA_PATH}")
        text_classifier = None
except Exception as e:
    print(f" WARNING: Transformers not available or error loading DistilRoBERTa: {e}")
    text_classifier = None

# --------------------
# YAMNet (Sound Events) - TensorFlow SavedModel
# --------------------
try:
    import tensorflow as tf
    if os.path.isdir(YAMNET_PATH):
        try:
            yamnet_model = tf.saved_model.load(YAMNET_PATH)
            class_map_path = os.path.join(YAMNET_PATH, "yamnet_class_map.csv")
            if os.path.isfile(class_map_path):
                with open(class_map_path, "r") as f:
                    lines = f.readlines()[1:]
                    yamnet_classes = [line.strip().split(",")[2] for line in lines]
            else:
                yamnet_classes = []
                print(f" WARNING: YAMNet class map not found at {class_map_path}")
            print(" YAMNet loaded")
        except Exception as e:
            print(f" WARNING: Failed to load YAMNet from {YAMNET_PATH}: {e}")
            yamnet_model = None
            yamnet_classes = []
    else:
        print(f" WARNING: YAMNet folder not found at: {YAMNET_PATH}")
        yamnet_model = None
        yamnet_classes = []
except Exception as e:
    print(f" WARNING: TensorFlow not available or error loading YAMNet: {e}")
    yamnet_model = None
    yamnet_classes = []

# --------------------
# Sentence-Transformer (embedder for semantic keyword matching)
# --------------------
try:
    from sentence_transformers import SentenceTransformer
    # Try to load local model folder first if present; otherwise attempt cache load
    local_embed_path = os.path.join(BASE, "all-MiniLM-L6-v2")
    if os.path.isdir(local_embed_path):
        try:
            embedder = SentenceTransformer(local_embed_path, device=TORCH_DEVICE.type)
            print(" SentenceTransformer loaded from local folder")
        except Exception:
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE.type)
            print(" SentenceTransformer loaded from cache/hub")
    else:
        # fallback to cached/hub model (this will require internet if not cached)
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE.type)
            print(" SentenceTransformer loaded")
        except Exception as e:
            print(f" WARNING: Failed to load SentenceTransformer: {e}")
            embedder = None
except Exception as e:
    print(f" WARNING: sentence-transformers not available: {e}")
    embedder = None

print("[INIT] All models initialized.\n")
