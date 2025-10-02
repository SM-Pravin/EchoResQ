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

# Device selection for PyTorch models
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# For huggingface pipeline device argument: 0 for cuda:0, -1 for CPU
PIPELINE_DEVICE = 0 if TORCH_DEVICE.type == "cuda" else -1

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

print("[INIT] Loading models... device:", TORCH_DEVICE)

# --------------------
# VOSK (Speech-to-Text)
# --------------------
try:
    from vosk import Model as VoskModel
    if os.path.isdir(VOSK_PATH):
        vosk_model = VoskModel(VOSK_PATH)
        print(" ✅ Vosk model loaded")
    else:
        print(f" ⚠️ Vosk model not found at: {VOSK_PATH}")
        vosk_model = None
except Exception as e:
    print(f" ⚠️ Failed to initialize Vosk model: {e}")
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
            print(" ✅ Wav2Vec2 model loaded (device: {})".format(TORCH_DEVICE))
        except Exception as e:
            print(f" ⚠️ Failed to load Wav2Vec2 from {WAV2VEC2_PATH}: {e}")
            audio_feature_extractor = None
            wav2vec_model = None
    else:
        print(f" ⚠️ Wav2Vec2 model folder not found at: {WAV2VEC2_PATH}")
        audio_feature_extractor = None
        wav2vec_model = None
except Exception as e:
    print(f" ⚠️ Transformers not available or error loading wav2vec2: {e}")
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
            print(" ✅ DistilRoBERTa loaded (device: {})".format(TORCH_DEVICE))
        except Exception as e:
            print(f" ⚠️ Failed to load DistilRoBERTa from {DISTILROBERTA_PATH}: {e}")
            text_classifier = None
    else:
        print(f" ⚠️ DistilRoBERTa model folder not found at: {DISTILROBERTA_PATH}")
        text_classifier = None
except Exception as e:
    print(f" ⚠️ Transformers not available or error loading DistilRoBERTa: {e}")
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
                print(f" ⚠️ YAMNet class map not found at {class_map_path}")
            print(" ✅ YAMNet loaded")
        except Exception as e:
            print(f" ⚠️ Failed to load YAMNet from {YAMNET_PATH}: {e}")
            yamnet_model = None
            yamnet_classes = []
    else:
        print(f" ⚠️ YAMNet folder not found at: {YAMNET_PATH}")
        yamnet_model = None
        yamnet_classes = []
except Exception as e:
    print(f" ⚠️ TensorFlow not available or error loading YAMNet: {e}")
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
            print(" ✅ SentenceTransformer loaded from local folder")
        except Exception:
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE.type)
            print(" ✅ SentenceTransformer loaded from cache/hub")
    else:
        # fallback to cached/hub model (this will require internet if not cached)
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE.type)
            print(" ✅ SentenceTransformer loaded")
        except Exception as e:
            print(f" ⚠️ Failed to load SentenceTransformer: {e}")
            embedder = None
except Exception as e:
    print(f" ⚠️ sentence-transformers not available: {e}")
    embedder = None

print("[INIT] All models initialized.\n")
