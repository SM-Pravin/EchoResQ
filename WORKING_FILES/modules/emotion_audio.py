# modules/emotion_audio.py
import torch
import numpy as np
import librosa
from modules.model_loader import audio_feature_extractor, wav2vec_model

def analyze_audio_emotion(audio_file):
    """
    Returns dict {label: score}. If model missing -> {}.
    """
    if audio_feature_extractor is None or wav2vec_model is None:
        return {}

    try:
        speech, sr = librosa.load(audio_file, sr=16000, mono=True)
        inputs = audio_feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = wav2vec_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        labels = wav2vec_model.config.id2label
        # id2label may be dict like {0: 'ang', 1: 'hap', ...}
        return {labels[int(i)].lower(): float(probs[int(i)]) for i in range(len(probs))}
    except Exception as e:
        print(f" ⚠️ Failed audio emotion analysis: {e}")
        return {}

