# modules/emotion_audio.py
import torch
import numpy as np
import librosa
from modules.model_loader import audio_feature_extractor, wav2vec_model, TORCH_DEVICE

def analyze_audio_emotion(audio_input, sr=16000):
    """
    Returns dict {label: score}.
    Accepts:
      - a str path to a file
      - a numpy ndarray (mono, float32, range -1..1)
      - a dict with keys { 'data': ndarray, 'sr': int }
    If model missing -> {}.
    """
    if audio_feature_extractor is None or wav2vec_model is None:
        return {}

    try:
        # get waveform and sampling rate
        if isinstance(audio_input, str):
            speech, s = librosa.load(audio_input, sr=sr, mono=True)
        elif isinstance(audio_input, dict) and 'data' in audio_input:
            speech = np.asarray(audio_input['data'])
            s = int(audio_input.get('sr', sr))
        elif isinstance(audio_input, np.ndarray):
            speech = audio_input
            s = sr
        else:
            return {}

        # ensure float32 and 1D
        speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        inputs = audio_feature_extractor(speech, sampling_rate=s, return_tensors="pt", padding=True)
        # Move tensors to model device if possible
        try:
            device = next(wav2vec_model.parameters()).device
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)
        except Exception:
            pass

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        labels = wav2vec_model.config.id2label
        # id2label may be dict like {0: 'ang', 1: 'hap', ...}
        return {labels[int(i)].lower(): float(probs[int(i)]) for i in range(len(probs))}
    except Exception as e:
        print(f" ⚠️ Failed audio emotion analysis: {e}")
        return {}
