import numpy as np
import librosa
from modules.model_loader import yamnet_model, yamnet_classes

CRITICAL_SOUNDS = {
    "siren": "peak emergency distress",
    "gunshot": "peak emergency distress",
    "explosion": "peak emergency distress",
    "screaming": "high distress",
    "fire alarm": "high distress",
    "crying": "medium distress",
    "glass breaking": "high distress",
}


def analyze_sound_events(audio_path, threshold=0.3):
    """Detect emergency-related sounds using YAMNet (if available)."""
    if yamnet_model is None or not yamnet_classes:
        print(" ⚠️ YAMNet model not available, skipping sound analysis.")
        return []

    try:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)

        scores, _, _ = yamnet_model(waveform)
        scores = scores.numpy()

        mean_scores = np.mean(scores, axis=0)
        top_idxs = np.argsort(mean_scores)[::-1][:10]

        detected = []
        for idx in top_idxs:
            sound_label = yamnet_classes[idx].lower()
            score = mean_scores[idx]
            if sound_label in CRITICAL_SOUNDS and score > threshold:
                detected.append((sound_label, float(score), CRITICAL_SOUNDS[sound_label]))

        return detected

    except Exception as e:
        print(f" ⚠️ Error analyzing sounds: {e}")
        return []

