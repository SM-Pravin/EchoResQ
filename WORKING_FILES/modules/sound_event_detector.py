# modules/sound_event_detector.py
import numpy as np
import librosa
from modules.model_loader import yamnet_model, yamnet_classes, get_model, get_models

CRITICAL_SOUNDS = {
    "siren": "peak emergency distress",
    "gunshot": "peak emergency distress",
    "explosion": "peak emergency distress",
    "screaming": "high distress",
    "fire alarm": "high distress",
    "crying": "medium distress",
    "glass breaking": "high distress",
}

def analyze_sound_events(audio_input, threshold=0.3):
    """Detect emergency-related sounds using YAMNet (if available).
    Accepts either a filename (str) or ndarray or dict {'data':ndarray,'sr':int}
    Returns list of tuples: (sound_label, score, mapped_distress)
    """
    ym = yamnet_model or get_model('yamnet_model')
    yc = yamnet_classes or get_model('yamnet_classes')
    if ym is None or not yc:
        return []

    try:
        if isinstance(audio_input, str):
            waveform, sr = librosa.load(audio_input, sr=16000, mono=True)
        elif isinstance(audio_input, dict) and 'data' in audio_input:
            waveform = np.asarray(audio_input['data'], dtype=np.float32)
            sr = int(audio_input.get('sr', 16000))
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input.astype(np.float32)
            sr = 16000
        else:
            return []

        # yamnet expects float32 1-D waveform at 16k
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        # run yamnet model
        scores, embeddings, spectrogram = ym(waveform)
        # scores: (frames, classes)
        scores = scores.numpy()
        mean_scores = np.mean(scores, axis=0)
        top_idxs = np.argsort(mean_scores)[::-1][:10]

        detected = []
        for idx in top_idxs:
            sound_label = yc[idx].lower()
            score = float(mean_scores[idx])
            if sound_label in CRITICAL_SOUNDS and score > threshold:
                detected.append((sound_label, score, CRITICAL_SOUNDS[sound_label]))
        return detected

    except Exception as e:
        print(f" [WARNING] Error analyzing sounds: {e}")
        return []
