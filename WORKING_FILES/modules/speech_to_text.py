# modules/speech_to_text.py
import json
import numpy as np
import soundfile as sf
import librosa
from vosk import KaldiRecognizer
from modules.model_loader import vosk_model

TARGET_SR = 16000
FRAME_SECONDS = 0.5  # feed Vosk 0.5s frames

def transcribe_audio(audio_path, sample_rate=TARGET_SR):
    """
    Robust Vosk transcription:
     - reads as float32
     - converts to mono
     - resamples if needed
     - converts to int16 PCM before feeding KaldiRecognizer
     - returns a single concatenated transcript string
    """
    if vosk_model is None:
        # model_loader already prints availability; be quiet here
        return ""

    try:
        # read as float32 (so we can safely resample / mono-average)
        audio, sr = sf.read(audio_path, dtype="float32")
        # audio shape: (n,) or (n, channels)
        if audio.ndim > 1:
            # average channels to mono
            audio = np.mean(audio, axis=1)

        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

        # normalize to -1..1 if not already
        maxv = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
        if maxv > 0:
            audio = audio / maxv

        # convert to PCM16
        pcm16 = (audio * 32767.0).astype('<i2')  # little-endian int16

        rec = KaldiRecognizer(vosk_model, sample_rate)
        rec.SetWords(True)

        transcript_parts = []
        frame_size = int(FRAME_SECONDS * sample_rate)
        for i in range(0, len(pcm16), frame_size):
            frame = pcm16[i:i+frame_size].tobytes()
            if rec.AcceptWaveform(frame):
                try:
                    res = json.loads(rec.Result())
                    t = res.get("text", "")
                    if t:
                        transcript_parts.append(t)
                except Exception:
                    pass

        # final
        try:
            final_res = json.loads(rec.FinalResult())
            if "text" in final_res and final_res["text"]:
                transcript_parts.append(final_res["text"])
        except Exception:
            pass

        return " ".join([t for t in transcript_parts if t]).strip()

    except Exception as e:
        # Don't crash main; upstream handles empty transcript
        print(f" ⚠️ Error in Vosk transcription: {e}")
        return ""

