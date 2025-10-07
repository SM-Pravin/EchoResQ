# modules/speech_to_text.py
import json
import numpy as np
import tempfile

# Optional imports with fallbacks
try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False
    print('[WARNING] soundfile not available in speech_to_text. Install with: pip install soundfile')
    # Minimal fallback for sf.write using wave module
    import wave
    import contextlib
    def _sf_write_fallback(path, data, samplerate):
        """Minimal WAV writer fallback using wave module."""
        data = np.asarray(data)
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1 if data.ndim == 1 else data.shape[1])
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(int(samplerate))
            # Convert to int16 PCM
            maxv = float(2 ** 15 - 1)
            intdata = (data * maxv).astype(np.int16)
            wf.writeframes(intdata.tobytes())
    
    class _SoundFileFallback:
        write = staticmethod(_sf_write_fallback)
    sf = _SoundFileFallback()

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print('[WARNING] librosa not available in speech_to_text. Install with: pip install librosa')
    # Minimal fallback for librosa.load using wave module only
    def _librosa_load_fallback(path, sr=None, mono=True):
        """Minimal audio loading fallback using wave module for WAV files."""
        # Fallback to wave module for WAV files
        import wave
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            sr_file = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            sampwidth = wf.getsampwidth()
            if sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            else:
                data = np.frombuffer(frames, dtype=np.float32)
            if sr is not None and sr_file != sr and len(data) > 0:
                duration = len(data) / float(sr_file)
                new_len = int(max(1, round(duration * sr)))
                old_times = np.linspace(0, duration, num=len(data))
                new_times = np.linspace(0, duration, num=new_len)
                data = np.interp(new_times, old_times, data).astype(np.float32)
                sr_file = sr
            return data, sr_file
    
    class _LibrosaFallback:
        load = staticmethod(_librosa_load_fallback)
    librosa = _LibrosaFallback()
from modules.model_loader import whisper_medium, get_model
from modules.keyword_detector import EMERGENCY_KEYWORDS

TARGET_SR = 16000
FRAME_SECONDS = 0.5

# Streaming configuration
STREAMING_BUFFER_SIZE = int(TARGET_SR * 2.0)  # 2 second buffer for streaming
PARTIAL_EMIT_THRESHOLD = 0.3  # seconds of audio before emitting partial results


def transcribe_audio_buffer(audio_buffer, sample_rate=TARGET_SR):
    """
    Transcribe AudioBuffer object directly without file I/O.
    """
    from modules.in_memory_audio import AudioBuffer
    
    # Prefer faster-whisper if available
    wm = whisper_medium or get_model('whisper_medium')
    if wm is not None:
        try:
            # Write buffer to a temp wav file and use faster-whisper file-based transcribe for stability
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tf:
                sf.write(tf.name, audio_buffer.data, audio_buffer.sample_rate)
                segments, info = wm.transcribe(tf.name, beam_size=5)
                # faster-whisper may return Segment objects (attrs) or dicts. Normalize.
                def _seg_get(s, k, default=None):
                    if hasattr(s, k):
                        return getattr(s, k)
                    try:
                        return s.get(k, default)
                    except Exception:
                        return default

                texts = [(_seg_get(seg, 'text', '') or '').strip() for seg in segments if (_seg_get(seg, 'text', '') or '').strip()]
                return ' '.join(texts).strip()
        except Exception as e:
            print(f"[WARNING] Whisper buffer transcription failed: {e}")

    # Use faster-whisper if available; otherwise return empty string
    wm = whisper_medium or get_model('whisper_medium')
    if wm is None:
        return ""

    try:
        # Get audio data from buffer
        audio_data = audio_buffer.data
        buffer_sr = audio_buffer.sample_rate
        
        # Resample if needed
        if buffer_sr != TARGET_SR:
            if LIBROSA_AVAILABLE:
                audio_data = librosa.resample(audio_data, orig_sr=buffer_sr, target_sr=TARGET_SR)
            else:
                # Simple linear interpolation fallback
                duration = len(audio_data) / float(buffer_sr)
                new_len = int(max(1, round(duration * TARGET_SR)))
                old_times = np.linspace(0, duration, num=len(audio_data))
                new_times = np.linspace(0, duration, num=new_len)
                audio_data = np.interp(new_times, old_times, audio_data).astype(np.float32)
        
        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Write to temp wav and use whisper model
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tf:
            sf.write(tf.name, audio_data, buffer_sr)
            segments, info = wm.transcribe(tf.name, beam_size=5)
            def _seg_get(s, k, default=None):
                if hasattr(s, k):
                    return getattr(s, k)
                try:
                    return s.get(k, default)
                except Exception:
                    return default

            texts = [(_seg_get(seg, 'text', '') or '').strip() for seg in segments if (_seg_get(seg, 'text', '') or '').strip()]
            return ' '.join(texts).strip()
    except Exception as e:
        print(f"[ERROR] Whisper buffer transcription error: {e}")
        return ""

def transcribe_audio(audio_path, sample_rate=TARGET_SR):
    """
    Transcribe an audio file using faster-whisper (Whisper medium).
    Falls back to returning an empty string if the whisper model is not available or fails.
    """
    wm = whisper_medium or get_model('whisper_medium')
    if wm is None:
        return ""

    try:
        segments, info = wm.transcribe(audio_path, beam_size=5)
        def _seg_get(s, k, default=None):
            if hasattr(s, k):
                return getattr(s, k)
            try:
                return s.get(k, default)
            except Exception:
                return default

        texts = [(_seg_get(seg, 'text', '') or '').strip() for seg in segments if (_seg_get(seg, 'text', '') or '').strip()]
        return ' '.join(texts).strip()
    except Exception as e:
        print(f" [WARNING] Whisper transcription failed: {e}")
        return ""


def transcribe_audio_streaming(audio_path, partial_callback=None, keyword_callback=None, sample_rate=TARGET_SR):
    """
    Streaming transcription with partial results and early keyword detection.
    
    Args:
        audio_path: path to audio file
        partial_callback: function called with partial transcription results
        keyword_callback: function called when emergency keywords are detected
        sample_rate: target sampling rate
    
    Returns:
        dict with:
            - final_transcript: complete transcription
            - partial_results: list of partial results
            - keywords_detected: list of (keyword, timestamp) tuples
    """
    # Use faster-whisper for streaming results (segments)
    wm = whisper_medium or get_model('whisper_medium')
    if wm is None:
        return {"final_transcript": "", "partial_results": [], "keywords_detected": []}

    results = {
        "final_transcript": "",
        "partial_results": [],
        "keywords_detected": []
    }

    try:
        # Load and prepare audio
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

        # Normalize
        maxv = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
        if maxv > 0:
            audio = audio / maxv

        # Use faster-whisper to get segments and timestamps
            try:
                segments, info = wm.transcribe(audio_path, beam_size=5)
                transcript_parts = []

                def _seg_get(s, k, default=None):
                    if hasattr(s, k):
                        return getattr(s, k)
                    try:
                        return s.get(k, default)
                    except Exception:
                        return default

                for seg in segments:
                    text = (_seg_get(seg, 'text', '') or '').strip()
                    start = float(_seg_get(seg, 'start', 0.0) or 0.0)
                    if text:
                        transcript_parts.append(text)
                        results['partial_results'].append({
                            'text': text,
                            'timestamp': start,
                            'is_final': True
                        })
                        if partial_callback:
                            try:
                                partial_callback(text, start, True)
                            except Exception:
                                pass

                        for keyword in EMERGENCY_KEYWORDS:
                            try:
                                if keyword.lower() in text.lower():
                                    kd = (keyword, start)
                                    results['keywords_detected'].append(kd)
                                    if keyword_callback:
                                        try:
                                            keyword_callback(keyword, start, text)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                results['final_transcript'] = ' '.join(transcript_parts).strip()
                return results
            except Exception as e:
                print(f" [WARNING] Whisper streaming transcription failed: {e}")
                return results
    except Exception as e:
        print(f" [ERROR] Whisper streaming transcription error: {e}")
        return results


def check_emergency_keywords_realtime(text, timestamp=0.0):
    """
    Quick emergency keyword detection for real-time processing.
    Returns list of (keyword, severity) tuples found in text.
    """
    if not text:
        return []
    
    detected = []
    text_lower = text.lower()
    
    # High priority keywords (immediate response needed)
    high_priority = ["help", "emergency", "fire", "police", "ambulance", "911", "urgent", "crisis"]
    
    # Critical keywords (life-threatening)
    critical = ["dying", "suicide", "kill", "murder", "rape", "attack", "bomb", "gun", "knife"]
    
    for keyword in critical:
        if keyword in text_lower:
            detected.append((keyword, "critical"))
    
    for keyword in high_priority:
        if keyword in text_lower:
            detected.append((keyword, "high"))
    
    # Check EMERGENCY_KEYWORDS for medium priority
    for keyword in EMERGENCY_KEYWORDS:
        if keyword.lower() in text_lower and not any(kw[0] == keyword for kw in detected):
            detected.append((keyword, "medium"))
    
    return detected

