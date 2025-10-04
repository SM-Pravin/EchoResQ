# modules/speech_to_text.py
import json
import numpy as np
import soundfile as sf
import librosa
from vosk import KaldiRecognizer
from modules.model_loader import vosk_model, get_model
from modules.keyword_detector import EMERGENCY_KEYWORDS

TARGET_SR = 16000
FRAME_SECONDS = 0.5  # feed Vosk 0.5s frames

# Streaming configuration
STREAMING_BUFFER_SIZE = int(TARGET_SR * 2.0)  # 2 second buffer for streaming
PARTIAL_EMIT_THRESHOLD = 0.3  # seconds of audio before emitting partial results


def transcribe_audio_buffer(audio_buffer, sample_rate=TARGET_SR):
    """
    Transcribe AudioBuffer object directly without file I/O.
    """
    from modules.in_memory_audio import AudioBuffer
    
    vm = vosk_model or get_model('vosk_model')
    if vm is None:
        return ""
    
    try:
        # Get audio data from buffer
        audio_data = audio_buffer.data
        buffer_sr = audio_buffer.sample_rate
        
        # Resample if needed
        if buffer_sr != TARGET_SR:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=buffer_sr, target_sr=TARGET_SR)
        
        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to int16 PCM for Vosk
        audio_data = np.clip(audio_data, -1.0, 1.0)
        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
        
        # Create recognizer
        rec = KaldiRecognizer(vm, TARGET_SR)
        rec.SetWords(True)
        
        # Process audio
        results = []
        chunk_size = int(TARGET_SR * FRAME_SECONDS) * 2  # 2 bytes per sample
        
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                if result.get("text", "").strip():
                    results.append(result["text"].strip())
        
        # Get final result
        final_result = json.loads(rec.FinalResult())
        if final_result.get("text", "").strip():
            results.append(final_result["text"].strip())
        
        return " ".join(results).strip()
        
    except Exception as e:
        print(f"[ERROR] Buffer transcription error: {e}")
        return ""

def transcribe_audio(audio_path, sample_rate=TARGET_SR):
    """
    Robust Vosk transcription:
     - reads as float32
     - converts to mono
     - resamples if needed
     - converts to int16 PCM before feeding KaldiRecognizer
     - returns a single concatenated transcript string
    """
    vm = vosk_model or get_model('vosk_model')
    if vm is None:
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

        rec = KaldiRecognizer(vm, sample_rate)
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
        print(f" [WARNING] Error in Vosk transcription: {e}")
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
    vm = vosk_model or get_model('vosk_model')
    if vm is None:
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

        # Convert to PCM16
        pcm16 = (audio * 32767.0).astype('<i2')
        
        rec = KaldiRecognizer(vm, sample_rate)
        rec.SetWords(True)

        transcript_parts = []
        frame_size = int(FRAME_SECONDS * sample_rate)
        time_offset = 0.0
        
        for i in range(0, len(pcm16), frame_size):
            frame = pcm16[i:i+frame_size].tobytes()
            current_time = i / sample_rate
            
            if rec.AcceptWaveform(frame):
                try:
                    res = json.loads(rec.Result())
                    text = res.get("text", "")
                    if text:
                        transcript_parts.append(text)
                        
                        # Check for emergency keywords
                        text_lower = text.lower()
                        for keyword in EMERGENCY_KEYWORDS:
                            if keyword.lower() in text_lower:
                                keyword_detection = (keyword, current_time)
                                results["keywords_detected"].append(keyword_detection)
                                if keyword_callback:
                                    try:
                                        keyword_callback(keyword, current_time, text)
                                    except Exception:
                                        pass
                        
                        # Add to results
                        results["partial_results"].append({
                            "text": text,
                            "timestamp": current_time,
                            "is_final": True
                        })
                        
                        if partial_callback:
                            try:
                                partial_callback(text, current_time, True)
                            except Exception:
                                pass
                                
                except Exception:
                    pass
            else:
                # Partial result
                try:
                    partial_res = json.loads(rec.PartialResult())
                    partial_text = partial_res.get("partial", "")
                    if partial_text and current_time >= PARTIAL_EMIT_THRESHOLD:
                        
                        # Check partial text for urgent keywords
                        partial_lower = partial_text.lower()
                        for keyword in EMERGENCY_KEYWORDS:
                            if keyword.lower() in partial_lower:
                                keyword_detection = (keyword, current_time)
                                results["keywords_detected"].append(keyword_detection)
                                if keyword_callback:
                                    try:
                                        keyword_callback(keyword, current_time, partial_text)
                                    except Exception:
                                        pass
                        
                        results["partial_results"].append({
                            "text": partial_text,
                            "timestamp": current_time,
                            "is_final": False
                        })
                        
                        if partial_callback:
                            try:
                                partial_callback(partial_text, current_time, False)
                            except Exception:
                                pass
                except Exception:
                    pass

        # Final result
        try:
            final_res = json.loads(rec.FinalResult())
            if "text" in final_res and final_res["text"]:
                final_text = final_res["text"]
                transcript_parts.append(final_text)
                
                # Final keyword check
                final_lower = final_text.lower()
                for keyword in EMERGENCY_KEYWORDS:
                    if keyword.lower() in final_lower:
                        keyword_detection = (keyword, len(pcm16) / sample_rate)
                        results["keywords_detected"].append(keyword_detection)
                        if keyword_callback:
                            try:
                                keyword_callback(keyword, len(pcm16) / sample_rate, final_text)
                            except Exception:
                                pass
                
                results["partial_results"].append({
                    "text": final_text,
                    "timestamp": len(pcm16) / sample_rate,
                    "is_final": True
                })
        except Exception:
            pass

        results["final_transcript"] = " ".join([t for t in transcript_parts if t]).strip()
        return results

    except Exception as e:
        print(f" [WARNING] Error in streaming transcription: {e}")
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

