"""
Main entrypoint for the emergency AI pipeline.
Extended to:
 - return per-chunk details for visualization
 - provide a streaming-style function that processes chunks sequentially
   and calls a user-supplied callback for each chunk (useful for live UI updates).
"""
import os
import sys
import tempfile
import warnings
import traceback
import time

# Environment tweaks should be set early
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore")
try:
    # Explicitly silence TensorFlow's Python-level warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import numpy as np
import soundfile as sf
import librosa
from modules.audio_preprocessing import preprocess_audio, split_audio_chunks
from modules.speech_to_text import transcribe_audio
from modules.emotion_audio import analyze_audio_emotion
from modules.emotion_text import analyze_text_emotion
from modules.fusion_engine import fuse_emotions
from modules.distress_mapper import get_distress_token
from modules.keyword_detector import check_keywords, severity_level
from modules.sound_event_detector import analyze_sound_events
from modules.logger import log_call

# Parameters
CHUNK_SIZE_SECONDS = 30
HOP_SIZE_SECONDS = 15
SILENCE_RMS_THRESHOLD_RATIO = 0.06
TEMP_DIR = "tmp_chunks"
# Sleep bounds used only when simulate_realtime=True in streaming function
STREAM_MIN_SLEEP = 0.15
STREAM_MAX_SLEEP = 1.2

# --- HELPER FUNCTIONS from original main.py (remove_temp_files, etc.) ---
def remove_temp_files(prefix="chunk_"):
    try:
        import glob
        files = glob.glob(os.path.join(TEMP_DIR, f"{prefix}*.wav"))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception:
        pass


def _safe_divide(a, b):
    return (a / b) if b else 0.0


# --- MAIN PROCESSING FUNCTION ---
def process_audio_file(audio_file, fast_mode=False, return_chunks_details=False):
    """
    Analyzes an audio file and returns a dictionary with the results.

    New fields when return_chunks_details=True:
      - 'fused_scores': dict of canonical emotion -> score (final fused)
      - 'chunks': list of chunk-level dicts:
          {index, start_s, end_s, rms, win_emotion, win_conf, win_fused, win_distress, sounds}
    """
    results = {
        "caller_id": "UNKNOWN", "transcript": "", "emotion": "UNKNOWN",
        "confidence": 0.0, "distress": "low distress", "sounds": [],
        "reason": "", "error": None, "fused_scores": {}, "chunks": []
    }
    fixed_file = None

    try:
        # Preprocess to 16k mono PCM WAV in a temporary directory
        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(audio_file)
        fixed_file = os.path.join(temp_dir, f"fixed_{base_name}")
        preprocess_audio(audio_file, fixed_file)

        caller_id = os.path.basename(audio_file).split(".")[0].upper()
        results["caller_id"] = caller_id

        # Full-file transcription
        transcript = transcribe_audio(fixed_file) or ""
        results["transcript"] = transcript

        # Text emotion (full transcript)
        text_scores = analyze_text_emotion(transcript)

        # FAST MODE
        if fast_mode:
            audio_scores = analyze_audio_emotion(fixed_file)
            confidence, emotion, fused_scores = fuse_emotions(audio_scores, text_scores)
            distress = get_distress_token(emotion, confidence)
            distress = check_keywords(transcript, distress)

            results.update({
                "emotion": emotion, "confidence": confidence, "distress": distress,
                "reason": "fast mode: skipped chunked audio & sound analysis",
                "fused_scores": fused_scores
            })
            log_call(caller_id, transcript, emotion, distress, {"fused": fused_scores, "sounds": []}, results["reason"])
            return results

        # --- FULL ANALYSIS (CHUNKED) ---
        waveform, sr = librosa.load(fixed_file, sr=16000, mono=True)
        if waveform is None or len(waveform) == 0:
            waveform = np.zeros(1, dtype=np.float32)

        chunks = split_audio_chunks(fixed_file, max_chunk=CHUNK_SIZE_SECONDS, overlap=int(HOP_SIZE_SECONDS))

        canonical_labels = ["angry", "happy", "neutral", "sad"]
        aggregated_fused = {k: 0.0 for k in canonical_labels}
        total_weight = 0.0
        sound_events_all = []
        reason = None
        final_distress = "low distress"
        final_emotion = "neutral"

        # Precompute RMS for weighting
        max_rms = 0.0
        chunk_rms_map = {}
        chunk_duration_map = {}
        for chunk_path in chunks:
            try:
                y, s = sf.read(chunk_path, dtype="float32")
                if y.ndim > 1: y = y.mean(axis=1)
                rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
                chunk_rms_map[chunk_path] = rms
                duration = len(y) / float(s) if s > 0 else 0.0
                chunk_duration_map[chunk_path] = duration
                if rms > max_rms: max_rms = rms
            except Exception:
                chunk_rms_map[chunk_path] = 0.0
                chunk_duration_map[chunk_path] = 0.0

        silence_thresh = max_rms * SILENCE_RMS_THRESHOLD_RATIO if max_rms > 0 else 0.0

        chunks_info = []

        # Analyze each chunk
        for idx, chunk_path in enumerate(chunks):
            start_s = idx * (CHUNK_SIZE_SECONDS - HOP_SIZE_SECONDS)
            end_s = start_s + chunk_duration_map.get(chunk_path, CHUNK_SIZE_SECONDS)
            rms = chunk_rms_map.get(chunk_path, 0.0)
            if max_rms > 0 and rms < silence_thresh:
                # still add a quiet chunk info for timeline (optional)
                chunks_info.append({
                    "index": idx, "start_s": float(start_s), "end_s": float(end_s),
                    "rms": float(rms), "win_emotion": None, "win_conf": 0.0,
                    "win_fused": {k: 0.0 for k in canonical_labels}, "win_distress": "low distress",
                    "sounds": []
                })
                continue

            try:
                audio_scores = analyze_audio_emotion(chunk_path)
                win_conf, win_emotion, win_fused = fuse_emotions(audio_scores, {})

                win_sounds = analyze_sound_events(chunk_path) or []
                if win_sounds:
                    sound_events_all.extend(win_sounds)

                win_distress = get_distress_token(win_emotion, win_conf)
                if severity_level(win_distress) < severity_level(final_distress):
                    final_distress = win_distress
                    final_emotion = win_emotion
                    reason = f"emotion escalation (chunk {idx}): {win_emotion} ({win_conf:.2f})"

                weight = rms if max_rms > 0 else 1.0
                for k in canonical_labels:
                    aggregated_fused[k] += win_fused.get(k, 0.0) * weight
                total_weight += weight

                chunks_info.append({
                    "index": idx, "start_s": float(start_s), "end_s": float(end_s),
                    "rms": float(rms), "win_emotion": win_emotion, "win_conf": float(win_conf),
                    "win_fused": {k: float(win_fused.get(k, 0.0)) for k in canonical_labels},
                    "win_distress": win_distress, "sounds": win_sounds
                })

            except Exception as e:
                print(f" ⚠️ Error processing chunk {chunk_path}: {e}")
                chunks_info.append({
                    "index": idx, "start_s": float(start_s), "end_s": float(end_s),
                    "rms": float(rms), "win_emotion": None, "win_conf": 0.0,
                    "win_fused": {k: 0.0 for k in canonical_labels}, "win_distress": "low distress",
                    "sounds": []
                })
                continue

        # Normalize aggregated scores
        aggregated_audio_scores = {k: _safe_divide(aggregated_fused[k], total_weight) for k in canonical_labels} if total_weight > 0 else {}

        # Final fusion and distress mapping
        confidence, emotion, fused_scores = fuse_emotions(aggregated_audio_scores, text_scores)
        distress = get_distress_token(emotion, confidence)
        distress_after_kw = check_keywords(transcript, distress)

        if severity_level(distress_after_kw) < severity_level(distress):
            reason = f"keyword escalation: {distress_after_kw}"
            distress = distress_after_kw

        # Check against chunk-level escalations
        if severity_level(final_distress) < severity_level(distress):
            distress = final_distress

        # Safe fallback
        if distress not in ["peak emergency distress", "high distress", "medium distress", "low distress"]:
            distress = "low distress"

        results.update({
            "emotion": emotion, "confidence": confidence, "distress": distress,
            "sounds": sound_events_all, "reason": reason, "fused_scores": fused_scores, "chunks": chunks_info
        })

        log_call(caller_id, transcript, emotion, distress, {"fused": fused_scores, "sounds": sound_events_all}, reason)
        return results

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        traceback.print_exc()
        results["error"] = str(e)
        return results
    finally:
        # Cleanup temp files
        if fixed_file and os.path.exists(fixed_file):
            try:
                os.remove(fixed_file)
            except Exception:
                pass
        remove_temp_files()
        

# --- STREAMING / SIMULATED REALTIME PROCESSOR ---
def process_audio_file_stream(audio_file, fast_mode=False, chunk_callback=None,
                              simulate_realtime=True, chunk_size_seconds=CHUNK_SIZE_SECONDS,
                              hop_size_seconds=HOP_SIZE_SECONDS):
    """
    Processes a file and calls `chunk_callback(chunk_result)` for each processed chunk.
    Useful to update a UI in near-real-time.

    chunk_callback receives a dict:
      {
        "index", "start_s", "end_s", "rms",
        "win_emotion", "win_conf", "win_fused", "win_distress", "sounds"
      }

    Returns the final result dict (same shape as process_audio_file).
    """
    # Reuse much of process_audio_file but stream chunks to callback
    final_results = {
        "caller_id": "UNKNOWN", "transcript": "", "emotion": "UNKNOWN",
        "confidence": 0.0, "distress": "low distress", "sounds": [],
        "reason": "", "error": None, "fused_scores": {}, "chunks": []
    }

    fixed_file = None
    try:
        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(audio_file)
        fixed_file = os.path.join(temp_dir, f"fixed_{base_name}")
        preprocess_audio(audio_file, fixed_file)

        caller_id = os.path.basename(audio_file).split(".")[0].upper()
        final_results["caller_id"] = caller_id

        transcript = transcribe_audio(fixed_file) or ""
        final_results["transcript"] = transcript

        text_scores = analyze_text_emotion(transcript)

        if fast_mode:
            # fast_mode stream -> just call single 'chunk' update once
            audio_scores = analyze_audio_emotion(fixed_file)
            conf, emo, fused = fuse_emotions(audio_scores, text_scores)
            distress = get_distress_token(emo, conf)
            distress = check_keywords(transcript, distress)
            chunk_result = {
                "index": 0, "start_s": 0.0, "end_s": 0.0, "rms": 0.0,
                "win_emotion": emo, "win_conf": conf, "win_fused": fused, "win_distress": distress, "sounds": []
            }
            if chunk_callback:
                try:
                    chunk_callback(chunk_result)
                except Exception:
                    pass
            final_results.update({
                "emotion": emo, "confidence": conf, "distress": distress,
                "fused_scores": fused, "chunks": [chunk_result]
            })
            log_call(caller_id, transcript, emo, distress, {"fused": fused, "sounds": []}, "fast stream")
            return final_results

        # FULL chunked stream
        chunks = split_audio_chunks(fixed_file, max_chunk=chunk_size_seconds, overlap=int(hop_size_seconds))
        canonical_labels = ["angry", "happy", "neutral", "sad"]
        aggregated_fused = {k: 0.0 for k in canonical_labels}
        total_weight = 0.0
        sound_events_all = []
        final_distress = "low distress"
        final_emotion = "neutral"
        reason = None

        # Precompute RMS and durations
        max_rms = 0.0
        chunk_rms_map = {}
        chunk_duration_map = {}
        for chunk_path in chunks:
            try:
                y, s = sf.read(chunk_path, dtype="float32")
                if y.ndim > 1: y = y.mean(axis=1)
                rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
                chunk_rms_map[chunk_path] = rms
                duration = len(y) / float(s) if s > 0 else 0.0
                chunk_duration_map[chunk_path] = duration
                if rms > max_rms: max_rms = rms
            except Exception:
                chunk_rms_map[chunk_path] = 0.0
                chunk_duration_map[chunk_path] = 0.0

        silence_thresh = max_rms * SILENCE_RMS_THRESHOLD_RATIO if max_rms > 0 else 0.0
        chunk_results = []

        for idx, chunk_path in enumerate(chunks):
            start_s = idx * (chunk_size_seconds - hop_size_seconds)
            end_s = start_s + chunk_duration_map.get(chunk_path, chunk_size_seconds)
            rms = chunk_rms_map.get(chunk_path, 0.0)

            if max_rms > 0 and rms < silence_thresh:
                chunk_result = {
                    "index": idx, "start_s": float(start_s), "end_s": float(end_s),
                    "rms": float(rms), "win_emotion": None, "win_conf": 0.0,
                    "win_fused": {k: 0.0 for k in canonical_labels}, "win_distress": "low distress", "sounds": []
                }
                chunk_results.append(chunk_result)
                if chunk_callback:
                    try:
                        chunk_callback(chunk_result)
                    except Exception:
                        pass
                # optionally simulate a small delay to allow UI updates
                if simulate_realtime:
                    sleep_for = min(max(chunk_duration_map.get(chunk_path, 0.2) * 0.4, STREAM_MIN_SLEEP), STREAM_MAX_SLEEP)
                    time.sleep(sleep_for)
                continue

            try:
                audio_scores = analyze_audio_emotion(chunk_path)
                win_conf, win_emotion, win_fused = fuse_emotions(audio_scores, {})

                win_sounds = analyze_sound_events(chunk_path) or []
                if win_sounds:
                    sound_events_all.extend(win_sounds)

                win_distress = get_distress_token(win_emotion, win_conf)
                if severity_level(win_distress) < severity_level(final_distress):
                    final_distress = win_distress
                    final_emotion = win_emotion
                    reason = f"emotion escalation (chunk {idx}): {win_emotion} ({win_conf:.2f})"

                weight = rms if max_rms > 0 else 1.0
                for k in canonical_labels:
                    aggregated_fused[k] += win_fused.get(k, 0.0) * weight
                total_weight += weight

                chunk_result = {
                    "index": idx, "start_s": float(start_s), "end_s": float(end_s),
                    "rms": float(rms), "win_emotion": win_emotion, "win_conf": float(win_conf),
                    "win_fused": {k: float(win_fused.get(k, 0.0)) for k in canonical_labels},
                    "win_distress": win_distress, "sounds": win_sounds
                }

                chunk_results.append(chunk_result)

                if chunk_callback:
                    try:
                        chunk_callback(chunk_result)
                    except Exception:
                        pass

                if simulate_realtime:
                    sleep_for = min(max(chunk_duration_map.get(chunk_path, 0.2) * 0.4, STREAM_MIN_SLEEP), STREAM_MAX_SLEEP)
                    time.sleep(sleep_for)

            except Exception as e:
                print(f" ⚠️ Error streaming chunk {chunk_path}: {e}")
                chunk_result = {
                    "index": idx, "start_s": float(start_s), "end_s": float(end_s),
                    "rms": float(rms), "win_emotion": None, "win_conf": 0.0,
                    "win_fused": {k: 0.0 for k in canonical_labels}, "win_distress": "low distress", "sounds": []
                }
                chunk_results.append(chunk_result)
                if chunk_callback:
                    try:
                        chunk_callback(chunk_result)
                    except Exception:
                        pass
                if simulate_realtime:
                    time.sleep(STREAM_MIN_SLEEP)
                continue

        # Normalize aggregated scores
        aggregated_audio_scores = {k: _safe_divide(aggregated_fused[k], total_weight) for k in canonical_labels} if total_weight > 0 else {}

        confidence, emotion, fused_scores = fuse_emotions(aggregated_audio_scores, text_scores)
        distress = get_distress_token(emotion, confidence)
        distress_after_kw = check_keywords(transcript, distress)

        if severity_level(distress_after_kw) < severity_level(distress):
            reason = f"keyword escalation: {distress_after_kw}"
            distress = distress_after_kw

        if severity_level(final_distress) < severity_level(distress):
            distress = final_distress

        if distress not in ["peak emergency distress", "high distress", "medium distress", "low distress"]:
            distress = "low distress"

        final_results.update({
            "emotion": emotion, "confidence": confidence, "distress": distress,
            "sounds": sound_events_all, "reason": reason, "fused_scores": fused_scores, "chunks": chunk_results
        })

        log_call(caller_id, transcript, emotion, distress, {"fused": fused_scores, "sounds": sound_events_all}, reason)
        return final_results

    except Exception as e:
        print(f"❌ An error occurred during streaming processing: {e}")
        traceback.print_exc()
        final_results["error"] = str(e)
        return final_results
    finally:
        if fixed_file and os.path.exists(fixed_file):
            try:
                os.remove(fixed_file)
            except Exception:
                pass
        remove_temp_files()
