"""
Main entrypoint for the emergency AI pipeline.
Extended to:
 - parallelize per-chunk analysis (ProcessPoolExecutor, spawn context)
 - preserve streaming behavior (sequential when simulating realtime)
 - use in-memory chunking (no unnecessary disk I/O)
 - return per-chunk details for visualization and callback use
"""

import os
import tempfile
import warnings
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

# Parameters (kept from original)
CHUNK_SIZE_SECONDS = 30
HOP_SIZE_SECONDS = 15
SILENCE_RMS_THRESHOLD_RATIO = 0.06
TEMP_DIR = "tmp_chunks"

# Batch processing parameters
ENABLE_BATCH_PROCESSING = os.environ.get("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
AUDIO_BATCH_SIZE = int(os.environ.get("AUDIO_BATCH_SIZE", "8"))

# Environment-controlled parallelism:
# Set PARALLEL_MAX_WORKERS env var to control worker count; defaults to os.cpu_count()
def _get_max_workers():
    try:
        v = os.environ.get("PARALLEL_MAX_WORKERS")
        if v:
            return max(1, int(v))
    except Exception:
        pass
    return max(1, (os.cpu_count() or 1))


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


# ----------------------
# Worker function with model initialization
# ----------------------

# Global worker state to avoid reloading models per chunk
_worker_models = None

def _init_worker_models():
    """Initialize models in worker process once on startup."""
    global _worker_models
    if _worker_models is not None:
        return
    
    import os
    # Set environment variables for model loading in worker
    os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # Force CPU in workers for stability
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    try:
        # Import all necessary modules and models
        from modules.emotion_audio import analyze_audio_emotion
        from modules.fusion_engine import fuse_emotions
        from modules.sound_event_detector import analyze_sound_events
        from modules.distress_mapper import get_distress_token
        
        _worker_models = {
            'analyze_audio_emotion': analyze_audio_emotion,
            'fuse_emotions': fuse_emotions,
            'analyze_sound_events': analyze_sound_events,
            'get_distress_token': get_distress_token
        }
    except Exception as e:
        print(f"⚠️ Worker failed to initialize models: {e}")
        _worker_models = {}

def _analyze_chunk_worker(args):
    """
    Enhanced worker entrypoint for a single chunk with pre-loaded models.
    Args is a tuple:
      (idx, chunk_dict, max_rms, silence_thresh, canonical_labels)
    Returns dict shaped like a single chunk result.
    """
    global _worker_models
    
    try:
        idx, chunk, max_rms, silence_thresh, canonical_labels = args
        
        # Ensure models are initialized
        if _worker_models is None:
            _init_worker_models()

        import numpy as _np
        
        y = chunk.get("data", _np.array([], dtype=_np.float32))
        sr = int(chunk.get("sr", 16000) or 16000)
        rms = float(_np.sqrt(_np.mean(y**2))) if y.size > 0 else 0.0
        start_s = float(chunk.get("start_s", 0.0))
        end_s = float(chunk.get("end_s", start_s + (len(y) / float(sr) if sr > 0 else 0.0)))

        # skip silent chunks
        if max_rms > 0 and rms < silence_thresh:
            return {
                "index": idx,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "rms": float(rms),
                "win_emotion": None,
                "win_conf": 0.0,
                "win_fused": {k: 0.0 for k in canonical_labels},
                "win_distress": "low distress",
                "sounds": []
            }

        # Use pre-loaded model functions
        audio_scores = _worker_models.get('analyze_audio_emotion', lambda *args, **kwargs: {})(y, sr=sr) or {}
        win_conf, win_emotion, win_fused = _worker_models.get('fuse_emotions', lambda *args, **kwargs: (0.0, None, {}))(audio_scores, {})

        # run sound event detector
        win_sounds = _worker_models.get('analyze_sound_events', lambda *args, **kwargs: [])({"data": y, "sr": sr}) or []

        # map to distress token
        win_distress = _worker_models.get('get_distress_token', lambda *args, **kwargs: "low distress")(win_emotion, win_conf)

        return {
            "index": idx,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "rms": float(rms),
            "win_emotion": win_emotion,
            "win_conf": float(win_conf),
            "win_fused": {k: float(win_fused.get(k, 0.0)) for k in canonical_labels},
            "win_distress": win_distress,
            "sounds": win_sounds
        }

    except Exception as e:
        print(f"⚠️ Worker error processing chunk {args[0] if args else 'unknown'}: {e}")
        # On any failure inside a worker, return a safe default chunk result
        return {
            "index": int(args[0] if args else 0),
            "start_s": 0.0,
            "end_s": 0.0,
            "rms": 0.0,
            "win_emotion": None,
            "win_conf": 0.0,
            "win_fused": {k: 0.0 for k in (args[4] if len(args) > 4 else ["angry","happy","neutral","sad"])},
            "win_distress": "low distress",
            "sounds": []
        }


def analyze_chunks_in_parallel(chunks, max_rms, silence_thresh, canonical_labels, max_workers=None):
    """
    Run _analyze_chunk_worker concurrently for all chunks and return
    a list of chunk result dicts ordered by chunk index.
    """
    if not chunks:
        return []

    max_workers = int(max_workers or _get_max_workers())
    # Build argument tuples
    job_args = [
        (int(chunk.get("index", idx)), chunk, float(max_rms), float(silence_thresh), list(canonical_labels))
        for idx, chunk in enumerate(chunks)
    ]

    results = []
    # Prefer spawn context for safety (especially when TF/CUDA present)
    try:
        ctx = multiprocessing.get_context("spawn")
    except Exception:
        ctx = None

    # Create executor with initializer to pre-load models in workers
    try:
        if ctx is not None:
            with ProcessPoolExecutor(
                max_workers=max_workers, 
                mp_context=ctx,
                initializer=_init_worker_models
            ) as ex:
                futures = [ex.submit(_analyze_chunk_worker, a) for a in job_args]
                for fut in as_completed(futures):
                    results.append(fut.result())
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker_models
            ) as ex:
                futures = [ex.submit(_analyze_chunk_worker, a) for a in job_args]
                for fut in as_completed(futures):
                    results.append(fut.result())
    except Exception as e:
        print(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
        # Fallback: If parallel executor fails (rare), run sequentially
        # This ensures the pipeline remains robust on minimal systems.
        results = []
        for a in job_args:
            results.append(_analyze_chunk_worker(a))

    # Ensure results are ordered by index (chunks may finish out of order)
    results.sort(key=lambda d: int(d.get("index", 0)))
    return results


def analyze_chunks_in_batch(chunks, max_rms, silence_thresh, canonical_labels):
    """
    Batch-enabled chunk analysis that processes non-silent chunks in batches for efficiency.
    Falls back to individual processing if batch processing fails.
    """
    if not chunks or not ENABLE_BATCH_PROCESSING:
        return analyze_chunks_in_parallel(chunks, max_rms, silence_thresh, canonical_labels)
    
    try:
        from modules.emotion_audio import analyze_audio_emotion_batch
        from modules.fusion_engine import fuse_emotions
        from modules.sound_event_detector import analyze_sound_events
        from modules.distress_mapper import get_distress_token
        
        results = []
        audio_inputs = []
        chunk_indices = []
        
        # Prepare chunks and filter out silent ones
        for idx, chunk in enumerate(chunks):
            y = chunk.get("data", np.array([], dtype=np.float32))
            sr = int(chunk.get("sr", 16000) or 16000)
            rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
            start_s = float(chunk.get("start_s", 0.0))
            end_s = float(chunk.get("end_s", start_s + (len(y) / float(sr) if sr > 0 else 0.0)))
            
            if max_rms > 0 and rms < silence_thresh:
                # Add silent chunk result directly
                results.append({
                    "index": idx,
                    "start_s": start_s,
                    "end_s": end_s,
                    "rms": rms,
                    "win_emotion": None,
                    "win_conf": 0.0,
                    "win_fused": {k: 0.0 for k in canonical_labels},
                    "win_distress": "low distress",
                    "sounds": []
                })
            else:
                # Collect for batch processing
                audio_inputs.append({"data": y, "sr": sr})
                chunk_indices.append((idx, chunk, rms, start_s, end_s))
        
        if audio_inputs:
            # Batch process audio emotion analysis
            batch_audio_scores = analyze_audio_emotion_batch(audio_inputs, sr=16000, max_batch_size=AUDIO_BATCH_SIZE)
            
            # Process each result
            for i, (idx, chunk, rms, start_s, end_s) in enumerate(chunk_indices):
                try:
                    y = chunk.get("data", np.array([], dtype=np.float32))
                    sr = int(chunk.get("sr", 16000) or 16000)
                    
                    # Get batch result
                    audio_scores = batch_audio_scores[i] if i < len(batch_audio_scores) else {}
                    win_conf, win_emotion, win_fused = fuse_emotions(audio_scores, {})
                    
                    # Sound events still processed individually (YAMNet doesn't batch well)
                    win_sounds = analyze_sound_events({"data": y, "sr": sr}) or []
                    
                    # Map to distress token
                    win_distress = get_distress_token(win_emotion, win_conf)
                    
                    results.append({
                        "index": idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "rms": rms,
                        "win_emotion": win_emotion,
                        "win_conf": float(win_conf),
                        "win_fused": {k: float(win_fused.get(k, 0.0)) for k in canonical_labels},
                        "win_distress": win_distress,
                        "sounds": win_sounds
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing batch chunk {idx}: {e}")
                    # Add error result
                    results.append({
                        "index": idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "rms": rms,
                        "win_emotion": None,
                        "win_conf": 0.0,
                        "win_fused": {k: 0.0 for k in canonical_labels},
                        "win_distress": "low distress",
                        "sounds": []
                    })
        
        # Sort by index
        results.sort(key=lambda d: int(d.get("index", 0)))
        return results
        
    except Exception as e:
        print(f"⚠️ Batch processing failed, falling back to parallel: {e}")
        return analyze_chunks_in_parallel(chunks, max_rms, silence_thresh, canonical_labels)


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

        # Full-file transcription (unchanged)
        transcript = transcribe_audio(fixed_file) or ""
        results["transcript"] = transcript

        # Text emotion (full transcript)
        text_scores = analyze_text_emotion(transcript)

        # FAST MODE (no chunk split or sound analysis)
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

        # --- FULL ANALYSIS (CHUNKED, in-memory) ---
        chunks = split_audio_chunks(fixed_file, max_chunk=CHUNK_SIZE_SECONDS, overlap=int(HOP_SIZE_SECONDS), in_memory=True)

        canonical_labels = ["angry", "happy", "neutral", "sad"]
        aggregated_fused = {k: 0.0 for k in canonical_labels}
        total_weight = 0.0
        sound_events_all = []
        reason = None
        final_distress = "low distress"
        final_emotion = "neutral"

        # Precompute RMS for weighting and durations
        max_rms = 0.0
        chunk_rms_map = {}
        chunk_duration_map = {}

        for idx, chunk in enumerate(chunks):
            y = chunk.get("data", np.array([], dtype=np.float32))
            sr = int(chunk.get("sr", 16000) or 16000)
            rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
            chunk_rms_map[idx] = rms
            chunk_duration_map[idx] = float(len(y) / float(sr)) if sr > 0 and y.size > 0 else 0.0
            if rms > max_rms: max_rms = rms

        silence_thresh = max_rms * SILENCE_RMS_THRESHOLD_RATIO if max_rms > 0 else 0.0

        # Choose processing method: batch > parallel > sequential
        if ENABLE_BATCH_PROCESSING and len(chunks) >= 2:
            chunks_info = analyze_chunks_in_batch(chunks, max_rms, silence_thresh, canonical_labels)
        else:
            max_workers = _get_max_workers()
            chunks_info = analyze_chunks_in_parallel(chunks, max_rms, silence_thresh, canonical_labels, max_workers=max_workers)

        # Aggregate results (same logic as before)
        for c in chunks_info:
            rms = float(c.get("rms", 0.0))
            # sounds
            win_sounds = c.get("sounds", []) or []
            if win_sounds:
                sound_events_all.extend(win_sounds)

            win_distress = c.get("win_distress", "low distress")
            win_conf = float(c.get("win_conf", 0.0) or 0.0)
            win_emotion = c.get("win_emotion", None)

            if severity_level(win_distress) < severity_level(final_distress):
                final_distress = win_distress
                final_emotion = win_emotion
                reason = f"emotion escalation (chunk {c.get('index')}): {win_emotion} ({win_conf:.2f})"

            weight = rms if max_rms > 0 else 1.0
            win_fused = c.get("win_fused", {}) or {}
            for k in canonical_labels:
                aggregated_fused[k] += win_fused.get(k, 0.0) * weight
            total_weight += weight

        # Normalize aggregated scores
        aggregated_audio_scores = {k: _safe_divide(aggregated_fused[k], total_weight) for k in canonical_labels} if total_weight > 0 else {}

        # Final fusion and distress mapping
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
        # Cleanup temp file
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
    Uses parallel processing when simulate_realtime==False (i.e., batch processing mode).
    """
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

        # Full chunked processing using in-memory chunks
        chunks = split_audio_chunks(fixed_file, max_chunk=chunk_size_seconds, overlap=int(hop_size_seconds), in_memory=True)
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
        for idx, chunk in enumerate(chunks):
            y = chunk.get("data", np.array([], dtype=np.float32))
            sr = int(chunk.get("sr", 16000) or 16000)
            rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
            chunk_rms_map[idx] = rms
            chunk_duration_map[idx] = float(len(y) / float(sr)) if sr > 0 and y.size > 0 else 0.0
            if rms > max_rms: max_rms = rms

        silence_thresh = max_rms * SILENCE_RMS_THRESHOLD_RATIO if max_rms > 0 else 0.0
        chunk_results = []

        # If simulate_realtime==False, process chunks in batch/parallel mode
        if not simulate_realtime:
            if ENABLE_BATCH_PROCESSING and len(chunks) >= 2:
                chunks_info = analyze_chunks_in_batch(chunks, max_rms, silence_thresh, canonical_labels)
            else:
                chunks_info = analyze_chunks_in_parallel(chunks, max_rms, silence_thresh, canonical_labels, max_workers=_get_max_workers())
            for c in chunks_info:
                # same aggregation/emit logic used earlier
                chunk_results.append(c)
                if chunk_callback:
                    try:
                        chunk_callback(c)
                    except Exception:
                        pass
            # now aggregate like process_audio_file
            for c in chunks_info:
                rms = float(c.get("rms", 0.0))
                win_sounds = c.get("sounds", []) or []
                if win_sounds:
                    sound_events_all.extend(win_sounds)
                win_distress = c.get("win_distress", "low distress")
                win_conf = float(c.get("win_conf", 0.0) or 0.0)
                win_emotion = c.get("win_emotion", None)
                if severity_level(win_distress) < severity_level(final_distress):
                    final_distress = win_distress
                    final_emotion = win_emotion
                    reason = f"emotion escalation (chunk {c.get('index')}): {win_emotion} ({win_conf:.2f})"
                weight = rms if max_rms > 0 else 1.0
                win_fused = c.get("win_fused", {}) or {}
                for k in canonical_labels:
                    aggregated_fused[k] += win_fused.get(k, 0.0) * weight
                total_weight += weight

        else:
            # sequential (real-time simulation) — keep original behavior with sleeps & callbacks
            for idx, chunk in enumerate(chunks):
                start_s = chunk.get("start_s", idx * (chunk_size_seconds - hop_size_seconds))
                end_s = chunk.get("end_s", start_s + chunk_duration_map.get(idx, chunk_size_seconds))
                y = chunk.get("data", np.array([], dtype=np.float32))
                sr = int(chunk.get("sr", 16000) or 16000)
                rms = chunk_rms_map.get(idx, 0.0)

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
                    if simulate_realtime:
                        sleep_for = min(max(chunk_duration_map.get(idx, 0.2) * 0.4, 0.15), 1.2)
                        time.sleep(sleep_for)
                    continue

                try:
                    audio_scores = analyze_audio_emotion(y, sr=sr)
                    win_conf, win_emotion, win_fused = fuse_emotions(audio_scores, {})

                    win_sounds = analyze_sound_events({"data": y, "sr": sr}) or []
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
                        sleep_for = min(max(chunk_duration_map.get(idx, 0.2) * 0.4, 0.15), 1.2)
                        time.sleep(sleep_for)

                except Exception as e:
                    print(f" ⚠️ Error streaming chunk {idx}: {e}")
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
                        time.sleep(0.15)
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
