# main.py
"""
Main entrypoint for the emergency AI pipeline.

- Full-file transcription (Vosk)
- Text-emotion on full transcript
- Sliding-window audio emotion & sound-event detection (with overlap)
- Keyword escalation and sound escalation
- --fast flag: skip heavy chunked audio emotion/sound detection (fast transcription+keywords)
"""

import os
import sys
import tempfile
import warnings

# Environment tweaks should be set early
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore")
try:
    # --- ADD THESE LINES ---
    # Explicitly silence TensorFlow's Python-level warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    # -----------------------

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
CHUNK_SIZE_SECONDS = 30     # larger chunk to keep sentence context
HOP_SIZE_SECONDS = 15       # 50% overlap
SILENCE_RMS_THRESHOLD_RATIO = 0.06
TEMP_DIR = "tmp_chunks"


def make_temp_wav(chunk_arr, sr):
    os.makedirs(TEMP_DIR, exist_ok=True)
    tf = tempfile.NamedTemporaryFile(prefix="chunk_", suffix=".wav", dir=TEMP_DIR, delete=False)
    tf.close()
    sf.write(tf.name, chunk_arr, sr, subtype="PCM_16")
    return tf.name


def compute_chunk_rms(waveform, sr, chunk_s, hop_s):
    n = len(waveform)
    n_chunk = int(chunk_s * sr)
    hop = int(hop_s * sr)
    rms_list = []
    for start in range(0, n, hop):
        end = min(start + n_chunk, n)
        seg = waveform[start:end]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size > 0 else 0.0
        rms_list.append((start, end, rms))
        if end == n:
            break
    return rms_list


def remove_temp_files(prefix="chunk_"):
    # utility to clean temp folder (not required, but helpful)
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


def usage_and_exit():
    print("Usage: python3 main.py <audio_file.wav> [--fast]")
    print("  --fast   : skip chunked audio emotion & sound analysis (fast path)")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage_and_exit()

    audio_file = sys.argv[1]
    fast_mode = "--fast" in sys.argv[2:]

    if not os.path.exists(audio_file):
        print(f"❌ Error: {audio_file} not found.")
        return

    # Preprocess to 16k mono PCM WAV
    fixed_file = audio_file.replace(".wav", "_fixed.wav")
    preprocess_audio(audio_file, fixed_file)

    caller_id = os.path.basename(audio_file).split(".")[0].upper()

    # Full-file transcription (single pass) - keep it intact (no chunking)
    transcript = transcribe_audio(fixed_file)
    if not transcript:
        transcript = ""

    # Text emotion (full transcript)
    text_scores = analyze_text_emotion(transcript)

    # If fast mode: skip heavy audio chunking/sound analysis and compute final results
    if fast_mode:
        # try quick audio emotion on the whole file if model available
        audio_scores = analyze_audio_emotion(fixed_file)
        confidence, emotion, fused_scores = fuse_emotions(audio_scores, text_scores)
        distress = get_distress_token(emotion, confidence)
        distress = check_keywords(transcript, distress)
        sound_events_all = []
        reason = "fast mode: skipped chunked audio & sound analysis"
        print("\n" + "=" * 50)
        print("      🚨 EMERGENCY CALL REPORT (FAST) 🚨")
        print("=" * 50)
        print(f" Caller ID     : {caller_id}")
        print(f" Transcript    : {transcript if transcript else '(empty)'}")
        print(f" Final Emotion : {str(emotion).upper() if emotion else 'UNKNOWN'}")
        print(f" Confidence    : {confidence:.2f}")
        print(f" Distress      : {distress.upper()}")
        print(f" Reason        : {reason}")
        print("=" * 50 + "\n")
        log_call(caller_id, transcript, emotion, distress, {"fused": fused_scores, "sounds": []}, reason)
        return

    # Load waveform for chunked audio emotion & sound detection
    waveform, sr = librosa.load(fixed_file, sr=16000, mono=True)
    if waveform is None or len(waveform) == 0:
        waveform = np.zeros(1, dtype=np.float32)

    # Use smarter splitting: do not split if file <= 60s
    chunks = split_audio_chunks(fixed_file, max_chunk=CHUNK_SIZE_SECONDS, overlap=int(HOP_SIZE_SECONDS))

    # We'll analyze each chunk for audio emotion and sounds, but STT/text done on full file
    canonical_labels = ["angry", "happy", "neutral", "sad"]
    aggregated_fused = {k: 0.0 for k in canonical_labels}
    total_weight = 0.0
    sound_events_all = []
    reason = None
    final_distress = "low distress"
    final_emotion = "neutral"

    # Precompute RMS for weighting per chunk (load per chunk is fine)
    max_rms = 0.0
    chunk_rms_map = {}
    for idx, chunk_path in enumerate(chunks):
        try:
            # load chunk for energy estimate
            y, s = sf.read(chunk_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            rms = float(np.sqrt(np.mean(y ** 2))) if y.size > 0 else 0.0
            chunk_rms_map[chunk_path] = rms
            if rms > max_rms:
                max_rms = rms
        except Exception:
            chunk_rms_map[chunk_path] = 0.0

    silence_thresh = max_rms * SILENCE_RMS_THRESHOLD_RATIO if max_rms > 0 else 0.0

    # Analyze each chunk
    for idx, chunk_path in enumerate(chunks):
        rms = chunk_rms_map.get(chunk_path, 0.0)
        if max_rms > 0 and rms < silence_thresh:
            # skip very silent chunk
            continue

        try:
            # Audio emotion for this chunk
            audio_scores = analyze_audio_emotion(chunk_path)  # dict or {}
            win_conf, win_emotion, win_fused = fuse_emotions(audio_scores, {})

            # Sound events
            win_sounds = analyze_sound_events(chunk_path) or []
            if win_sounds:
                sound_events_all.extend(win_sounds)
                for lbl, conf, mapped in win_sounds:
                    if severity_level(mapped) < severity_level(final_distress):
                        final_distress = mapped
                        final_emotion = win_emotion
                        reason = f"sound escalation: {lbl} ({conf:.2f})"
                        if final_distress == "peak emergency distress":
                            break

            # Emotion-based escalation per window
            win_distress = get_distress_token(win_emotion, win_conf)
            if severity_level(win_distress) < severity_level(final_distress):
                final_distress = win_distress
                final_emotion = win_emotion
                reason = f"emotion escalation (chunk {idx}): {win_emotion} ({win_conf:.2f})"
                if final_distress == "peak emergency distress":
                    break

            # Weight the window's fused canonical scores by RMS and aggregate
            weight = rms if max_rms > 0 else 1.0
            for k in canonical_labels:
                aggregated_fused[k] += win_fused.get(k, 0.0) * weight
            total_weight += weight

        except Exception as e:
            # be resilient: continue to next chunk
            print(f" ⚠️ Error processing chunk {chunk_path}: {e}")
            continue

    # Normalize aggregated audio fused scores
    if total_weight > 0:
        aggregated_audio_scores = {k: aggregated_fused[k] / total_weight for k in canonical_labels}
    else:
        aggregated_audio_scores = {}

    # Final fusion (aggregated audio + text)
    confidence, emotion, fused_scores = fuse_emotions(aggregated_audio_scores, analyze_text_emotion(transcript))

    # Distress mapping + keyword escalation
    distress = get_distress_token(emotion, confidence)
    distress_after_kw = check_keywords(transcript, distress)
    if severity_level(distress_after_kw) < severity_level(distress):
        reason = f"keyword escalation: {distress_after_kw}"
        distress = distress_after_kw

    # Final sound escalation check
    for lbl, conf, mapped in sound_events_all:
        if severity_level(mapped) < severity_level(distress):
            reason = f"sound escalation: {lbl} ({conf:.2f})"
            distress = mapped
            break

    # Prefer any earlier final_distress found during chunk loop
    if severity_level(final_distress) < severity_level(distress):
        distress = final_distress

    # Safe fallback
    if distress not in ["peak emergency distress", "high distress", "medium distress", "low distress"]:
        distress = "low distress"

    # Pretty print final report
    print("\n" + "=" * 50)
    print("      🚨 EMERGENCY CALL REPORT 🚨")
    print("=" * 50)
    print(f" Caller ID     : {caller_id}")
    print(f" Transcript    : {transcript if transcript else '(empty)'}")
    print(f" Final Emotion : {str(emotion).upper() if emotion else 'UNKNOWN'}")
    print(f" Confidence    : {confidence:.2f}")
    print(f" Distress      : {distress.upper()}")
    if sound_events_all:
        print("\n Background Sounds Detected:")
        for s, c, m in sound_events_all:
            print(f"  - {s} ({c:.2f}) -> {m.upper()}")
    if reason:
        print(f"\n Reason        : {reason}")
    print("=" * 50 + "\n")

    # Log final entry
    log_call(
        caller_id=caller_id,
        transcript=transcript,
        final_emotion=emotion,
        distress_token=distress,
        scores={"fused": fused_scores, "sounds": sound_events_all},
        reason=reason
    )
    print(f"[LOG] Entry saved for caller {caller_id}\n")

    # cleanup temp chunk files
    remove_temp_files()


if __name__ == "__main__":
    main()

