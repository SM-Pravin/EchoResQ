from pathlib import Path
import sys
import traceback

audio = Path("WORKING_FILES") / "audio_samples" / "0.wav"
if not audio.exists():
    print("Audio sample not found:", audio)
    sys.exit(2)

# Prefer the snapshot location created by snapshot_download
local_snapshot = Path("WORKING_FILES") / "models" / "whisper-medium" / "models--openai--whisper-medium"

try:
    from faster_whisper import WhisperModel
except Exception as e:
    print("faster_whisper not installed:", e)
    sys.exit(3)

model = None
try:
    if local_snapshot.exists():
        print("Attempting to load Whisper model from local snapshot:", local_snapshot)
        try:
            model = WhisperModel(str(local_snapshot), device="cpu", compute_type="float32")
        except Exception as e_local:
            print("Local snapshot load failed (not in expected converted format):", e_local)
            print("Falling back to loading by model name 'medium' so faster-whisper can convert/cache the model.")
            model = WhisperModel("medium", device="cpu", compute_type="float32")
    else:
        print("Local snapshot not found; loading by name 'medium' (will use HF cache if present)")
        model = WhisperModel("medium", device="cpu", compute_type="float32")
except Exception:
    print("Model load failed:")
    traceback.print_exc()
    sys.exit(4)

print("Model loaded. Transcribing (this may take a while on CPU)...")
try:
    segments, info = model.transcribe(str(audio), beam_size=5)
    print("Segments:")
    def seg_get(seg, key, default=None):
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)

    for s in segments:
        start = seg_get(s, 'start', 0.0) or 0.0
        end = seg_get(s, 'end', 0.0) or 0.0
        text = (seg_get(s, 'text', '') or '').strip()
        try:
            print(f"  [{float(start):.2f}-{float(end):.2f}] {text}")
        except Exception:
            print(f"  [ {start} - {end} ] {text}")

    # Build full transcript safely
    full_texts = []
    for s in segments:
        txt = (seg_get(s, 'text', '') or '').strip()
        if txt:
            full_texts.append(txt)
    full = " ".join(full_texts)
    print("\nFull transcript:\n", full)
except Exception:
    print("Transcription failed:")
    traceback.print_exc()
    sys.exit(5)
