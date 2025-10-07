from pathlib import Path
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
MODELDIR = Path("WORKING_FILES") / "models" / "whisper-medium"
MODELDIR.mkdir(parents=True, exist_ok=True)
try:
    print("Attempting snapshot_download(openai/whisper-medium) ->", MODELDIR)
    snapshot_download(repo_id="openai/whisper-medium", cache_dir=str(MODELDIR), repo_type="model")
    print("snapshot_download completed")
except Exception as e:
    print("snapshot_download failed:", e)
    print("Falling back to instantiating faster-whisper (this will download to HF cache).")
    try:
        _ = WhisperModel("medium", device="cpu")
        print("faster-whisper instantiation complete (model in HF cache).")
    except Exception as e2:
        print("faster-whisper fallback failed:", e2)
