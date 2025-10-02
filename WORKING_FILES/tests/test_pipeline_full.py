# tests/test_pipeline_full.py
import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import soundfile as sf
from analysis_pipeline import process_audio_file   # <-- FIXED

def create_long_sine_wav(path, freq=440, sr=16000, seconds=65):
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    y = 0.1 * np.sin(2*np.pi*freq*t).astype('float32')
    sf.write(path, y, sr)

def test_full_pipeline_chunking(tmp_path):
    test_file = tmp_path / "long_test.wav"
    create_long_sine_wav(str(test_file), seconds=65)
    result = process_audio_file(str(test_file), fast_mode=False, return_chunks_details=True)
    assert isinstance(result, dict)
    chunks = result.get("chunks", [])
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    print("Full pipeline result summary:", {
        "caller_id": result.get("caller_id"),
        "num_chunks": len(chunks),
        "distress": result.get("distress"),
        "emotion": result.get("emotion"),
        "confidence": result.get("confidence")
    })
