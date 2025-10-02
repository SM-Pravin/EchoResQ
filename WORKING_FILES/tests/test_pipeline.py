# tests/test_pipeline.py
import os
import numpy as np
import soundfile as sf
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis_pipeline import process_audio_file

def create_sine_wav(path, freq=440, sr=16000, seconds=2):
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    y = 0.05 * np.sin(2*np.pi*freq*t)  # low amplitude so VAD might ignore; good sanity check
    sf.write(path, y.astype('float32'), sr)

def test_basic_pipeline(tmp_path):
    test_file = tmp_path / "test.wav"
    create_sine_wav(str(test_file))
    result = process_audio_file(str(test_file), fast_mode=True, return_chunks_details=True)
    assert isinstance(result, dict)
    assert "transcript" in result
    assert "emotion" in result
    assert "distress" in result
    print("Test pipeline result:", result)
