# tests/test_denoise.py
# tests/test_denoise.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import soundfile as sf
import tempfile
from modules.audio_preprocessing import preprocess_audio, denoise_with_spectral_gating


def test_denoise_reduces_noise():
    sr = 16000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Clean signal: 440Hz sine wave
    clean = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Noise: white noise
    noise = 0.3 * np.random.randn(len(t))

    # Noisy signal = clean + noise
    noisy = clean + noise

    # Save noisy to a temp wav
    tmp_in = os.path.join(tempfile.gettempdir(), "noisy_test.wav")
    tmp_out = os.path.join(tempfile.gettempdir(), "denoised_test.wav")
    sf.write(tmp_in, noisy, sr)

    # Run preprocessing (includes denoise)
    preprocess_audio(tmp_in, tmp_out, target_sr=sr)

    # Load denoised
    denoised, _ = sf.read(tmp_out)

    # Compare RMS energy of noise portion
    noisy_rms = np.sqrt(np.mean((noisy - clean) ** 2))
    denoised_rms = np.sqrt(np.mean((denoised - clean[:len(denoised)]) ** 2))

    print("Noisy RMS:", noisy_rms, "Denoised RMS:", denoised_rms)

    # Assert denoised RMS < noisy RMS (i.e. noise reduced)
    assert denoised_rms < noisy_rms
