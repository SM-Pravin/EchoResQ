# modules/audio_preprocessing.py
"""
Audio preprocessing helpers.

- preprocess_audio(input_path, output_path, target_sr=16000)
    (keeps original behavior for backwards compatibility)

- split_audio_chunks(input_path, max_chunk=45, overlap=5, target_sr=16000, in_memory=False)
    If in_memory=False: retains legacy behavior (writes chunk files, returns paths).
    If in_memory=True: returns a list of dicts:
       { "data": np.ndarray(float32, mono, -1..1), "sr": int, "start_s": float, "end_s": float }

- Uses WebRTC VAD (webrtcvad). If it's not installed, raises ImportError.
"""

import os
import math
import soundfile as sf
import librosa
import numpy as np
import webrtcvad
import struct
from typing import List
import noisereduce as nr   # [OK] added noise reduction

# Constants / defaults
DEFAULT_SR = 16000
FRAME_MS = 30  # frame size for VAD
VAD_AGGRESSIVENESS = 2  # 0..3


# --------- Noise reduction helper ---------
def denoise_with_spectral_gating(y, sr=DEFAULT_SR):
    """
    Apply noise reduction using spectral gating.
    Uses the first 0.5 seconds as noise profile.
    If audio is too short, returns unchanged y.
    """
    if len(y) < sr // 2:
        return y
    noise_clip = y[:sr // 2]
    return nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)


def preprocess_audio(input_path, output_path, target_sr=DEFAULT_SR):
    """
    Convert input audio to 16kHz mono PCM WAV (no trimming).
    Includes:
      - resampling
      - DC offset removal
      - spectral gating denoise
      - normalization (0.97 peak)
    Overwrites output_path.
    (Kept to preserve backwards compatibility with existing code.)
    """
    y, sr = librosa.load(input_path, sr=None, mono=False)  # load with native sr
    if y.ndim > 1:
        y = librosa.to_mono(y)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # [OK] remove DC offset
    y = y - np.mean(y)

    # [OK] apply noise reduction
    y = denoise_with_spectral_gating(y, sr)

    # normalize to avoid clipping
    maxv = max(1e-9, float(max(abs(y.min()), abs(y.max()))))
    if maxv > 0:
        y = y / maxv * 0.97

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, y, sr, subtype="PCM_16")
    return output_path


# -------- WebRTC VAD helpers --------
class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def _frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Yield Frame objects containing PCM16 bytes of duration frame_duration_ms
    audio: numpy float array in [-1, 1]
    """
    if audio is None or len(audio) == 0:
        return
    frame_length = int(sample_rate * frame_duration_ms / 1000.0)
    offset = 0
    pcm16 = (audio * 32767.0).astype('<i2').tobytes()
    total_samples = len(audio)
    bytes_per_sample = 2
    frame_byte_length = frame_length * bytes_per_sample
    while offset + frame_length <= total_samples:
        start_byte = offset * bytes_per_sample
        frame_bytes = pcm16[start_byte:start_byte + frame_byte_length]
        timestamp = offset / float(sample_rate)
        duration = frame_length / float(sample_rate)
        yield Frame(frame_bytes, timestamp, duration)
        offset += frame_length


def _vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    Collect voiced frames using typical WebRTC VAD approach with padding.
    Returns raw PCM16 bytes concatenated for voiced regions.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    import collections
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_bytes = bytearray()

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_bytes.extend(f.bytes)
                ring_buffer.clear()
        else:
            voiced_bytes.extend(frame.bytes)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                ring_buffer.clear()
    return bytes(voiced_bytes)


def apply_webrtc_vad(y: np.ndarray, sr=DEFAULT_SR, frame_ms=FRAME_MS, padding_ms=300, aggressiveness=VAD_AGGRESSIVENESS):
    """
    Apply WebRTC VAD to float audio `y` (range -1..1). Returns voiced audio as float32 array (mono).
    If VAD finds nothing, returns empty array.
    """
    if y is None or len(y) == 0:
        return np.array([], dtype=np.float32)

    vad = webrtcvad.Vad(aggressiveness)
    frames = list(_frame_generator(frame_ms, y, sr))
    if not frames:
        return np.array([], dtype=np.float32)
    voiced_pcm = _vad_collector(sr, frame_ms, padding_ms, vad, frames)
    if not voiced_pcm:
        return np.array([], dtype=np.float32)
    arr = np.frombuffer(voiced_pcm, dtype='<i2').astype(np.float32) / 32767.0
    return arr


# -------- Chunking API (supports in-memory chunks) --------
def split_audio_chunks(input_path, max_chunk=45, overlap=5, target_sr=DEFAULT_SR, in_memory=False):
    """
    Smart splitter. Behaviors:
      - If in_memory=False (legacy): identical to original function: writes chunk .wav files and returns list of paths.
      - If in_memory=True: returns list of dicts { "data": np.ndarray(float32), "sr": target_sr, "start_s": float, "end_s": float }.

    If file <= 60s and in_memory=False, returns [input_path] (no split).
    """
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    duration_s = len(y) / float(sr) if sr > 0 else 0.0

    try:
        y_voiced = apply_webrtc_vad(y, sr=sr)
        if y_voiced is None or len(y_voiced) < 0.05 * len(y):
            y_voiced = y
    except Exception:
        y_voiced = y

    if not in_memory:
        if duration_s <= 60:
            return [input_path]

        step = max_chunk - overlap
        chunks = []
        start = 0
        total = int(math.ceil(duration_s))
        while start < total:
            s_sample = int(start * sr)
            e_sample = int(min((start + max_chunk) * sr, len(y)))
            chunk = y[s_sample:e_sample]
            out_path = input_path.replace(".wav", f"_chunk{start}.wav")
            sf.write(out_path, chunk, sr, subtype="PCM_16")
            chunks.append(out_path)
            start += step
        return chunks
    else:
        data = y_voiced
        dur = len(data) / float(sr) if sr > 0 else 0.0
        if dur <= 0:
            return []

        if dur <= 60:
            return [{"data": data.astype(np.float32), "sr": sr, "start_s": 0.0, "end_s": dur}]

        step = int((max_chunk - overlap) * sr)
        chunk_len = int(max_chunk * sr)
        chunks = []
        idx = 0
        for start in range(0, len(data) - chunk_len + 1, step):
            end = start + chunk_len
            chunk = data[start:end]
            s_time = start / float(sr)
            e_time = end / float(sr)
            chunks.append({
                "data": chunk.astype(np.float32),
                "sr": sr,
                "start_s": float(s_time),
                "end_s": float(e_time),
                "index": idx
            })
            idx += 1

        if len(data) % step != 0 and (len(data) - chunk_len) > 0:
            start = max(0, len(data) - chunk_len)
            end = len(data)
            chunk = data[start:end]
            s_time = start / float(sr)
            e_time = end / float(sr)
            chunks.append({
                "data": chunk.astype(np.float32),
                "sr": sr,
                "start_s": float(s_time),
                "end_s": float(e_time),
                "index": idx
            })

        return chunks
