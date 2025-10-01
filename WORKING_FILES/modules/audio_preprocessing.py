# modules/audio_preprocessing.py
import librosa
import soundfile as sf
import os

def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    Convert input audio to 16kHz mono PCM WAV (no trimming).
    Overwrites output_path.
    """
    y, sr = librosa.load(input_path, sr=None, mono=False)  # load with native sr
    # y shape: (n,) or (channels, n)
    if y.ndim > 1:
        # librosa returns (channels, n) when mono=False
        y = librosa.to_mono(y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # normalize to avoid clipping
    maxv = max(1e-9, float(max(abs(y.min()), abs(y.max()))))
    if maxv > 0:
        y = y / maxv * 0.97

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, y, sr, subtype="PCM_16")
    return output_path


def split_audio_chunks(input_path, max_chunk=45, overlap=5, target_sr=16000):
    """
    Smart splitter: if file <= 60s, return [input_path] (no split).
    Otherwise split into max_chunk windows with overlap seconds (both ints).
    """
    import soundfile as sf
    import math

    info = sf.info(input_path)
    duration_s = info.frames / float(info.samplerate)
    if duration_s <= 60:
        return [input_path]

    # load once and cut
    y, sr = sf.read(input_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
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

