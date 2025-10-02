# tests/bench_parallel.py
import os, time, tempfile
import numpy as np
import soundfile as sf
from analysis_pipeline import process_audio_file

def make_sine_wav(path, duration_s=12.0, sr=16000, freq=440.0):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * freq * t).astype('float32')
    sf.write(path, y, sr, subtype='PCM_16')

tmp = tempfile.gettempdir()
sample = os.path.join(tmp, "test_parallel_sample.wav")
make_sine_wav(sample, duration_s=40.0)  # 40s will produce multiple chunks with your default settings

# sequential run (force single worker)
os.environ['PARALLEL_MAX_WORKERS'] = '1'
t0 = time.perf_counter()
r1 = process_audio_file(sample)
t1 = time.perf_counter()
print("Sequential (1 worker) runtime:", t1-t0, "chunks:", len(r1.get("chunks", [])))

# parallel run (2..N workers)
os.environ['PARALLEL_MAX_WORKERS'] = '4'
t0 = time.perf_counter()
r2 = process_audio_file(sample)
t1 = time.perf_counter()
print("Parallel (4 workers) runtime:", t1-t0, "chunks:", len(r2.get("chunks", [])))
