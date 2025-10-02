# tests/bench_parallel.py
import os, time, tempfile
import numpy as np
import soundfile as sf
from analysis_pipeline import process_audio_file

def make_sine_wav(path, duration_s=12.0, sr=16000, freq=440.0):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * freq * t).astype('float32')
    sf.write(path, y, sr, subtype='PCM_16')

def benchmark_configurations():
    """Enhanced benchmark testing multiple configurations."""
    print("Enhanced Performance Benchmark")
    print("="*50)
    
    tmp = tempfile.gettempdir()
    sample = os.path.join(tmp, "test_parallel_sample.wav")
    make_sine_wav(sample, duration_s=90.0)  # 90s for more chunks
    
    configurations = [
        {"workers": 1, "batch": False, "name": "Sequential (1 worker, no batch)"},
        {"workers": 1, "batch": True, "name": "Sequential (1 worker, with batch)"},
        {"workers": 4, "batch": False, "name": "Parallel (4 workers, no batch)"},
        {"workers": 4, "batch": True, "name": "Parallel (4 workers, with batch)"},
        {"workers": 8, "batch": True, "name": "High Parallel (8 workers, with batch)"},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        # Set environment
        os.environ['PARALLEL_MAX_WORKERS'] = str(config['workers'])
        os.environ['ENABLE_BATCH_PROCESSING'] = str(config['batch']).lower()
        
        try:
            # Warm-up run
            _ = process_audio_file(sample, fast_mode=True)
            
            # Benchmark run
            t0 = time.perf_counter()
            result = process_audio_file(sample, fast_mode=False)
            t1 = time.perf_counter()
            
            duration = t1 - t0
            chunk_count = len(result.get("chunks", []))
            transcript_len = len(result.get("transcript", ""))
            emotion = result.get("emotion", "unknown")
            
            results.append({
                "config": config['name'],
                "duration": duration,
                "chunks": chunk_count,
                "transcript_length": transcript_len,
                "emotion": emotion,
                "chunks_per_second": chunk_count / duration if duration > 0 else 0
            })
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Chunks: {chunk_count}")
            print(f"  Transcript: {transcript_len} chars")
            print(f"  Emotion: {emotion}")
            print(f"  Throughput: {chunk_count/duration:.1f} chunks/sec")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "config": config['name'],
                "error": str(e)
            })
    
    # Performance comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    successful_results = [r for r in results if 'duration' in r]
    if len(successful_results) >= 2:
        baseline = successful_results[0]
        print(f"Baseline: {baseline['config']} ({baseline['duration']:.2f}s)")
        
        for result in successful_results[1:]:
            speedup = baseline['duration'] / result['duration']
            print(f"{result['config']}: {speedup:.2f}x speedup ({result['duration']:.2f}s)")
    
    # Cleanup
    try:
        os.remove(sample)
    except Exception:
        pass
    
    print("\nBenchmark completed!")
    return results

if __name__ == "__main__":
    benchmark_configurations()
