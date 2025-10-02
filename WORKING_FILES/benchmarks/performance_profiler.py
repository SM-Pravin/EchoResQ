#!/usr/bin/env python3
"""
Comprehensive performance profiling and benchmarking suite for Emergency AI.
Provides detailed timing, memory usage, and bottleneck analysis.
"""

import os
import sys
import time
import psutil
import traceback
import tempfile
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
import threading
import queue

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from analysis_pipeline import process_audio_file, process_audio_file_stream

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    duration_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: str
    additional_info: Dict[str, Any] = None

class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_memory = 0
        self.initial_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        self.memory_queue = queue.Queue()
    
    def start(self):
        """Start monitoring memory usage."""
        process = psutil.Process(os.getpid())
        self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    self.memory_queue.put(memory_mb)
                    time.sleep(self.interval)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring and return peak memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        return {
            'peak_mb': self.peak_memory,
            'delta_mb': self.peak_memory - self.initial_memory,
            'initial_mb': self.initial_memory
        }

@contextmanager
def profile_operation(operation_name: str, additional_info: Dict = None):
    """Context manager for profiling operations."""
    # Initial measurements
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_cpu = process.cpu_percent()
    
    # Start memory monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    try:
        yield
    finally:
        # Final measurements
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Stop memory monitoring
        memory_info = memory_monitor.stop()
        
        # CPU usage (average during operation)
        final_cpu = process.cpu_percent()
        avg_cpu = (start_cpu + final_cpu) / 2
        
        # Create metrics
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration_ms=duration_ms,
            memory_peak_mb=memory_info['peak_mb'],
            memory_delta_mb=memory_info['delta_mb'],
            cpu_percent=avg_cpu,
            timestamp=datetime.now().isoformat(),
            additional_info=additional_info or {}
        )
        
        # Store metrics globally for reporting
        if not hasattr(profile_operation, 'metrics'):
            profile_operation.metrics = []
        profile_operation.metrics.append(metrics)
        
        print(f"📊 {operation_name}: {duration_ms:.1f}ms, {memory_info['delta_mb']:.1f}MB delta, {avg_cpu:.1f}% CPU")

def create_test_audio(duration_s=30, sample_rate=16000, freq=440, output_path=None):
    """Create test audio file for benchmarking."""
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"benchmark_audio_{duration_s}s.wav")
    
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Create more complex audio with multiple frequencies and some noise
    audio = (0.3 * np.sin(2 * np.pi * freq * t) + 
             0.2 * np.sin(2 * np.pi * freq * 1.5 * t) + 
             0.1 * np.random.normal(0, 0.1, len(t)))
    
    audio = audio.astype(np.float32)
    sf.write(output_path, audio, sample_rate, subtype='PCM_16')
    return output_path

def benchmark_audio_durations():
    """Benchmark processing times for different audio durations."""
    print("🎯 Benchmarking different audio durations...")
    
    durations = [10, 30, 60, 120, 300]  # seconds
    results = {}
    
    for duration in durations:
        print(f"\n📏 Testing {duration}s audio...")
        
        # Create test audio
        test_file = create_test_audio(duration)
        
        try:
            # Benchmark full processing
            with profile_operation(f"process_audio_file_{duration}s", {"duration": duration, "mode": "full"}):
                result = process_audio_file(test_file, fast_mode=False)
                chunk_count = len(result.get("chunks", []))
            
            # Benchmark fast mode
            with profile_operation(f"process_audio_file_{duration}s_fast", {"duration": duration, "mode": "fast"}):
                result_fast = process_audio_file(test_file, fast_mode=True)
            
            results[duration] = {
                'chunk_count': chunk_count,
                'transcript_length': len(result.get("transcript", "")),
                'emotion': result.get("emotion"),
                'distress': result.get("distress")
            }
            
        except Exception as e:
            print(f"❌ Error processing {duration}s audio: {e}")
            results[duration] = {'error': str(e)}
        
        finally:
            # Cleanup
            try:
                os.remove(test_file)
            except Exception:
                pass
    
    return results

def benchmark_parallelization():
    """Benchmark different parallelization settings."""
    print("🔄 Benchmarking parallelization settings...")
    
    test_file = create_test_audio(60)  # 1-minute audio
    worker_counts = [1, 2, 4, 8]
    results = {}
    
    for workers in worker_counts:
        print(f"\n⚙️ Testing with {workers} workers...")
        
        try:
            # Set environment variable for worker count
            old_workers = os.environ.get("PARALLEL_MAX_WORKERS")
            os.environ["PARALLEL_MAX_WORKERS"] = str(workers)
            
            with profile_operation(f"parallel_processing_{workers}_workers", {"workers": workers}):
                result = process_audio_file(test_file, fast_mode=False)
                chunk_count = len(result.get("chunks", []))
            
            results[workers] = {
                'chunk_count': chunk_count,
                'success': True
            }
            
            # Restore old value
            if old_workers is not None:
                os.environ["PARALLEL_MAX_WORKERS"] = old_workers
            elif "PARALLEL_MAX_WORKERS" in os.environ:
                del os.environ["PARALLEL_MAX_WORKERS"]
                
        except Exception as e:
            print(f"❌ Error with {workers} workers: {e}")
            results[workers] = {'error': str(e), 'success': False}
    
    # Cleanup
    try:
        os.remove(test_file)
    except Exception:
        pass
    
    return results

def benchmark_batch_processing():
    """Benchmark batch vs individual processing."""
    print("📦 Benchmarking batch processing...")
    
    test_file = create_test_audio(90)  # 1.5-minute audio for multiple chunks
    results = {}
    
    # Test with batch processing enabled
    try:
        old_batch = os.environ.get("ENABLE_BATCH_PROCESSING")
        
        # Batch processing ON
        os.environ["ENABLE_BATCH_PROCESSING"] = "true"
        with profile_operation("batch_processing_enabled", {"batch_enabled": True}):
            result_batch = process_audio_file(test_file, fast_mode=False)
        
        # Batch processing OFF
        os.environ["ENABLE_BATCH_PROCESSING"] = "false"
        with profile_operation("batch_processing_disabled", {"batch_enabled": False}):
            result_no_batch = process_audio_file(test_file, fast_mode=False)
        
        results = {
            'batch_enabled': {
                'chunk_count': len(result_batch.get("chunks", [])),
                'emotion': result_batch.get("emotion"),
                'success': True
            },
            'batch_disabled': {
                'chunk_count': len(result_no_batch.get("chunks", [])),
                'emotion': result_no_batch.get("emotion"),
                'success': True
            }
        }
        
        # Restore old value
        if old_batch is not None:
            os.environ["ENABLE_BATCH_PROCESSING"] = old_batch
        elif "ENABLE_BATCH_PROCESSING" in os.environ:
            del os.environ["ENABLE_BATCH_PROCESSING"]
            
    except Exception as e:
        print(f"❌ Error in batch processing benchmark: {e}")
        results = {'error': str(e)}
    
    # Cleanup
    try:
        os.remove(test_file)
    except Exception:
        pass
    
    return results

def benchmark_streaming_vs_batch():
    """Compare streaming vs batch processing."""
    print("🌊 Benchmarking streaming vs batch processing...")
    
    test_file = create_test_audio(45)  # 45s audio
    results = {}
    
    try:
        # Streaming mode (simulated realtime)
        def chunk_callback(chunk_result):
            pass  # Just receive chunks
        
        with profile_operation("streaming_realtime", {"mode": "streaming", "realtime": True}):
            result_stream = process_audio_file_stream(
                test_file, 
                fast_mode=False, 
                chunk_callback=chunk_callback,
                simulate_realtime=True
            )
        
        # Streaming mode (batch processing)
        with profile_operation("streaming_batch", {"mode": "streaming", "realtime": False}):
            result_batch = process_audio_file_stream(
                test_file, 
                fast_mode=False, 
                chunk_callback=chunk_callback,
                simulate_realtime=False
            )
        
        # Regular batch processing
        with profile_operation("regular_batch", {"mode": "regular"}):
            result_regular = process_audio_file(test_file, fast_mode=False)
        
        results = {
            'streaming_realtime': {
                'chunk_count': len(result_stream.get("chunks", [])),
                'emotion': result_stream.get("emotion"),
                'success': True
            },
            'streaming_batch': {
                'chunk_count': len(result_batch.get("chunks", [])),
                'emotion': result_batch.get("emotion"),
                'success': True
            },
            'regular_batch': {
                'chunk_count': len(result_regular.get("chunks", [])),
                'emotion': result_regular.get("emotion"),
                'success': True
            }
        }
        
    except Exception as e:
        print(f"❌ Error in streaming benchmark: {e}")
        results = {'error': str(e)}
    
    # Cleanup
    try:
        os.remove(test_file)
    except Exception:
        pass
    
    return results

def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("💾 Benchmarking memory usage patterns...")
    
    # Test with different audio sizes
    durations = [30, 120, 300]  # 30s, 2min, 5min
    results = {}
    
    for duration in durations:
        test_file = create_test_audio(duration)
        
        try:
            with profile_operation(f"memory_test_{duration}s", {"duration": duration}):
                result = process_audio_file(test_file, fast_mode=False)
                
            results[duration] = {
                'success': True,
                'chunks': len(result.get("chunks", [])),
                'transcript_length': len(result.get("transcript", ""))
            }
            
        except Exception as e:
            print(f"❌ Memory test failed for {duration}s: {e}")
            results[duration] = {'error': str(e), 'success': False}
        
        finally:
            try:
                os.remove(test_file)
            except Exception:
                pass
    
    return results

def generate_performance_report():
    """Generate comprehensive performance report."""
    print("📋 Generating performance report...")
    
    # Get all collected metrics
    metrics = getattr(profile_operation, 'metrics', [])
    if not metrics:
        print("❌ No performance metrics collected")
        return None
    
    # Analyze metrics
    report = {
        'summary': {
            'total_operations': len(metrics),
            'total_duration_ms': sum(m.duration_ms for m in metrics),
            'avg_duration_ms': sum(m.duration_ms for m in metrics) / len(metrics),
            'peak_memory_mb': max(m.memory_peak_mb for m in metrics),
            'total_memory_delta_mb': sum(m.memory_delta_mb for m in metrics),
            'avg_cpu_percent': sum(m.cpu_percent for m in metrics) / len(metrics)
        },
        'operations': {},
        'slowest_operations': [],
        'memory_intensive_operations': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Group by operation type
    op_groups = {}
    for metric in metrics:
        op_type = metric.operation
        if op_type not in op_groups:
            op_groups[op_type] = []
        op_groups[op_type].append(metric)
    
    # Analyze each operation type
    for op_type, op_metrics in op_groups.items():
        report['operations'][op_type] = {
            'count': len(op_metrics),
            'avg_duration_ms': sum(m.duration_ms for m in op_metrics) / len(op_metrics),
            'min_duration_ms': min(m.duration_ms for m in op_metrics),
            'max_duration_ms': max(m.duration_ms for m in op_metrics),
            'avg_memory_delta_mb': sum(m.memory_delta_mb for m in op_metrics) / len(op_metrics),
            'max_memory_peak_mb': max(m.memory_peak_mb for m in op_metrics),
            'avg_cpu_percent': sum(m.cpu_percent for m in op_metrics) / len(op_metrics)
        }
    
    # Find slowest operations
    sorted_by_duration = sorted(metrics, key=lambda m: m.duration_ms, reverse=True)
    report['slowest_operations'] = [
        {
            'operation': m.operation,
            'duration_ms': m.duration_ms,
            'memory_delta_mb': m.memory_delta_mb,
            'additional_info': m.additional_info
        }
        for m in sorted_by_duration[:5]
    ]
    
    # Find memory-intensive operations
    sorted_by_memory = sorted(metrics, key=lambda m: m.memory_delta_mb, reverse=True)
    report['memory_intensive_operations'] = [
        {
            'operation': m.operation,
            'memory_delta_mb': m.memory_delta_mb,
            'duration_ms': m.duration_ms,
            'additional_info': m.additional_info
        }
        for m in sorted_by_memory[:5]
    ]
    
    return report

def save_report(report, output_file="performance_report.json"):
    """Save performance report to file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Performance report saved to: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        return False

def print_summary(report):
    """Print performance summary to console."""
    if not report:
        return
    
    summary = report['summary']
    print("\n" + "="*60)
    print("🏆 PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Operations: {summary['total_operations']}")
    print(f"Total Duration: {summary['total_duration_ms']:.1f}ms ({summary['total_duration_ms']/1000:.1f}s)")
    print(f"Average Duration: {summary['avg_duration_ms']:.1f}ms")
    print(f"Peak Memory Usage: {summary['peak_memory_mb']:.1f}MB")
    print(f"Total Memory Delta: {summary['total_memory_delta_mb']:.1f}MB")
    print(f"Average CPU Usage: {summary['avg_cpu_percent']:.1f}%")
    
    print("\n🐌 SLOWEST OPERATIONS:")
    for i, op in enumerate(report['slowest_operations'][:3], 1):
        print(f"  {i}. {op['operation']}: {op['duration_ms']:.1f}ms")
    
    print("\n💾 MEMORY INTENSIVE OPERATIONS:")
    for i, op in enumerate(report['memory_intensive_operations'][:3], 1):
        print(f"  {i}. {op['operation']}: {op['memory_delta_mb']:.1f}MB")
    
    print("\n📊 OPERATION BREAKDOWN:")
    for op_type, stats in report['operations'].items():
        print(f"  {op_type}: {stats['avg_duration_ms']:.1f}ms avg ({stats['count']} runs)")
    
    print("="*60)

def run_full_benchmark():
    """Run the complete benchmark suite."""
    print("🚀 Starting comprehensive performance benchmark...")
    print("⏰ This may take several minutes...")
    
    # Initialize metrics collection
    profile_operation.metrics = []
    
    try:
        # Run all benchmarks
        duration_results = benchmark_audio_durations()
        parallel_results = benchmark_parallelization()
        batch_results = benchmark_batch_processing()
        streaming_results = benchmark_streaming_vs_batch()
        memory_results = benchmark_memory_usage()
        
        # Generate and save report
        report = generate_performance_report()
        if report:
            # Add benchmark-specific results
            report['benchmark_results'] = {
                'audio_durations': duration_results,
                'parallelization': parallel_results,
                'batch_processing': batch_results,
                'streaming_comparison': streaming_results,
                'memory_usage': memory_results
            }
            
            # Save to files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"performance_report_{timestamp}.json"
            save_report(report, report_file)
            
            # Print summary
            print_summary(report)
            
            return report
        else:
            print("❌ Failed to generate performance report")
            return None
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set optimal environment for benchmarking
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
    
    run_full_benchmark()
