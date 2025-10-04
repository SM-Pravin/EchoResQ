#!/usr/bin/env python3
"""
Enhanced performance profiling and benchmarking suite for Emergency AI.
Provides detailed timing, memory usage, bottleneck analysis, and regression testing.
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

# Try to import memory profiler (optional dependency)
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available. Install with: pip install memory-profiler")

@dataclass
class PerformanceMetrics:
    """Container for performance metrics with latency targeting."""
    operation: str
    duration_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: str
    target_latency_ms: float = 300.0  # Default target
    meets_target: bool = None
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.meets_target is None:
            self.meets_target = self.duration_ms <= self.target_latency_ms


@dataclass
class PerformanceBenchmark:
    """Detailed performance benchmark results."""
    test_name: str
    timestamp: str
    audio_duration_seconds: float
    processing_time_ms: float
    processing_speed_ratio: float  # processing_time / audio_duration
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    confidence_score: float
    transcript_length: int
    models_loaded: List[str]
    bottlenecks: List[str]
    metadata: Dict[str, Any] = None


@dataclass
class SystemProfile:
    """System resource profiling information."""
    cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    python_version: str
    tensorflow_version: Optional[str]
    torch_version: Optional[str]
    cuda_available: bool
    gpu_memory_gb: Optional[float]

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
def profile_operation(operation_name: str, additional_info: Dict = None, target_latency_ms: float = 300.0):
    """Context manager for profiling operations with latency targeting."""
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
        
        # Create metrics with target tracking
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration_ms=duration_ms,
            memory_peak_mb=memory_info['peak_mb'],
            memory_delta_mb=memory_info['delta_mb'],
            cpu_percent=avg_cpu,
            timestamp=datetime.now().isoformat(),
            target_latency_ms=target_latency_ms,
            additional_info=additional_info or {}
        )
        
        # Store metrics globally for reporting
        if not hasattr(profile_operation, 'metrics'):
            profile_operation.metrics = []
        profile_operation.metrics.append(metrics)
        
        # Enhanced logging with target status
        status = "[OK]" if metrics.meets_target else "[ERROR]"
        print(f"[DASHBOARD] {status} {operation_name}: {duration_ms:.1f}ms (target: {target_latency_ms}ms), " +
              f"{memory_info['delta_mb']:.1f}MB delta, {avg_cpu:.1f}% CPU")

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

class EnhancedPerformanceProfiler:
    """Enhanced performance profiler with comprehensive benchmarking capabilities."""
    
    def __init__(self):
        from modules.enhanced_logger import get_logger
        from modules.config_manager import get_config_manager
        
        self.logger = get_logger()
        self.config = get_config_manager().config
        self.system_profile = self._get_system_profile()
        self.benchmarks: List[PerformanceBenchmark] = []
    
    def _get_system_profile(self) -> SystemProfile:
        """Get system resource information."""
        import platform
        
        # Check for optional dependencies
        tf_version = None
        torch_version = None
        cuda_available = False
        gpu_memory = None
        
        try:
            import tensorflow as tf
            tf_version = tf.__version__
        except ImportError:
            pass
        
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        return SystemProfile(
            cpu_cores=psutil.cpu_count(),
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            available_memory_gb=psutil.virtual_memory().available / (1024**3),
            python_version=platform.python_version(),
            tensorflow_version=tf_version,
            torch_version=torch_version,
            cuda_available=cuda_available,
            gpu_memory_gb=gpu_memory
        )
    
    def profile_audio_processing(self, audio_file: str, 
                               test_name: str = "audio_processing") -> PerformanceBenchmark:
        """Profile audio processing with detailed metrics."""
        self.logger.info(f"Profiling {test_name} for file: {audio_file}")
        
        # Get audio duration
        audio_info = sf.info(audio_file)
        audio_duration = audio_info.duration
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # CPU monitoring thread
        cpu_samples = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                try:
                    cpu_samples.append(psutil.cpu_percent(interval=0.1))
                except:
                    break
        
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        
        try:
            # Profile the processing
            start_time = time.time()
            
            with profile_operation("benchmark", test_name):
                result = process_audio_file(audio_file)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Stop monitoring
            monitoring = False
            cpu_thread.join()
            
            # Get final resource usage
            final_memory = process.memory_info().rss / (1024**2)  # MB
            peak_memory = max(initial_memory, final_memory)
            avg_cpu = sum(cpu_samples) / max(len(cpu_samples), 1) if cpu_samples else 0
            
            # Identify potential bottlenecks
            bottlenecks = self._identify_bottlenecks(
                processing_time, audio_duration, peak_memory, avg_cpu
            )
            
            benchmark = PerformanceBenchmark(
                test_name=test_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                audio_duration_seconds=audio_duration,
                processing_time_ms=processing_time,
                processing_speed_ratio=processing_time / 1000 / audio_duration,
                memory_usage_mb=final_memory - initial_memory,
                peak_memory_mb=peak_memory,
                cpu_usage_percent=avg_cpu,
                confidence_score=result.get('confidence', 0.0),
                transcript_length=len(result.get('transcript', '')),
                models_loaded=self._get_loaded_models(),
                bottlenecks=bottlenecks,
                metadata={
                    'audio_file': os.path.basename(audio_file),
                    'sample_rate': audio_info.samplerate,
                    'channels': audio_info.channels,
                    'system_profile': asdict(self.system_profile)
                }
            )
            
            self.benchmarks.append(benchmark)
            return benchmark
            
        except Exception as e:
            monitoring = False
            if cpu_thread.is_alive():
                cpu_thread.join()
            self.logger.error(f"Profiling failed for {test_name}: {e}")
            raise
    
    def _identify_bottlenecks(self, processing_time_ms: float, audio_duration: float,
                            memory_mb: float, cpu_percent: float) -> List[str]:
        """Identify performance bottlenecks based on metrics."""
        bottlenecks = []
        
        # Processing speed bottleneck
        speed_ratio = processing_time_ms / 1000 / audio_duration
        if speed_ratio > 2.0:  # Taking more than 2x real-time
            bottlenecks.append("slow_processing")
        
        # Memory bottleneck
        if memory_mb > 2000:  # More than 2GB
            bottlenecks.append("high_memory_usage")
        
        # CPU bottleneck
        if cpu_percent > 90:
            bottlenecks.append("high_cpu_usage")
        elif cpu_percent < 20:
            bottlenecks.append("underutilized_cpu")
        
        # Model loading bottleneck (if processing is slow but CPU is low)
        if speed_ratio > 1.5 and cpu_percent < 50:
            bottlenecks.append("model_loading_overhead")
        
        return bottlenecks
    
    def _get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        models = []
        
        if self.config.get('models', {}).get('speech_to_text', {}).get('enabled', True):
            models.append('speech_to_text')
        
        if self.config.get('models', {}).get('emotion_detection', {}).get('enabled', True):
            models.append('emotion_detection')
        
        if self.config.get('models', {}).get('sound_classification', {}).get('enabled', True):
            models.append('sound_classification')
        
        return models
    
    def benchmark_suite(self, test_audio_files: List[str]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        self.logger.info(f"Running benchmark suite on {len(test_audio_files)} files")
        
        suite_start = time.time()
        benchmarks = []
        
        for i, audio_file in enumerate(test_audio_files):
            if not os.path.exists(audio_file):
                self.logger.warning(f"Skipping missing file: {audio_file}")
                continue
            
            test_name = f"benchmark_{i+1}_{os.path.basename(audio_file)}"
            
            try:
                benchmark = self.profile_audio_processing(audio_file, test_name)
                benchmarks.append(benchmark)
            except Exception as e:
                self.logger.error(f"Benchmark failed for {audio_file}: {e}")
        
        suite_duration = time.time() - suite_start
        
        # Calculate suite statistics
        if benchmarks:
            avg_processing_time = sum(b.processing_time_ms for b in benchmarks) / len(benchmarks)
            avg_speed_ratio = sum(b.processing_speed_ratio for b in benchmarks) / len(benchmarks)
            avg_memory = sum(b.memory_usage_mb for b in benchmarks) / len(benchmarks)
            avg_confidence = sum(b.confidence_score for b in benchmarks) / len(benchmarks)
            
            # Find most common bottlenecks
            all_bottlenecks = [b for benchmark in benchmarks for b in benchmark.bottlenecks]
            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            
            common_bottlenecks = sorted(bottleneck_counts.items(), 
                                      key=lambda x: x[1], reverse=True)
        else:
            avg_processing_time = 0
            avg_speed_ratio = 0 
            avg_memory = 0
            avg_confidence = 0
            common_bottlenecks = []
        
        suite_results = {
            'suite_duration_seconds': suite_duration,
            'total_files': len(test_audio_files),
            'successful_benchmarks': len(benchmarks),
            'failed_benchmarks': len(test_audio_files) - len(benchmarks),
            'avg_processing_time_ms': avg_processing_time,
            'avg_speed_ratio': avg_speed_ratio,
            'avg_memory_usage_mb': avg_memory,
            'avg_confidence_score': avg_confidence,
            'common_bottlenecks': common_bottlenecks,
            'system_profile': asdict(self.system_profile),
            'individual_benchmarks': [asdict(b) for b in benchmarks]
        }
        
        return suite_results


def benchmark_audio_durations():
    """Benchmark processing times for different audio durations."""
    print("[TARGET] Benchmarking different audio durations...")
    
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
            print(f"[ERROR] Error processing {duration}s audio: {e}")
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
            print(f"[ERROR] Error with {workers} workers: {e}")
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
        print(f"[ERROR] Error in batch processing benchmark: {e}")
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
        print(f"[ERROR] Error in streaming benchmark: {e}")
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
            print(f"[ERROR] Memory test failed for {duration}s: {e}")
            results[duration] = {'error': str(e), 'success': False}
        
        finally:
            try:
                os.remove(test_file)
            except Exception:
                pass
    
    return results

def generate_performance_report():
    """Generate comprehensive performance report with latency analysis."""
    print("📋 Generating enhanced performance report...")
    
    # Get all collected metrics
    metrics = getattr(profile_operation, 'metrics', [])
    if not metrics:
        print("[ERROR] No performance metrics collected")
        return None
    
    # Target analysis
    target_met_count = sum(1 for m in metrics if m.meets_target)
    target_missed_count = len(metrics) - target_met_count
    
    # Latency breakdown by operation type
    operation_groups = {}
    for metric in metrics:
        op_name = metric.operation
        if op_name not in operation_groups:
            operation_groups[op_name] = []
        operation_groups[op_name].append(metric)
    
    # Analyze each operation group
    operation_analysis = {}
    for op_name, op_metrics in operation_groups.items():
        durations = [m.duration_ms for m in op_metrics]
        operation_analysis[op_name] = {
            'call_count': len(op_metrics),
            'total_duration_ms': sum(durations),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'median_duration_ms': sorted(durations)[len(durations)//2],
            'target_met_count': sum(1 for m in op_metrics if m.meets_target),
            'target_miss_count': sum(1 for m in op_metrics if not m.meets_target),
            'target_success_rate': sum(1 for m in op_metrics if m.meets_target) / len(op_metrics) * 100,
            'avg_memory_delta_mb': sum(m.memory_delta_mb for m in op_metrics) / len(op_metrics),
            'avg_cpu_percent': sum(m.cpu_percent for m in op_metrics) / len(op_metrics)
        }
    
    # Find bottlenecks
    bottlenecks = []
    for op_name, analysis in operation_analysis.items():
        if analysis['target_success_rate'] < 80:  # Less than 80% success rate
            bottlenecks.append({
                'operation': op_name,
                'avg_duration_ms': analysis['avg_duration_ms'],
                'success_rate': analysis['target_success_rate'],
                'worst_case_ms': analysis['max_duration_ms']
            })
    
    bottlenecks.sort(key=lambda x: x['avg_duration_ms'], reverse=True)
    
    # Analyze metrics
    report = {
        'summary': {
            'total_operations': len(metrics),
            'total_duration_ms': sum(m.duration_ms for m in metrics),
            'avg_duration_ms': sum(m.duration_ms for m in metrics) / len(metrics),
            'peak_memory_mb': max(m.memory_peak_mb for m in metrics),
            'total_memory_delta_mb': sum(m.memory_delta_mb for m in metrics),
            'avg_cpu_percent': sum(m.cpu_percent for m in metrics) / len(metrics),
            'target_met_count': target_met_count,
            'target_missed_count': target_missed_count,
            'overall_success_rate': (target_met_count / len(metrics)) * 100
        },
        'operations': operation_analysis,
        'bottlenecks': bottlenecks,
        'slowest_operations': sorted(metrics, key=lambda m: m.duration_ms, reverse=True)[:10],
        'memory_intensive_operations': sorted(metrics, key=lambda m: m.memory_delta_mb, reverse=True)[:10],
        'timestamp': datetime.now().isoformat()
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Operations: {report['summary']['total_operations']}")
    print(f"Average Duration: {report['summary']['avg_duration_ms']:.1f}ms")
    print(f"Target Success Rate: {report['summary']['overall_success_rate']:.1f}%")
    print(f"Operations Meeting Target: {target_met_count}/{len(metrics)}")
    
    if bottlenecks:
        print(f"\n[EMERGENCY] PERFORMANCE BOTTLENECKS ({len(bottlenecks)}):")
        for bottleneck in bottlenecks[:5]:
            print(f"  • {bottleneck['operation']}: {bottleneck['avg_duration_ms']:.1f}ms avg, " +
                  f"{bottleneck['success_rate']:.1f}% success rate")
    
    print(f"\n[DASHBOARD] OPERATION BREAKDOWN:")
    for op_name, analysis in operation_analysis.items():
        status = "[OK]" if analysis['target_success_rate'] >= 80 else "[ERROR]"
        print(f"  {status} {op_name}: {analysis['avg_duration_ms']:.1f}ms avg " +
              f"({analysis['target_success_rate']:.1f}% success, {analysis['call_count']} calls)")
    
    print(f"{'='*60}\n")
    
    return report
    
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
        print(f"[ERROR] Failed to save report: {e}")
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
    
    print("\n[DASHBOARD] OPERATION BREAKDOWN:")
    for op_type, stats in report['operations'].items():
        print(f"  {op_type}: {stats['avg_duration_ms']:.1f}ms avg ({stats['count']} runs)")
    
    print("="*60)

def run_full_benchmark():
    """Run the complete benchmark suite."""
    print("[ROCKET] Starting comprehensive performance benchmark...")
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
            print("[ERROR] Failed to generate performance report")
            return None
            
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        traceback.print_exc()
        return None


def run_performance_benchmarks(audio_files: List[str]) -> Dict[str, Any]:
    """Run performance benchmarks on specific audio files."""
    profiler = EnhancedPerformanceProfiler()
    
    try:
        results = profiler.benchmark_suite(audio_files)
        return {
            'avg_processing_time_ms': results.get('avg_processing_time_ms', 0),
            'avg_memory_usage_mb': results.get('avg_memory_usage_mb', 0),
            'total_files_processed': len(audio_files),
            'success_rate': results.get('success_rate', 0),
            'benchmark_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"[ERROR] Performance benchmarks failed: {e}")
        return {}


def run_memory_profiling() -> Dict[str, Any]:
    """Run memory profiling analysis."""
    try:
        # Try to import memory_profiler, fallback if not available
        try:
            import memory_profiler
            MEMORY_PROFILER_AVAILABLE = True
        except ImportError:
            MEMORY_PROFILER_AVAILABLE = False
        
        if not MEMORY_PROFILER_AVAILABLE:
            return {
                'model_loading': {'total_model_memory_mb': 0, 'error': 'memory_profiler not available'},
                'processing': {'peak_memory_mb': 0, 'error': 'memory_profiler not available'}
            }
        
        # Run basic memory profiling
        import psutil
        process = psutil.Process()
        
        # Memory before and after model loading
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        # Simulate model loading
        from modules.model_loader import ModelLoader
        loader = ModelLoader()
        # Just initialize, don't load all models to avoid excessive memory usage
        
        memory_after = process.memory_info().rss / (1024 * 1024)
        
        return {
            'model_loading': {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'total_model_memory_mb': memory_after - memory_before
            },
            'processing': {
                'current_memory_mb': memory_after,
                'peak_memory_mb': memory_after  # Simplified for now
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'model_loading': {'total_model_memory_mb': 0},
            'processing': {'peak_memory_mb': 0}
        }


if __name__ == "__main__":
    # Set optimal environment for benchmarking
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
    
    run_full_benchmark()
