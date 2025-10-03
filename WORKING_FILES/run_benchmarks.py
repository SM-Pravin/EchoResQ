#!/usr/bin/env python3
"""
Advanced benchmark runner for Emergency AI performance testing.
Includes detailed per-module timing, latency breakdown, and regression testing.
Target: <300ms latency per audio chunk.
"""

import os
import sys
import argparse
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from datetime import datetime
import tempfile
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
from modules.memory_management import get_memory_manager, check_memory_pressure
from modules.in_memory_audio import get_audio_processor
from modules.model_loader import get_device_manager

class ModuleTimer:
    """Detailed timing for individual modules with latency tracking."""
    
    def __init__(self, target_latency_ms: float = 300.0):
        self.target_latency_ms = target_latency_ms
        self.timings = {}
        self.call_counts = {}
        self.memory_usage = []
        self.start_time = None
        self.total_time = 0.0
        self._lock = threading.Lock()
    
    @contextmanager
    def time_module(self, module_name: str):
        """Context manager for timing individual modules."""
        start_time = time.perf_counter()
        
        # Track memory before
        memory_manager = get_memory_manager()
        memory_before = memory_manager.get_comprehensive_stats()['current_memory']
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Track memory after
            memory_after = memory_manager.get_comprehensive_stats()['current_memory']
            memory_delta = memory_after.get('process_memory_mb', 0) - memory_before.get('process_memory_mb', 0)
            
            with self._lock:
                if module_name not in self.timings:
                    self.timings[module_name] = []
                    self.call_counts[module_name] = 0
                
                self.timings[module_name].append(duration_ms)
                self.call_counts[module_name] += 1
                
                self.memory_usage.append({
                    'module': module_name,
                    'memory_delta_mb': memory_delta,
                    'timestamp': time.time()
                })
    
    def get_module_stats(self, module_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific module."""
        if module_name not in self.timings:
            return {}
        
        times = self.timings[module_name]
        return {
            'call_count': self.call_counts[module_name],
            'total_time_ms': sum(times),
            'average_time_ms': statistics.mean(times),
            'median_time_ms': statistics.median(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'target_met': statistics.mean(times) < self.target_latency_ms,
            'target_latency_ms': self.target_latency_ms,
            'performance_ratio': self.target_latency_ms / statistics.mean(times) if times else 0.0
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_calls = sum(self.call_counts.values())
        total_time = sum(sum(times) for times in self.timings.values())
        
        module_stats = {name: self.get_module_stats(name) for name in self.timings.keys()}
        
        # Find bottlenecks
        bottlenecks = []
        for name, stats in module_stats.items():
            if not stats.get('target_met', True):
                bottlenecks.append({
                    'module': name,
                    'average_ms': stats['average_time_ms'],
                    'target_ms': stats['target_latency_ms'],
                    'overrun_factor': stats['average_time_ms'] / stats['target_latency_ms']
                })
        
        bottlenecks.sort(key=lambda x: x['overrun_factor'], reverse=True)
        
        # Memory analysis
        memory_heavy_modules = []
        for usage in self.memory_usage:
            if usage['memory_delta_mb'] > 10:  # >10MB delta is significant
                memory_heavy_modules.append(usage)
        
        return {
            'summary': {
                'total_modules': len(self.timings),
                'total_calls': total_calls,
                'total_time_ms': total_time,
                'average_call_time_ms': total_time / total_calls if total_calls > 0 else 0.0,
                'target_met_overall': all(stats.get('target_met', True) for stats in module_stats.values()),
                'bottleneck_count': len(bottlenecks)
            },
            'modules': module_stats,
            'bottlenecks': bottlenecks,
            'memory_analysis': {
                'heavy_modules': memory_heavy_modules,
                'total_memory_usage': self.memory_usage
            },
            'timestamp': datetime.now().isoformat()
        }


def create_test_audio_samples() -> List[str]:
    """Create test audio samples of various lengths."""
    samples = []
    durations = [5, 15, 30, 60, 120]  # Different audio lengths
    
    for duration in durations:
        temp_file = tempfile.NamedTemporaryFile(suffix=f'_test_{duration}s.wav', delete=False)
        temp_file.close()
        
        # Generate complex test audio
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Multi-frequency signal with noise
        audio = (0.3 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz tone
                0.2 * np.sin(2 * np.pi * 880 * t) +  # 880 Hz harmonic
                0.1 * np.sin(2 * np.pi * 220 * t) +  # 220 Hz sub-harmonic
                0.05 * np.random.normal(0, 1, len(t)))  # Background noise
        
        # Add some amplitude variation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        sf.write(temp_file.name, audio.astype(np.float32), sr)
        samples.append(temp_file.name)
    
    return samples


def benchmark_module_performance(test_files: List[str], timer: ModuleTimer) -> Dict[str, Any]:
    """Benchmark individual module performance with detailed timing."""
    results = {'files_processed': 0, 'total_chunks': 0, 'errors': []}
    
    for file_path in test_files:
        try:
            print(f"  📁 Processing: {Path(file_path).name}")
            
            # Test async pipeline
            with timer.time_module('async_pipeline_full'):
                from async_pipeline import process_audio_file_async
                import asyncio
                
                try:
                    result = asyncio.run(process_audio_file_async(
                        file_path, 
                        fast_mode=False, 
                        return_chunks_details=True
                    ))
                    results['total_chunks'] += len(result.get('chunks', []))
                except Exception as e:
                    # Fallback to sync wrapper
                    from async_pipeline import process_audio_file_sync_wrapper
                    result = process_audio_file_sync_wrapper(
                        file_path, 
                        fast_mode=False, 
                        return_chunks_details=True
                    )
                    results['total_chunks'] += len(result.get('chunks', []))
            
            # Test individual components
            with timer.time_module('audio_preprocessing'):
                from modules.in_memory_audio import preprocess_audio_in_memory
                audio_buffer = preprocess_audio_in_memory(file_path)
            
            with timer.time_module('audio_chunking'):
                from modules.in_memory_audio import split_audio_in_memory
                chunks = split_audio_in_memory(audio_buffer, chunk_duration=30.0)
            
            # Test individual inference modules
            if chunks:
                test_chunk = chunks[0]
                
                with timer.time_module('speech_to_text'):
                    from modules.speech_to_text import transcribe_audio
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        sf.write(tmp.name, test_chunk.data, test_chunk.sample_rate)
                        transcript = transcribe_audio(tmp.name)
                        os.unlink(tmp.name)
                
                with timer.time_module('audio_emotion'):
                    from modules.emotion_audio import analyze_audio_emotion
                    audio_emotion = analyze_audio_emotion({
                        'data': test_chunk.data, 
                        'sr': test_chunk.sample_rate
                    })
                
                with timer.time_module('text_emotion'):
                    from modules.emotion_text import analyze_text_emotion
                    text_emotion = analyze_text_emotion(transcript or "test text")
                
                with timer.time_module('sound_detection'):
                    from modules.sound_event_detector import analyze_sound_events
                    sound_events = analyze_sound_events({
                        'data': test_chunk.data, 
                        'sr': test_chunk.sample_rate
                    })
                
                with timer.time_module('emotion_fusion'):
                    from modules.fusion_engine import fuse_emotions
                    confidence, emotion, fused = fuse_emotions(audio_emotion or {}, text_emotion or {})
            
            results['files_processed'] += 1
            
        except Exception as e:
            error_info = {
                'file': file_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results['errors'].append(error_info)
            print(f"    ❌ Error processing {Path(file_path).name}: {e}")
    
    return results


def run_regression_tests(baseline_file: Optional[str] = None) -> Dict[str, Any]:
    """Run automated performance regression tests."""
    print("🔍 Running performance regression tests...")
    
    # Load baseline if available
    baseline = None
    if baseline_file and os.path.exists(baseline_file):
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            print(f"    📊 Loaded baseline from {baseline_file}")
        except Exception as e:
            print(f"    ⚠️ Failed to load baseline: {e}")
    
    # Run current benchmarks
    timer = ModuleTimer(target_latency_ms=300.0)
    test_files = create_test_audio_samples()
    
    try:
        benchmark_results = benchmark_module_performance(test_files, timer)
        current_report = timer.get_comprehensive_report()
        
        # Compare with baseline if available
        regression_analysis = {}
        if baseline:
            regression_analysis = analyze_regression(baseline, current_report)
        
        # Save current results as new baseline
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"benchmark_results_{timestamp}.json"
        
        results = {
            'benchmark_results': benchmark_results,
            'performance_report': current_report,
            'regression_analysis': regression_analysis,
            'test_configuration': {
                'target_latency_ms': timer.target_latency_ms,
                'test_files_count': len(test_files),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"    💾 Results saved to {results_file}")
        
        return results
        
    finally:
        # Cleanup test files
        for file_path in test_files:
            try:
                os.unlink(file_path)
            except Exception:
                pass


def analyze_regression(baseline: Dict, current: Dict) -> Dict[str, Any]:
    """Analyze performance regression between baseline and current results."""
    analysis = {
        'regressions': [],
        'improvements': [],
        'new_modules': [],
        'removed_modules': [],
        'overall_change_percent': 0.0
    }
    
    baseline_modules = baseline.get('modules', {})
    current_modules = current.get('modules', {})
    
    # Check for new/removed modules
    baseline_names = set(baseline_modules.keys())
    current_names = set(current_modules.keys())
    
    analysis['new_modules'] = list(current_names - baseline_names)
    analysis['removed_modules'] = list(baseline_names - current_names)
    
    # Compare common modules
    common_modules = baseline_names & current_names
    
    total_baseline_time = 0.0
    total_current_time = 0.0
    
    for module_name in common_modules:
        baseline_avg = baseline_modules[module_name].get('average_time_ms', 0)
        current_avg = current_modules[module_name].get('average_time_ms', 0)
        
        total_baseline_time += baseline_avg
        total_current_time += current_avg
        
        if baseline_avg > 0:  # Avoid division by zero
            change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
            
            change_info = {
                'module': module_name,
                'baseline_ms': baseline_avg,
                'current_ms': current_avg,
                'change_percent': change_percent,
                'change_ms': current_avg - baseline_avg
            }
            
            if change_percent > 10:  # >10% slower is a regression
                analysis['regressions'].append(change_info)
            elif change_percent < -10:  # >10% faster is an improvement
                analysis['improvements'].append(change_info)
    
    # Overall performance change
    if total_baseline_time > 0:
        analysis['overall_change_percent'] = ((total_current_time - total_baseline_time) / total_baseline_time) * 100
    
    return analysis


def print_performance_summary(results: Dict[str, Any]):
    """Print a comprehensive performance summary."""
    report = results['performance_report']
    summary = report['summary']
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Target Latency: <300ms per module")
    print(f"Total Modules Tested: {summary['total_modules']}")
    print(f"Total Function Calls: {summary['total_calls']}")
    print(f"Average Call Time: {summary['average_call_time_ms']:.1f}ms")
    print(f"Target Met Overall: {'✅ YES' if summary['target_met_overall'] else '❌ NO'}")
    
    if summary['bottleneck_count'] > 0:
        print(f"\n🚫 PERFORMANCE BOTTLENECKS ({summary['bottleneck_count']}):")
        for bottleneck in report['bottlenecks'][:5]:  # Top 5
            print(f"  • {bottleneck['module']}: {bottleneck['average_ms']:.1f}ms " +
                  f"({bottleneck['overrun_factor']:.1f}x over target)")
    
    print(f"\n📊 MODULE PERFORMANCE:")
    for module_name, stats in report['modules'].items():
        status = "✅" if stats['target_met'] else "❌"
        print(f"  {status} {module_name}: {stats['average_time_ms']:.1f}ms " +
              f"(calls: {stats['call_count']})")
    
    # Regression analysis
    if 'regression_analysis' in results and results['regression_analysis']:
        reg_analysis = results['regression_analysis']
        print(f"\n📈 REGRESSION ANALYSIS:")
        print(f"  Overall Change: {reg_analysis['overall_change_percent']:.1f}%")
        
        if reg_analysis['regressions']:
            print(f"  Regressions: {len(reg_analysis['regressions'])}")
            for reg in reg_analysis['regressions'][:3]:
                print(f"    • {reg['module']}: +{reg['change_percent']:.1f}% slower")
        
        if reg_analysis['improvements']:
            print(f"  Improvements: {len(reg_analysis['improvements'])}")
            for imp in reg_analysis['improvements'][:3]:
                print(f"    • {imp['module']}: {abs(imp['change_percent']):.1f}% faster")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run Enhanced Emergency AI benchmarks")
    parser.add_argument("--type", choices=["quick", "full", "stress", "regression"], default="quick",
                       help="Type of benchmark to run")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers to test")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch processing")
    parser.add_argument("--gpu", action="store_true",
                       help="Enable GPU processing")
    parser.add_argument("--target-latency", type=float, default=300.0,
                       help="Target latency in milliseconds (default: 300ms)")
    parser.add_argument("--baseline", type=str, default=None,
                       help="Baseline results file for regression testing")
    parser.add_argument("--save-baseline", action="store_true",
                       help="Save results as new baseline")
    
    args = parser.parse_args()
    
    # Set environment based on arguments
    if args.workers:
        os.environ["PARALLEL_MAX_WORKERS"] = str(args.workers)
    
    if args.batch:
        os.environ["ENABLE_BATCH_PROCESSING"] = "true"
    else:
        os.environ["ENABLE_BATCH_PROCESSING"] = "false"
    
    if not args.gpu:
        os.environ["FORCE_CPU"] = "true"
    
    # Suppress verbose logging
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
    
    print("🚀 Enhanced Emergency AI Performance Benchmarks")
    print("="*60)
    print(f"Benchmark Type: {args.type}")
    print(f"Target Latency: {args.target_latency}ms")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Batch Processing: {args.batch}")
    print(f"GPU Enabled: {args.gpu}")
    if args.baseline:
        print(f"Baseline File: {args.baseline}")
    print("="*60)
    
    # Initialize system components
    print("\n🔧 Initializing system components...")
    device_manager = get_device_manager()
    memory_manager = get_memory_manager()
    
    print(f"Device: {device_manager.optimal_device}")
    memory_info = check_memory_pressure()
    print(f"Memory Pressure: {memory_info['pressure_level']}")
    
    try:
        if args.type == "quick":
            print("\n⚡ Running quick module benchmark...")
            timer = ModuleTimer(target_latency_ms=args.target_latency)
            test_files = create_test_audio_samples()[:2]  # Just 2 files for quick test
            
            benchmark_results = benchmark_module_performance(test_files, timer)
            performance_report = timer.get_comprehensive_report()
            
            results = {
                'benchmark_results': benchmark_results,
                'performance_report': performance_report
            }
            
            print_performance_summary(results)
            
            # Cleanup
            for file_path in test_files:
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
        
        elif args.type == "full":
            print("\n📊 Running comprehensive performance benchmark...")
            try:
                from benchmarks.performance_profiler import run_full_benchmark
                run_full_benchmark()
            except ImportError:
                print("Full benchmark module not available, running enhanced quick benchmark instead...")
                timer = ModuleTimer(target_latency_ms=args.target_latency)
                test_files = create_test_audio_samples()
                
                benchmark_results = benchmark_module_performance(test_files, timer)
                performance_report = timer.get_comprehensive_report()
                
                results = {
                    'benchmark_results': benchmark_results,
                    'performance_report': performance_report
                }
                
                print_performance_summary(results)
                
                # Cleanup
                for file_path in test_files:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass
        
        elif args.type == "stress":
            print("\n🔥 Running stress testing suite...")
            try:
                from benchmarks.stress_test import StressTestRunner
                runner = StressTestRunner()
                runner.run_full_stress_test()
            except ImportError:
                print("Stress test module not available, running intensive benchmark instead...")
                # Run multiple iterations
                timer = ModuleTimer(target_latency_ms=args.target_latency)
                
                for iteration in range(5):
                    print(f"  Iteration {iteration + 1}/5...")
                    test_files = create_test_audio_samples()
                    benchmark_module_performance(test_files, timer)
                    
                    # Cleanup
                    for file_path in test_files:
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass
                
                performance_report = timer.get_comprehensive_report()
                results = {'performance_report': performance_report}
                print_performance_summary(results)
        
        elif args.type == "regression":
            print("\n🔍 Running regression analysis...")
            results = run_regression_tests(args.baseline)
            print_performance_summary(results)
            
            if args.save_baseline:
                baseline_file = "performance_baseline.json"
                with open(baseline_file, 'w') as f:
                    json.dump(results['performance_report'], f, indent=2)
                print(f"\n💾 Baseline saved to {baseline_file}")
        
        # Final memory cleanup
        print("\n🧹 Cleaning up...")
        memory_manager.cleanup_all()
        
        print("\n✅ Benchmarking completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
