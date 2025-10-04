#!/usr/bin/env python3
"""
Stress testing suite for Emergency AI pipeline.
Tests system behavior under heavy load, concurrent requests, and edge cases.
"""

import os
import sys
import time
import threading
import tempfile
import concurrent.futures
from pathlib import Path
from datetime import datetime
import json
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from analysis_pipeline import process_audio_file, process_audio_file_stream

class StressTestRunner:
    """Manages stress testing operations."""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        
    def create_test_audio(self, duration_s=30, complexity="normal"):
        """Create test audio with different complexity levels."""
        sample_rate = 16000
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        
        if complexity == "simple":
            # Simple sine wave
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        elif complexity == "complex":
            # Complex audio with multiple frequencies and varying amplitude
            audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 
                    0.2 * np.sin(2 * np.pi * 880 * t) + 
                    0.15 * np.sin(2 * np.pi * 1320 * t) +
                    0.1 * np.random.normal(0, 0.1, len(t)) +
                    0.1 * np.sin(2 * np.pi * 220 * t * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))))
        else:  # normal
            # Normal complexity
            audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 
                    0.2 * np.sin(2 * np.pi * 660 * t) + 
                    0.1 * np.random.normal(0, 0.05, len(t)))
        
        # Add some dynamic range
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        audio = audio * envelope
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio.astype(np.float32), sample_rate, subtype='PCM_16')
        temp_file.close()
        
        return temp_file.name
    
    def single_processing_task(self, task_id, audio_file, fast_mode=False):
        """Single processing task for concurrent testing."""
        try:
            start_time = time.perf_counter()
            result = process_audio_file(audio_file, fast_mode=fast_mode)
            end_time = time.perf_counter()
            
            return {
                'task_id': task_id,
                'success': True,
                'duration': end_time - start_time,
                'transcript_length': len(result.get('transcript', '')),
                'emotion': result.get('emotion'),
                'distress': result.get('distress'),
                'chunks': len(result.get('chunks', [])),
                'error': None
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'duration': 0,
                'error': str(e)
            }
    
    def concurrent_load_test(self, num_concurrent=5, num_tasks_per_worker=3, audio_duration=30):
        """Test concurrent processing with multiple workers."""
        print(f"🔄 Running concurrent load test: {num_concurrent} workers, {num_tasks_per_worker} tasks each")
        
        # Create test audio files
        test_files = []
        for i in range(num_concurrent):
            complexity = ["simple", "normal", "complex"][i % 3]
            test_file = self.create_test_audio(audio_duration, complexity)
            test_files.append(test_file)
        
        results = []
        start_time = time.perf_counter()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                # Submit all tasks
                futures = []
                task_id = 0
                
                for worker_id in range(num_concurrent):
                    for task_num in range(num_tasks_per_worker):
                        audio_file = test_files[worker_id]
                        fast_mode = (task_num % 2 == 0)  # Alternate between fast and full mode
                        
                        future = executor.submit(
                            self.single_processing_task, 
                            task_id, 
                            audio_file, 
                            fast_mode
                        )
                        futures.append(future)
                        task_id += 1
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=120)  # 2-minute timeout per task
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'task_id': -1,
                            'success': False,
                            'duration': 0,
                            'error': f"Future failed: {str(e)}"
                        })
        
        except Exception as e:
            print(f"[ERROR] Concurrent test failed: {e}")
        
        finally:
            # Cleanup test files
            for test_file in test_files:
                try:
                    os.unlink(test_file)
                except Exception:
                    pass
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Analyze results
        successful_tasks = [r for r in results if r['success']]
        failed_tasks = [r for r in results if not r['success']]
        
        analysis = {
            'total_tasks': len(results),
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(successful_tasks) / len(results) if results else 0,
            'total_duration': total_duration,
            'avg_task_duration': np.mean([r['duration'] for r in successful_tasks]) if successful_tasks else 0,
            'max_task_duration': max([r['duration'] for r in successful_tasks]) if successful_tasks else 0,
            'min_task_duration': min([r['duration'] for r in successful_tasks]) if successful_tasks else 0,
            'throughput_tasks_per_second': len(results) / total_duration,
            'errors': [r['error'] for r in failed_tasks if r['error']]
        }
        
        print(f"[OK] Concurrent test completed: {analysis['success_rate']:.1%} success rate")
        print(f"[DASHBOARD] Throughput: {analysis['throughput_tasks_per_second']:.2f} tasks/second")
        
        return analysis
    
    def memory_stress_test(self, num_iterations=10, audio_duration=120):
        """Test memory usage under repeated processing."""
        print(f"💾 Running memory stress test: {num_iterations} iterations of {audio_duration}s audio")
        
        import psutil
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        memory_readings = [initial_memory]
        
        test_file = self.create_test_audio(audio_duration, "complex")
        
        try:
            for i in range(num_iterations):
                print(f"  Iteration {i+1}/{num_iterations}")
                
                # Process audio
                try:
                    result = process_audio_file(test_file, fast_mode=False)
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_readings.append(current_memory)
                    peak_memory = max(peak_memory, current_memory)
                    
                except Exception as e:
                    print(f"    [ERROR] Error in iteration {i+1}: {e}")
        
        finally:
            try:
                os.unlink(test_file)
            except Exception:
                pass
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        analysis = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': final_memory - initial_memory,
            'peak_increase_mb': peak_memory - initial_memory,
            'avg_memory_mb': np.mean(memory_readings),
            'memory_readings': memory_readings,
            'iterations_completed': num_iterations
        }
        
        print(f"[CHART] Memory delta: {analysis['memory_increase_mb']:.1f}MB, Peak: {analysis['peak_increase_mb']:.1f}MB")
        
        return analysis
    
    def edge_case_test(self):
        """Test edge cases and error handling."""
        print("[TARGET] Running edge case tests...")
        
        tests = []
        
        # Test 1: Very short audio
        try:
            short_file = self.create_test_audio(0.5, "simple")  # 0.5 second
            start_time = time.perf_counter()
            result = process_audio_file(short_file, fast_mode=False)
            duration = time.perf_counter() - start_time
            
            tests.append({
                'test': 'very_short_audio',
                'success': True,
                'duration': duration,
                'result_summary': {
                    'emotion': result.get('emotion'),
                    'chunks': len(result.get('chunks', [])),
                    'transcript_length': len(result.get('transcript', ''))
                }
            })
            os.unlink(short_file)
            
        except Exception as e:
            tests.append({
                'test': 'very_short_audio',
                'success': False,
                'error': str(e)
            })
        
        # Test 2: Very long audio
        try:
            long_file = self.create_test_audio(600, "normal")  # 10 minutes
            start_time = time.perf_counter()
            result = process_audio_file(long_file, fast_mode=True)  # Use fast mode for speed
            duration = time.perf_counter() - start_time
            
            tests.append({
                'test': 'very_long_audio',
                'success': True,
                'duration': duration,
                'result_summary': {
                    'emotion': result.get('emotion'),
                    'chunks': len(result.get('chunks', [])),
                    'transcript_length': len(result.get('transcript', ''))
                }
            })
            os.unlink(long_file)
            
        except Exception as e:
            tests.append({
                'test': 'very_long_audio',
                'success': False,
                'error': str(e)
            })
        
        # Test 3: Silent audio
        try:
            # Create silent audio
            silent_audio = np.zeros(16000 * 30, dtype=np.float32)  # 30s of silence
            silent_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(silent_file.name, silent_audio, 16000, subtype='PCM_16')
            silent_file.close()
            
            start_time = time.perf_counter()
            result = process_audio_file(silent_file.name, fast_mode=False)
            duration = time.perf_counter() - start_time
            
            tests.append({
                'test': 'silent_audio',
                'success': True,
                'duration': duration,
                'result_summary': {
                    'emotion': result.get('emotion'),
                    'chunks': len(result.get('chunks', [])),
                    'transcript_length': len(result.get('transcript', ''))
                }
            })
            os.unlink(silent_file.name)
            
        except Exception as e:
            tests.append({
                'test': 'silent_audio',
                'success': False,
                'error': str(e)
            })
        
        # Test 4: High-noise audio
        try:
            # Create noisy audio
            noise_audio = np.random.normal(0, 0.3, 16000 * 30).astype(np.float32)
            noise_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(noise_file.name, noise_audio, 16000, subtype='PCM_16')
            noise_file.close()
            
            start_time = time.perf_counter()
            result = process_audio_file(noise_file.name, fast_mode=False)
            duration = time.perf_counter() - start_time
            
            tests.append({
                'test': 'high_noise_audio',
                'success': True,
                'duration': duration,
                'result_summary': {
                    'emotion': result.get('emotion'),
                    'chunks': len(result.get('chunks', [])),
                    'transcript_length': len(result.get('transcript', ''))
                }
            })
            os.unlink(noise_file.name)
            
        except Exception as e:
            tests.append({
                'test': 'high_noise_audio',
                'success': False,
                'error': str(e)
            })
        
        successful_tests = [t for t in tests if t['success']]
        failed_tests = [t for t in tests if not t['success']]
        
        print(f"[OK] Edge case tests: {len(successful_tests)}/{len(tests)} passed")
        
        return {
            'total_tests': len(tests),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'tests': tests
        }
    
    def run_full_stress_test(self):
        """Run the complete stress test suite."""
        print("[ROCKET] Starting comprehensive stress test...")
        self.start_time = time.perf_counter()
        
        try:
            # Test 1: Concurrent load
            concurrent_results = self.concurrent_load_test(
                num_concurrent=3, 
                num_tasks_per_worker=2, 
                audio_duration=30
            )
            
            # Test 2: Memory stress
            memory_results = self.memory_stress_test(
                num_iterations=5, 
                audio_duration=60
            )
            
            # Test 3: Edge cases
            edge_case_results = self.edge_case_test()
            
            # Test 4: High concurrent load
            high_load_results = self.concurrent_load_test(
                num_concurrent=8, 
                num_tasks_per_worker=1, 
                audio_duration=15
            )
            
            self.end_time = time.perf_counter()
            total_duration = self.end_time - self.start_time
            
            # Compile results
            final_results = {
                'test_summary': {
                    'total_duration_seconds': total_duration,
                    'timestamp': datetime.now().isoformat(),
                    'system_info': self._get_system_info()
                },
                'concurrent_load_test': concurrent_results,
                'memory_stress_test': memory_results,
                'edge_case_test': edge_case_results,
                'high_concurrent_load_test': high_load_results
            }
            
            # Save results
            self._save_results(final_results)
            self._print_summary(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"[ERROR] Stress test suite failed: {e}")
            traceback.print_exc()
            return None
    
    def _get_system_info(self):
        """Get system information for the report."""
        try:
            import psutil
            return {
                'cpu_count': os.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'platform': sys.platform
            }
        except Exception:
            return {'error': 'Could not collect system info'}
    
    def _save_results(self, results):
        """Save stress test results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stress_test_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Stress test results saved to: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")
    
    def _print_summary(self, results):
        """Print stress test summary."""
        print("\n" + "="*60)
        print("🏆 STRESS TEST SUMMARY")
        print("="*60)
        
        total_duration = results['test_summary']['total_duration_seconds']
        print(f"Total Test Duration: {total_duration:.1f} seconds")
        
        # Concurrent load test
        concurrent = results['concurrent_load_test']
        print(f"\n🔄 Concurrent Load Test:")
        print(f"  Success Rate: {concurrent['success_rate']:.1%}")
        print(f"  Throughput: {concurrent['throughput_tasks_per_second']:.2f} tasks/sec")
        print(f"  Avg Task Duration: {concurrent['avg_task_duration']:.2f}s")
        
        # Memory stress test  
        memory = results['memory_stress_test']
        print(f"\n💾 Memory Stress Test:")
        print(f"  Memory Increase: {memory['memory_increase_mb']:.1f}MB")
        print(f"  Peak Memory Increase: {memory['peak_increase_mb']:.1f}MB")
        print(f"  Iterations Completed: {memory['iterations_completed']}")
        
        # Edge case test
        edge_cases = results['edge_case_test']
        print(f"\n[TARGET] Edge Case Test:")
        print(f"  Tests Passed: {edge_cases['successful_tests']}/{edge_cases['total_tests']}")
        
        # High load test
        high_load = results['high_concurrent_load_test']
        print(f"\n⚡ High Concurrent Load Test:")
        print(f"  Success Rate: {high_load['success_rate']:.1%}")
        print(f"  Throughput: {high_load['throughput_tasks_per_second']:.2f} tasks/sec")
        
        print("="*60)

def main():
    """Main entry point for stress testing."""
    # Set environment for testing
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
    
    runner = StressTestRunner()
    results = runner.run_full_stress_test()
    
    if results:
        print("[OK] Stress testing completed successfully")
        return 0
    else:
        print("[ERROR] Stress testing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
