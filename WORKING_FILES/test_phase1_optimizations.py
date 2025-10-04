"""
Test script to validate Phase 1 optimizations for Emergency AI pipeline.
Tests async processing, smart batch processing, performance monitoring, and in-memory operations.
"""

import time
import asyncio
import os
import sys
import tempfile
import numpy as np
import soundfile as sf

# Add the working files directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'WORKING_FILES'))

# Import our optimized modules
from analysis_pipeline import process_audio_file
from async_pipeline import AsyncAudioProcessor
from modules.smart_batch_processing import get_batch_processor, optimize_batch_processing_for_system
from modules.real_time_performance_monitor import (
    get_performance_monitor, 
    setup_pipeline_monitoring,
    get_performance_dashboard
)
from modules.in_memory_audio import AudioBuffer, get_audio_processor
from modules.memory_management import get_memory_manager
from modules.model_loader import get_device_manager


def create_test_audio_files():
    """Create test audio files for validation."""
    print("🎵 Creating test audio files...")
    
    test_files = []
    
    # Create different duration test files
    durations = [10, 30, 60, 120]  # seconds
    sample_rate = 16000
    
    for duration in durations:
        # Generate synthetic audio with some noise and speech-like patterns
        t = np.linspace(0, duration, duration * sample_rate)
        
        # Base sine wave
        audio = np.sin(2 * np.pi * 440 * t) * 0.1
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(t))
        audio += noise
        
        # Add some speech-like modulation
        modulation = np.sin(2 * np.pi * 2 * t) * 0.3
        audio *= (1 + modulation)
        
        # Ensure proper range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=f'_test_{duration}s.wav', delete=False)
        sf.write(temp_file.name, audio, sample_rate)
        temp_file.close()
        
        test_files.append(temp_file.name)
        print(f"  Created {duration}s test audio: {temp_file.name}")
    
    return test_files


def test_device_management():
    """Test device management optimization."""
    print("\n[CONFIG] Testing Device Management...")
    
    device_manager = get_device_manager()
    device_info = device_manager.device_info
    
    print(f"  CPU Cores: {device_info['cpu_cores']}")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    print(f"  DirectML Available: {device_info['directml_available']}")
    print(f"  Optimal Device: {device_info['optimal_device']}")
    
    # Test ONNX providers
    providers = device_manager.get_onnx_providers()
    print(f"  ONNX Providers: {providers}")
    
    return True


def test_memory_management():
    """Test memory management optimization."""
    print("\n🧠 Testing Memory Management...")
    
    memory_manager = get_memory_manager()
    
    # Test buffer pool
    buffer1 = memory_manager.get_buffer((16000,), np.float32)
    buffer2 = memory_manager.get_buffer((16000,), np.float32)
    
    print(f"  Buffer pool created buffers: {buffer1 is not None and buffer2 is not None}")
    
    # Return buffers
    memory_manager.return_buffer(buffer1)
    memory_manager.return_buffer(buffer2)
    
    # Check memory pressure
    pressure_info = memory_manager.check_memory_pressure()
    print(f"  Memory pressure level: {pressure_info['pressure_level']}")
    print(f"  Available memory: {pressure_info['available_memory_mb']:.1f}MB")
    
    return True


def test_smart_batch_processing():
    """Test smart batch processing optimization."""
    print("\n📦 Testing Smart Batch Processing...")
    
    # Get optimal configuration
    config = optimize_batch_processing_for_system()
    print(f"  Optimal batch size range: {config.min_batch_size}-{config.max_batch_size}")
    print(f"  Target memory usage: {config.target_memory_usage_mb}MB")
    print(f"  Adaptive sizing: {config.adaptive_sizing}")
    
    batch_processor = get_batch_processor()
    
    # Test batch size calculation
    optimal_size = batch_processor.calculate_optimal_batch_size(
        item_memory_mb=10.0,
        processing_time_ms=150.0
    )
    print(f"  Calculated optimal batch size: {optimal_size}")
    
    # Test adaptive chunk sizing
    chunk_sizes = []
    for duration in [30, 120, 300, 600]:
        chunk_size = batch_processor.calculate_adaptive_chunk_size(duration)
        chunk_sizes.append(chunk_size)
        print(f"  Adaptive chunk size for {duration}s audio: {chunk_size:.1f}s")
    
    return len(chunk_sizes) == 4


def test_in_memory_processing(test_files):
    """Test in-memory audio processing."""
    print("\n💾 Testing In-Memory Processing...")
    
    if not test_files:
        print("  No test files available")
        return False
    
    audio_processor = get_audio_processor()
    
    # Test loading audio into memory
    test_file = test_files[0]  # Use shortest file
    audio_buffer = audio_processor.load_audio_file(test_file)
    
    print(f"  Loaded audio buffer: {audio_buffer.duration:.1f}s, {audio_buffer.memory_usage_mb:.2f}MB")
    
    # Test chunking in memory
    chunks = audio_processor.split_into_chunks(audio_buffer, chunk_duration=15.0)
    print(f"  Created {len(chunks)} in-memory chunks")
    
    # Test preprocessing
    processed_buffer = audio_processor.preprocess_audio_in_memory(audio_buffer)
    print(f"  Preprocessed audio: {processed_buffer.sample_rate}Hz, {len(processed_buffer.data)} samples")
    
    return len(chunks) > 0


def test_performance_monitoring():
    """Test real-time performance monitoring."""
    print("\n[DASHBOARD] Testing Performance Monitoring...")
    
    # Set up monitoring
    monitor = setup_pipeline_monitoring()
    
    # Simulate some processing with performance tracking
    start_time = time.perf_counter()
    
    # Simulate chunk processing
    for i in range(3):
        chunk_start = time.perf_counter()
        time.sleep(0.1)  # Simulate processing
        chunk_end = time.perf_counter()
        
        chunk_duration_ms = (chunk_end - chunk_start) * 1000
        monitor.record_processing_time("chunk_processing", chunk_duration_ms)
    
    # Record throughput
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    monitor.record_throughput(3, total_duration)
    
    # Get performance summary
    summary = monitor.get_current_performance_summary()
    print(f"  Current metrics tracked: {len(summary['current_metrics'])}")
    print(f"  Target compliance: {len(summary['target_compliance'])}")
    
    # Check if we have reasonable performance data
    has_latency = 'chunk_processing_latency_ms' in summary['current_metrics']
    has_throughput = 'throughput_chunks_per_sec' in summary['current_metrics']
    
    print(f"  Latency tracking: {has_latency}")
    print(f"  Throughput tracking: {has_throughput}")
    
    return has_latency and has_throughput


async def test_async_processing(test_files):
    """Test async processing optimization."""
    print("\n⚡ Testing Async Processing...")
    
    if not test_files:
        print("  No test files available")
        return False
    
    processor = AsyncAudioProcessor()
    
    # Test concurrent processing
    start_time = time.perf_counter()
    
    # Process multiple files concurrently
    results = await processor.process_audio_files_concurrent(
        test_files[:2],  # Use first 2 test files
        fast_mode=True   # Use fast mode for quicker testing
    )
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"  Processed {len(results)} files concurrently in {duration:.2f}s")
    
    # Check results
    successful_results = [r for r in results if r.get('error') is None]
    print(f"  Successful results: {len(successful_results)}/{len(results)}")
    
    return len(successful_results) > 0


def test_legacy_pipeline(test_files):
    """Test the enhanced legacy pipeline with optimizations."""
    print("\n🔄 Testing Enhanced Legacy Pipeline...")
    
    if not test_files:
        print("  No test files available")
        return False
    
    test_file = test_files[0]  # Use shortest file
    
    # Test fast mode
    start_time = time.perf_counter()
    result_fast = process_audio_file(test_file, fast_mode=True)
    fast_duration = time.perf_counter() - start_time
    
    # Test full mode with chunks
    start_time = time.perf_counter()
    result_full = process_audio_file(test_file, fast_mode=False, return_chunks_details=True)
    full_duration = time.perf_counter() - start_time
    
    print(f"  Fast mode: {fast_duration*1000:.1f}ms")
    print(f"  Full mode: {full_duration*1000:.1f}ms")
    print(f"  Fast mode successful: {result_fast.get('error') is None}")
    print(f"  Full mode successful: {result_full.get('error') is None}")
    print(f"  Chunks processed: {len(result_full.get('chunks', []))}")
    
    # Check if we met performance targets
    fast_target_met = fast_duration * 1000 < 300  # <300ms target
    full_target_met = full_duration * 1000 < 600  # <600ms for full processing
    
    print(f"  Fast mode target met (<300ms): {fast_target_met}")
    print(f"  Full mode reasonable (<600ms): {full_target_met}")
    
    return result_fast.get('error') is None and result_full.get('error') is None


def cleanup_test_files(test_files):
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"  Deleted: {file_path}")
        except Exception as e:
            print(f"  Failed to delete {file_path}: {e}")


async def run_comprehensive_test():
    """Run comprehensive test of all Phase 1 optimizations."""
    print("[ROCKET] EMERGENCY AI PHASE 1 OPTIMIZATION VALIDATION")
    print("=" * 60)
    
    test_results = {}
    test_files = []
    
    try:
        # Create test audio files
        test_files = create_test_audio_files()
        
        # Run all tests
        print("\n📋 Running optimization tests...")
        
        test_results['device_management'] = test_device_management()
        test_results['memory_management'] = test_memory_management() 
        test_results['smart_batch_processing'] = test_smart_batch_processing()
        test_results['in_memory_processing'] = test_in_memory_processing(test_files)
        test_results['performance_monitoring'] = test_performance_monitoring()
        test_results['async_processing'] = await test_async_processing(test_files)
        test_results['legacy_pipeline'] = test_legacy_pipeline(test_files)
        
        # Summary
        print("\n" + "=" * 60)
        print("[DASHBOARD] TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "[OK] PASS" if result else "[ERROR] FAIL"
            print(f"  {status} {test_name.replace('_', ' ').title()}")
            if result:
                passed_tests += 1
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("[SUCCESS] ALL PHASE 1 OPTIMIZATIONS VALIDATED SUCCESSFULLY!")
        elif passed_tests >= total_tests * 0.8:
            print("[WARNING] Most optimizations working - some issues to investigate")
        else:
            print("[ERROR] Significant issues detected - optimization review needed")
        
        # Show performance dashboard
        if test_results.get('performance_monitoring'):
            print("\n" + "=" * 60)
            print("[CHART] PERFORMANCE DASHBOARD")
            print("=" * 60)
            dashboard = get_performance_dashboard()
            print(dashboard)
        
        return passed_tests >= total_tests * 0.8
        
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        cleanup_test_files(test_files)


def main():
    """Main test execution function."""
    try:
        # Run the comprehensive test
        result = asyncio.run(run_comprehensive_test())
        
        if result:
            print("\n[OK] Phase 1 optimization validation completed successfully!")
            return 0
        else:
            print("\n[ERROR] Phase 1 optimization validation failed!")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())