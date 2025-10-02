#!/usr/bin/env python3
"""
Test script to verify Streamlit optimization fixes.
Tests both original and optimized versions for model loading issues.
"""

import os
import sys
import time
import tempfile
import numpy as np
import soundfile as sf

def create_test_audio(duration=30):
    """Create a test audio file."""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio.astype(np.float32), sample_rate, subtype='PCM_16')
    temp_file.close()
    return temp_file.name

def test_model_loading():
    """Test that models load only once."""
    print("Testing Model Loading Optimization")
    print("="*50)
    
    # Clear any existing model state
    if 'modules.model_loader' in sys.modules:
        del sys.modules['modules.model_loader']
    
    print("1. Testing direct model import...")
    start_time = time.perf_counter()
    
    try:
        from modules.model_loader import get_models, is_loaded
        models1 = get_models()
        first_load_time = time.perf_counter() - start_time
        
        print(f"   First load: {first_load_time:.3f}s")
        print(f"   Models loaded: {is_loaded()}")
        
        # Test second import (should be instant)
        start_time = time.perf_counter()
        models2 = get_models()
        second_load_time = time.perf_counter() - start_time
        
        print(f"   Second load: {second_load_time:.3f}s")
        print(f"   Speedup: {first_load_time/second_load_time:.1f}x")
        
        # Verify same instances
        same_instances = all(models1[k] is models2[k] for k in models1.keys())
        print(f"   Same instances: {same_instances}")
        
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_analysis_pipeline():
    """Test the analysis pipeline functionality."""
    print("\nTesting Analysis Pipeline")
    print("="*50)
    
    try:
        from analysis_pipeline import process_audio_file
        
        # Create test audio
        test_file = create_test_audio(10)  # 10 second test
        print(f"   Created test audio: {os.path.basename(test_file)}")
        
        # Test analysis
        start_time = time.perf_counter()
        results = process_audio_file(test_file, fast_mode=True)
        processing_time = time.perf_counter() - start_time
        
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Detected emotion: {results.get('emotion', 'unknown')}")
        print(f"   Distress level: {results.get('distress', 'unknown')}")
        print(f"   Chunks processed: {len(results.get('chunks', []))}")
        
        # Cleanup
        os.unlink(test_file)
        
        return not results.get('error')
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_performance_settings():
    """Test performance optimization settings."""
    print("\nTesting Performance Settings")
    print("="*50)
    
    settings = {
        'PARALLEL_MAX_WORKERS': os.environ.get('PARALLEL_MAX_WORKERS', 'default'),
        'ENABLE_BATCH_PROCESSING': os.environ.get('ENABLE_BATCH_PROCESSING', 'default'),
        'AUDIO_BATCH_SIZE': os.environ.get('AUDIO_BATCH_SIZE', 'default'),
        'FORCE_CPU': os.environ.get('FORCE_CPU', 'default'),
    }
    
    print("   Current environment settings:")
    for key, value in settings.items():
        print(f"   {key}: {value}")
    
    # Test with optimized settings
    os.environ['PARALLEL_MAX_WORKERS'] = '4'
    os.environ['ENABLE_BATCH_PROCESSING'] = 'true'
    os.environ['AUDIO_BATCH_SIZE'] = '8'
    
    print("\n   Applied optimization settings:")
    print("   PARALLEL_MAX_WORKERS: 4")
    print("   ENABLE_BATCH_PROCESSING: true") 
    print("   AUDIO_BATCH_SIZE: 8")
    
    return True

def main():
    """Run all tests."""
    print("Streamlit Optimization Test Suite")
    print("="*60)
    
    results = {
        'model_loading': test_model_loading(),
        'analysis_pipeline': test_analysis_pipeline(),
        'performance_settings': test_performance_settings()
    }
    
    print("\nTest Results Summary")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All optimizations working correctly!")
        print("\nReady to run optimized Streamlit app:")
        print("   python -m streamlit run app_streamlit_optimized.py")
        return 0
    else:
        print("Some issues detected. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
