"""
Comprehensive testing framework for Emergency AI with stress tests and regression benchmarks.
Includes edge-case handling, performance testing, and automated quality assurance.
"""

import os
import sys
import time
import json
import pytest
import numpy as np
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import soundfile as sf
import librosa

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.enhanced_logger import get_logger, track_operation, PerformanceMetrics
from modules.config_manager import get_config_manager
from modules.in_memory_audio import AudioBuffer, get_audio_processor
from analysis_pipeline import process_audio_file


@dataclass
class TestResult:
    """Test result with performance metrics."""
    test_name: str
    success: bool
    duration_ms: float
    confidence: Optional[float]
    distress_score: Optional[float]
    transcript: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class BenchmarkResult:
    """Benchmark result for regression testing."""
    test_suite: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    avg_processing_time_ms: float
    avg_confidence: float
    avg_distress_accuracy: float
    performance_regression: bool
    accuracy_regression: bool
    results: List[TestResult] = None


class AudioTestGenerator:
    """Generate synthetic audio for testing various scenarios."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = get_logger()
    
    def generate_silent_audio(self, duration: float) -> np.ndarray:
        """Generate silent audio for testing."""
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)
    
    def generate_white_noise(self, duration: float, amplitude: float = 0.1) -> np.ndarray:
        """Generate white noise for testing noise handling."""
        samples = int(duration * self.sample_rate)
        return np.random.normal(0, amplitude, samples).astype(np.float32)
    
    def generate_sine_wave(self, frequency: float, duration: float, amplitude: float = 0.5) -> np.ndarray:
        """Generate sine wave for testing tonal audio."""
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    def generate_mixed_audio(self, speech_amplitude: float = 0.7, 
                           noise_amplitude: float = 0.3, duration: float = 10.0) -> np.ndarray:
        """Generate mixed speech and noise for testing robustness."""
        # Simulate speech with varying frequencies
        speech = np.zeros(int(duration * self.sample_rate))
        for i in range(0, int(duration), 2):
            freq = np.random.uniform(200, 800)  # Human speech range
            segment_duration = min(2.0, duration - i)
            segment = self.generate_sine_wave(freq, segment_duration, speech_amplitude)
            start_idx = i * self.sample_rate
            end_idx = start_idx + len(segment)
            if end_idx <= len(speech):
                speech[start_idx:end_idx] = segment
        
        # Add background noise
        noise = self.generate_white_noise(duration, noise_amplitude)
        
        return (speech + noise).astype(np.float32)
    
    def generate_whisper_audio(self, duration: float = 5.0) -> np.ndarray:
        """Generate very quiet audio to test whisper detection."""
        base_audio = self.generate_mixed_audio(duration=duration)
        return (base_audio * 0.05).astype(np.float32)  # Very low amplitude
    
    def generate_loud_audio(self, duration: float = 5.0) -> np.ndarray:
        """Generate loud audio to test clipping handling."""
        base_audio = self.generate_mixed_audio(duration=duration)
        return np.clip(base_audio * 3.0, -1.0, 1.0).astype(np.float32)
    
    def generate_rapid_speech(self, duration: float = 10.0) -> np.ndarray:
        """Generate rapidly changing audio to simulate fast speech."""
        samples = int(duration * self.sample_rate)
        audio = np.zeros(samples)
        
        # Rapid frequency changes
        for i in range(0, samples, 1000):  # Change every ~62ms
            freq = np.random.uniform(300, 1000)
            segment_size = min(1000, samples - i)
            t = np.linspace(0, segment_size / self.sample_rate, segment_size, False)
            segment = 0.5 * np.sin(2 * np.pi * freq * t)
            audio[i:i + segment_size] = segment
        
        return audio.astype(np.float32)
    
    def generate_overlapping_speech(self, duration: float = 15.0) -> np.ndarray:
        """Generate overlapping speech patterns."""
        # Two speakers with different frequency ranges
        speaker1 = self.generate_mixed_audio(0.6, 0.1, duration)  # Lower noise
        speaker2_freqs = [freq * 1.5 for freq in [200, 400, 600]]  # Higher pitch
        
        speaker2 = np.zeros_like(speaker1)
        for i in range(0, int(duration), 3):  # Overlapping segments
            for freq in speaker2_freqs:
                segment_duration = min(1.5, duration - i)
                segment = self.generate_sine_wave(freq, segment_duration, 0.4)
                start_idx = i * self.sample_rate
                end_idx = start_idx + len(segment)
                if end_idx <= len(speaker2):
                    speaker2[start_idx:end_idx] += segment
        
        # Combine with partial overlap
        combined = np.zeros_like(speaker1)
        mid_point = len(combined) // 2
        combined[:mid_point] += speaker1[:mid_point]
        combined[mid_point//2:] += speaker2[mid_point//2:]
        
        return np.clip(combined, -1.0, 1.0).astype(np.float32)


class StressTestSuite:
    """Comprehensive stress testing for Emergency AI system."""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config_manager().config
        self.audio_generator = AudioTestGenerator()
        self.test_results: List[TestResult] = []
    
    def run_quick_validation(self) -> bool:
        """Run quick validation tests for Phase 3 verification."""
        try:
            # Test basic functionality
            test_audio = self.audio_generator.generate_test_tone(duration=1.0)
            temp_file = self.audio_generator.create_temp_file(test_audio)
            
            result = process_audio_file(temp_file, fast_mode=True)
            
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
            # Validate result structure
            required_keys = ['transcript', 'confidence', 'distress_score']
            return all(key in result for key in required_keys)
            
        except Exception as e:
            self.logger.error(f"Quick validation failed: {e}")
            return False
    
    def run_all_stress_tests(self) -> BenchmarkResult:
        """Run complete stress test suite."""
        self.logger.info("Starting comprehensive stress test suite")
        start_time = time.time()
        
        test_methods = [
            self.test_long_audio_processing,
            self.test_noisy_audio_handling,
            self.test_whisper_detection,
            self.test_loud_audio_clipping,
            self.test_rapid_speech_changes,
            self.test_overlapping_speech,
            self.test_silent_audio_handling,
            self.test_memory_intensive_processing,
            self.test_concurrent_processing,
            self.test_edge_case_durations,
            self.test_phase3_integration
        ]
        
        for test_method in test_methods:
            try:
                with track_operation("stress_test", test_method.__name__):
                    test_method()
            except Exception as e:
                self.logger.error(f"Stress test {test_method.__name__} failed: {e}")
                self.test_results.append(TestResult(
                    test_name=test_method.__name__,
                    success=False,
                    duration_ms=0.0,
                    confidence=None,
                    distress_score=None,
                    transcript="",
                    error_message=str(e)
                ))
        
        total_time = (time.time() - start_time) * 1000
        passed = sum(1 for r in self.test_results if r.success)
        
        # Calculate averages
        successful_results = [r for r in self.test_results if r.success]
        avg_time = sum(r.duration_ms for r in successful_results) / max(len(successful_results), 1)
        avg_confidence = sum(r.confidence for r in successful_results if r.confidence) / max(
            len([r for r in successful_results if r.confidence]), 1)
        
        benchmark = BenchmarkResult(
            test_suite="stress_tests",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(self.test_results),
            passed=passed,
            failed=len(self.test_results) - passed,
            avg_processing_time_ms=avg_time,
            avg_confidence=avg_confidence,
            avg_distress_accuracy=0.85,  # Placeholder - would need labeled data
            performance_regression=False,  # Would compare against baseline
            accuracy_regression=False,
            results=self.test_results
        )
        
        self.logger.info(f"Stress test suite completed: {passed}/{len(self.test_results)} passed")
        return benchmark
    
    def test_long_audio_processing(self):
        """Test processing of very long audio files."""
        self.logger.info("Testing long audio processing (10+ minutes)")
        
        # Generate 12-minute audio file
        long_audio = self.audio_generator.generate_mixed_audio(duration=720.0)  # 12 minutes
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, long_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="long_audio_processing",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'audio_duration': 720.0}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_noisy_audio_handling(self):
        """Test processing of very noisy audio."""
        self.logger.info("Testing noisy audio handling")
        
        # Generate audio with high noise levels
        noisy_audio = self.audio_generator.generate_mixed_audio(
            speech_amplitude=0.3, noise_amplitude=0.8, duration=30.0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, noisy_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="noisy_audio_handling",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'noise_level': 'high'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_whisper_detection(self):
        """Test detection of very quiet speech."""
        self.logger.info("Testing whisper detection")
        
        whisper_audio = self.audio_generator.generate_whisper_audio(duration=15.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, whisper_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="whisper_detection",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'volume_level': 'whisper'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_loud_audio_clipping(self):
        """Test handling of clipped/loud audio."""
        self.logger.info("Testing loud audio clipping handling")
        
        loud_audio = self.audio_generator.generate_loud_audio(duration=20.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, loud_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="loud_audio_clipping",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'volume_level': 'loud'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_rapid_speech_changes(self):
        """Test handling of rapidly changing speech patterns."""
        self.logger.info("Testing rapid speech changes")
        
        rapid_audio = self.audio_generator.generate_rapid_speech(duration=25.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, rapid_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="rapid_speech_changes",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'speech_type': 'rapid'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_overlapping_speech(self):
        """Test handling of overlapping speech from multiple speakers."""
        self.logger.info("Testing overlapping speech handling")
        
        overlap_audio = self.audio_generator.generate_overlapping_speech(duration=30.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, overlap_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="overlapping_speech",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'speech_type': 'overlapping'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_silent_audio_handling(self):
        """Test handling of completely silent audio."""
        self.logger.info("Testing silent audio handling")
        
        silent_audio = self.audio_generator.generate_silent_audio(duration=10.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, silent_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="silent_audio_handling",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'audio_type': 'silent'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_memory_intensive_processing(self):
        """Test memory usage with large audio files."""
        self.logger.info("Testing memory-intensive processing")
        
        # Generate large audio file (30 minutes)
        large_audio = self.audio_generator.generate_mixed_audio(duration=1800.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, large_audio, self.audio_generator.sample_rate)
            temp_path = tmp.name
        
        try:
            start_time = time.time()
            result = process_audio_file(temp_path)
            duration_ms = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="memory_intensive_processing",
                success=True,
                duration_ms=duration_ms,
                confidence=result.get('confidence', 0.0),
                distress_score=result.get('distress_score', 0.0),
                transcript=result.get('transcript', ''),
                metadata={'audio_duration': 1800.0, 'test_type': 'memory_intensive'}
            ))
            
        finally:
            os.unlink(temp_path)
    
    def test_concurrent_processing(self):
        """Test concurrent processing of multiple audio files."""
        self.logger.info("Testing concurrent processing")
        
        # Generate multiple test files
        test_files = []
        for i in range(5):
            audio = self.audio_generator.generate_mixed_audio(duration=30.0)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, self.audio_generator.sample_rate)
                test_files.append(tmp.name)
        
        results = []
        errors = []
        
        def process_file(file_path):
            try:
                start_time = time.time()
                result = process_audio_file(file_path)
                duration_ms = (time.time() - start_time) * 1000
                results.append((True, duration_ms, result))
            except Exception as e:
                errors.append(str(e))
                results.append((False, 0.0, {}))
        
        # Process files concurrently
        threads = []
        start_time = time.time()
        
        for file_path in test_files:
            thread = threading.Thread(target=process_file, args=(file_path,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Clean up files
        for file_path in test_files:
            os.unlink(file_path)
        
        # Record results
        success = len(errors) == 0
        avg_confidence = sum(r[2].get('confidence', 0.0) for r in results if r[0]) / max(
            len([r for r in results if r[0]]), 1)
        
        self.test_results.append(TestResult(
            test_name="concurrent_processing",
            success=success,
            duration_ms=total_duration_ms,
            confidence=avg_confidence,
            distress_score=0.0,
            transcript="",
            metadata={
                'concurrent_files': len(test_files),
                'errors': len(errors),
                'error_messages': errors[:3]  # First 3 errors
            }
        ))
    
    def test_edge_case_durations(self):
        """Test edge cases with very short and specific durations."""
        self.logger.info("Testing edge case durations")
        
        edge_durations = [0.1, 0.5, 1.0, 2.5, 3.7, 7.3, 13.1]  # Various odd durations
        
        for duration in edge_durations:
            audio = self.audio_generator.generate_mixed_audio(duration=duration)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, self.audio_generator.sample_rate)
                temp_path = tmp.name
            
            try:
                start_time = time.time()
                result = process_audio_file(temp_path)
                processing_time = (time.time() - start_time) * 1000
                
                self.test_results.append(TestResult(
                    test_name=f"edge_duration_{duration}s",
                    success=True,
                    duration_ms=processing_time,
                    confidence=result.get('confidence', 0.0),
                    distress_score=result.get('distress_score', 0.0),
                    transcript=result.get('transcript', ''),
                    metadata={'audio_duration': duration}
                ))
                
            except Exception as e:
                self.test_results.append(TestResult(
                    test_name=f"edge_duration_{duration}s",
                    success=False,
                    duration_ms=0.0,
                    confidence=None,
                    distress_score=None,
                    transcript="",
                    error_message=str(e),
                    metadata={'audio_duration': duration}
                ))
            
            finally:
                os.unlink(temp_path)
    
    def test_phase3_integration(self):
        """Test integration of all Phase 3 components."""
        self.logger.info("Testing Phase 3 component integration")
        start_time = time.time()
        
        try:
            # Test 1: Logging integration
            try:
                from modules.enhanced_logger import get_logger, track_operation, PerformanceMetrics
                logger = get_logger()
                with track_operation("phase3_integration_test") as tracker:
                    tracker.add_metric("test_metric", 100)
                integration_score = 25  # 25% for logging
            except Exception as e:
                self.logger.error(f"Logging integration failed: {e}")
                integration_score = 0
            
            # Test 2: Visualization components
            try:
                from modules.visualization_dashboard import DeveloperDashboard
                dashboard = DeveloperDashboard()
                if hasattr(dashboard, 'render_dashboard'):
                    integration_score += 25  # 25% for visualization
            except Exception as e:
                self.logger.error(f"Visualization integration failed: {e}")
            
            # Test 3: CLI/GUI tools
            try:
                from cli import main as cli_main
                from gui import EmergencyAIGUI
                if callable(cli_main) and EmergencyAIGUI:
                    integration_score += 25  # 25% for tools
            except Exception as e:
                self.logger.error(f"Tools integration failed: {e}")
            
            # Test 4: Complete pipeline with enhanced features
            try:
                test_audio = self.audio_generator.generate_test_tone(duration=2.0)
                temp_file = self.audio_generator.create_temp_file(test_audio)
                
                # Process with enhanced pipeline
                result = process_audio_file(temp_file)
                
                if all(key in result for key in ['transcript', 'confidence', 'distress_score']):
                    integration_score += 25  # 25% for pipeline
                
                os.unlink(temp_file)
                
            except Exception as e:
                self.logger.error(f"Pipeline integration failed: {e}")
            
            # Record result
            duration_ms = (time.time() - start_time) * 1000
            success = integration_score >= 75  # Require 75% success rate
            
            self.test_results.append(TestResult(
                test_name="phase3_integration", 
                success=success,
                duration_ms=duration_ms,
                confidence=float(integration_score / 100),
                distress_score=0.0,
                transcript=f"Integration score: {integration_score}%",
                error_message=None if success else f"Integration score too low: {integration_score}%",
                metadata={'integration_score': integration_score}
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="phase3_integration",
                success=False, 
                duration_ms=duration_ms,
                confidence=None,
                distress_score=None,
                transcript="",
                error_message=str(e),
                metadata={'error_type': type(e).__name__}
            ))


class RegressionTestRunner:
    """Automated regression testing for performance and accuracy."""
    
    def __init__(self, baseline_file: str = "tests/regression_baseline.json"):
        self.baseline_file = baseline_file
        self.logger = get_logger()
        self.stress_tester = StressTestSuite()
    
    def run_regression_tests(self) -> BenchmarkResult:
        """Run full regression test suite."""
        self.logger.info("Starting regression test suite")
        
        # Run stress tests
        benchmark = self.stress_tester.run_all_stress_tests()
        
        # Load baseline if exists
        baseline = self.load_baseline()
        
        if baseline:
            # Check for regressions
            benchmark.performance_regression = self.check_performance_regression(benchmark, baseline)
            benchmark.accuracy_regression = self.check_accuracy_regression(benchmark, baseline)
        
        # Save new baseline
        self.save_baseline(benchmark)
        
        return benchmark
    
    def load_baseline(self) -> Optional[BenchmarkResult]:
        """Load regression baseline from file."""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    return BenchmarkResult(**data)
        except Exception as e:
            self.logger.warning(f"Could not load baseline: {e}")
        return None
    
    def save_baseline(self, benchmark: BenchmarkResult):
        """Save current results as new baseline."""
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
            
            # Convert to dict for JSON serialization  
            data = {
                'test_suite': benchmark.test_suite,
                'timestamp': benchmark.timestamp,
                'total_tests': benchmark.total_tests,
                'passed': benchmark.passed,
                'failed': benchmark.failed,
                'avg_processing_time_ms': benchmark.avg_processing_time_ms,
                'avg_confidence': benchmark.avg_confidence,
                'avg_distress_accuracy': benchmark.avg_distress_accuracy,
                'performance_regression': benchmark.performance_regression,
                'accuracy_regression': benchmark.accuracy_regression
            }
            
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save baseline: {e}")
    
    def check_performance_regression(self, current: BenchmarkResult, 
                                   baseline: BenchmarkResult) -> bool:
        """Check if performance has regressed."""
        # Allow 20% performance degradation before flagging regression
        threshold = 1.2
        return current.avg_processing_time_ms > (baseline.avg_processing_time_ms * threshold)
    
    def check_accuracy_regression(self, current: BenchmarkResult, 
                                baseline: BenchmarkResult) -> bool:
        """Check if accuracy has regressed."""
        # Allow 5% accuracy degradation before flagging regression
        threshold = 0.05
        return current.avg_confidence < (baseline.avg_confidence - threshold)


def run_stress_tests():
    """CLI entry point for stress testing."""
    suite = StressTestSuite()
    results = suite.run_all_stress_tests()
    
    print("\n" + "="*60)
    print("STRESS TEST RESULTS")
    print("="*60)
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed}")
    print(f"Failed: {results.failed}")
    print(f"Success Rate: {results.passed/results.total_tests*100:.1f}%")
    print(f"Average Processing Time: {results.avg_processing_time_ms:.1f}ms")
    print(f"Average Confidence: {results.avg_confidence:.3f}")
    
    if results.failed > 0:
        print("\nFailed Tests:")
        failed_tests = [r for r in results.results if not r.success]
        for test in failed_tests:
            print(f"  - {test.test_name}: {test.error_message}")
    
    return results


def run_regression_tests():
    """CLI entry point for regression testing."""
    runner = RegressionTestRunner()
    results = runner.run_regression_tests()
    
    print("\n" + "="*60)
    print("REGRESSION TEST RESULTS")
    print("="*60)
    print(f"Performance Regression: {'YES' if results.performance_regression else 'NO'}")
    print(f"Accuracy Regression: {'YES' if results.accuracy_regression else 'NO'}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency AI Testing Framework")
    parser.add_argument('--stress', action='store_true', help='Run stress tests')
    parser.add_argument('--regression', action='store_true', help='Run regression tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.stress or args.all:
        run_stress_tests()
    
    if args.regression or args.all:
        run_regression_tests()
    
    if not any([args.stress, args.regression, args.all]):
        parser.print_help()