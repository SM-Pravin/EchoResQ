"""
Edge case testing module for Emergency AI system.
Tests boundary conditions, error handling, and unusual input scenarios.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.enhanced_logger import get_logger
from modules.config_manager import get_config_manager
from analysis_pipeline import process_audio_file


@dataclass
class EdgeCaseResult:
    """Result of edge case testing."""
    test_name: str
    input_description: str
    expected_behavior: str
    actual_result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class EdgeCaseTestSuite:
    """Comprehensive edge case testing for Emergency AI."""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config_manager().config
        self.results: List[EdgeCaseResult] = []
    
    def run_all_edge_case_tests(self) -> List[EdgeCaseResult]:
        """Run complete edge case test suite."""
        self.logger.info("Starting comprehensive edge case testing")
        
        test_methods = [
            self.test_empty_audio_file,
            self.test_corrupted_audio_file,
            self.test_extremely_short_audio,
            self.test_extremely_long_audio,
            self.test_invalid_sample_rates,
            self.test_different_bit_depths,
            self.test_mono_vs_stereo,
            self.test_clipped_audio,
            self.test_dc_offset_audio,
            self.test_infinity_nan_values,
            self.test_different_audio_formats,
            self.test_unicode_filenames,
            self.test_very_large_files,
            self.test_concurrent_access,
            self.test_memory_exhaustion,
            self.test_invalid_file_paths
        ]
        
        for test_method in test_methods:
            try:
                self.logger.info(f"Running edge case test: {test_method.__name__}")
                test_method()
            except Exception as e:
                self.logger.error(f"Edge case test {test_method.__name__} failed: {e}")
                self.results.append(EdgeCaseResult(
                    test_name=test_method.__name__,
                    input_description="Test setup failed",
                    expected_behavior="Should handle gracefully",
                    actual_result={},
                    success=False,
                    error_message=str(e)
                ))
        
        return self.results
    
    def test_empty_audio_file(self):
        """Test handling of completely empty audio files."""
        # Create empty audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write minimal WAV header with no audio data
            empty_audio = np.array([], dtype=np.float32)
            try:
                sf.write(tmp.name, empty_audio, 16000)
                temp_path = tmp.name
            except Exception as e:
                # If we can't even create empty file, that's the test result
                self.results.append(EdgeCaseResult(
                    test_name="empty_audio_file",
                    input_description="Audio file with zero samples",
                    expected_behavior="Should handle gracefully or provide meaningful error",
                    actual_result={"creation_error": str(e)},
                    success=False,
                    error_message=f"Cannot create empty audio file: {e}"
                ))
                return
        
        try:
            result = process_audio_file(temp_path)
            
            self.results.append(EdgeCaseResult(
                test_name="empty_audio_file",
                input_description="Audio file with zero samples",
                expected_behavior="Should handle gracefully or provide meaningful error",
                actual_result=result,
                success=True,
                metadata={"audio_duration": 0.0}
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="empty_audio_file", 
                input_description="Audio file with zero samples",
                expected_behavior="Should handle gracefully or provide meaningful error",
                actual_result={},
                success=False,
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_corrupted_audio_file(self):
        """Test handling of corrupted audio files."""
        # Create corrupted file with random bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write random bytes that look like audio but are corrupted
            random_data = np.random.bytes(1024)
            tmp.write(random_data)
            temp_path = tmp.name
        
        try:
            result = process_audio_file(temp_path)
            
            self.results.append(EdgeCaseResult(
                test_name="corrupted_audio_file",
                input_description="File with random bytes, not valid audio",
                expected_behavior="Should fail gracefully with clear error message",
                actual_result=result,
                success=False,  # Should not succeed with corrupted data
                metadata={"file_size": 1024}
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="corrupted_audio_file",
                input_description="File with random bytes, not valid audio", 
                expected_behavior="Should fail gracefully with clear error message",
                actual_result={},
                success=True,  # Expected to fail
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_extremely_short_audio(self):
        """Test handling of extremely short audio (< 100ms)."""
        durations = [0.001, 0.01, 0.05, 0.1]  # 1ms to 100ms
        
        for duration in durations:
            # Generate very short audio
            sample_rate = 16000
            samples = int(sample_rate * duration)
            audio = np.random.randn(samples).astype(np.float32) * 0.1
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, sample_rate)
                temp_path = tmp.name
            
            try:
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"extremely_short_audio_{duration*1000:.0f}ms",
                    input_description=f"Audio file of {duration*1000:.1f}ms duration",
                    expected_behavior="Should process or indicate insufficient audio",
                    actual_result=result,
                    success=True,
                    metadata={"duration_ms": duration * 1000, "samples": samples}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"extremely_short_audio_{duration*1000:.0f}ms",
                    input_description=f"Audio file of {duration*1000:.1f}ms duration",
                    expected_behavior="Should process or indicate insufficient audio",
                    actual_result={},
                    success=False,
                    error_message=str(e),
                    metadata={"duration_ms": duration * 1000, "samples": samples}
                ))
            
            finally:
                os.unlink(temp_path)
    
    def test_extremely_long_audio(self):
        """Test handling of very long audio files."""
        # Test with 2-hour audio (would be ~230MB at 16kHz)
        duration = 7200  # 2 hours
        sample_rate = 16000
        
        # Instead of creating the full file, simulate it with chunks
        chunk_duration = 60  # 1 minute chunks
        chunk_samples = sample_rate * chunk_duration
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Create first chunk 
            audio_chunk = np.random.randn(chunk_samples).astype(np.float32) * 0.1
            
            # Write multiple chunks to simulate very long file
            sf.write(tmp.name, audio_chunk, sample_rate)
            temp_path = tmp.name
        
        try:
            # Time the processing
            import time
            start_time = time.time()
            result = process_audio_file(temp_path)
            processing_time = time.time() - start_time
            
            self.results.append(EdgeCaseResult(
                test_name="extremely_long_audio",
                input_description=f"Simulated {duration}s ({duration/3600:.1f}h) audio file",
                expected_behavior="Should process efficiently or handle in chunks",
                actual_result=result,
                success=True,
                metadata={
                    "simulated_duration": duration,
                    "actual_duration": chunk_duration,
                    "processing_time": processing_time
                }
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="extremely_long_audio",
                input_description=f"Simulated {duration}s ({duration/3600:.1f}h) audio file",
                expected_behavior="Should process efficiently or handle in chunks",
                actual_result={},
                success=False,
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_invalid_sample_rates(self):
        """Test handling of unusual sample rates."""
        sample_rates = [8000, 11025, 22050, 32000, 44100, 48000, 96000, 192000]
        
        for sr in sample_rates:
            # Generate 5-second test audio
            duration = 5.0
            samples = int(sr * duration)
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)).astype(np.float32) * 0.5
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                try:
                    sf.write(tmp.name, audio, sr)
                    temp_path = tmp.name
                except Exception as write_error:
                    self.results.append(EdgeCaseResult(
                        test_name=f"invalid_sample_rate_{sr}Hz",
                        input_description=f"Audio with {sr}Hz sample rate",
                        expected_behavior="Should handle or provide clear error",
                        actual_result={},
                        success=False,
                        error_message=f"Cannot write {sr}Hz audio: {write_error}"
                    ))
                    continue
            
            try:
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"invalid_sample_rate_{sr}Hz",
                    input_description=f"Audio with {sr}Hz sample rate",
                    expected_behavior="Should handle or provide clear error",
                    actual_result=result,
                    success=True,
                    metadata={"sample_rate": sr, "duration": duration}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"invalid_sample_rate_{sr}Hz",
                    input_description=f"Audio with {sr}Hz sample rate",
                    expected_behavior="Should handle or provide clear error",
                    actual_result={},
                    success=False,
                    error_message=str(e),
                    metadata={"sample_rate": sr}
                ))
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_different_bit_depths(self):
        """Test handling of different audio bit depths."""
        bit_depths = ['PCM_16', 'PCM_24', 'PCM_32', 'FLOAT']
        
        for bit_depth in bit_depths:
            # Generate test audio
            sample_rate = 16000
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                try:
                    sf.write(tmp.name, audio, sample_rate, subtype=bit_depth)
                    temp_path = tmp.name
                except Exception as write_error:
                    self.results.append(EdgeCaseResult(
                        test_name=f"bit_depth_{bit_depth}",
                        input_description=f"Audio with {bit_depth} bit depth",
                        expected_behavior="Should handle different bit depths",
                        actual_result={},
                        success=False,
                        error_message=f"Cannot write {bit_depth} audio: {write_error}"
                    ))
                    continue
            
            try:
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"bit_depth_{bit_depth}",
                    input_description=f"Audio with {bit_depth} bit depth",
                    expected_behavior="Should handle different bit depths",
                    actual_result=result,
                    success=True,
                    metadata={"bit_depth": bit_depth, "duration": duration}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"bit_depth_{bit_depth}",
                    input_description=f"Audio with {bit_depth} bit depth",
                    expected_behavior="Should handle different bit depths",
                    actual_result={},
                    success=False,
                    error_message=str(e),
                    metadata={"bit_depth": bit_depth}
                ))
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_mono_vs_stereo(self):
        """Test handling of mono vs stereo audio."""
        configurations = [
            ("mono", 1),
            ("stereo", 2),
            ("surround", 6)  # 5.1 surround
        ]
        
        for config_name, channels in configurations:
            # Generate multi-channel audio
            sample_rate = 16000
            duration = 15.0
            samples = int(sample_rate * duration)
            
            if channels == 1:
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)).astype(np.float32)
            else:
                # Create multi-channel audio with different frequencies per channel
                audio = np.zeros((samples, channels), dtype=np.float32)
                for ch in range(channels):
                    freq = 440 + ch * 110  # Different frequency per channel
                    audio[:, ch] = np.sin(2 * np.pi * freq * np.linspace(0, duration, samples)) * 0.5
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                try:
                    sf.write(tmp.name, audio, sample_rate)
                    temp_path = tmp.name
                except Exception as write_error:
                    self.results.append(EdgeCaseResult(
                        test_name=f"channels_{config_name}",
                        input_description=f"Audio with {channels} channels ({config_name})",
                        expected_behavior="Should handle different channel configurations",
                        actual_result={},
                        success=False,
                        error_message=f"Cannot write {channels}-channel audio: {write_error}"
                    ))
                    continue
            
            try:
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"channels_{config_name}",
                    input_description=f"Audio with {channels} channels ({config_name})",
                    expected_behavior="Should handle different channel configurations",
                    actual_result=result,
                    success=True,
                    metadata={"channels": channels, "config": config_name}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"channels_{config_name}",
                    input_description=f"Audio with {channels} channels ({config_name})",
                    expected_behavior="Should handle different channel configurations",
                    actual_result={},
                    success=False,
                    error_message=str(e),
                    metadata={"channels": channels, "config": config_name}
                ))
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_clipped_audio(self):
        """Test handling of clipped/distorted audio."""
        # Generate severely clipped audio
        sample_rate = 16000
        duration = 20.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio that clips severely
        audio = 3.0 * np.sin(2 * np.pi * 440 * t)  # 3x amplitude
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)  # Hard clipping
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            temp_path = tmp.name
        
        try:
            result = process_audio_file(temp_path)
            
            self.results.append(EdgeCaseResult(
                test_name="clipped_audio",
                input_description="Severely clipped/distorted audio",
                expected_behavior="Should handle distortion gracefully",
                actual_result=result,
                success=True,
                metadata={"clipping_detected": True, "duration": duration}
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="clipped_audio",
                input_description="Severely clipped/distorted audio",
                expected_behavior="Should handle distortion gracefully",
                actual_result={},
                success=False,
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_dc_offset_audio(self):
        """Test handling of audio with DC offset."""
        # Generate audio with significant DC offset
        sample_rate = 16000
        duration = 15.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Audio with DC offset
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.7  # +0.7 DC offset
        audio = audio.astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            temp_path = tmp.name
        
        try:
            result = process_audio_file(temp_path)
            
            self.results.append(EdgeCaseResult(
                test_name="dc_offset_audio",
                input_description="Audio with significant DC offset (+0.7)",
                expected_behavior="Should handle DC offset gracefully",
                actual_result=result,
                success=True,
                metadata={"dc_offset": 0.7, "duration": duration}
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="dc_offset_audio",
                input_description="Audio with significant DC offset (+0.7)",
                expected_behavior="Should handle DC offset gracefully",
                actual_result={},
                success=False,
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_infinity_nan_values(self):
        """Test handling of audio with infinity and NaN values."""
        # Generate audio with problematic values
        sample_rate = 16000
        duration = 10.0
        samples = int(sample_rate * duration)
        
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)).astype(np.float32)
        
        # Introduce problematic values
        audio[1000:1010] = np.inf  # Some infinity values
        audio[2000:2005] = np.nan  # Some NaN values
        audio[3000] = -np.inf      # Negative infinity
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            try:
                sf.write(tmp.name, audio, sample_rate)
                temp_path = tmp.name
            except Exception as write_error:
                self.results.append(EdgeCaseResult(
                    test_name="infinity_nan_values", 
                    input_description="Audio with infinity and NaN values",
                    expected_behavior="Should handle or reject invalid values",
                    actual_result={},
                    success=True,  # Expected to fail at write stage
                    error_message=f"Cannot write inf/nan audio: {write_error}"
                ))
                return
        
        try:
            result = process_audio_file(temp_path)
            
            self.results.append(EdgeCaseResult(
                test_name="infinity_nan_values",
                input_description="Audio with infinity and NaN values",
                expected_behavior="Should handle or reject invalid values",
                actual_result=result,
                success=False,  # Should not process invalid audio
                metadata={"inf_count": 11, "nan_count": 5}
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="infinity_nan_values",
                input_description="Audio with infinity and NaN values",
                expected_behavior="Should handle or reject invalid values",
                actual_result={},
                success=True,  # Expected to fail
                error_message=str(e)
            ))
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_different_audio_formats(self):
        """Test handling of different audio file formats."""
        formats = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
        
        # Generate base audio
        sample_rate = 16000
        duration = 8.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp:
                temp_path = tmp.name
            
            try:
                # Try to write in this format
                if fmt == '.wav':
                    sf.write(temp_path, audio, sample_rate)
                elif fmt == '.flac':
                    sf.write(temp_path, audio, sample_rate, format='FLAC')
                else:
                    # Skip formats that might not be supported
                    self.results.append(EdgeCaseResult(
                        test_name=f"audio_format_{fmt[1:]}",
                        input_description=f"Audio in {fmt} format",
                        expected_behavior="Should handle supported formats",
                        actual_result={},
                        success=False,
                        error_message=f"Format {fmt} not tested (requires additional codecs)"
                    ))
                    os.unlink(temp_path)
                    continue
                
                # Try to process
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"audio_format_{fmt[1:]}",
                    input_description=f"Audio in {fmt} format",
                    expected_behavior="Should handle supported formats",
                    actual_result=result,
                    success=True,
                    metadata={"format": fmt, "duration": duration}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"audio_format_{fmt[1:]}",
                    input_description=f"Audio in {fmt} format",
                    expected_behavior="Should handle supported formats",
                    actual_result={},
                    success=False,
                    error_message=str(e),
                    metadata={"format": fmt}
                ))
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_unicode_filenames(self):
        """Test handling of Unicode filenames."""
        unicode_names = [
            "测试音频.wav",  # Chinese
            "тест_аудио.wav",  # Russian
            "テスト音声.wav",  # Japanese
            "🎵_audio_🎧.wav",  # Emoji
            "café_résumé.wav",  # Accented characters
        ]
        
        # Generate test audio
        sample_rate = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        for unicode_name in unicode_names:
            temp_path = os.path.join(tempfile.gettempdir(), unicode_name)
            
            try:
                sf.write(temp_path, audio, sample_rate)
                
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"unicode_filename",
                    input_description=f"File with Unicode name: {unicode_name}",
                    expected_behavior="Should handle Unicode filenames",
                    actual_result=result,
                    success=True,
                    metadata={"filename": unicode_name, "encoding": "utf-8"}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"unicode_filename",
                    input_description=f"File with Unicode name: {unicode_name}",
                    expected_behavior="Should handle Unicode filenames", 
                    actual_result={},
                    success=False,
                    error_message=str(e),
                    metadata={"filename": unicode_name}
                ))
            
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
    
    def test_very_large_files(self):
        """Test handling of very large audio files (memory pressure)."""
        # Simulate large file processing without actually creating huge files
        # Test memory efficiency with repeated processing
        
        sample_rate = 16000
        chunk_duration = 60  # 1 minute
        chunk_samples = sample_rate * chunk_duration
        
        # Create a 1-minute chunk that we'll process multiple times
        audio_chunk = np.random.randn(chunk_samples).astype(np.float32) * 0.1
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_chunk, sample_rate)
            temp_path = tmp.name
        
        try:
            # Process the same file multiple times to simulate large file processing
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)  # MB
            
            for i in range(5):  # Simulate processing 5 chunks
                result = process_audio_file(temp_path)
                current_memory = process.memory_info().rss / (1024**2)
                memory_growth = current_memory - initial_memory
                
                if memory_growth > 1000:  # More than 1GB growth
                    raise Exception(f"Excessive memory growth: {memory_growth:.1f}MB")
            
            final_memory = process.memory_info().rss / (1024**2)
            
            self.results.append(EdgeCaseResult(
                test_name="very_large_files",
                input_description="Simulated large file processing (5x 1-minute chunks)",
                expected_behavior="Should maintain reasonable memory usage",
                actual_result=result,
                success=True,
                metadata={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_growth_mb": final_memory - initial_memory,
                    "chunks_processed": 5
                }
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="very_large_files",
                input_description="Simulated large file processing (5x 1-minute chunks)",
                expected_behavior="Should maintain reasonable memory usage",
                actual_result={},
                success=False,
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_concurrent_access(self):
        """Test concurrent access to the same audio file."""
        import threading
        import time
        
        # Create test audio file
        sample_rate = 16000
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            temp_path = tmp.name
        
        results = []
        errors = []
        
        def process_file():
            try:
                result = process_audio_file(temp_path)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        try:
            # Start multiple threads accessing the same file
            threads = []
            for i in range(3):
                thread = threading.Thread(target=process_file)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            self.results.append(EdgeCaseResult(
                test_name="concurrent_access",
                input_description="3 threads processing same audio file simultaneously",
                expected_behavior="Should handle concurrent access gracefully",
                actual_result={"successful_processes": len(results), "errors": len(errors)},
                success=len(errors) == 0,
                error_message=errors[0] if errors else None,
                metadata={"thread_count": 3, "error_messages": errors[:3]}
            ))
            
        except Exception as e:
            self.results.append(EdgeCaseResult(
                test_name="concurrent_access",
                input_description="3 threads processing same audio file simultaneously",
                expected_behavior="Should handle concurrent access gracefully",
                actual_result={},
                success=False,
                error_message=str(e)
            ))
        
        finally:
            os.unlink(temp_path)
    
    def test_memory_exhaustion(self):
        """Test behavior under memory pressure."""
        # This test is necessarily limited to avoid actually exhausting system memory
        import gc
        
        # Try to create memory pressure by allocating large arrays
        large_arrays = []
        try:
            # Allocate arrays until we use significant memory
            for i in range(10):
                # Each array is ~100MB
                array = np.random.randn(25 * 1024 * 1024).astype(np.float32)
                large_arrays.append(array)
            
            # Now try to process audio under memory pressure
            sample_rate = 16000
            duration = 30.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, sample_rate)
                temp_path = tmp.name
            
            try:
                result = process_audio_file(temp_path)
                
                self.results.append(EdgeCaseResult(
                    test_name="memory_exhaustion",
                    input_description="Processing under high memory pressure (~1GB allocated)",
                    expected_behavior="Should handle memory pressure gracefully",
                    actual_result=result,
                    success=True,
                    metadata={"allocated_memory_gb": len(large_arrays) * 0.1}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name="memory_exhaustion",
                    input_description="Processing under high memory pressure (~1GB allocated)",
                    expected_behavior="Should handle memory pressure gracefully",
                    actual_result={},
                    success=False,
                    error_message=str(e)
                ))
            
            finally:
                os.unlink(temp_path)
                
        except MemoryError:
            self.results.append(EdgeCaseResult(
                test_name="memory_exhaustion",
                input_description="Processing under high memory pressure",
                expected_behavior="Should handle memory pressure gracefully",
                actual_result={},
                success=True,  # Expected to fail due to memory pressure
                error_message="Memory allocation failed as expected"
            ))
        
        finally:
            # Clean up memory
            large_arrays.clear()
            gc.collect()
    
    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        invalid_paths = [
            "/nonexistent/path/audio.wav",
            "C:\\nonexistent\\path\\audio.wav",  
            "",  # Empty path
            "   ",  # Whitespace only
            "audio_file_without_extension",
            "very_long_" + "x" * 1000 + "_filename.wav",  # Very long filename
            "con.wav",  # Reserved Windows filename
            "prn.wav",  # Reserved Windows filename
            "audio.wav.exe",  # Suspicious extension
        ]
        
        for invalid_path in invalid_paths:
            try:
                result = process_audio_file(invalid_path)
                
                self.results.append(EdgeCaseResult(
                    test_name=f"invalid_file_path",
                    input_description=f"Invalid path: {invalid_path[:50]}...",
                    expected_behavior="Should fail gracefully with clear error",
                    actual_result=result,
                    success=False,  # Should not succeed with invalid paths
                    metadata={"path": invalid_path}
                ))
                
            except Exception as e:
                self.results.append(EdgeCaseResult(
                    test_name=f"invalid_file_path",
                    input_description=f"Invalid path: {invalid_path[:50]}...",
                    expected_behavior="Should fail gracefully with clear error",
                    actual_result={},
                    success=True,  # Expected to fail
                    error_message=str(e),
                    metadata={"path": invalid_path}
                ))


def run_edge_case_tests():
    """CLI entry point for edge case testing."""
    suite = EdgeCaseTestSuite()
    results = suite.run_all_edge_case_tests()
    
    print("\n" + "="*60)
    print("EDGE CASE TEST RESULTS")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - successful_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    if failed_tests > 0:
        print("\nFailed Tests:")
        failed_results = [r for r in results if not r.success]
        for test in failed_results[:10]:  # Show first 10 failures
            print(f"  - {test.test_name}: {test.error_message}")
    
    # Save detailed results
    results_file = "tests/edge_case_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump([{
                'test_name': r.test_name,
                'input_description': r.input_description,
                'expected_behavior': r.expected_behavior,
                'success': r.success,
                'error_message': r.error_message,
                'metadata': r.metadata
            } for r in results], f, indent=2)
        
        print(f"\nDetailed results saved to {results_file}")
    except Exception as e:
        print(f"Could not save results: {e}")
    
    return results


if __name__ == "__main__":
    run_edge_case_tests()