"""
In-memory audio processing system to eliminate disk I/O operations.
Provides AudioBuffer class and InMemoryAudioProcessor for complete in-memory pipeline.
"""

import os
import time
import gc
import tempfile
import threading
import uuid
import io
from typing import Dict, List, Optional, Any, Union, Tuple
from contextlib import contextmanager
import numpy as np
# soundfile is optional; provide a lightweight fallback for basic WAV I/O
try:
    import soundfile as sf
    SF_AVAILABLE = True
except Exception:
    SF_AVAILABLE = False
    print('[WARNING] soundfile not available. Falling back to scipy/wave reader for basic WAV support. Install with: pip install soundfile')
    try:
        from scipy.io import wavfile as _scipy_wavfile
        _SCIPY_AVAILABLE = True
    except Exception:
        _SCIPY_AVAILABLE = False

    import wave as _wave
    import contextlib as _contextlib

    def _sf_read(path_or_file):
        """Read WAV data from a path or file-like object. Returns (data, samplerate).

        This is a minimal fallback and supports only PCM WAV. For broader format
        support (FLAC/OGG/MP3) install `soundfile`.
        """
        # If it's a path string, try scipy first (if available)
        if isinstance(path_or_file, (str, bytes)):
            p = path_or_file
            if _SCIPY_AVAILABLE:
                sr, data = _scipy_wavfile.read(p)
                data = data.astype('float32')
                # Normalize integer PCM to float32 range
                if data.dtype.kind in ('i', 'u'):
                    maxv = float(2 ** (8 * data.dtype.itemsize - 1))
                    data = data.astype('float32') / maxv
                return data, int(sr)
            else:
                # Use wave module
                with _contextlib.closing(_wave.open(p, 'rb')) as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    import numpy as _np
                    # Interpret bytes according to sample width
                    sampwidth = wf.getsampwidth()
                    dtype = None
                    if sampwidth == 1:
                        dtype = _np.uint8
                    elif sampwidth == 2:
                        dtype = _np.int16
                    elif sampwidth == 4:
                        dtype = _np.int32
                    else:
                        # fallback: return raw bytes as float32 zeros
                        return _np.frombuffer(frames, dtype=_np.float32), sr
                    data = _np.frombuffer(frames, dtype=dtype).astype(_np.float32)
                    if sampwidth != 1:
                        maxv = float(2 ** (8 * sampwidth - 1))
                        data = data / maxv
                    return data, int(sr)

        # file-like object (BytesIO etc.)
        try:
            b = path_or_file.read()
        except Exception:
            raise ValueError('Unsupported file-like object for audio read')

        # Try scipy from buffer
        import io as _io
        bio = _io.BytesIO(b)
        if _SCIPY_AVAILABLE:
            sr, data = _scipy_wavfile.read(bio)
            data = data.astype('float32')
            if data.dtype.kind in ('i', 'u'):
                maxv = float(2 ** (8 * data.dtype.itemsize - 1))
                data = data.astype('float32') / maxv
            return data, int(sr)

        # wave from buffer
        with _contextlib.closing(_wave.open(bio, 'rb')) as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            import numpy as _np
            sampwidth = wf.getsampwidth()
            dtype = None
            if sampwidth == 1:
                dtype = _np.uint8
            elif sampwidth == 2:
                dtype = _np.int16
            elif sampwidth == 4:
                dtype = _np.int32
            else:
                return _np.frombuffer(frames, dtype=_np.float32), sr
            data = _np.frombuffer(frames, dtype=dtype).astype(_np.float32)
            if sampwidth != 1:
                maxv = float(2 ** (8 * sampwidth - 1))
                data = data / maxv
            return data, int(sr)

    # Provide a minimal write if other code expects sf.write (not extensively used)
    def _sf_write(path, data, samplerate):
        # Use scipy or wave
        import numpy as _np
        arr = _np.asarray(data)
        if _SCIPY_AVAILABLE:
            _scipy_wavfile.write(path, int(samplerate), arr)
            return
        # wave writer
        with _contextlib.closing(_wave.open(path, 'wb')) as wf:
            wf.setnchannels(1 if arr.ndim == 1 else arr.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(int(samplerate))
            # Convert to int16 PCM
            maxv = float(2 ** 15 - 1)
            intdata = (arr * maxv).astype(_np.int16)
            wf.writeframes(intdata.tobytes())

    # Minimal namespace replacement
    class _SoundFileFallback:
        read = staticmethod(_sf_read)
        write = staticmethod(_sf_write)

    sf = _SoundFileFallback()

# librosa is optional; provide a lightweight fallback using soundfile + numpy
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False
    print('[WARNING] librosa not available. Falling back to soundfile + numpy for basic audio I/O. Install with: pip install librosa')

    def _load_with_soundfile(path_or_file, sr=None, mono=True):
        # soundfile accepts path or file-like
        data, srate = sf.read(path_or_file)
        # Convert multi-channel to mono if requested
        if data.ndim > 1 and mono:
            data = np.mean(data, axis=1)
        # Resample if target sr provided
        if sr is not None and srate != sr and len(data) > 0:
            duration = len(data) / float(srate)
            new_len = int(max(1, round(duration * sr)))
            old_times = np.linspace(0, duration, num=len(data))
            new_times = np.linspace(0, duration, num=new_len)
            data = np.interp(new_times, old_times, data).astype(np.float32)
            srate = sr
        return data.astype(np.float32), srate

    def _to_mono(data):
        if data.ndim == 1:
            return data
        return np.mean(data, axis=1)

    def _resample(data, orig_sr, target_sr):
        if orig_sr == target_sr or len(data) == 0:
            return data
        duration = len(data) / float(orig_sr)
        new_len = int(max(1, round(duration * target_sr)))
        old_times = np.linspace(0, duration, num=len(data))
        new_times = np.linspace(0, duration, num=new_len)
        return np.interp(new_times, old_times, data).astype(np.float32)

    # Create a minimal librosa-like namespace so existing calls (librosa.load, .to_mono, .resample)
    # continue to work without changing remainder of the module.
    class _LibrosaFallback:
        load = staticmethod(_load_with_soundfile)
        to_mono = staticmethod(_to_mono)
        resample = staticmethod(_resample)

    librosa = _LibrosaFallback()

# Audio processing constants
DEFAULT_SR = 16000
VAD_AGGRESSIVENESS = 2
MAX_MEMORY_CACHE_MB = 512.0


class AudioBuffer:
    """In-memory audio buffer with metadata and processing capabilities."""
    
    def __init__(self, data: np.ndarray, sample_rate: int, start_time: float = 0.0, 
                 end_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        self.data = np.asarray(data, dtype=np.float32)
        self.sample_rate = int(sample_rate)
        self.start_time = float(start_time)
        self.end_time = end_time if end_time is not None else start_time + self.duration
        self.metadata = metadata or {}
        self._id = str(uuid.uuid4())[:8]
        
        # Register with memory manager
        AudioBufferManager.register_buffer(self)
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate if len(self.data) > 0 else 0.0
    
    @property
    def samples(self) -> int:
        """Number of samples."""
        return len(self.data)
    
    @property
    def memory_usage_mb(self) -> float:
        """Memory usage in MB."""
        return self.data.nbytes / (1024 * 1024)
    
    def get_chunk(self, start_time: float, end_time: float) -> 'AudioBuffer':
        """Extract a chunk of audio data."""
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Clamp to valid ranges
        start_sample = max(0, start_sample)
        end_sample = min(len(self.data), end_sample)
        
        if start_sample >= end_sample:
            return AudioBuffer(np.array([]), self.sample_rate, start_time)
        
        chunk_data = self.data[start_sample:end_sample].copy()
        absolute_start_time = self.start_time + start_time
        
        return AudioBuffer(
            chunk_data, 
            self.sample_rate, 
            absolute_start_time,
            metadata={**self.metadata, 'parent_id': self._id, 'chunk': True}
        )
    
    def apply_preprocessing(self, normalize: bool = True, 
                          remove_silence: bool = False) -> 'AudioBuffer':
        """Apply preprocessing to the audio buffer."""
        processed_data = self.data.copy()
        
        if normalize and len(processed_data) > 0:
            # Normalize to prevent clipping
            max_val = np.max(np.abs(processed_data))
            if max_val > 0:
                processed_data = processed_data / max_val * 0.9
        
        if remove_silence and len(processed_data) > 0:
            # Simple silence removal based on RMS threshold
            rms = np.sqrt(np.mean(processed_data**2))
            threshold = rms * 0.1
            
            # Find segments above threshold
            above_threshold = np.abs(processed_data) > threshold
            if np.any(above_threshold):
                start_idx = np.where(above_threshold)[0][0]
                end_idx = np.where(above_threshold)[0][-1] + 1
                processed_data = processed_data[start_idx:end_idx]
        
        return AudioBuffer(
            processed_data, 
            self.sample_rate, 
            self.start_time,
            metadata={**self.metadata, 'preprocessed': True}
        )
    
    def to_mono(self) -> 'AudioBuffer':
        """Convert to mono if stereo."""
        if self.data.ndim == 1:
            return self
        
        mono_data = librosa.to_mono(self.data.T)  # librosa expects (channels, samples)
        return AudioBuffer(
            mono_data, 
            self.sample_rate, 
            self.start_time,
            metadata={**self.metadata, 'converted_to_mono': True}
        )
    
    def resample(self, target_sr: int) -> 'AudioBuffer':
        """Resample to target sample rate."""
        if self.sample_rate == target_sr:
            return self
        
        resampled_data = librosa.resample(
            self.data, 
            orig_sr=self.sample_rate, 
            target_sr=target_sr
        )
        
        return AudioBuffer(
            resampled_data, 
            target_sr, 
            self.start_time,
            metadata={**self.metadata, 'resampled': True, 'original_sr': self.sample_rate}
        )
    
    def get_rms(self) -> float:
        """Get RMS (Root Mean Square) value."""
        return float(np.sqrt(np.mean(self.data**2))) if len(self.data) > 0 else 0.0
    
    def get_peak(self) -> float:
        """Get peak amplitude."""
        return float(np.max(np.abs(self.data))) if len(self.data) > 0 else 0.0
    
    def __del__(self):
        """Cleanup when buffer is deleted."""
        AudioBufferManager.unregister_buffer(self._id)


class AudioBufferManager:
    """Global manager for AudioBuffer memory tracking."""
    
    _buffers = {}
    _lock = threading.Lock()
    _total_memory_mb = 0.0
    
    @classmethod
    def register_buffer(cls, buffer: AudioBuffer):
        """Register a new buffer."""
        with cls._lock:
            cls._buffers[buffer._id] = {
                'buffer': buffer,
                'created_at': time.time(),
                'memory_mb': buffer.memory_usage_mb
            }
            cls._total_memory_mb += buffer.memory_usage_mb
            
            # Check memory limits
            if cls._total_memory_mb > MAX_MEMORY_CACHE_MB:
                cls._cleanup_old_buffers()
    
    @classmethod
    def unregister_buffer(cls, buffer_id: str):
        """Unregister a buffer."""
        with cls._lock:
            if buffer_id in cls._buffers:
                memory_mb = cls._buffers[buffer_id]['memory_mb']
                del cls._buffers[buffer_id]
                cls._total_memory_mb -= memory_mb
    
    @classmethod
    def _cleanup_old_buffers(cls):
        """Clean up old buffers when memory limit is exceeded."""
        current_time = time.time()
        old_threshold = 300  # 5 minutes
        
        to_remove = []
        for buffer_id, info in cls._buffers.items():
            if current_time - info['created_at'] > old_threshold:
                to_remove.append(buffer_id)
        
        for buffer_id in to_remove:
            cls.unregister_buffer(buffer_id)
        
        # Force garbage collection
        gc.collect()
    
    @classmethod
    def get_memory_usage(cls) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with cls._lock:
            return {
                'total_buffers': len(cls._buffers),
                'total_memory_mb': cls._total_memory_mb,
                'memory_limit_mb': MAX_MEMORY_CACHE_MB,
                'memory_usage_percent': (cls._total_memory_mb / MAX_MEMORY_CACHE_MB) * 100
            }
    
    @classmethod
    def clear_all_buffers(cls):
        """Clear all registered buffers."""
        with cls._lock:
            cls._buffers.clear()
            cls._total_memory_mb = 0.0
        gc.collect()


class InMemoryAudioProcessor:
    """Complete in-memory audio processing system."""
    
    def __init__(self, target_sr: int = DEFAULT_SR, max_cache_size: int = 100):
        self.target_sr = target_sr
        self.max_cache_size = max_cache_size
        self._vad_cache = {}
        self._preprocessing_cache = {}
        self._lock = threading.Lock()
    
    def load_from_file(self, file_path: str, 
                      preprocess: bool = True,
                      cache_key: Optional[str] = None) -> AudioBuffer:
        """Load audio file into memory buffer."""
        try:
            # Check cache first
            if cache_key and cache_key in self._preprocessing_cache:
                return self._preprocessing_cache[cache_key]
            
            # Load audio data
            audio_data, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            buffer = AudioBuffer(
                audio_data, 
                sr, 
                metadata={'source': 'file', 'file_path': file_path}
            )
            
            if preprocess:
                buffer = buffer.apply_preprocessing()
            
            # Cache if key provided
            if cache_key:
                self._preprocessing_cache[cache_key] = buffer
            
            return buffer
            
        except Exception as e:
            raise ValueError(f"Failed to load audio from {file_path}: {e}")
    
    def load_from_bytes(self, audio_bytes: bytes,
                       format: str = 'WAV',
                       preprocess: bool = True) -> AudioBuffer:
        """Load audio from bytes buffer."""
        try:
            with io.BytesIO(audio_bytes) as buffer:
                audio_data, sr = librosa.load(buffer, sr=self.target_sr, mono=True)
            
            buffer = AudioBuffer(audio_data, sr, metadata={'source': 'bytes'})
            
            if preprocess:
                buffer = buffer.apply_preprocessing()
            
            return buffer
            
        except Exception as e:
            raise ValueError(f"Failed to load audio from bytes: {e}")
    
    def load_from_array(self, audio_data: np.ndarray, 
                       sample_rate: int,
                       preprocess: bool = True) -> AudioBuffer:
        """Load audio from numpy array."""
        try:
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # Resample if needed
            if sample_rate != self.target_sr:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.target_sr
                )
                sample_rate = self.target_sr
            
            buffer = AudioBuffer(audio_data, sample_rate, metadata={'source': 'array'})
            
            if preprocess:
                buffer = buffer.apply_preprocessing()
            
            return buffer
            
        except Exception as e:
            raise ValueError(f"Failed to load audio from array: {e}")
    
    def split_into_chunks(self, buffer: AudioBuffer,
                         chunk_duration: float = 30.0,
                         overlap_duration: float = 15.0) -> List[AudioBuffer]:
        """Split audio buffer into overlapping chunks."""
        if buffer.duration <= chunk_duration:
            return [buffer]
        
        chunks = []
        step_duration = chunk_duration - overlap_duration
        current_time = 0.0
        chunk_index = 0
        
        while current_time < buffer.duration:
            end_time = min(current_time + chunk_duration, buffer.duration)
            
            chunk = buffer.get_chunk(current_time, end_time)
            chunk.metadata.update({
                'chunk_index': chunk_index,
                'chunk_start': current_time,
                'chunk_end': end_time,
                'parent_duration': buffer.duration
            })
            
            chunks.append(chunk)
            
            current_time += step_duration
            chunk_index += 1
            
            # Stop if we've covered the full duration
            if end_time >= buffer.duration:
                break
        
        return chunks
    
    def apply_vad(self, buffer: AudioBuffer, 
                 aggressiveness: int = VAD_AGGRESSIVENESS) -> AudioBuffer:
        """Apply Voice Activity Detection to remove silence."""
        cache_key = f"{buffer._id}_{aggressiveness}"
        
        if cache_key in self._vad_cache:
            return self._vad_cache[cache_key]
        
        try:
            import webrtcvad
            from modules.audio_preprocessing import apply_webrtc_vad
            
            voiced_data = apply_webrtc_vad(
                buffer.data, 
                sr=buffer.sample_rate,
                aggressiveness=aggressiveness
            )
            
            if len(voiced_data) == 0:
                voiced_data = buffer.data  # Fallback to original
            
            result = AudioBuffer(
                voiced_data, 
                buffer.sample_rate, 
                buffer.start_time,
                metadata={**buffer.metadata, 'vad_applied': True}
            )
            
            self._vad_cache[cache_key] = result
            return result
            
        except ImportError:
            # WebRTC VAD not available, return original
            return buffer
        except Exception:
            # VAD failed, return original
            return buffer
    
    def resample_buffer(self, buffer: AudioBuffer, 
                       target_sr: int) -> AudioBuffer:
        """Resample audio buffer to target sample rate."""
        if buffer.sample_rate == target_sr:
            return buffer
        
        resampled_data = librosa.resample(
            buffer.data,
            orig_sr=buffer.sample_rate,
            target_sr=target_sr
        )
        
        return AudioBuffer(
            resampled_data,
            target_sr,
            buffer.start_time,
            metadata={**buffer.metadata, 'resampled': True, 'original_sr': buffer.sample_rate}
        )
    
    def concatenate_buffers(self, buffers: List[AudioBuffer]) -> AudioBuffer:
        """Concatenate multiple audio buffers into one."""
        if not buffers:
            return AudioBuffer(np.array([]), self.target_sr)
        
        if len(buffers) == 1:
            return buffers[0]
        
        # Ensure all buffers have the same sample rate
        target_sr = buffers[0].sample_rate
        for i, buffer in enumerate(buffers):
            if buffer.sample_rate != target_sr:
                buffers[i] = self.resample_buffer(buffer, target_sr)
        
        # Concatenate data
        concatenated_data = np.concatenate([buf.data for buf in buffers])
        
        # Combine metadata
        combined_metadata = {
            'source': 'concatenated',
            'buffer_count': len(buffers),
            'original_durations': [buf.duration for buf in buffers]
        }
        
        return AudioBuffer(
            concatenated_data,
            target_sr,
            buffers[0].start_time,
            metadata=combined_metadata
        )
    
    def save_buffer_to_file(self, buffer: AudioBuffer, 
                           output_path: str,
                           format: str = 'PCM_16') -> str:
        """Save audio buffer to file (only when necessary)."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, buffer.data, buffer.sample_rate, subtype=format)
            return output_path
        except Exception as e:
            raise ValueError(f"Failed to save buffer to {output_path}: {e}")
    
    def get_buffer_info(self, buffer: AudioBuffer) -> Dict[str, Any]:
        """Get comprehensive information about an audio buffer."""
        return {
            'duration_seconds': buffer.duration,
            'sample_rate': buffer.sample_rate,
            'samples': buffer.samples,
            'memory_usage_mb': buffer.memory_usage_mb,
            'start_time': buffer.start_time,
            'end_time': buffer.end_time,
            'data_type': str(buffer.data.dtype),
            'data_shape': buffer.data.shape,
            'rms': float(np.sqrt(np.mean(buffer.data**2))),
            'peak': float(np.max(np.abs(buffer.data))),
            'metadata': buffer.metadata
        }
    
    def clear_caches(self):
        """Clear all internal caches."""
        self._vad_cache.clear()
        self._preprocessing_cache.clear()
        gc.collect()


# Global processor instance
_processor = None


def get_audio_processor() -> InMemoryAudioProcessor:
    """Get or create global audio processor."""
    global _processor
    if _processor is None:
        _processor = InMemoryAudioProcessor()
    return _processor


@contextmanager
def temporary_audio_file(buffer: AudioBuffer, suffix: str = '.wav'):
    """Context manager for temporary audio files when legacy file-based processing is needed."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            temp_file = f.name
        
        processor = get_audio_processor()
        processor.save_buffer_to_file(buffer, temp_file)
        
        yield temp_file
        
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def preprocess_audio_in_memory(audio_input: Union[str, bytes, np.ndarray],
                              sample_rate: Optional[int] = None,
                              target_sr: int = DEFAULT_SR,
                              apply_vad: bool = False) -> AudioBuffer:
    """Preprocess audio entirely in memory without disk I/O."""
    processor = get_audio_processor()
    
    # Load based on input type
    if isinstance(audio_input, str):
        buffer = processor.load_from_file(audio_input, preprocess=True)
    elif isinstance(audio_input, bytes):
        buffer = processor.load_from_bytes(audio_input, preprocess=True)
    elif isinstance(audio_input, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided for numpy array input")
        buffer = processor.load_from_array(audio_input, sample_rate, preprocess=True)
    else:
        raise ValueError(f"Unsupported input type: {type(audio_input)}")
    
    # Apply VAD if requested
    if apply_vad:
        buffer = processor.apply_vad(buffer)
    
    # Resample if needed
    if buffer.sample_rate != target_sr:
        buffer = processor.resample_buffer(buffer, target_sr)
    
    return buffer


def split_audio_in_memory(audio_input: Union[str, AudioBuffer],
                         chunk_duration: float = 30.0,
                         overlap_duration: float = 15.0,
                         apply_vad: bool = False) -> List[AudioBuffer]:
    """Split audio into chunks entirely in memory."""
    processor = get_audio_processor()
    
    # Get buffer
    if isinstance(audio_input, str):
        buffer = processor.load_from_file(audio_input, preprocess=True)
    elif isinstance(audio_input, AudioBuffer):
        buffer = audio_input
    else:
        raise ValueError(f"Unsupported input type: {type(audio_input)}")
    
    # Apply VAD if requested
    if apply_vad:
        buffer = processor.apply_vad(buffer)
    
    # Split into chunks
    return processor.split_into_chunks(buffer, chunk_duration, overlap_duration)


def get_memory_usage_info() -> Dict[str, Any]:
    """Get comprehensive memory usage information."""
    buffer_stats = AudioBufferManager.get_memory_usage()
    
    try:
        import psutil
        process = psutil.Process()
        system_memory = {
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'system_memory_percent': process.memory_percent(),
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    except ImportError:
        system_memory = {'status': 'psutil not available'}
    
    return {
        'audio_buffers': buffer_stats,
        'system_memory': system_memory,
        'memory_limit_mb': MAX_MEMORY_CACHE_MB
    }


def cleanup_memory():
    """Force cleanup of all audio processing memory."""
    global _processor
    
    AudioBufferManager.clear_all_buffers()
    
    if _processor:
        _processor.clear_caches()
    
    gc.collect()
    print("🧹 Audio processing memory cleaned up")