"""
Advanced memory management system for audio processing with buffer pools,
memory-mapped files, and intelligent garbage collection.
"""

import time
import gc
import mmap
import threading
import weakref
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from contextlib import contextmanager
import numpy as np

# Try to import the AudioBuffer class - fallback if not available
try:
    from modules.in_memory_audio import AudioBuffer
except ImportError:
    # Create a minimal AudioBuffer class if not available
    class AudioBuffer:
        def __init__(self, data, sample_rate, start_time=0.0, metadata=None):
            self.data = data
            self.sample_rate = sample_rate
            self.start_time = start_time
            self.end_time = start_time + (len(data) / sample_rate if len(data) > 0 else 0)
            self.metadata = metadata or {}
        
        @property
        def duration(self):
            return len(self.data) / self.sample_rate if len(self.data) > 0 else 0.0
        
        @property
        def memory_usage_mb(self):
            return self.data.nbytes / (1024 * 1024) if hasattr(self.data, 'nbytes') else 0.0

# Default sample rate
DEFAULT_SR = 16000


class BufferPool:
    """Memory buffer pool for efficient audio data management."""
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self._pools = {}  # size_dtype -> list of buffers
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_allocated': 0,
            'total_returned': 0
        }
        self._lock = threading.Lock()
    
    def get_buffer(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Get a buffer from the pool or create a new one."""
        key = (shape, dtype)
        
        with self._lock:
            pool = self._pools.get(key, [])
            
            if pool:
                buffer = pool.pop()
                self._stats['hits'] += 1
                # Clear the buffer
                buffer.fill(0)
                return buffer
            else:
                self._stats['misses'] += 1
                self._stats['total_allocated'] += 1
                return np.zeros(shape, dtype=dtype)
    
    def return_buffer(self, buffer: np.ndarray):
        """Return a buffer to the pool."""
        if buffer is None:
            return
        
        key = (buffer.shape, buffer.dtype)
        
        with self._lock:
            self._stats['total_returned'] += 1
            
            pool = self._pools.setdefault(key, [])
            
            if len(pool) < self.max_pool_size:
                pool.append(buffer)
    
    def clear_pool(self):
        """Clear all buffers from the pool."""
        with self._lock:
            self._pools.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer pool statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'total_allocated': self._stats['total_allocated'],
                'total_returned': self._stats['total_returned'],
                'total_pooled_buffers': sum(len(pool) for pool in self._pools.values()),
                'pool_types': len(self._pools)
            }


class MemoryMappedAudioFile:
    """Memory-mapped audio file for efficient large file handling."""
    
    def __init__(self, file_path: str, sample_rate: int = DEFAULT_SR):
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self._mmap = None
        self._file_handle = None
        self._audio_data = None
        self._metadata = {}
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the memory-mapped file or load into memory."""
        try:
            # Get file metadata
            if self.file_path.exists():
                file_size = self.file_path.stat().st_size
                self._metadata = {
                    'file_size': file_size,
                    'file_path': str(self.file_path)
                }
            
            # For large files (>50MB), use memory mapping
            if self._metadata['file_size'] > 50 * 1024 * 1024:  # 50MB threshold
                self._init_memory_map()
            else:
                # For smaller files, load normally
                import librosa
                self._audio_data, sr = librosa.load(str(self.file_path), sr=self.sample_rate, mono=True)
                if sr != self.sample_rate:
                    self._audio_data = librosa.resample(
                        self._audio_data, orig_sr=sr, target_sr=self.sample_rate
                    )
        
        except Exception as e:
            raise ValueError(f"Failed to initialize memory-mapped file {self.file_path}: {e}")
    
    def _init_memory_map(self):
        """Initialize memory mapping for large files."""
        try:
            # Convert audio to raw format first if needed
            raw_path = self.file_path.with_suffix('.raw')
            
            if not raw_path.exists() or raw_path.stat().st_mtime < self.file_path.stat().st_mtime:
                self._create_raw_file(raw_path)
            
            # Memory map the raw file
            self._file_handle = open(raw_path, 'rb')
            self._mmap = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            
        except Exception as e:
            self._cleanup_mmap()
            raise ValueError(f"Failed to create memory map: {e}")
    
    def _create_raw_file(self, raw_path: Path):
        """Convert audio file to raw float32 format."""
        import librosa
        import soundfile as sf
        
        # Load and convert
        audio_data, sr = librosa.load(str(self.file_path), sr=self.sample_rate, mono=True)
        
        # Save as raw float32
        audio_data.astype(np.float32).tofile(str(raw_path))
    
    def get_chunk(self, start_sec: float, duration_sec: float) -> AudioBuffer:
        """Get a chunk of audio data."""
        start_sample = int(start_sec * self.sample_rate)
        num_samples = int(duration_sec * self.sample_rate)
        
        if self._mmap is not None:
            # Read from memory map
            start_byte = start_sample * 4  # 4 bytes per float32
            num_bytes = num_samples * 4
            
            # Ensure we don't read beyond file
            max_bytes = len(self._mmap) - start_byte
            num_bytes = min(num_bytes, max_bytes)
            
            if num_bytes <= 0:
                return AudioBuffer(np.array([]), self.sample_rate, start_sec)
            
            # Read data from memory map
            data_bytes = self._mmap[start_byte:start_byte + num_bytes]
            audio_data = np.frombuffer(data_bytes, dtype=np.float32)
            
        else:
            # Read from loaded data
            end_sample = min(start_sample + num_samples, len(self._audio_data))
            audio_data = self._audio_data[start_sample:end_sample].copy()
        
        return AudioBuffer(
            audio_data, 
            self.sample_rate, 
            start_sec,
            metadata={'source': 'memory_mapped', 'file': str(self.file_path)}
        )
    
    def get_full_audio(self) -> AudioBuffer:
        """Get the complete audio file as an AudioBuffer."""
        if self._mmap is not None:
            # Read entire memory mapped file
            audio_data = np.frombuffer(self._mmap, dtype=np.float32)
        else:
            audio_data = self._audio_data.copy()
        
        return AudioBuffer(
            audio_data,
            self.sample_rate,
            0.0,
            metadata={'source': 'memory_mapped_full', 'file': str(self.file_path)}
        )
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._metadata.get('duration', 0.0)
    
    @property
    def file_size_mb(self) -> float:
        """File size in MB."""
        return self._metadata.get('file_size', 0) / (1024 * 1024)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Clean up resources."""
        self._cleanup_mmap()
    
    def _cleanup_mmap(self):
        """Clean up memory mapping resources."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def __del__(self):
        self._cleanup_mmap()


class SmartGarbageCollector:
    """Intelligent garbage collection for audio processing."""
    
    def __init__(self, 
                 memory_threshold_mb: float = 512.0,
                 check_interval_sec: float = 30.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.check_interval_sec = check_interval_sec
        self._last_check = 0.0
        self._stats = {
            'collections_triggered': 0,
            'memory_freed_mb': 0.0,
            'last_collection_time': 0.0
        }
    
    def check_and_collect(self, force: bool = False) -> Dict[str, Any]:
        """Check memory usage and trigger GC if needed."""
        current_time = time.time()
        
        if not force and (current_time - self._last_check) < self.check_interval_sec:
            return {'status': 'skipped', 'reason': 'too_soon'}
        
        self._last_check = current_time
        
        # Get memory usage
        memory_info = self._get_memory_usage()
        current_memory_mb = memory_info.get('process_memory_mb', 0)
        
        if force or current_memory_mb > self.memory_threshold_mb:
            return self._perform_collection(memory_info)
        
        return {'status': 'no_collection_needed', 'memory_mb': current_memory_mb}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return {
                'process_memory_mb': process.memory_info().rss / (1024 * 1024),
                'memory_percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
        except ImportError:
            return {'process_memory_mb': 0, 'memory_percent': 0, 'available_mb': 0}
    
    def _perform_collection(self, before_memory: Dict[str, float]) -> Dict[str, Any]:
        """Perform garbage collection and measure results."""
        before_mb = before_memory.get('process_memory_mb', 0)
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory usage after collection
        after_memory = self._get_memory_usage()
        after_mb = after_memory.get('process_memory_mb', 0)
        
        freed_mb = before_mb - after_mb
        
        # Update stats
        self._stats['collections_triggered'] += 1
        self._stats['memory_freed_mb'] += freed_mb
        self._stats['last_collection_time'] = time.time()
        
        return {
            'status': 'collected',
            'objects_collected': collected,
            'memory_before_mb': before_mb,
            'memory_after_mb': after_mb,
            'memory_freed_mb': freed_mb,
            'stats': self._stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        return self._stats.copy()


class AdvancedMemoryManager:
    """Advanced memory management system for audio processing."""
    
    def __init__(self):
        self.buffer_pool = BufferPool()
        self.gc_manager = SmartGarbageCollector()
        self._memory_mapped_files = weakref.WeakValueDictionary()
        self._lock = threading.Lock()
        
    def get_pooled_buffer(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Get a buffer from the pool."""
        return self.buffer_pool.get_buffer(shape, dtype)
    
    def return_pooled_buffer(self, buffer: np.ndarray):
        """Return a buffer to the pool."""
        self.buffer_pool.return_buffer(buffer)
    
    def open_memory_mapped_file(self, file_path: str, 
                               sample_rate: int = DEFAULT_SR) -> MemoryMappedAudioFile:
        """Open a memory-mapped audio file with caching."""
        file_key = f"{file_path}_{sample_rate}"
        
        with self._lock:
            if file_key in self._memory_mapped_files:
                return self._memory_mapped_files[file_key]
            
            mm_file = MemoryMappedAudioFile(file_path, sample_rate)
            self._memory_mapped_files[file_key] = mm_file
            return mm_file
    
    def create_optimized_audio_buffer(self, size: int, 
                                    sample_rate: int = DEFAULT_SR,
                                    start_time: float = 0.0) -> AudioBuffer:
        """Create an AudioBuffer using pooled memory."""
        data = self.get_pooled_buffer((size,), np.float32)
        
        # Custom AudioBuffer that returns buffer to pool on deletion
        class PooledAudioBuffer(AudioBuffer):
            def __init__(self, data, sr, start_time, manager):
                super().__init__(data, sr, start_time)
                self._manager = manager
                self._original_data = data
            
            def __del__(self):
                if hasattr(self, '_manager') and hasattr(self, '_original_data'):
                    self._manager.return_pooled_buffer(self._original_data)
                super().__del__()
        
        return PooledAudioBuffer(data, sample_rate, start_time, self)
    
    def check_memory_and_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """Check memory usage and perform cleanup if needed."""
        result = self.gc_manager.check_and_collect(force)
        
        # Also clean up buffer pools if memory pressure is high
        if result.get('status') == 'collected' and result.get('memory_freed_mb', 0) < 50:
            self.buffer_pool.clear_pool()
            result['buffer_pool_cleared'] = True
        
        return result
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory management statistics."""
        return {
            'buffer_pool': self.buffer_pool.get_stats(),
            'garbage_collector': self.gc_manager.get_stats(),
            'memory_mapped_files': len(self._memory_mapped_files),
            'current_memory': self.gc_manager._get_memory_usage()
        }
    
    def cleanup_all(self):
        """Clean up all managed resources."""
        with self._lock:
            # Close all memory-mapped files
            for mm_file in list(self._memory_mapped_files.values()):
                try:
                    mm_file.close()
                except Exception:
                    pass
            self._memory_mapped_files.clear()
            
            # Clear buffer pools
            self.buffer_pool.clear_pool()
            
            # Force garbage collection
            self.gc_manager.check_and_collect(force=True)


# Global memory manager instance
_memory_manager = None


def get_memory_manager() -> AdvancedMemoryManager:
    """Get or create global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AdvancedMemoryManager()
    return _memory_manager


@contextmanager
def managed_audio_file(file_path: str, sample_rate: int = DEFAULT_SR):
    """Context manager for memory-mapped audio files."""
    manager = get_memory_manager()
    mm_file = manager.open_memory_mapped_file(file_path, sample_rate)
    try:
        yield mm_file
    finally:
        # File will be automatically cleaned up by weak references
        pass


def create_optimized_buffer(size: int, sample_rate: int = DEFAULT_SR) -> AudioBuffer:
    """Create an optimized audio buffer using memory management."""
    manager = get_memory_manager()
    return manager.create_optimized_audio_buffer(size, sample_rate)


def check_memory_pressure() -> Dict[str, Any]:
    """Check current memory pressure and get recommendations."""
    manager = get_memory_manager()
    stats = manager.get_comprehensive_stats()
    
    current_memory = stats['current_memory'].get('process_memory_mb', 0)
    memory_percent = stats['current_memory'].get('memory_percent', 0)
    
    # Determine pressure level
    if memory_percent > 80 or current_memory > 1024:  # 1GB
        pressure_level = 'high'
        recommendation = 'Consider forcing garbage collection and clearing caches'
    elif memory_percent > 60 or current_memory > 512:  # 512MB
        pressure_level = 'medium'
        recommendation = 'Monitor memory usage closely'
    else:
        pressure_level = 'low'
        recommendation = 'Memory usage is healthy'
    
    return {
        'pressure_level': pressure_level,
        'recommendation': recommendation,
        'stats': stats,
        'current_memory_mb': current_memory,
        'memory_percent': memory_percent,
        'available_memory_mb': stats['current_memory'].get('available_mb', 0)
    }


def optimize_memory_usage(aggressive: bool = False) -> Dict[str, Any]:
    """Optimize memory usage across all audio processing components."""
    manager = get_memory_manager()
    
    results = {
        'before': manager.get_comprehensive_stats(),
        'actions_taken': []
    }
    
    # Force garbage collection
    gc_result = manager.check_memory_and_cleanup(force=True)
    results['actions_taken'].append('garbage_collection')
    
    if aggressive:
        # Clear all buffer pools
        manager.buffer_pool.clear_pool()
        results['actions_taken'].append('buffer_pool_cleared')
        
        # Clean up all resources
        manager.cleanup_all()
        results['actions_taken'].append('full_cleanup')
    
    results['after'] = manager.get_comprehensive_stats()
    results['garbage_collection'] = gc_result
    
    return results


def get_memory_usage_summary() -> str:
    """Get a human-readable memory usage summary."""
    pressure_info = check_memory_pressure()
    stats = pressure_info['stats']
    
    summary = f"""Memory Usage Summary:
{'='*40}
Pressure Level: {pressure_info['pressure_level'].upper()}
Current Memory: {pressure_info['current_memory_mb']:.1f} MB ({pressure_info['memory_percent']:.1f}%)

Buffer Pool:
  Total Requests: {stats['buffer_pool']['hits'] + stats['buffer_pool']['misses']}
  Hit Rate: {stats['buffer_pool']['hit_rate']:.1%}
  Pooled Buffers: {stats['buffer_pool']['total_pooled_buffers']}

Garbage Collection:
  Collections: {stats['garbage_collector']['collections_triggered']}
  Memory Freed: {stats['garbage_collector']['memory_freed_mb']:.1f} MB

Memory-Mapped Files: {stats['memory_mapped_files']}

Recommendation: {pressure_info['recommendation']}
{'='*40}"""
    
    return summary


def force_cleanup():
    """Force immediate cleanup of all memory resources."""
    manager = get_memory_manager()
    return manager.check_memory_and_cleanup(force=True)