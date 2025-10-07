"""
Smart batch processing system with dynamic sizing based on available memory and CPU cores.
Includes adaptive chunk sizing for different audio lengths and optimal resource utilization.
"""

import os
import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil

from modules.memory_management import get_memory_manager, check_memory_pressure
from modules.model_loader import get_device_manager
from modules.in_memory_audio import AudioBuffer, get_audio_processor


@dataclass
class BatchProcessingConfig:
    """Configuration for adaptive batch processing."""
    min_batch_size: int = 1
    max_batch_size: int = 32
    target_memory_usage_mb: float = 256.0
    cpu_utilization_target: float = 0.8
    adaptive_sizing: bool = True
    prefer_threading: bool = True  # Threading vs multiprocessing
    chunk_size_strategy: str = "adaptive"  # adaptive, fixed, audio_length_based


class AdaptiveBatchProcessor:
    """Intelligent batch processor that adapts to system resources and workload."""
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        self.device_manager = get_device_manager()
        self.memory_manager = get_memory_manager()
        
        # Performance tracking
        self._performance_history = []
        self._last_optimization = 0.0
        self._current_batch_size = self.config.min_batch_size
        self._current_chunk_duration = 30.0  # Default chunk duration
        
        # Resource monitoring
        self._cpu_count = os.cpu_count() or 4
        self._memory_limit_mb = self._get_memory_limit()
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _get_memory_limit(self) -> float:
        """Calculate safe memory limit for batch processing."""
        try:
            system_memory = psutil.virtual_memory()
            # Use 50% of available memory, capped at config target
            available_mb = system_memory.available / (1024 * 1024)
            safe_limit = min(available_mb * 0.5, self.config.target_memory_usage_mb)
            return max(safe_limit, 128.0)  # Minimum 128MB
        except Exception:
            return self.config.target_memory_usage_mb
    
    def calculate_optimal_batch_size(self, 
                                   item_memory_mb: float,
                                   processing_time_ms: float = None) -> int:
        """Calculate optimal batch size based on memory and performance constraints."""
        
        # Memory-based calculation
        if item_memory_mb > 0:
            max_memory_batch = int(self._memory_limit_mb / item_memory_mb)
        else:
            max_memory_batch = self.config.max_batch_size
        
        # CPU-based calculation
        cpu_batch = min(self._cpu_count * 2, self.config.max_batch_size)
        
        # Performance-based adjustment
        if processing_time_ms and self._performance_history:
            # If items process quickly, increase batch size
            # If items process slowly, decrease batch size
            avg_time = np.mean([p['processing_time_ms'] for p in self._performance_history[-10:]])
            if processing_time_ms < avg_time * 0.5:
                performance_multiplier = 1.5
            elif processing_time_ms > avg_time * 1.5:
                performance_multiplier = 0.7
            else:
                performance_multiplier = 1.0
            
            cpu_batch = int(cpu_batch * performance_multiplier)
        
        # Take the minimum of all constraints
        optimal_size = min(
            max_memory_batch,
            cpu_batch,
            self.config.max_batch_size
        )
        
        return max(optimal_size, self.config.min_batch_size)
    
    def calculate_adaptive_chunk_size(self, audio_duration: float) -> float:
        """Calculate optimal chunk size based on audio length and system resources."""
        
        if self.config.chunk_size_strategy == "fixed":
            return self._current_chunk_duration
        
        elif self.config.chunk_size_strategy == "audio_length_based":
            # Shorter chunks for longer audio to maintain responsiveness
            if audio_duration <= 30:
                return 15.0
            elif audio_duration <= 120:
                return 30.0
            elif audio_duration <= 300:
                return 45.0
            else:
                return 60.0
        
        else:  # adaptive strategy
            # Consider system resources and performance history
            memory_pressure = check_memory_pressure()
            
            base_chunk_size = 30.0
            
            # Adjust based on memory pressure
            if memory_pressure['pressure_level'] == 'high':
                base_chunk_size = 15.0
            elif memory_pressure['pressure_level'] == 'low':
                base_chunk_size = 45.0
            
            # Adjust based on CPU cores (more cores can handle larger chunks)
            cpu_factor = min(self._cpu_count / 4.0, 2.0)
            adaptive_size = base_chunk_size * cpu_factor
            
            # Performance-based adjustment
            if self._performance_history:
                recent_performance = self._performance_history[-5:]
                avg_throughput = np.mean([p.get('throughput_chunks_per_sec', 1.0) 
                                        for p in recent_performance])
                
                if avg_throughput > 2.0:  # High throughput, can handle larger chunks
                    adaptive_size *= 1.2
                elif avg_throughput < 0.5:  # Low throughput, use smaller chunks
                    adaptive_size *= 0.8
            
            # Clamp to reasonable bounds
            return max(15.0, min(adaptive_size, 60.0))
    
    def process_batch_async(self,
                          items: List[Any],
                          processor_func: Callable,
                          batch_size: Optional[int] = None,
                          max_workers: Optional[int] = None) -> List[Any]:
        """Process items in batches with optimal resource utilization."""
        
        if not items:
            return []
        
        start_time = time.perf_counter()
        
        # Estimate memory usage per item
        estimated_memory_mb = self._estimate_item_memory(items[0]) if items else 1.0
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self.calculate_optimal_batch_size(estimated_memory_mb)
        
        # Calculate optimal worker count
        if max_workers is None:
            max_workers = min(len(items), self._cpu_count, batch_size)
        
        print(f"🔄 Processing {len(items)} items in batches of {batch_size} "
              f"with {max_workers} workers")
        
        results = []
        
        try:
            # Process in batches
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            if self.config.prefer_threading:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self._process_single_batch, batch, processor_func): batch
                        for batch in batches
                    }
                    
                    for future in as_completed(future_to_batch):
                        batch_results = future.result()
                        results.extend(batch_results)
            else:
                # Use multiprocessing for CPU-intensive tasks
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self._process_single_batch, batch, processor_func): batch
                        for batch in batches
                    }
                    
                    for future in as_completed(future_to_batch):
                        batch_results = future.result()
                        results.extend(batch_results)
        
        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")
            # Fallback to sequential processing
            results = [processor_func(item) for item in items]
        
        # Track performance
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        self._record_performance(
            batch_size=batch_size,
            item_count=len(items),
            processing_time_ms=processing_time_ms,
            memory_usage_mb=estimated_memory_mb * len(items)
        )
        
        return results
    
    def _process_single_batch(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single batch of items."""
        return [processor_func(item) for item in batch]
    
    def _estimate_item_memory(self, item: Any) -> float:
        """Estimate memory usage of a single item in MB."""
        try:
            if isinstance(item, AudioBuffer):
                return item.memory_usage_mb
            elif isinstance(item, np.ndarray):
                return item.nbytes / (1024 * 1024)
            elif isinstance(item, dict) and 'data' in item:
                data = item['data']
                if isinstance(data, np.ndarray):
                    return data.nbytes / (1024 * 1024)
            
            # Default estimate for unknown types
            return 5.0
            
        except Exception:
            return 5.0  # Conservative estimate
    
    def _record_performance(self, 
                          batch_size: int,
                          item_count: int,
                          processing_time_ms: float,
                          memory_usage_mb: float):
        """Record performance metrics for adaptive optimization."""
        
        throughput = item_count / (processing_time_ms / 1000.0) if processing_time_ms > 0 else 0
        
        performance_record = {
            'timestamp': time.time(),
            'batch_size': batch_size,
            'item_count': item_count,
            'processing_time_ms': processing_time_ms,
            'memory_usage_mb': memory_usage_mb,
            'throughput_items_per_sec': throughput,
            'throughput_chunks_per_sec': throughput  # Alias for compatibility
        }
        
        with self._lock:
            self._performance_history.append(performance_record)
            
            # Keep only recent history (last 50 records)
            if len(self._performance_history) > 50:
                self._performance_history = self._performance_history[-50:]
            
            # Adaptive optimization
            if self.config.adaptive_sizing:
                self._optimize_parameters()
    
    def _optimize_parameters(self):
        """Optimize batch processing parameters based on performance history."""
        current_time = time.time()
        
        # Only optimize every 60 seconds to avoid thrashing
        if current_time - self._last_optimization < 60.0:
            return
        
        if len(self._performance_history) < 5:
            return
        
        self._last_optimization = current_time
        
        # Analyze recent performance
        recent_records = self._performance_history[-10:]
        avg_throughput = np.mean([r['throughput_items_per_sec'] for r in recent_records])
        avg_memory = np.mean([r['memory_usage_mb'] for r in recent_records])
        
        # Adjust batch size based on performance
        if avg_throughput > 5.0 and avg_memory < self._memory_limit_mb * 0.7:
            # Good performance, can increase batch size
            self._current_batch_size = min(
                int(self._current_batch_size * 1.2),
                self.config.max_batch_size
            )
        elif avg_throughput < 1.0 or avg_memory > self._memory_limit_mb * 0.9:
            # Poor performance or high memory usage, decrease batch size
            self._current_batch_size = max(
                int(self._current_batch_size * 0.8),
                self.config.min_batch_size
            )
        
        print(f"[TARGET] Optimized batch size to {self._current_batch_size} "
              f"(throughput: {avg_throughput:.1f} items/sec)")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self._performance_history:
            return {'status': 'no_data'}
        
        recent_records = self._performance_history[-20:]
        
        return {
            'total_records': len(self._performance_history),
            'recent_avg_throughput': np.mean([r['throughput_items_per_sec'] for r in recent_records]),
            'recent_avg_memory_mb': np.mean([r['memory_usage_mb'] for r in recent_records]),
            'recent_avg_processing_time_ms': np.mean([r['processing_time_ms'] for r in recent_records]),
            'current_batch_size': self._current_batch_size,
            'current_chunk_duration': self._current_chunk_duration,
            'memory_limit_mb': self._memory_limit_mb,
            'cpu_count': self._cpu_count,
            'config': {
                'min_batch_size': self.config.min_batch_size,
                'max_batch_size': self.config.max_batch_size,
                'adaptive_sizing': self.config.adaptive_sizing,
                'prefer_threading': self.config.prefer_threading
            }
        }
    
    def reset_performance_history(self):
        """Reset performance tracking history."""
        with self._lock:
            self._performance_history.clear()
            self._current_batch_size = self.config.min_batch_size
            self._last_optimization = 0.0


# Audio-specific batch processing functions
def process_audio_chunks_smart_batch(chunks: List[AudioBuffer],
                                   processor_func: Callable,
                                   config: BatchProcessingConfig = None) -> List[Any]:
    """Process audio chunks with smart batching."""
    
    processor = AdaptiveBatchProcessor(config)
    
    # Group chunks by similar characteristics for optimal batching
    chunk_groups = _group_chunks_by_characteristics(chunks)
    
    all_results = []
    
    for group_name, chunk_group in chunk_groups.items():
        print(f"📦 Processing {group_name} group: {len(chunk_group)} chunks")
        
        # Calculate optimal batch size for this group
        if chunk_group:
            sample_chunk = chunk_group[0]
            estimated_memory = sample_chunk.memory_usage_mb
            optimal_batch_size = processor.calculate_optimal_batch_size(estimated_memory)
        else:
            optimal_batch_size = processor.config.min_batch_size
        
        # Process this group
        group_results = processor.process_batch_async(
            chunk_group,
            processor_func,
            batch_size=optimal_batch_size
        )
        
        all_results.extend(group_results)
    
    return all_results


def _group_chunks_by_characteristics(chunks: List[AudioBuffer]) -> Dict[str, List[AudioBuffer]]:
    """Group chunks by similar characteristics for optimal batch processing."""
    
    groups = {
        'short': [],    # < 20 seconds
        'medium': [],   # 20-40 seconds
        'long': [],     # > 40 seconds
    }
    
    for chunk in chunks:
        duration = chunk.duration
        
        if duration < 20:
            groups['short'].append(chunk)
        elif duration < 40:
            groups['medium'].append(chunk)
        else:
            groups['long'].append(chunk)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def adaptive_audio_chunking(audio_buffer: AudioBuffer,
                          config: BatchProcessingConfig = None) -> List[AudioBuffer]:
    """Adaptively chunk audio based on system resources and audio characteristics."""
    
    processor = AdaptiveBatchProcessor(config)
    chunk_duration = processor.calculate_adaptive_chunk_size(audio_buffer.duration)
    
    print(f"🎵 Adaptive chunking: {chunk_duration:.1f}s chunks for {audio_buffer.duration:.1f}s audio")
    
    # Use the in-memory audio processor for chunking
    audio_proc = get_audio_processor()
    return audio_proc.split_into_chunks(
        audio_buffer, 
        chunk_duration=chunk_duration,
        overlap_duration=chunk_duration * 0.3  # 30% overlap
    )


# Global processor instance
_global_processor = None


def get_batch_processor() -> AdaptiveBatchProcessor:
    """Get or create global batch processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = AdaptiveBatchProcessor()
    return _global_processor


@contextmanager
def smart_batch_processing(config: BatchProcessingConfig = None):
    """Context manager for smart batch processing."""
    processor = AdaptiveBatchProcessor(config)
    try:
        yield processor
    finally:
        # Optional cleanup
        pass


def optimize_batch_processing_for_system() -> BatchProcessingConfig:
    """Create optimized batch processing configuration for current system."""
    
    # Analyze system resources
    cpu_count = os.cpu_count() or 4
    
    try:
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
    except Exception:
        available_memory_gb = 4.0  # Conservative estimate
    
    device_manager = get_device_manager()
    has_gpu = device_manager.device_info['cuda_available']
    
    # Configure based on system capabilities
    if available_memory_gb > 8 and cpu_count >= 8:
        # High-end system
        config = BatchProcessingConfig(
            min_batch_size=2,
            max_batch_size=16,
            target_memory_usage_mb=512.0,
            cpu_utilization_target=0.9,
            prefer_threading=True
        )
    elif available_memory_gb > 4 and cpu_count >= 4:
        # Mid-range system
        config = BatchProcessingConfig(
            min_batch_size=1,
            max_batch_size=8,
            target_memory_usage_mb=256.0,
            cpu_utilization_target=0.8,
            prefer_threading=True
        )
    else:
        # Low-end system
        config = BatchProcessingConfig(
            min_batch_size=1,
            max_batch_size=4,
            target_memory_usage_mb=128.0,
            cpu_utilization_target=0.7,
            prefer_threading=True
        )
    
    # Adjust for GPU availability
    if has_gpu:
        config.max_batch_size = min(config.max_batch_size * 2, 32)
        config.target_memory_usage_mb *= 1.5
    
    print(f"[TARGET] Optimized batch config: batch_size={config.min_batch_size}-{config.max_batch_size}, "
          f"memory_target={config.target_memory_usage_mb:.0f}MB")
    
    return config


def get_batch_processing_recommendations() -> Dict[str, Any]:
    """Get recommendations for optimal batch processing settings."""
    
    config = optimize_batch_processing_for_system()
    processor = get_batch_processor()
    stats = processor.get_performance_stats()
    
    memory_pressure = check_memory_pressure()
    
    recommendations = {
        'optimal_config': config,
        'current_performance': stats,
        'memory_status': memory_pressure,
        'recommendations': []
    }
    
    # Generate specific recommendations
    if memory_pressure['pressure_level'] == 'high':
        recommendations['recommendations'].append(
            "Reduce batch size and chunk duration due to high memory pressure"
        )
    
    if stats.get('recent_avg_throughput', 0) < 1.0:
        recommendations['recommendations'].append(
            "Consider enabling GPU acceleration or reducing chunk complexity"
        )
    
    if stats.get('recent_avg_processing_time_ms', 0) > 500:
        recommendations['recommendations'].append(
            "Enable adaptive sizing and increase parallelism"
        )
    
    return recommendations