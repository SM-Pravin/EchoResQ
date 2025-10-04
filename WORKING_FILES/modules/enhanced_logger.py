"""
Enhanced structured logging system for Emergency AI with performance metrics.
Uses loguru for structured JSON logging with per-module confidence and latency tracking.
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from modules.config_manager import get_config_manager


@dataclass
class PerformanceMetrics:
    """Performance metrics for module execution."""
    module_name: str
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    confidence: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.metadata is None:
            data['metadata'] = {}
        return data


@dataclass
class CallAnalysisLog:
    """Complete call analysis logging structure."""
    call_id: str
    timestamp: str
    caller_id: Optional[str]
    duration_seconds: float
    transcript: str
    final_emotion: str
    distress_score: float
    distress_level: str
    confidence: float
    keywords_detected: List[str]
    sound_events: List[Dict[str, Any]]
    processing_metrics: List[PerformanceMetrics]
    total_processing_time_ms: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['processing_metrics'] = [m.to_dict() if isinstance(m, PerformanceMetrics) else m 
                                    for m in self.processing_metrics]
        return data


class PerformanceTracker:
    """Thread-safe performance tracking for modules."""
    
    def __init__(self):
        self._metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._active_operations: Dict[str, float] = {}
    
    def start_operation(self, module_name: str, operation: str) -> str:
        """Start tracking an operation."""
        operation_id = f"{module_name}.{operation}.{int(time.time() * 1000000)}"
        with self._lock:
            self._active_operations[operation_id] = time.perf_counter()
        return operation_id
    
    def end_operation(self, operation_id: str, 
                     confidence: Optional[float] = None,
                     input_size: Optional[int] = None,
                     output_size: Optional[int] = None,
                     success: bool = True,
                     error_message: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """End tracking an operation and return metrics."""
        end_time = time.perf_counter()
        
        with self._lock:
            start_time = self._active_operations.pop(operation_id, end_time)
        
        duration_ms = (end_time - start_time) * 1000
        
        # Parse operation details from ID
        parts = operation_id.split('.')
        module_name = parts[0] if len(parts) > 0 else "unknown"
        operation = parts[1] if len(parts) > 1 else "unknown"
        
        # Get system metrics if available
        memory_usage_mb = None
        cpu_percent = None
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
            except Exception:
                pass
        
        metrics = PerformanceMetrics(
            module_name=module_name,
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            confidence=confidence,
            input_size=input_size,
            output_size=output_size,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics.append(metrics)
        
        return metrics
    
    def get_metrics(self, module_name: Optional[str] = None) -> List[PerformanceMetrics]:
        """Get performance metrics, optionally filtered by module."""
        with self._lock:
            if module_name:
                return [m for m in self._metrics if m.module_name == module_name]
            return self._metrics.copy()
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        with self._lock:
            self._metrics.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if not self._metrics:
                return {}
            
            by_module = {}
            for metric in self._metrics:
                if metric.module_name not in by_module:
                    by_module[metric.module_name] = {
                        'total_operations': 0,
                        'total_duration_ms': 0,
                        'avg_duration_ms': 0,
                        'min_duration_ms': float('inf'),
                        'max_duration_ms': 0,
                        'success_rate': 0,
                        'avg_confidence': 0,
                        'operations': {}
                    }
                
                module_stats = by_module[metric.module_name]
                module_stats['total_operations'] += 1
                module_stats['total_duration_ms'] += metric.duration_ms
                module_stats['min_duration_ms'] = min(module_stats['min_duration_ms'], metric.duration_ms)
                module_stats['max_duration_ms'] = max(module_stats['max_duration_ms'], metric.duration_ms)
                
                if metric.operation not in module_stats['operations']:
                    module_stats['operations'][metric.operation] = {
                        'count': 0,
                        'total_duration_ms': 0,
                        'avg_duration_ms': 0
                    }
                
                op_stats = module_stats['operations'][metric.operation]
                op_stats['count'] += 1
                op_stats['total_duration_ms'] += metric.duration_ms
                op_stats['avg_duration_ms'] = op_stats['total_duration_ms'] / op_stats['count']
            
            # Calculate averages and success rates
            for module_name, stats in by_module.items():
                if stats['total_operations'] > 0:
                    stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['total_operations']
                    
                    successful = sum(1 for m in self._metrics 
                                   if m.module_name == module_name and m.success)
                    stats['success_rate'] = successful / stats['total_operations']
                    
                    confidences = [m.confidence for m in self._metrics 
                                 if m.module_name == module_name and m.confidence is not None]
                    if confidences:
                        stats['avg_confidence'] = sum(confidences) / len(confidences)
            
            return by_module


class EnhancedLogger:
    """Enhanced logging system with structured JSON output and performance tracking."""
    
    def __init__(self, config=None):
        self.config = config or get_config_manager().config
        self.performance_tracker = PerformanceTracker()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup loguru logging with JSON structure."""
        if not LOGURU_AVAILABLE:
            print("[WARNING] Loguru not available, falling back to standard logging")
            return
        
        # Remove default handler
        logger.remove()
        
        # Get logging config
        log_config = self.config.logging
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/emergency_ai.log')
        max_size = f"{log_config.get('max_file_size_mb', 50)} MB"
        backup_count = log_config.get('backup_count', 3)
        console_output = log_config.get('console_output', True)
        
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Console handler with rich formatting
        if console_output:
            logger.add(
                sys.stdout,
                level=log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # JSON file handler for structured logging
        logger.add(
            log_file,
            level=log_level,
            format=self._json_formatter,
            rotation=max_size,
            retention=backup_count,
            compression="gz",
            serialize=False,  # We handle JSON ourselves
            backtrace=True,
            diagnose=True
        )
        
        # Performance metrics file
        perf_log_file = log_file.replace('.log', '_performance.jsonl')
        logger.add(
            perf_log_file,
            level="INFO",
            format="{message}",
            filter=lambda record: record["extra"].get("performance_metric", False),
            rotation=max_size,
            retention=backup_count,
            compression="gz"
        )
    
    def _json_formatter(self, record):
        """Custom JSON formatter for structured logging."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "process_id": record["process"].id,
            "thread_id": record["thread"].id,
        }
        
        # Add extra fields
        if record["extra"]:
            log_entry["extra"] = record["extra"]
        
        # Add exception info if present
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry, default=str)
    
    @contextmanager
    def track_operation(self, module_name: str, operation: str, **kwargs):
        """Context manager for tracking operation performance."""
        operation_id = self.performance_tracker.start_operation(module_name, operation)
        
        try:
            yield operation_id
            # Operation succeeded
            metrics = self.performance_tracker.end_operation(
                operation_id, success=True, **kwargs
            )
            self.log_performance_metric(metrics)
            
        except Exception as e:
            # Operation failed
            metrics = self.performance_tracker.end_operation(
                operation_id, success=False, error_message=str(e), **kwargs
            )
            self.log_performance_metric(metrics)
            raise
    
    def log_performance_metric(self, metric: PerformanceMetrics):
        """Log a performance metric."""
        if LOGURU_AVAILABLE:
            logger.bind(performance_metric=True).info(json.dumps(metric.to_dict(), default=str))
    
    def log_call_analysis(self, call_log: CallAnalysisLog):
        """Log complete call analysis results."""
        logger.info(
            "Call analysis completed",
            extra={
                "call_analysis": call_log.to_dict(),
                "call_id": call_log.call_id,
                "distress_score": call_log.distress_score,
                "processing_time_ms": call_log.total_processing_time_ms
            }
        )
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        logger.debug(message, **kwargs)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance tracking summary."""
        return self.performance_tracker.get_summary()


# Global logger instance
_enhanced_logger: Optional[EnhancedLogger] = None


def get_logger() -> EnhancedLogger:
    """Get or create global enhanced logger instance."""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger()
    return _enhanced_logger


def setup_logging(config=None):
    """Setup logging system with configuration."""
    global _enhanced_logger
    _enhanced_logger = EnhancedLogger(config)
    return _enhanced_logger


# Convenience functions
def track_operation(module_name: str, operation: str, **kwargs):
    """Context manager for tracking operation performance."""
    return get_logger().track_operation(module_name, operation, **kwargs)


def log_info(message: str, **kwargs):
    """Log info message."""
    get_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message."""
    get_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log error message."""
    get_logger().error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message."""
    get_logger().debug(message, **kwargs)


def log_performance_metric(metric: PerformanceMetrics):
    """Log a performance metric."""
    get_logger().log_performance_metric(metric)


def log_call_analysis(call_log: CallAnalysisLog):
    """Log complete call analysis results."""
    get_logger().log_call_analysis(call_log)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance tracking summary."""
    return get_logger().get_performance_summary()


# Legacy compatibility functions
def log_call(caller_id: str, transcript: str, final_emotion: str, 
            distress_token: str, scores: Dict[str, Any], reason: str = ""):
    """Legacy logging function for backward compatibility."""
    call_log = CallAnalysisLog(
        call_id=f"call_{int(time.time())}",
        timestamp=datetime.now().isoformat(),
        caller_id=caller_id,
        duration_seconds=0.0,
        transcript=transcript,
        final_emotion=final_emotion,
        distress_score=scores.get('distress_score', 0.0),
        distress_level=distress_token,
        confidence=scores.get('confidence', 0.0),
        keywords_detected=scores.get('keywords', []),
        sound_events=scores.get('sounds', []),
        processing_metrics=[],
        total_processing_time_ms=0.0,
        success=True
    )
    log_call_analysis(call_log)