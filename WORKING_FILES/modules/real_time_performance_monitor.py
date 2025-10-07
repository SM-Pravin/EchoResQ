"""
Real-time performance monitoring system for Emergency AI pipeline.
Tracks latency, throughput, memory usage, and automatically adjusts processing parameters
for optimal performance targeting <300ms per chunk.
"""

import time
import threading
import statistics
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta
import json
import os

from modules.memory_management import get_memory_manager, check_memory_pressure
from modules.model_loader import get_device_manager
from modules.smart_batch_processing import get_batch_processor


class PerformanceMetric(NamedTuple):
    """Single performance measurement."""
    timestamp: float
    metric_name: str
    value: float
    metadata: Dict[str, Any] = {}


@dataclass
class PerformanceTarget:
    """Performance target configuration."""
    name: str
    target_value: float
    tolerance: float = 0.1  # 10% tolerance by default
    unit: str = "ms"
    direction: str = "lower"  # "lower" means better when value is lower
    
    def is_met(self, current_value: float) -> bool:
        """Check if performance target is met."""
        if self.direction == "lower":
            return current_value <= self.target_value * (1 + self.tolerance)
        else:  # "higher" direction
            return current_value >= self.target_value * (1 - self.tolerance)
    
    def deviation_percent(self, current_value: float) -> float:
        """Calculate deviation from target as percentage."""
        if self.target_value == 0:
            return 0.0
        return ((current_value - self.target_value) / self.target_value) * 100


@dataclass
class SystemOptimizationRule:
    """Rule for automatic system optimization."""
    name: str
    condition: Callable[[Dict[str, float]], bool]
    action: Callable[[Dict[str, Any]], None]
    cooldown_seconds: float = 60.0
    description: str = ""
    last_triggered: float = field(default=0.0)
    
    def can_trigger(self) -> bool:
        """Check if rule can be triggered (respecting cooldown)."""
        return time.time() - self.last_triggered >= self.cooldown_seconds
    
    def trigger(self, current_metrics: Dict[str, float], system_state: Dict[str, Any]):
        """Trigger the optimization rule."""
        if self.can_trigger() and self.condition(current_metrics):
            try:
                self.action(system_state)
                self.last_triggered = time.time()
                return True
            except Exception as e:
                print(f"[ERROR] Failed to execute optimization rule '{self.name}': {e}")
                return False
        return False


class RealTimePerformanceMonitor:
    """Real-time performance monitoring and optimization system."""
    
    def __init__(self, 
                 max_history_size: int = 1000,
                 monitoring_interval: float = 1.0,
                 optimization_interval: float = 30.0):
        
        self.max_history_size = max_history_size
        self.monitoring_interval = monitoring_interval
        self.optimization_interval = optimization_interval
        
        # Metrics storage
        self._metrics_history: deque = deque(maxlen=max_history_size)
        self._current_metrics: Dict[str, float] = {}
        self._metric_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'count': 0, 'sum': 0.0, 'min': float('inf'), 'max': 0.0, 'avg': 0.0
        })
        
        # Performance targets
        self.targets = {
            'chunk_processing_latency_ms': PerformanceTarget(
                name='Chunk Processing Latency',
                target_value=300.0,
                tolerance=0.2,  # 20% tolerance (360ms max)
                unit='ms',
                direction='lower'
            ),
            'total_pipeline_latency_ms': PerformanceTarget(
                name='Total Pipeline Latency',
                target_value=500.0,
                tolerance=0.3,
                unit='ms',
                direction='lower'
            ),
            'throughput_chunks_per_sec': PerformanceTarget(
                name='Throughput',
                target_value=2.0,
                tolerance=0.2,
                unit='chunks/sec',
                direction='higher'
            ),
            'memory_usage_mb': PerformanceTarget(
                name='Memory Usage',
                target_value=512.0,
                tolerance=0.5,  # 50% tolerance for memory
                unit='MB',
                direction='lower'
            ),
            'cpu_utilization_percent': PerformanceTarget(
                name='CPU Utilization',
                target_value=80.0,
                tolerance=0.25,
                unit='%',
                direction='lower'
            )
        }
        
        # System optimization rules
        self.optimization_rules = self._create_optimization_rules()
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._optimization_thread = None
        self._last_optimization = 0.0
        self._system_state = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance alerts
        self._alerts: List[Dict[str, Any]] = []
        self._alert_thresholds = {
            'critical_latency_ms': 600.0,  # 2x target
            'critical_memory_mb': 1024.0,  # 2x target
            'critical_cpu_percent': 95.0
        }
        
        print("[SEARCH] Real-time performance monitor initialized")
    
    def _create_optimization_rules(self) -> List[SystemOptimizationRule]:
        """Create system optimization rules."""
        rules = []
        
        # Rule 1: High latency optimization
        def high_latency_condition(metrics: Dict[str, float]) -> bool:
            return metrics.get('chunk_processing_latency_ms', 0) > 400
        
        def reduce_latency_action(system_state: Dict[str, Any]):
            print("[TARGET] Auto-optimization: Reducing latency by decreasing batch size")
            batch_processor = get_batch_processor()
            if hasattr(batch_processor, '_current_batch_size'):
                batch_processor._current_batch_size = max(
                    batch_processor._current_batch_size - 1,
                    batch_processor.config.min_batch_size
                )
        
        rules.append(SystemOptimizationRule(
            name="high_latency_reduction",
            condition=high_latency_condition,
            action=reduce_latency_action,
            cooldown_seconds=60.0,
            description="Reduce batch size when latency exceeds 400ms"
        ))
        
        # Rule 2: Memory pressure optimization
        def high_memory_condition(metrics: Dict[str, float]) -> bool:
            return metrics.get('memory_usage_mb', 0) > 800
        
        def reduce_memory_action(system_state: Dict[str, Any]):
            print("🧹 Auto-optimization: Reducing memory usage")
            memory_manager = get_memory_manager()
            if hasattr(memory_manager, 'force_cleanup'):
                memory_manager.force_cleanup()
            
            # Also reduce batch size
            batch_processor = get_batch_processor()
            if hasattr(batch_processor, '_current_batch_size'):
                batch_processor._current_batch_size = max(
                    int(batch_processor._current_batch_size * 0.7),
                    batch_processor.config.min_batch_size
                )
        
        rules.append(SystemOptimizationRule(
            name="memory_pressure_reduction",
            condition=high_memory_condition,
            action=reduce_memory_action,
            cooldown_seconds=45.0,
            description="Clean up memory and reduce batch size when usage exceeds 800MB"
        ))
        
        # Rule 3: Low throughput optimization
        def low_throughput_condition(metrics: Dict[str, float]) -> bool:
            return (metrics.get('throughput_chunks_per_sec', 0) < 1.0 and 
                    metrics.get('cpu_utilization_percent', 0) < 60)
        
        def increase_throughput_action(system_state: Dict[str, Any]):
            print("[CHART] Auto-optimization: Increasing throughput by optimizing batch size")
            batch_processor = get_batch_processor()
            if hasattr(batch_processor, '_current_batch_size'):
                current_memory = get_current_memory_usage_mb()
                if current_memory < 400:  # Safe to increase
                    batch_processor._current_batch_size = min(
                        batch_processor._current_batch_size + 1,
                        batch_processor.config.max_batch_size
                    )
        
        rules.append(SystemOptimizationRule(
            name="throughput_optimization",
            condition=low_throughput_condition,
            action=increase_throughput_action,
            cooldown_seconds=90.0,
            description="Increase batch size when throughput is low and CPU usage is low"
        ))
        
        return rules
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitoring_thread.start()
        
        # Start optimization thread
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True,
            name="AutoOptimizer"
        )
        self._optimization_thread.start()
        
        print("[ROCKET] Real-time performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2.0)
        
        if self._optimization_thread and self._optimization_thread.is_alive():
            self._optimization_thread.join(timeout=2.0)
        
        print("⏹️ Real-time performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"[ERROR] Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _optimization_loop(self):
        """Automatic optimization loop."""
        while self._monitoring_active:
            try:
                current_time = time.time()
                
                if current_time - self._last_optimization >= self.optimization_interval:
                    self._run_optimization_rules()
                    self._last_optimization = current_time
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                print(f"[ERROR] Error in optimization loop: {e}")
                time.sleep(5.0)
    
    def _collect_system_metrics(self):
        """Collect current system performance metrics."""
        try:
            # Memory metrics
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Update current metrics
            with self._lock:
                self._current_metrics.update({
                    'system_memory_used_mb': memory_info.used / (1024 * 1024),
                    'system_memory_available_mb': memory_info.available / (1024 * 1024),
                    'process_memory_mb': process_memory.rss / (1024 * 1024),
                    'memory_usage_mb': process_memory.rss / (1024 * 1024),  # Alias
                    'cpu_utilization_percent': cpu_percent,
                    'system_load_avg': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
                })
                
                # Get metrics from other components
                try:
                    batch_processor = get_batch_processor()
                    batch_stats = batch_processor.get_performance_stats()
                    
                    if batch_stats.get('recent_avg_throughput'):
                        self._current_metrics['throughput_chunks_per_sec'] = batch_stats['recent_avg_throughput']
                    
                    if batch_stats.get('recent_avg_processing_time_ms'):
                        self._current_metrics['chunk_processing_latency_ms'] = batch_stats['recent_avg_processing_time_ms']
                        
                except Exception:
                    pass  # Batch processor might not be available
                
                # Update statistics
                self._update_metric_stats()
        
        except Exception as e:
            print(f"[ERROR] Error collecting system metrics: {e}")
    
    def _update_metric_stats(self):
        """Update rolling statistics for metrics."""
        timestamp = time.time()
        
        for metric_name, value in self._current_metrics.items():
            # Add to history
            metric = PerformanceMetric(timestamp, metric_name, value)
            self._metrics_history.append(metric)
            
            # Update statistics
            stats = self._metric_stats[metric_name]
            stats['count'] += 1
            stats['sum'] += value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['avg'] = stats['sum'] / stats['count']
    
    def _check_alerts(self):
        """Check for performance alerts."""
        current_time = time.time()
        
        # Check critical latency
        latency = self._current_metrics.get('chunk_processing_latency_ms', 0)
        if latency > self._alert_thresholds['critical_latency_ms']:
            self._add_alert('critical_latency', f"Critical latency: {latency:.1f}ms", 'high')
        
        # Check critical memory
        memory = self._current_metrics.get('memory_usage_mb', 0)
        if memory > self._alert_thresholds['critical_memory_mb']:
            self._add_alert('critical_memory', f"Critical memory usage: {memory:.1f}MB", 'high')
        
        # Check critical CPU
        cpu = self._current_metrics.get('cpu_utilization_percent', 0)
        if cpu > self._alert_thresholds['critical_cpu_percent']:
            self._add_alert('critical_cpu', f"Critical CPU usage: {cpu:.1f}%", 'medium')
        
        # Clean old alerts (older than 5 minutes)
        cutoff_time = current_time - 300
        self._alerts = [alert for alert in self._alerts if alert['timestamp'] > cutoff_time]
    
    def _add_alert(self, alert_type: str, message: str, severity: str):
        """Add a performance alert."""
        # Check if similar alert was recently added
        recent_alerts = [a for a in self._alerts[-10:] if a['type'] == alert_type]
        if recent_alerts:
            last_alert_time = max(a['timestamp'] for a in recent_alerts)
            if time.time() - last_alert_time < 60:  # Don't spam alerts
                return
        
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self._alerts.append(alert)
        print(f"[WARNING] Performance Alert [{severity.upper()}]: {message}")
    
    def _run_optimization_rules(self):
        """Execute automatic optimization rules."""
        with self._lock:
            current_metrics = self._current_metrics.copy()
        
        triggered_rules = []
        
        for rule in self.optimization_rules:
            try:
                if rule.trigger(current_metrics, self._system_state):
                    triggered_rules.append(rule.name)
            except Exception as e:
                print(f"[ERROR] Error executing rule '{rule.name}': {e}")
        
        if triggered_rules:
            print(f"[TARGET] Auto-optimization triggered rules: {', '.join(triggered_rules)}")
    
    def record_processing_time(self, operation: str, duration_ms: float, metadata: Dict[str, Any] = None):
        """Record processing time for a specific operation."""
        with self._lock:
            if operation == 'chunk_processing':
                self._current_metrics['chunk_processing_latency_ms'] = duration_ms
            elif operation == 'total_pipeline':
                self._current_metrics['total_pipeline_latency_ms'] = duration_ms
            
            # Also add to history
            metric = PerformanceMetric(
                timestamp=time.time(),
                metric_name=f"{operation}_latency_ms",
                value=duration_ms,
                metadata=metadata or {}
            )
            self._metrics_history.append(metric)
    
    def record_throughput(self, items_processed: int, duration_seconds: float):
        """Record throughput measurement."""
        throughput = items_processed / duration_seconds if duration_seconds > 0 else 0
        
        with self._lock:
            self._current_metrics['throughput_chunks_per_sec'] = throughput
            
            metric = PerformanceMetric(
                timestamp=time.time(),
                metric_name='throughput_chunks_per_sec',
                value=throughput,
                metadata={'items_processed': items_processed, 'duration_seconds': duration_seconds}
            )
            self._metrics_history.append(metric)
    
    def get_current_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        with self._lock:
            current_metrics = self._current_metrics.copy()
        
        summary = {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'target_compliance': {},
            'recent_alerts': self._alerts[-5:],  # Last 5 alerts
            'system_recommendations': []
        }
        
        # Check target compliance
        for metric_name, target in self.targets.items():
            current_value = current_metrics.get(metric_name, 0)
            summary['target_compliance'][metric_name] = {
                'current_value': current_value,
                'target_value': target.target_value,
                'unit': target.unit,
                'is_met': target.is_met(current_value),
                'deviation_percent': target.deviation_percent(current_value)
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(current_metrics)
        summary['system_recommendations'] = recommendations
        
        return summary
    
    def get_performance_trends(self, lookback_minutes: int = 15) -> Dict[str, Any]:
        """Get performance trends over specified time period."""
        cutoff_time = time.time() - (lookback_minutes * 60)
        
        with self._lock:
            recent_metrics = [m for m in self._metrics_history if m.timestamp >= cutoff_time]
        
        trends = {}
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        # Calculate trends
        for metric_name, values in metric_groups.items():
            if len(values) >= 2:
                trends[metric_name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'median': statistics.median(values),
                    'trend': 'improving' if values[-1] < values[0] else 'degrading' if values[-1] > values[0] else 'stable'
                }
                
                if len(values) >= 3:
                    trends[metric_name]['std_dev'] = statistics.stdev(values)
        
        return {
            'lookback_minutes': lookback_minutes,
            'trends': trends,
            'summary': {
                'total_measurements': len(recent_metrics),
                'metrics_tracked': len(metric_groups)
            }
        }
    
    def _generate_recommendations(self, current_metrics: Dict[str, float]) -> List[str]:
        """Generate system optimization recommendations."""
        recommendations = []
        
        latency = current_metrics.get('chunk_processing_latency_ms', 0)
        memory = current_metrics.get('memory_usage_mb', 0)
        cpu = current_metrics.get('cpu_utilization_percent', 0)
        throughput = current_metrics.get('throughput_chunks_per_sec', 0)
        
        # Latency recommendations
        if latency > 400:
            recommendations.append("Consider reducing batch size or enabling GPU acceleration")
        elif latency > 300:
            recommendations.append("Monitor latency closely - approaching target limit")
        
        # Memory recommendations
        if memory > 700:
            recommendations.append("High memory usage detected - consider memory cleanup")
        
        # CPU recommendations
        if cpu < 40 and throughput < 1.5:
            recommendations.append("Low CPU utilization - can increase parallelism")
        elif cpu > 90:
            recommendations.append("High CPU usage - consider reducing concurrent operations")
        
        # Throughput recommendations
        if throughput < 1.0:
            recommendations.append("Low throughput - check for bottlenecks in pipeline")
        
        return recommendations
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics history to file."""
        try:
            with self._lock:
                data = {
                    'export_timestamp': time.time(),
                    'targets': {name: {
                        'target_value': target.target_value,
                        'unit': target.unit,
                        'direction': target.direction
                    } for name, target in self.targets.items()},
                    'metrics_history': [
                        {
                            'timestamp': m.timestamp,
                            'metric_name': m.metric_name,
                            'value': m.value,
                            'metadata': m.metadata
                        } for m in self._metrics_history
                    ],
                    'current_metrics': self._current_metrics.copy(),
                    'alerts': self._alerts.copy()
                }
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            print(f"[DASHBOARD] Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to export metrics: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all metrics and history."""
        with self._lock:
            self._metrics_history.clear()
            self._current_metrics.clear()
            self._metric_stats.clear()
            self._alerts.clear()
            
            # Reset optimization rule cooldowns
            for rule in self.optimization_rules:
                rule.last_triggered = 0.0
        
        print("🔄 Performance metrics reset")


# Convenience functions and context managers

@contextmanager
def performance_tracking(monitor: RealTimePerformanceMonitor, operation: str):
    """Context manager for tracking operation performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        monitor.record_processing_time(operation, duration_ms)


def get_current_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_system_performance_snapshot() -> Dict[str, Any]:
    """Get quick system performance snapshot."""
    try:
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'timestamp': time.time(),
            'memory_used_mb': memory_info.used / (1024 * 1024),
            'memory_available_mb': memory_info.available / (1024 * 1024),
            'memory_percent': memory_info.percent,
            'cpu_percent': cpu_percent,
            'process_memory_mb': get_current_memory_usage_mb()
        }
    except Exception as e:
        return {'error': str(e), 'timestamp': time.time()}


# Global monitor instance
_global_monitor = None


def get_performance_monitor() -> RealTimePerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealTimePerformanceMonitor()
    return _global_monitor


def start_global_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    return monitor


def stop_global_monitoring():
    """Stop global performance monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


def get_performance_dashboard() -> str:
    """Get formatted performance dashboard string."""
    monitor = get_performance_monitor()
    summary = monitor.get_current_performance_summary()
    trends = monitor.get_performance_trends(lookback_minutes=10)
    
    dashboard = []
    dashboard.append("=" * 60)
    dashboard.append("[SEARCH] EMERGENCY AI PERFORMANCE DASHBOARD")
    dashboard.append("=" * 60)
    
    # Current metrics
    dashboard.append("\n[DASHBOARD] CURRENT METRICS:")
    for metric_name, value in summary['current_metrics'].items():
        if metric_name in monitor.targets:
            target = monitor.targets[metric_name]
            compliance = summary['target_compliance'][metric_name]
            status = "[OK]" if compliance['is_met'] else "[ERROR]"
            deviation = compliance['deviation_percent']
            
            dashboard.append(f"  {status} {target.name}: {value:.1f}{target.unit} "
                           f"(target: {target.target_value:.1f}{target.unit}, "
                           f"deviation: {deviation:+.1f}%)")
    
    # Recent alerts
    if summary['recent_alerts']:
        dashboard.append("\n[WARNING] RECENT ALERTS:")
        for alert in summary['recent_alerts'][-3:]:
            alert_time = datetime.fromtimestamp(alert['timestamp']).strftime("%H:%M:%S")
            dashboard.append(f"  [{alert_time}] {alert['message']}")
    
    # Recommendations
    if summary['system_recommendations']:
        dashboard.append("\n💡 RECOMMENDATIONS:")
        for rec in summary['system_recommendations'][:3]:
            dashboard.append(f"  • {rec}")
    
    # Performance trends
    if trends['trends']:
        dashboard.append("\n[CHART] TRENDS (Last 10 min):")
        for metric_name, trend_data in list(trends['trends'].items())[:3]:
            if metric_name.endswith('_ms') or metric_name.endswith('_per_sec'):
                trend_icon = "[CHART]" if trend_data['trend'] == 'improving' else "📉" if trend_data['trend'] == 'degrading' else "➡️"
                dashboard.append(f"  {trend_icon} {metric_name}: avg {trend_data['avg']:.1f}, "
                               f"range {trend_data['min']:.1f}-{trend_data['max']:.1f}")
    
    dashboard.append("=" * 60)
    
    return "\n".join(dashboard)


# Integration with existing pipeline
def setup_pipeline_monitoring():
    """Set up performance monitoring for the emergency AI pipeline."""
    monitor = get_performance_monitor()
    
    # Customize targets for emergency AI pipeline
    monitor.targets['chunk_processing_latency_ms'].target_value = 250.0  # Stricter target
    monitor.targets['total_pipeline_latency_ms'].target_value = 400.0
    monitor.targets['throughput_chunks_per_sec'].target_value = 3.0      # Higher throughput target
    
    # Start monitoring
    monitor.start_monitoring()
    
    print("[TARGET] Emergency AI pipeline monitoring configured and started")
    return monitor