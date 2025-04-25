from typing import Optional, Dict, Any
from contextlib import contextmanager
import time
from prometheus_client import Counter, Histogram, CollectorRegistry, push_to_gateway
from app.core.config import settings

class MetricsClient:
    """Client for collecting and reporting metrics."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.counters = {
            'child_metrics_success': Counter(
                'child_metrics_success_total',
                'Number of successful child metrics retrievals',
                registry=self.registry
            ),
            'child_metrics_validation_error': Counter(
                'child_metrics_validation_error_total',
                'Number of validation errors in child metrics',
                registry=self.registry
            ),
            'child_metrics_db_error': Counter(
                'child_metrics_db_error_total',
                'Number of database errors in child metrics',
                registry=self.registry
            ),
            'child_metrics_unknown_error': Counter(
                'child_metrics_unknown_error_total',
                'Number of unknown errors in child metrics',
                registry=self.registry
            ),
            'get_events_success': Counter(
                'get_events_success_total',
                'Number of successful event retrievals',
                registry=self.registry
            ),
            'get_events_validation_error': Counter(
                'get_events_validation_error_total',
                'Number of validation errors in event retrieval',
                registry=self.registry
            ),
            'get_events_db_error': Counter(
                'get_events_db_error_total',
                'Number of database errors in event retrieval',
                registry=self.registry
            ),
            'get_events_unknown_error': Counter(
                'get_events_unknown_error_total',
                'Number of unknown errors in event retrieval',
                registry=self.registry
            )
        }
        
        self.histograms = {
            'child_metrics_duration': Histogram(
                'child_metrics_duration_seconds',
                'Time spent retrieving child metrics',
                registry=self.registry
            ),
            'get_events_duration': Histogram(
                'get_events_duration_seconds',
                'Time spent retrieving events',
                registry=self.registry
            )
        }
    
    def increment(self, metric: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric: Name of the metric to increment
            value: Value to increment by (default: 1)
            labels: Optional labels for the metric
        """
        if metric in self.counters:
            if labels:
                self.counters[metric].labels(**labels).inc(value)
            else:
                self.counters[metric].inc(value)
    
    @contextmanager
    def timer(self, metric: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            metric: Name of the metric to time
            labels: Optional labels for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if metric in self.histograms:
                if labels:
                    self.histograms[metric].labels(**labels).observe(duration)
                else:
                    self.histograms[metric].observe(duration)
    
    def push_metrics(self) -> None:
        """Push metrics to Prometheus gateway."""
        if settings.PROMETHEUS_GATEWAY_URL:
            push_to_gateway(
                settings.PROMETHEUS_GATEWAY_URL,
                job='metrics_service',
                registry=self.registry
            )

# Create singleton instance
metrics_client = MetricsClient() 