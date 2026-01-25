"""
Argus Core - Prometheus Metrics
===============================
Metrics collection for observability and monitoring.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - utils/metrics.py
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional

# ============== APPLICATION INFO ==============

app_info = Info(
    "argus_app",
    "Application information"
)

# ============== REQUEST COUNTERS ==============

analysis_requests_total = Counter(
    "argus_analysis_requests_total",
    "Total analysis requests",
    ["status", "modality"]
)

http_requests_total = Counter(
    "argus_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

# ============== HISTOGRAMS ==============

analysis_duration_seconds = Histogram(
    "argus_analysis_duration_seconds",
    "Analysis processing time",
    ["modality"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0)
)

inference_duration_seconds = Histogram(
    "argus_inference_duration_seconds",
    "Model inference time",
    ["model_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

http_request_duration_seconds = Histogram(
    "argus_http_request_duration_seconds",
    "HTTP request processing time",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# ============== GAUGES ==============

model_vram_usage_bytes = Gauge(
    "argus_model_vram_usage_bytes",
    "Current VRAM usage",
    ["model"]
)

active_analyses = Gauge(
    "argus_active_analyses",
    "Currently processing analyses"
)

model_cache_size = Gauge(
    "argus_model_cache_size",
    "Number of models in cache"
)

queue_size = Gauge(
    "argus_queue_size",
    "Number of jobs in queue",
    ["queue_name"]
)


# ============== HELPER FUNCTIONS ==============

def record_analysis_request(status: str, modality: str) -> None:
    """Record an analysis request metric."""
    analysis_requests_total.labels(status=status, modality=modality).inc()


def record_analysis_duration(modality: str, duration: float) -> None:
    """Record analysis duration metric."""
    analysis_duration_seconds.labels(modality=modality).observe(duration)


def record_inference_duration(model_name: str, duration: float) -> None:
    """Record model inference duration metric."""
    inference_duration_seconds.labels(model_name=model_name).observe(duration)


def record_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float
) -> None:
    """Record HTTP request metrics."""
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()
    
    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def update_vram_usage(model: str, bytes_used: int) -> None:
    """Update VRAM usage gauge."""
    model_vram_usage_bytes.labels(model=model).set(bytes_used)


def set_active_analyses(count: int) -> None:
    """Set current active analyses count."""
    active_analyses.set(count)


def set_queue_size(queue_name: str, size: int) -> None:
    """Set queue size gauge."""
    queue_size.labels(queue_name=queue_name).set(size)


def init_app_info(version: str, environment: str) -> None:
    """Initialize application info metric."""
    app_info.info({
        "version": version,
        "environment": environment,
        "service": "argus-core"
    })
