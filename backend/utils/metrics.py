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

# ============== MODEL MONITORING ==============

model_load_total = Counter(
    "argus_model_load_total",
    "Total model load attempts",
    ["model_name", "status"]
)

model_inference_total = Counter(
    "argus_model_inference_total",
    "Total model inference calls",
    ["model_name", "status"]
)

model_confidence_histogram = Histogram(
    "argus_model_confidence",
    "Model prediction confidence distribution",
    ["model_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

model_latency_seconds = Histogram(
    "argus_model_latency_seconds",
    "Model inference latency with percentiles",
    ["model_name"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
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

model_loaded = Gauge(
    "argus_model_loaded",
    "Whether a model is currently loaded",
    ["model_name"]
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


# ============== MODEL MONITORING HELPERS ==============

def record_model_load(model_name: str, success: bool) -> None:
    """Record model load attempt."""
    status = "success" if success else "failure"
    model_load_total.labels(model_name=model_name, status=status).inc()
    model_loaded.labels(model_name=model_name).set(1 if success else 0)


def record_model_inference(
    model_name: str,
    success: bool,
    latency_seconds: float,
    confidence: Optional[float] = None
) -> None:
    """
    Record model inference metrics.
    
    Args:
        model_name: Name of the model
        success: Whether inference succeeded
        latency_seconds: Inference latency in seconds
        confidence: Optional prediction confidence (0-1)
    """
    status = "success" if success else "failure"
    model_inference_total.labels(model_name=model_name, status=status).inc()
    model_latency_seconds.labels(model_name=model_name).observe(latency_seconds)
    
    if confidence is not None and success:
        model_confidence_histogram.labels(model_name=model_name).observe(confidence)


def record_model_unload(model_name: str) -> None:
    """Record model unload event."""
    model_loaded.labels(model_name=model_name).set(0)
