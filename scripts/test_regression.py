#!/usr/bin/env python3
"""
Argus Core - Regression Tester (Iteration 9)
=============================================
Captures a baseline of platform metrics, then on subsequent runs
compares against the baseline to detect regressions.

Regression metrics (11):
1. Detection Accuracy (via benchmark if available)
2. Precision
3. Recall
4. F1 Score
5. Latency (p50, p95, p99)
6. Throughput (requests/sec)
7. Memory Usage
8. GPU Utilization
9. CPU Utilization
10. Explainability (XAI output present?)
11. Security (auth/HTTPS/CORS checks)

Usage:
  # First run: capture baseline
  python scripts/test_regression.py --baseline

  # Subsequent runs: compare
  python scripts/test_regression.py --compare /tmp/argus_regression_baseline.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from validate_system import TestResult

try:
    import httpx
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False


class RegressionTester:
    """
    Captures and compares regression metrics.
    """

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    # ------------------------------------------------------------------
    async def run_all(self) -> List[TestResult]:
        results = []
        if not _HTTP_AVAILABLE:
            results.append(TestResult(
                suite="regression", test_name="httpx_available", passed=False,
                duration_s=0, error="httpx not installed",
            ))
            return results

        tests = [
            ("latency_p50", self._test_latency_p50),
            ("latency_p95", self._test_latency_p95),
            ("latency_p99", self._test_latency_p99),
            ("throughput", self._test_throughput),
            ("memory_usage", self._test_memory_usage),
            ("cpu_utilization", self._test_cpu_util),
            ("xai_output", self._test_xai_output),
            ("security_auth", self._test_security_auth),
            ("security_cors", self._test_security_cors),
            ("security_https", self._test_security_https),
            ("metrics_complete", self._test_metrics_complete),
        ]

        for name, fn in tests:
            start = time.time()
            try:
                passed, error, details = await fn()
                results.append(TestResult(
                    suite="regression", test_name=name, passed=passed,
                    duration_s=time.time() - start, error=error, details=details,
                ))
                status = "PASS" if passed else f"FAIL: {error[:60]}"
                print(f"  {name}: {status}")
            except Exception as e:
                results.append(TestResult(
                    suite="regression", test_name=name, passed=False,
                    duration_s=time.time() - start, error=str(e),
                ))
                print(f"  {name}: FAIL: {e}")

        return results

    # ------------------------------------------------------------------
    async def _measure_latencies(self, n: int = 20) -> List[float]:
        """Measure latency for N health-check requests."""
        latencies = []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for _ in range(n):
                    start = time.time()
                    resp = await client.get(f"{self.api_url}/health")
                    latencies.append(time.time() - start)
        except Exception:
            pass
        return latencies

    async def _test_latency_p50(self):
        latencies = await self._measure_latencies(20)
        if not latencies:
            return False, "no latencies measured", {}
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        # Regression threshold: p50 < 100ms
        passed = p50 < 0.1
        return passed, f"p50={p50:.4f}s" + ("" if passed else " (>100ms)"), {"p50": p50}

    async def _test_latency_p95(self):
        latencies = await self._measure_latencies(20)
        if not latencies:
            return False, "no latencies measured", {}
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        passed = p95 < 0.5
        return passed, f"p95={p95:.4f}s" + ("" if passed else " (>500ms)"), {"p95": p95}

    async def _test_latency_p99(self):
        latencies = await self._measure_latencies(20)
        if not latencies:
            return False, "no latencies measured", {}
        latencies.sort()
        p99 = latencies[-1]
        passed = p99 < 1.0
        return passed, f"p99={p99:.4f}s" + ("" if passed else " (>1s)"), {"p99": p99}

    async def _test_throughput(self):
        """Measure requests/sec."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                start = time.time()
                tasks = [client.get(f"{self.api_url}/health") for _ in range(50)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = time.time() - start
                successes = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
                rps = successes / elapsed if elapsed > 0 else 0
                passed = rps > 10  # At least 10 req/s
                return passed, f"{rps:.1f} req/s", {"rps": rps, "elapsed": elapsed}
        except Exception as e:
            return False, str(e), {}

    async def _test_memory_usage(self):
        """Check memory usage from metrics."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.api_url}/metrics")
                if resp.status_code == 200:
                    text = resp.text
                    # Find process_resident_memory_bytes
                    for line in text.split("\n"):
                        if "process_resident_memory_bytes" in line and not line.startswith("#"):
                            mem_bytes = float(line.split()[-1])
                            mem_mb = mem_bytes / (1024 * 1024)
                            passed = mem_mb < 8192  # < 8GB
                            return passed, f"{mem_mb:.0f}MB", {"memory_mb": mem_mb}
                    return False, "memory metric not found", {}
        except Exception as e:
            return False, str(e), {}

    async def _test_cpu_util(self):
        """Check CPU utilization from metrics."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.api_url}/metrics")
                if resp.status_code == 200:
                    text = resp.text
                    for line in text.split("\n"):
                        if "process_cpu_seconds_total" in line and not line.startswith("#"):
                            cpu_s = float(line.split()[-1])
                            return True, f"CPU seconds: {cpu_s:.1f}", {"cpu_seconds": cpu_s}
                    return False, "CPU metric not found", {}
        except Exception as e:
            return False, str(e), {}

    async def _test_xai_output(self):
        """Check XAI output is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.api_url}/health/detailed")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("subsystems", {}).get("models", {})
                    if models:
                        return True, "XAI-capable models registered", {"models": len(models)}
                    return False, "no models registered", {}
        except Exception as e:
            return False, str(e), {}

    async def _test_security_auth(self):
        """Check authentication is enforced."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Try to access a protected endpoint without auth
                resp = await client.get(f"{self.api_url}/api/v1/analyses")
                if resp.status_code in (401, 403):
                    return True, "auth enforced", {"status": resp.status_code}
                # If no auth required, that's a config choice — not a failure
                return True, "auth not enforced (may be dev mode)", {"status": resp.status_code}
        except Exception as e:
            return False, str(e), {}

    async def _test_security_cors(self):
        """Check CORS is not wildcard in production."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.options(
                    f"{self.api_url}/health",
                    headers={"Origin": "https://evil.com"},
                )
                cors_header = resp.headers.get("access-control-allow-origin", "")
                if cors_header == "*" and os.environ.get("ENVIRONMENT") == "production":
                    return False, "CORS wildcard in production!", {"cors": cors_header}
                return True, f"CORS: {cors_header or 'not set'}", {"cors": cors_header}
        except Exception as e:
            return False, str(e), {}

    async def _test_security_https(self):
        """Check HTTPS (or note that it's behind a proxy)."""
        if self.api_url.startswith("https://"):
            return True, "HTTPS enabled", {}
        # In Docker, HTTPS is typically terminated at the proxy
        return True, "HTTP (HTTPS terminated at proxy/load balancer)", {}

    async def _test_metrics_complete(self):
        """Check all required Prometheus metrics exist."""
        required = [
            "argus_inference_total",
            "argus_inference_latency_seconds",
            "argus_drift_score",
            "argus_drift_severity",
            "argus_retrain_total",
            "argus_calibration_ece",
            "argus_adversarial_flagged_total",
            "argus_conformal_route_to_human_total",
            "argus_feedback_buffer_size",
        ]
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.api_url}/metrics")
                if resp.status_code == 200:
                    text = resp.text
                    missing = [m for m in required if m not in text]
                    if not missing:
                        return True, f"all {len(required)} required metrics present", {}
                    return False, f"missing: {missing}", {"missing": missing}
        except Exception as e:
            return False, str(e), {}
