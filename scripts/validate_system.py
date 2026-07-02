#!/usr/bin/env python3
"""
Argus Core - System Validation Orchestrator (Iteration 9)
==========================================================
Master script that runs ALL validation suites and produces a unified
report. This is the single entry point for the SYSTEM VALIDATION &
CONTINUOUS VERIFICATION PROTOCOL.

Usage:
  python scripts/validate_system.py
  python scripts/validate_system.py --suite e2e
  python scripts/validate_system.py --suite endpoints
  python scripts/validate_system.py --suite failures
  python scripts/validate_system.py --suite regression
  python scripts/validate_system.py --suite all --api-url http://localhost:8000

Suites:
  e2e         — End-to-end user flow (28 stages)
  endpoints   — REST API endpoint validation (15 scenarios per endpoint)
  failures    — Failure simulation / chaos engineering (12 failure modes)
  regression  — Regression baseline + comparison (11 metrics)
  unit        — Unit tests (pytest)
  all         — Run everything (default)

Output:
  /tmp/argus_validation_report.json — structured report
  /tmp/argus_validation_report.md   — human-readable markdown
  stdout                            — live progress
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    suite: str
    test_name: str
    passed: bool
    duration_s: float
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    api_url: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_s: float
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "api_url": self.api_url,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_s": round(self.duration_s, 2),
            "results": [asdict(r) for r in self.results],
            "summary": self.summary,
        }


# =====================================================================
# Suite runners
# =====================================================================

async def run_e2e_suite(api_url: str) -> List[TestResult]:
    """Run end-to-end user flow tests."""
    print("\n" + "=" * 60)
    print("SUITE: End-to-End User Flow")
    print("=" * 60)
    results = []
    try:
        from test_end_to_end import EndToEndValidator
        validator = EndToEndValidator(api_url)
        results = await validator.run_all()
    except Exception as e:
        print(f"  E2E suite failed to start: {e}")
        results.append(TestResult(
            suite="e2e", test_name="suite_init", passed=False,
            duration_s=0, error=str(e),
        ))
    return results


async def run_endpoint_suite(api_url: str) -> List[TestResult]:
    """Run endpoint validation tests."""
    print("\n" + "=" * 60)
    print("SUITE: Endpoint Validation")
    print("=" * 60)
    results = []
    try:
        from test_endpoints import EndpointValidator
        validator = EndpointValidator(api_url)
        results = await validator.run_all()
    except Exception as e:
        print(f"  Endpoint suite failed to start: {e}")
        results.append(TestResult(
            suite="endpoints", test_name="suite_init", passed=False,
            duration_s=0, error=str(e),
        ))
    return results


async def run_failure_suite(api_url: str) -> List[TestResult]:
    """Run failure simulation tests."""
    print("\n" + "=" * 60)
    print("SUITE: Failure Simulation")
    print("=" * 60)
    results = []
    try:
        from simulate_failures import FailureSimulator
        simulator = FailureSimulator(api_url)
        results = await simulator.run_all()
    except Exception as e:
        print(f"  Failure suite failed to start: {e}")
        results.append(TestResult(
            suite="failures", test_name="suite_init", passed=False,
            duration_s=0, error=str(e),
        ))
    return results


async def run_regression_suite(api_url: str) -> List[TestResult]:
    """Run regression tests."""
    print("\n" + "=" * 60)
    print("SUITE: Regression")
    print("=" * 60)
    results = []
    try:
        from test_regression import RegressionTester
        tester = RegressionTester(api_url)
        results = await tester.run_all()
    except Exception as e:
        print(f"  Regression suite failed to start: {e}")
        results.append(TestResult(
            suite="regression", test_name="suite_init", passed=False,
            duration_s=0, error=str(e),
        ))
    return results


def run_unit_suite() -> List[TestResult]:
    """Run pytest unit tests."""
    print("\n" + "=" * 60)
    print("SUITE: Unit Tests (pytest)")
    print("=" * 60)
    results = []
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(backend_dir / "tests"),
             "--tb=short", "-q", "--json-report",
             f"--json-report-file=/tmp/argus_pytest.json"],
            capture_output=True, text=True, timeout=300,
        )
        passed = proc.returncode == 0
        results.append(TestResult(
            suite="unit", test_name="pytest", passed=passed,
            duration_s=0,  # filled from report
            details={"stdout": proc.stdout[-500:], "stderr": proc.stderr[-500:]},
        ))
        print(f"  pytest: {'PASS' if passed else 'FAIL'}")
    except subprocess.TimeoutExpired:
        results.append(TestResult(
            suite="unit", test_name="pytest", passed=False,
            duration_s=300, error="timeout (>300s)",
        ))
    except Exception as e:
        # pytest not available or tests dir missing — skip, not fail
        results.append(TestResult(
            suite="unit", test_name="pytest", passed=False,
            duration_s=0, error=str(e),
        ))
        print(f"  pytest: SKIPPED ({e})")
    return results


# =====================================================================
# Main orchestrator
# =====================================================================

async def main_async(args):
    from datetime import datetime, timezone

    api_url = args.api_url
    print("=" * 60)
    print("ARGUS CORE - SYSTEM VALIDATION")
    print("=" * 60)
    print(f"  API URL: {api_url}")
    print(f"  Suite:   {args.suite}")
    print(f"  Time:    {datetime.now(timezone.utc).isoformat()}")

    all_results: List[TestResult] = []
    start = time.time()

    # Run selected suites
    if args.suite in ("e2e", "all"):
        all_results.extend(await run_e2e_suite(api_url))
    if args.suite in ("endpoints", "all"):
        all_results.extend(await run_endpoint_suite(api_url))
    if args.suite in ("failures", "all"):
        all_results.extend(await run_failure_suite(api_url))
    if args.suite in ("regression", "all"):
        all_results.extend(await run_regression_suite(api_url))
    if args.suite in ("unit", "all"):
        all_results.extend(run_unit_suite())

    duration = time.time() - start

    # Build report
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed and r.error != "skipped")
    skipped = sum(1 for r in all_results if r.error == "skipped")

    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        api_url=api_url,
        total_tests=len(all_results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        duration_s=duration,
        results=all_results,
        summary={
            "pass_rate": round(passed / max(len(all_results), 1) * 100, 1),
            "suites_run": args.suite,
        },
    )

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total:  {report.total_tests}")
    print(f"  Passed: {report.passed}")
    print(f"  Failed: {report.failed}")
    print(f"  Skipped: {report.skipped}")
    print(f"  Pass rate: {report.summary['pass_rate']}%")
    print(f"  Duration: {report.duration_s:.1f}s")

    # Print failures
    failures = [r for r in all_results if not r.passed and r.error != "skipped"]
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURES ({len(failures)})")
        print(f"{'='*60}")
        for r in failures:
            print(f"  [{r.suite}] {r.test_name}: {r.error[:100]}")

    # Write JSON report
    with open(args.output_json, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"\nJSON report: {args.output_json}")

    # Write Markdown report
    md = generate_markdown_report(report)
    with open(args.output_md, "w") as fh:
        fh.write(md)
    print(f"Markdown report: {args.output_md}")

    # Exit code
    if failed > 0 and not args.allow_failures:
        print(f"\nEXIT 1: {failed} test(s) failed")
        return 1
    print(f"\nEXIT 0: All tests passed (or --allow-failures)")
    return 0


def generate_markdown_report(report: ValidationReport) -> str:
    """Generate a human-readable Markdown report."""
    lines = [
        "# Argus Core — System Validation Report",
        "",
        f"**Timestamp:** {report.timestamp}",
        f"**API URL:** {report.api_url}",
        f"**Duration:** {report.duration_s:.1f}s",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tests | {report.total_tests} |",
        f"| Passed | {report.passed} |",
        f"| Failed | {report.failed} |",
        f"| Skipped | {report.skipped} |",
        f"| Pass rate | {report.summary.get('pass_rate', 0)}% |",
        "",
        "## Results by Suite",
        "",
    ]
    suites = {}
    for r in report.results:
        if r.suite not in suites:
            suites[r.suite] = []
        suites[r.suite].append(r)

    for suite, results in suites.items():
        s_passed = sum(1 for r in results if r.passed)
        s_total = len(results)
        lines.append(f"### {suite} ({s_passed}/{s_total} passed)")
        lines.append("")
        lines.append("| Test | Status | Duration | Error |")
        lines.append("|------|--------|----------|-------|")
        for r in results:
            status = "✅ PASS" if r.passed else "❌ FAIL" if r.error != "skipped" else "⏭️ SKIP"
            error = r.error[:60].replace("|", "\\|") if r.error else ""
            lines.append(f"| {r.test_name} | {status} | {r.duration_s:.2f}s | {error} |")
        lines.append("")

    if report.failed > 0:
        lines.append("## Failures Detail")
        lines.append("")
        for r in report.results:
            if not r.passed and r.error != "skipped":
                lines.append(f"### [{r.suite}] {r.test_name}")
                lines.append(f"```\n{r.error}\n```")
                lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Argus system validation orchestrator")
    parser.add_argument("--suite", default="all",
                        choices=["all", "e2e", "endpoints", "failures", "regression", "unit"])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output-json", default="/tmp/argus_validation_report.json")
    parser.add_argument("--output-md", default="/tmp/argus_validation_report.md")
    parser.add_argument("--allow-failures", action="store_true",
                        help="Exit 0 even if tests fail (for CI dev runs)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
