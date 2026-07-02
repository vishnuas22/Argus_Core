"""
Argus Core - Forensics Module
=============================
C2PA integration, PDF report generation, and audit logging.

Implements: PRIME_ARGUS_DOCUMENT.md - Layer 7: Forensics
"""

from forensics.audit import AuditLogger, get_audit_logger, AuditEventType
from forensics.forensics import ForensicsEngine, get_forensics_engine
from forensics.report import ReportGenerator, get_report_generator

__all__ = [
    "AuditLogger",
    "get_audit_logger",
    "AuditEventType",
    "ForensicsEngine",
    "get_forensics_engine",
    "ReportGenerator",
    "get_report_generator"
]
