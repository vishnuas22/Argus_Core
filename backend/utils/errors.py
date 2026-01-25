"""
Argus Core - Custom Exception Classes
=====================================
Structured error handling with HTTP status code mapping.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - utils/errors.py
"""

from typing import Optional, Dict, Any


class ArgusError(Exception):
    """
    Base exception for all Argus errors.
    
    All custom exceptions inherit from this class to enable
    consistent error handling and HTTP response mapping.
    """
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class InvalidFileError(ArgusError):
    """Raised when uploaded file fails validation."""
    status_code = 400
    error_code = "INVALID_FILE"
    
    def __init__(self, message: str = "Invalid file", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class AnalysisNotFoundError(ArgusError):
    """Raised when requested analysis doesn't exist."""
    status_code = 404
    error_code = "ANALYSIS_NOT_FOUND"
    
    def __init__(self, analysis_id: str):
        super().__init__(
            f"Analysis not found: {analysis_id}",
            {"analysis_id": analysis_id}
        )


class ModelLoadError(ArgusError):
    """Raised when ML model fails to load."""
    status_code = 500
    error_code = "MODEL_LOAD_FAILED"
    
    def __init__(self, model_name: str, reason: str = "Unknown error"):
        super().__init__(
            f"Failed to load model: {model_name}",
            {"model_name": model_name, "reason": reason}
        )


class InferenceError(ArgusError):
    """Raised when model inference fails."""
    status_code = 500
    error_code = "INFERENCE_FAILED"
    
    def __init__(self, model_name: str, reason: str = "Unknown error"):
        super().__init__(
            f"Inference failed for model: {model_name}",
            {"model_name": model_name, "reason": reason}
        )


class StorageError(ArgusError):
    """Raised when storage operations fail."""
    status_code = 500
    error_code = "STORAGE_ERROR"
    
    def __init__(self, operation: str, reason: str = "Unknown error"):
        super().__init__(
            f"Storage operation failed: {operation}",
            {"operation": operation, "reason": reason}
        )


class ValidationError(ArgusError):
    """Raised when input validation fails."""
    status_code = 400
    error_code = "VALIDATION_ERROR"
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Validation failed for {field}: {reason}",
            {"field": field, "reason": reason}
        )


class ConfigurationError(ArgusError):
    """Raised when configuration is invalid or missing."""
    status_code = 500
    error_code = "CONFIGURATION_ERROR"
    
    def __init__(self, key: str, reason: str = "Missing or invalid"):
        super().__init__(
            f"Configuration error for {key}: {reason}",
            {"key": key, "reason": reason}
        )


class RateLimitError(ArgusError):
    """Raised when rate limit is exceeded."""
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    
    def __init__(self, limit: int, window_seconds: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            {"limit": limit, "window_seconds": window_seconds}
        )


class PreprocessingError(ArgusError):
    """Raised when media preprocessing fails."""
    status_code = 500
    error_code = "PREPROCESSING_FAILED"
    
    def __init__(self, stage: str, reason: str = "Unknown error"):
        super().__init__(
            f"Preprocessing failed at {stage}: {reason}",
            {"stage": stage, "reason": reason}
        )


class FusionError(ArgusError):
    """Raised when multi-modal fusion fails."""
    status_code = 500
    error_code = "FUSION_FAILED"
    
    def __init__(self, reason: str = "Unable to aggregate results"):
        super().__init__(
            f"Multi-modal fusion failed: {reason}",
            {"reason": reason}
        )


class ReportGenerationError(ArgusError):
    """Raised when PDF report generation fails."""
    status_code = 500
    error_code = "REPORT_GENERATION_FAILED"
    
    def __init__(self, analysis_id: str, reason: str = "Unknown error"):
        super().__init__(
            f"Report generation failed for analysis {analysis_id}",
            {"analysis_id": analysis_id, "reason": reason}
        )


class AuthenticationError(ArgusError):
    """Raised for authentication failures."""
    status_code = 401
    error_code = "AUTHENTICATION_FAILED"
    
    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(
            f"Authentication failed: {reason}",
            {"reason": reason}
        )


class AuthorizationError(ArgusError):
    """Raised for authorization failures."""
    status_code = 403
    error_code = "AUTHORIZATION_FAILED"
    
    def __init__(self, resource: str, action: str):
        super().__init__(
            f"Not authorized to {action} {resource}",
            {"resource": resource, "action": action}
        )
