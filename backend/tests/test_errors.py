"""
Argus Core - Custom Exception Tests
=====================================
Tests for all custom exception classes in utils/errors.py.

Validates:
- Exception hierarchy (all inherit from ArgusError)
- HTTP status code mapping
- Error code strings
- Serialization to dict
- Details preservation

No mocks. Real exception instantiation and serialization.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from utils.errors import (
    ArgusError, InvalidFileError, AnalysisNotFoundError,
    ModelLoadError, InferenceError, StorageError, ValidationError,
    ConfigurationError, RateLimitError, PreprocessingError,
    FusionError, ReportGenerationError, AuthenticationError,
    AuthorizationError, XAIError,
)


class TestArgusErrorBase:
    """Test base ArgusError class."""

    def test_base_error_is_exception(self) -> None:
        assert issubclass(ArgusError, Exception)

    def test_base_error_default_status(self) -> None:
        err = ArgusError("test error")
        assert err.status_code == 500
        assert err.error_code == "INTERNAL_ERROR"

    def test_base_error_message(self) -> None:
        err = ArgusError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"

    def test_base_error_details(self) -> None:
        err = ArgusError("error", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_base_error_default_details(self) -> None:
        err = ArgusError("error")
        assert err.details == {}

    def test_to_dict(self) -> None:
        err = ArgusError("test error", details={"field": "value"})
        result = err.to_dict()
        assert result["error_code"] == "INTERNAL_ERROR"
        assert result["message"] == "test error"
        assert result["details"] == {"field": "value"}

    def test_to_dict_empty_details(self) -> None:
        err = ArgusError("error")
        result = err.to_dict()
        assert result["details"] == {}


class TestExceptionHierarchy:
    """Test that all exceptions inherit from ArgusError."""

    @pytest.mark.parametrize("exc_class", [
        InvalidFileError, AnalysisNotFoundError, ModelLoadError,
        InferenceError, StorageError, ValidationError, ConfigurationError,
        RateLimitError, PreprocessingError, FusionError,
        ReportGenerationError, AuthenticationError, AuthorizationError, XAIError,
    ])
    def test_inherits_from_argus_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, ArgusError)

    def test_exceptions_catchable_as_argus_error(self) -> None:
        """All exceptions should be catchable as ArgusError."""
        exceptions = [
            InvalidFileError("test"),
            AnalysisNotFoundError("test-001"),
            ModelLoadError("model"),
            InferenceError("model"),
            StorageError("operation"),
            ValidationError("field", "reason"),
            ConfigurationError("key"),
            RateLimitError(100, 60),
            PreprocessingError("stage"),
            FusionError("reason"),
            ReportGenerationError("analysis_id"),
            AuthenticationError("reason"),
            AuthorizationError("resource", "action"),
            XAIError("reason"),
        ]
        for exc in exceptions:
            with pytest.raises(ArgusError):
                raise exc


class TestStatusCodes:
    """Test HTTP status code mapping for all exceptions."""

    @pytest.mark.parametrize("exc_class,expected_code", [
        (InvalidFileError, 400),
        (ValidationError, 400),
        (AuthenticationError, 401),
        (AuthorizationError, 403),
        (AnalysisNotFoundError, 404),
        (RateLimitError, 429),
        (ModelLoadError, 500),
        (InferenceError, 500),
        (StorageError, 500),
        (ConfigurationError, 500),
        (PreprocessingError, 500),
        (FusionError, 500),
        (ReportGenerationError, 500),
        (XAIError, 500),
    ])
    def test_status_code(self, exc_class: type, expected_code: int) -> None:
        err = exc_class.__new__(exc_class)
        # Initialize with minimal args
        if exc_class == AnalysisNotFoundError:
            exc_class.__init__(err, "test-001")
        elif exc_class == ModelLoadError:
            exc_class.__init__(err, "model_name")
        elif exc_class == InferenceError:
            exc_class.__init__(err, "model_name")
        elif exc_class == StorageError:
            exc_class.__init__(err, "operation")
        elif exc_class == ValidationError:
            exc_class.__init__(err, "field", "reason")
        elif exc_class == ConfigurationError:
            exc_class.__init__(err, "key")
        elif exc_class == RateLimitError:
            exc_class.__init__(err, 100, 60)
        elif exc_class == PreprocessingError:
            exc_class.__init__(err, "stage")
        elif exc_class == ReportGenerationError:
            exc_class.__init__(err, "analysis_id")
        elif exc_class == AuthorizationError:
            exc_class.__init__(err, "resource", "action")
        else:
            exc_class.__init__(err, "message")
        assert err.status_code == expected_code


class TestErrorCodes:
    """Test error code strings."""

    def test_invalid_file_code(self) -> None:
        assert InvalidFileError("test").error_code == "INVALID_FILE"

    def test_analysis_not_found_code(self) -> None:
        assert AnalysisNotFoundError("001").error_code == "ANALYSIS_NOT_FOUND"

    def test_model_load_code(self) -> None:
        assert ModelLoadError("model").error_code == "MODEL_LOAD_FAILED"

    def test_inference_code(self) -> None:
        assert InferenceError("model").error_code == "INFERENCE_FAILED"

    def test_storage_code(self) -> None:
        assert StorageError("op").error_code == "STORAGE_ERROR"

    def test_validation_code(self) -> None:
        assert ValidationError("field", "reason").error_code == "VALIDATION_ERROR"

    def test_configuration_code(self) -> None:
        assert ConfigurationError("key").error_code == "CONFIGURATION_ERROR"

    def test_rate_limit_code(self) -> None:
        assert RateLimitError(100, 60).error_code == "RATE_LIMIT_EXCEEDED"

    def test_preprocessing_code(self) -> None:
        assert PreprocessingError("stage").error_code == "PREPROCESSING_FAILED"

    def test_fusion_code(self) -> None:
        assert FusionError("reason").error_code == "FUSION_FAILED"

    def test_report_generation_code(self) -> None:
        assert ReportGenerationError("id").error_code == "REPORT_GENERATION_FAILED"

    def test_authentication_code(self) -> None:
        assert AuthenticationError("reason").error_code == "AUTHENTICATION_FAILED"

    def test_authorization_code(self) -> None:
        assert AuthorizationError("res", "act").error_code == "AUTHORIZATION_FAILED"

    def test_xai_code(self) -> None:
        assert XAIError("reason").error_code == "XAI_GENERATION_FAILED"


class TestSpecificExceptions:
    """Test specific exception constructors."""

    def test_invalid_file_default_message(self) -> None:
        err = InvalidFileError()
        assert err.message == "Invalid file"

    def test_invalid_file_with_details(self) -> None:
        err = InvalidFileError("File too large", {"size": 1000})
        assert err.message == "File too large"
        assert err.details["size"] == 1000

    def test_analysis_not_found_message(self) -> None:
        err = AnalysisNotFoundError("abc-123")
        assert "abc-123" in err.message
        assert err.details["analysis_id"] == "abc-123"

    def test_model_load_with_reason(self) -> None:
        err = ModelLoadError("efficientnet", reason="CUDA OOM")
        assert "efficientnet" in err.message
        assert err.details["reason"] == "CUDA OOM"

    def test_storage_error_details(self) -> None:
        err = StorageError("upload", reason="connection timeout")
        assert err.details["operation"] == "upload"
        assert err.details["reason"] == "connection timeout"

    def test_validation_error_details(self) -> None:
        err = ValidationError("email", "invalid format")
        assert err.details["field"] == "email"
        assert err.details["reason"] == "invalid format"

    def test_rate_limit_error_details(self) -> None:
        err = RateLimitError(100, 60)
        assert err.details["limit"] == 100
        assert err.details["window_seconds"] == 60

    def test_authorization_error_details(self) -> None:
        err = AuthorizationError("analysis", "delete")
        assert err.details["resource"] == "analysis"
        assert err.details["action"] == "delete"

    def test_report_generation_error_details(self) -> None:
        err = ReportGenerationError("analysis-001", reason="PDF generation failed")
        assert err.details["analysis_id"] == "analysis-001"
        assert err.details["reason"] == "PDF generation failed"


class TestExceptionSerialization:
    """Test exception serialization for API responses."""

    def test_all_exceptions_serialize_to_dict(self) -> None:
        exceptions = [
            InvalidFileError("test"),
            AnalysisNotFoundError("test-001"),
            ValidationError("field", "reason"),
            AuthenticationError("reason"),
            RateLimitError(100, 60),
            StorageError("op", "reason"),
            XAIError("reason"),
        ]
        
        for exc in exceptions:
            d = exc.to_dict()
            assert "error_code" in d
            assert "message" in d
            assert "details" in d
            assert isinstance(d["error_code"], str)
            assert isinstance(d["message"], str)
            assert isinstance(d["details"], dict)

    def test_serialization_round_trip(self) -> None:
        """Verify serialized error contains original information."""
        err = AnalysisNotFoundError("my-analysis-id")
        d = err.to_dict()
        assert d["error_code"] == "ANALYSIS_NOT_FOUND"
        assert "my-analysis-id" in d["message"]
        assert d["details"]["analysis_id"] == "my-analysis-id"
