"""
Argus Core - Authentication API
================================
Endpoints for JWT token generation.

Provides:
- POST /api/v1/auth/anonymous: Generate guest JWT token
- POST /api/v1/auth/refresh: Refresh existing token

Guest tokens are valid for 24 hours and require no credentials.
This enables anonymous usage while maintaining JWT security.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Header
from pydantic import BaseModel, Field

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

auth_router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


class BaseSchema(BaseModel):
    """Base schema with standard configuration."""
    model_config = {"extra": "ignore", "from_attributes": True}


class AuthToken(BaseSchema):
    """JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user_id: str = Field(..., description="User identifier")


@auth_router.post(
    "/anonymous",
    response_model=AuthToken,
    summary="Get anonymous JWT token",
    description="Generate a guest JWT token for anonymous access.",
)
async def anonymous_login() -> AuthToken:
    """
    Generate an anonymous JWT token for guest access.
    
    Creates a token with a unique guest user ID that is valid
    for 24 hours. No credentials required.
    
    Returns:
        AuthToken with access_token and metadata
    """
    try:
        import jwt
        import uuid
        
        user_id = f"guest-{uuid.uuid4().hex[:12]}"
        expires_in = config.jwt_expire_minutes * 60
        
        payload = {
            "sub": user_id,
            "email": f"{user_id}@guest.argus.dev",
            "roles": ["user", "analyst"],
            "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            "iat": datetime.now(timezone.utc),
            "type": "anonymous",
        }
        
        access_token = jwt.encode(
            payload,
            config.jwt_secret,
            algorithm=config.jwt_algorithm,
        )
        
        logger.info(f"Anonymous token generated for {user_id}")
        
        return AuthToken(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user_id=user_id,
        )
        
    except Exception as exc:
        logger.error(f"Failed to generate anonymous token: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TOKEN_GENERATION_FAILED",
                "message": "Failed to generate authentication token",
            },
        )


@auth_router.post(
    "/refresh",
    response_model=AuthToken,
    summary="Refresh JWT token",
    description="Refresh an existing JWT token before expiration. "
                "Pass the current token via the Authorization: Bearer <token> header.",
)
async def refresh_token(
    # H1 fix: read token from Authorization header, NOT query param.
    # This prevents JWT leakage into URL/proxy logs.
    authorization: str = Header(
        ...,
        description="Bearer <current_token>",
    ),
) -> AuthToken:
    """
    Refresh an existing JWT token.

    Validates the current token from the Authorization header and
    generates a new one with extended expiration.

    Args:
        authorization: "Bearer <current_token>" header

    Returns:
        AuthToken with new access_token
    """
    try:
        import jwt
        import uuid

        # H1 fix: parse Bearer token from Authorization header
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "INVALID_AUTH_HEADER",
                    "message": "Authorization header must be 'Bearer <token>'",
                },
            )
        current_token = authorization.split(" ", 1)[1].strip()

        if not current_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "TOKEN_REQUIRED", "message": "Current token required"},
            )

        # Decode current token
        payload = jwt.decode(
            current_token,
            config.jwt_secret,
            algorithms=[config.jwt_algorithm],
        )
        
        user_id = payload.get("sub", f"guest-{uuid.uuid4().hex[:12]}")
        email = payload.get("email", f"{user_id}@argus.dev")
        roles = payload.get("roles", ["user"])
        
        expires_in = config.jwt_expire_minutes * 60
        
        new_payload = {
            "sub": user_id,
            "email": email,
            "roles": roles,
            "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            "iat": datetime.now(timezone.utc),
            "type": payload.get("type", "anonymous"),
        }
        
        access_token = jwt.encode(
            new_payload,
            config.jwt_secret,
            algorithm=config.jwt_algorithm,
        )
        
        logger.info(f"Token refreshed for {user_id}")
        
        return AuthToken(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user_id=user_id,
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "TOKEN_EXPIRED", "message": "Token has expired"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_TOKEN", "message": "Invalid token"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to refresh token: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "REFRESH_FAILED", "message": "Failed to refresh token"},
        )
