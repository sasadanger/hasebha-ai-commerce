"""Shared-secret authentication for the AI service's inference endpoints.

This service is publicly reachable (Railway) but is designed to be called
only by the Medusa backend's order-placed subscriber and internal
smoke-tests -- it is not a customer-facing API. Without this check, anyone
with the URL could call inference endpoints directly with no credentials,
which is a real, exploitable issue on a metered host (resource-consumption
abuse / cost-DoS) even though no customer data is exposed by the calls
themselves (the service holds no database connection and returns only a
function of the caller-supplied input).

Deliberately NOT applied to /health or /ready: those must stay reachable
by Railway's own health checks and external uptime monitoring without a
credential.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

_API_KEY_ENV_VAR = "AI_SERVICE_API_KEY"


def require_api_key(x_internal_api_key: str = Header(default="")) -> None:
    expected = os.environ.get(_API_KEY_ENV_VAR, "")
    if not expected:
        # Fail closed: an unconfigured secret must never mean "open to
        # everyone". Local dev without the var set will 503, not silently
        # allow all traffic.
        raise HTTPException(status_code=503, detail="AI service authentication not configured")
    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Api-Key")
