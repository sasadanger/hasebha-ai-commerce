"""Shared pytest fixtures for the AI service test suite.

Sets a fixed test API key before any test client is constructed, so the
`require_api_key` dependency (src/ai_service/auth.py) has something to
check against -- it fails closed (503) if AI_SERVICE_API_KEY is unset,
which would otherwise break every existing inference-endpoint test.
"""
import os

TEST_API_KEY = "test-suite-shared-secret-do-not-use-in-production"

os.environ.setdefault("AI_SERVICE_API_KEY", TEST_API_KEY)
