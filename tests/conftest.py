"""Shared fixtures and constants for all tests.

Provides:
- ``PROJECT_ROOT`` — absolute path to the repository root.
- ``TEST_SETS_ROOT`` — absolute path to ``data/test_sets/``.
"""

from pathlib import Path

import pytest

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SETS_ROOT = PROJECT_ROOT / "data" / "test_sets"


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root as a session-scoped fixture."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_sets_root() -> Path:
    """Path to ``data/test_sets/``."""
    return TEST_SETS_ROOT
