"""Shared fixtures for integration tests.

Resets flask-limiter state between tests so rate-limit counters from one
test don't bleed into another. memory:// storage is per-worker, which
means it's per-pytest-process and accumulates across tests without a
reset.

Also resets the Database singleton before each integration test so that
a unit-test run (which resets the singleton in teardown) preceding an
integration-test run does not leave the singleton pointing at /app/data
(the hard-coded default) rather than the tmpdir set by MINUSPOD_DATA_DIR.
"""
import os
import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear in-memory limiter counters before each test."""
    try:
        from api import limiter
        limiter.reset()
    except Exception:
        # Limiter may not be initialised yet when the first test of a
        # module runs (Flask app import is lazy). Safe to skip.
        pass
    yield


@pytest.fixture(autouse=True)
def _reset_singletons_if_stale():
    """Reset Database and Storage singletons when their path doesn't match MINUSPOD_DATA_DIR.

    Unit-test fixtures call `Database._instance = None` in teardown. When an
    integration test then calls `Database()` (no args), the constructor defaults
    to '/app/data' instead of the tmpdir set by MINUSPOD_DATA_DIR. Clearing both
    singletons here forces a fresh initialisation with the correct path so
    integration tests pass regardless of test collection order.
    """
    data_dir = os.environ.get('MINUSPOD_DATA_DIR')
    if not data_dir:
        yield
        return
    try:
        from database import Database
        inst = Database._instance
        if inst is None or str(inst.data_dir) != data_dir:
            Database._instance = None
            # Storage holds a reference to the old Database; reset it too so
            # the next Storage() call re-initialises and creates a fresh Database
            # pointing at data_dir.
            try:
                from storage import Storage
                Storage._instance = None
            except Exception:
                pass
    except Exception:
        pass
    yield
