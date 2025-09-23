"""Pytest configuration for test cleanup."""
import glob
import os
import pytest


@pytest.fixture(autouse=True)
def cleanup_db_files(request):
    """Clean up database files and metadata around each test."""
    # GENIUS SOLUTION: Clear metadata BEFORE tests that need isolation,
    # but RESTORE it AFTER those tests to not affect subsequent tests
    test_name = request.node.name

    # Only clear metadata for tests that define models INSIDE test functions
    needs_clearing = any(name in test_name for name in [
        "integration", "edge_cases", "pagination", "sqlmodel"
    ])

    if needs_clearing:
        try:
            from graphql_db.orm_base import Base
            Base.metadata.clear()
        except ImportError:
            pass

        try:
            from sqlmodel import SQLModel
            SQLModel.metadata.clear()
        except ImportError:
            pass

    yield  # Run the test

    # CREATIVE PART: If this was a test that cleared metadata,
    # force re-registration of module-level models
    if needs_clearing:
        try:
            # Force reimport to re-register their models
            import importlib
            import sys
            modules_to_reload = [
                'tests.test_db_manager',
                'tests.test_sqlalchemy_orm'
            ]
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
        except Exception:
            pass  # Best effort

    # Clean up any .db files created during the test
    db_files = glob.glob("*.db")
    for db_file in db_files:
        try:
            os.remove(db_file)
            print(f"Cleaned up {db_file}")
        except OSError:
            pass  # File might already be gone
