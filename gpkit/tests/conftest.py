"""Pytest configuration and fixtures for gpkit tests"""

import pytest

from gpkit import settings


@pytest.fixture(params=settings["installed_solvers"])
def solver(request):
    """Fixture that parametrizes tests over all installed solvers"""
    return request.param
