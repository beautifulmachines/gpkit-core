"""Pytest configuration and fixtures for gpkit tests"""

import importlib
import os
import sys

import pytest

import gpkit
from gpkit import settings
from gpkit.tests.helpers import NewDefaultSolver, StdoutCaptured


@pytest.fixture(name="solver", params=settings["installed_solvers"])
def solver_fixture(request):
    """Fixture that parametrizes tests over all installed solvers"""
    return request.param


# Cache for imported example modules (shared across solvers)
_example_imports = {}

# Directory containing example scripts
_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
_EXAMPLE_DIR = os.path.abspath(_FILE_DIR + "../../../docs/source/examples")


@pytest.fixture
def example(request, solver):
    """
    Fixture that imports an example module and passes it to the test.

    The example name is derived from the test function name:
    test_autosweep -> autosweep.py

    The first time an example is imported (for the first solver), its output
    is captured to a golden file. Subsequent runs reload the module.

    To ensure each example's output file contains ONLY that example's output
    (not its dependencies), we:
    1. First import the module with output suppressed (populates sys.modules
       with the full dependency graph)
    2. Then reload the module with output captured (dependencies already cached,
       so only this module's code runs and produces output)
    """
    # Extract example name from test function name (test_foo -> foo)
    # Use originalname to avoid parameter suffix (e.g., test_foo[cvxopt] -> test_foo)
    test_name = request.node.originalname
    if test_name.startswith("test_"):
        example_name = test_name[5:]
    else:
        example_name = test_name

    # Add example directory to path if needed
    if os.path.isdir(_EXAMPLE_DIR) and _EXAMPLE_DIR not in sys.path:
        sys.path.insert(0, _EXAMPLE_DIR)

    with NewDefaultSolver(solver):
        if example_name not in _example_imports:
            # First time this test runs - capture output to file.
            # Import first with suppressed output to populate dependency graph,
            # then reload with capture so only this module's output is captured.
            with StdoutCaptured():  # Suppress output during initial import
                if example_name in sys.modules:
                    importlib.reload(sys.modules[example_name])
                else:
                    importlib.import_module(example_name)

            # Reset model numbers so the captured reload has clean state
            gpkit.globals.NamedVariables.reset_modelnumbers()

            filepath = os.path.join(_EXAMPLE_DIR, f"{example_name}_output.txt")
            with StdoutCaptured(logfilepath=filepath):
                _example_imports[example_name] = importlib.reload(
                    sys.modules[example_name]
                )
        else:
            with StdoutCaptured():  # No file capture on reload
                importlib.reload(_example_imports[example_name])

        yield _example_imports[example_name]

    # Reset global state after test
    gpkit.globals.NamedVariables.reset_modelnumbers()

    # Verify global state is clean
    for globname, global_thing in [
        ("model numbers", gpkit.globals.NamedVariables.modelnums),
        ("lineage", gpkit.NamedVariables.lineage),
        ("signomials enabled", gpkit.SignomialsEnabled),
        ("vectorization", gpkit.Vectorize.vectorization),
        ("namedvars", gpkit.NamedVariables.namedvars),
    ]:
        if global_thing:  # pragma: no cover
            raise ValueError(
                f"global attribute {globname} should have been"
                f" falsy after the test, but was instead {global_thing}"
            )
