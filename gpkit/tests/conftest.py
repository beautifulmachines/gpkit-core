"""Pytest configuration and fixtures for gpkit tests"""

import importlib
import os
import sys

import numpy as np
import pytest

import gpkit
from gpkit import settings
from gpkit.util.small_scripts import mag


def assert_logtol(first, second, logtol=1e-6):
    "Asserts that the logs of two arrays have a given abstol"
    np.testing.assert_allclose(
        np.log(mag(first)), np.log(mag(second)), atol=logtol, rtol=0
    )


class NullFile:
    "A fake file interface that does nothing"

    def write(self, string):
        "Do not write, do not pass go."

    def close(self):
        "Having not written, cease."


class NewDefaultSolver:
    "Creates an environment with a different default solver"

    def __init__(self, solver):
        self.solver = solver
        self.prev_default_solver = None

    def __enter__(self):
        "Change default solver."
        self.prev_default_solver = gpkit.settings["default_solver"]
        gpkit.settings["default_solver"] = self.solver

    def __exit__(self, *args):
        "Reset default solver."
        gpkit.settings["default_solver"] = self.prev_default_solver


class StdoutCaptured:
    "Puts everything that would have printed to stdout in a log file instead"

    def __init__(self, logfilepath=None):
        self.logfilepath = logfilepath
        self.original_stdout = None

    def __enter__(self):
        "Capture stdout"
        self.original_stdout = sys.stdout
        sys.stdout = (
            open(self.logfilepath, mode="w", encoding="utf-8")
            if self.logfilepath
            else NullFile()
        )

    def __exit__(self, *args):
        "Return stdout"
        sys.stdout.close()
        sys.stdout = self.original_stdout


@pytest.fixture(name="solver", params=settings["installed_solvers"])
def solver_fixture(request):
    """Fixture that parametrizes tests over all installed solvers"""
    return request.param


# Cache for imported example modules (shared across solvers)
_example_imports = {}

# Directory for captured example output (docs artifacts)
_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
_OUTPUT_DIR = os.path.abspath(os.path.join(_FILE_DIR, "../../docs/source/examples"))


def _verify_clean_global_state():
    """Verify gpkit global state is reset after test."""
    for name, value in [
        ("model numbers", gpkit.globals.NamedVariables.modelnums),
        ("lineage", gpkit.NamedVariables.lineage),
        ("signomials enabled", gpkit.SignomialsEnabled),
        ("vectorization", gpkit.Vectorize.vectorization),
        ("namedvars", gpkit.NamedVariables.namedvars),
    ]:
        if value:  # pragma: no cover
            raise ValueError(
                f"global attribute {name} should be falsy after test, was {value}"
            )


def _import_example(name):
    """
    Import or reload an example from gpkit.examples.

    On first import, uses two-pass approach to isolate output:
    1. Import with suppressed output (populates sys.modules with dependencies)
    2. Reload with captured output (only this module's output goes to file)

    This ensures each example's *_output.txt contains only that example's
    output, not output from any dependencies it imports.
    """
    module_name = f"gpkit.examples.{name}"
    if name not in _example_imports:
        # First time: two-pass to isolate this example's output
        with StdoutCaptured():  # Suppress during initial import
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)

        # Reset model numbers so captured reload has clean state
        gpkit.globals.NamedVariables.reset_modelnumbers()

        filepath = os.path.join(_OUTPUT_DIR, f"{name}_output.txt")
        with StdoutCaptured(logfilepath=filepath):
            _example_imports[name] = importlib.reload(sys.modules[module_name])
    else:
        # Already imported: just reload with suppressed output
        with StdoutCaptured():
            importlib.reload(_example_imports[name])

    return _example_imports[name]


@pytest.fixture
def example(request, solver):
    """
    Fixture that imports an example module and yields it for testing.

    Example name derived from test function: test_autosweep -> autosweep.py
    Imports from gpkit.examples subpackage.
    First run captures output to docs/source/examples/*_output.txt.
    """
    # Extract example name from test function (use originalname to strip params)
    test_name = request.node.originalname
    example_name = test_name[5:] if test_name.startswith("test_") else test_name

    with NewDefaultSolver(solver):
        mod = _import_example(example_name)
        yield mod

    gpkit.globals.NamedVariables.reset_modelnumbers()
    _verify_clean_global_state()
