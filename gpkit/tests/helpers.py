"""Convenience classes and functions for unit testing"""

import sys


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
        import gpkit  # pylint: disable=import-outside-toplevel

        self.prev_default_solver = gpkit.settings["default_solver"]
        gpkit.settings["default_solver"] = self.solver

    def __exit__(self, *args):
        "Reset default solver."
        import gpkit  # pylint: disable=import-outside-toplevel

        gpkit.settings["default_solver"] = self.prev_default_solver


class StdoutCaptured:
    "Puts everything that would have printed to stdout in a log file instead"

    def __init__(self, logfilepath=None):
        self.logfilepath = logfilepath
        self.original_stdout = None
        self.original_unit_printing = None

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
