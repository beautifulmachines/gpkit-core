"Choosing the solver a program is handed to."

from contextvars import ContextVar

from ..util.globals import settings

# The solver to use when a solve names none.  A ContextVar, not a plain
# setting, so that one thread (or asyncio task) choosing a solver cannot
# retarget a solve running concurrently in another.
_default_solver = ContextVar("default_solver", default=None)


def default_solver():
    """The solver to hand a program that names none.

    The innermost active `DefaultSolver`, else the solver found when gpkit
    was built.  A name ("cvxopt", "mosek_cli", "mosek_conif") or a callable.
    """
    solver = _default_solver.get()
    if solver is None:
        solver = settings["default_solver"]
    if not solver:
        raise ValueError(
            "No default solver was set during build, so"
            " solvers must be manually specified."
        )
    return solver


class DefaultSolver:
    """Chooses the solver for solves that don't name one.

    Example
    -------
        >>> from gpkit.solvers import DefaultSolver
        >>> with DefaultSolver("mosek_conif"):
        >>>     sol = m.solve()
    """

    def __init__(self, solver):
        self.solver = solver

    def __enter__(self):
        self._token = _default_solver.set(self.solver)

    def __exit__(self, type_, val, traceback):
        _default_solver.reset(self._token)
