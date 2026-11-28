"Tests for choosing the solver a program is handed to"

import threading

import pytest

from gpkit import Model, Variable, settings
from gpkit.solvers import DefaultSolver, default_solver
from gpkit.solvers.cvxopt import optimize as cvxopt_optimize
from gpkit.tests.conftest import run_threads


def test_default_is_the_solver_chosen_at_build():
    assert default_solver() == settings["default_solver"]


def test_override_is_scoped_to_its_block():
    with DefaultSolver("a_solver"):
        assert default_solver() == "a_solver"
    assert default_solver() == settings["default_solver"]


def test_override_is_reentrant():
    "Exiting an inner block must restore the outer one, not the build default."
    with DefaultSolver("outer"):
        with DefaultSolver("inner"):
            assert default_solver() == "inner"
        assert default_solver() == "outer"


def test_no_solver_available_says_so():
    "A build that found no solver gives a clear error, not an empty name."
    build_default = settings["default_solver"]
    settings["default_solver"] = ""
    try:
        with pytest.raises(ValueError, match="No default solver"):
            default_solver()
    finally:
        settings["default_solver"] = build_default


def test_thread_isolation():
    "One thread's override must not retarget another thread's solve."
    override_set = threading.Barrier(4)
    reads_done = threading.Barrier(4)
    seen = {}

    def choose(i):
        if i == 0:
            with DefaultSolver("only_mine"):
                override_set.wait(timeout=10)  # hold the override while others read
                seen[i] = default_solver()
                reads_done.wait(timeout=10)
        else:
            override_set.wait(timeout=10)
            seen[i] = default_solver()
            reads_done.wait(timeout=10)

    run_threads(choose, count=4)
    assert seen[0] == "only_mine"
    assert [seen[i] for i in (1, 2, 3)] == [settings["default_solver"]] * 3


def test_bare_solve_uses_the_active_default():
    "A solve that names no solver is handed the contextual default."
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return cvxopt_optimize(*args, **kwargs)

    x = Variable("x")
    m = Model(x, [x >= 2])
    with DefaultSolver(spy):
        sol = m.solve(verbosity=0)
    assert len(calls) == 1
    assert sol.cost == pytest.approx(2, rel=1e-4)


def test_named_solver_beats_the_default():
    "An explicit solver= argument wins over the contextual default."
    x = Variable("x")
    m = Model(x, [x >= 2])
    with DefaultSolver("not_a_solver"):
        sol = m.solve(solver="cvxopt", verbosity=0)
    assert sol.cost == pytest.approx(2, rel=1e-4)
