"""Tests for Solution class"""

import sys
import threading

import numpy as np
import pytest

import gpkit
from gpkit import (
    Model,
    SignomialsEnabled,
    Variable,
    VectorVariable,
    breakdowns,
    printing,
)
from gpkit.tests.conftest import run_threads
from gpkit.util.small_classes import Quantity, Strings


class TestSolution:
    """Unit tests for the Solution class"""

    def test_getitem(self):
        A = Variable("A", "-", "Test Variable")
        prob = Model(A, [A >= 1])
        sol = prob.solve(verbosity=0)
        assert sol[A] == pytest.approx(1.0, abs=1e-8)

    def test_getitem_units(self):
        # test from issue541
        x = Variable("x", 10, "ft")
        y = Variable("y", "m")
        m = Model(y, [y >= x])
        sol = m.solve(verbosity=0)
        assert sol[y] / sol[x] == pytest.approx(1.0, abs=1e-6)
        assert sol[x] / sol[y] == pytest.approx(1.0, abs=1e-6)

    def test_call_vector(self):
        n = 5
        x = VectorVariable(n, "x")
        prob = Model(sum(x), [x >= 2.5])
        sol = prob.solve(verbosity=0)
        solx = sol[x]
        assert isinstance(solx, Quantity)
        assert isinstance(solx.magnitude, np.ndarray)
        assert solx.shape == (n,)
        for i in range(n):
            assert solx[i] == pytest.approx(2.5, abs=1e-4)

    def test_subinto(self):
        h_max = Variable("h_max", 10, "m", "Length")
        a_min = Variable("A_min", 10, "m^2", "Area")
        p_max = Variable("P", "m", "Perimeter")
        h = Variable("h", "m", "Length")
        w = Variable("w", "m", "width")
        m = Model(12 / (w * h**3), [h <= h_max, h * w >= a_min, p_max >= 2 * h + 2 * w])
        p_vals = np.linspace(13, 24, 20)
        sweepsol = m.sweep({p_max: p_vals}, verbosity=0)
        p_sol = [sol.subinto(p_max) for sol in sweepsol]
        assert len(p_sol) == 20
        for pv, ps in zip(p_vals, p_sol):
            assert 0 * gpkit.ureg.m == pytest.approx(pv * gpkit.ureg.m - ps)

    def test_table(self):
        x = Variable("x")
        gp = Model(x, [x >= 12])
        sol = gp.solve(verbosity=0)
        tab = sol.table()
        assert isinstance(tab, Strings)

    def test_units_sub(self):
        # issue 809
        t = Variable("t", "N", "thrust")
        tmin = Variable("t_{min}", "N", "minimum thrust")
        m = Model(t, [t >= tmin])
        tminsub = 1000 * gpkit.ureg.lbf
        m.substitutions.update({tmin: tminsub})
        sol = m.solve(verbosity=0)
        assert round(sol[tmin] - tminsub, 7) == 0
        assert "1000N" not in sol.table().replace(" ", "").replace("[", "").replace(
            "]", ""
        )

    def test_key_options(self):
        # issue 993
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(y, [y + 6 * x >= 13 + x**2])
        msol = m.localsolve(verbosity=0)
        spsol = m.sp().localsolve(verbosity=0)
        gpsol = m.program.gps[-1].solve(verbosity=0)
        assert msol[x] == spsol[x]
        assert msol[x] == gpsol[x]

    def test_solution_section(self):
        """Solution table has a single combined 'Solution' section."""
        x = Variable("x", "m", "free variable")
        c = Variable("c", 2.0, "m", "fixed variable")
        m = Model(x, [x >= c])
        sol = m.solve(verbosity=0)
        tab = sol.table()
        assert "Solution" in tab
        assert "Free Variables" not in tab
        assert "Fixed Variables" not in tab
        # fixed var sensitivity appears in the table
        assert "(+" in tab or "(~0)" in tab
        # free var appears before "...constants", fixed var after
        sol_section = tab[tab.index("Solution") :]
        assert sol_section.index("x") < sol_section.index("...constants")
        assert sol_section.index("c") > sol_section.index("...constants")

    def test_result_access(self):
        """Test result table access from SP solution"""
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            sig = y + 6 * x >= 13 + x**2
        m = Model(y, [sig])
        sol = m.localsolve(verbosity=0)
        assert all(isinstance(gp.result.table(), Strings) for gp in m.program.gps)
        assert sol.cost / 4.0 == pytest.approx(1.0, abs=1e-5)
        assert sol[x] / 3.0 == pytest.approx(1.0, abs=1e-3)


def test_tables_for_a_model_with_no_constraints():
    """Unconstrained posynomial minimization is a classical GP, and its
    solution must print: with no constraints there are no sensitivities to
    attribute, so the model-sensitivity breakdown has nothing to lay out."""
    x, y = Variable("x"), Variable("y")
    sol = Model(x + 1 / x + y + 1 / y, []).solve(verbosity=0)
    assert sol.cost == pytest.approx(4, rel=1e-4)
    assert not sol.sens.constraints
    assert not sol.model_sens_breakdown()
    assert sol.table()
    assert sol.summary()


def test_concurrent_tables_dont_corrupt_stdout(monkeypatch):
    """Building a table must leave sys.stdout alone.

    The breakdown sections used to be rendered by printing them and capturing
    the output, which meant swapping sys.stdout -- a process-wide mutation. Two
    threads doing that at once traded streams: one restored the other's capture
    as "the original", leaving it installed for the rest of the process, and a
    thread still mid-capture read .lines() off whatever was there. Syncing once
    inside the breakdown puts all four threads there at the same moment.
    """
    x, y = Variable("x"), Variable("y")
    sol = Model(x + y, [x >= 2, y >= 3 * x]).solve(verbosity=0)

    real_graph = breakdowns.graph
    barrier = threading.Barrier(4)
    once = threading.local()

    def synced_graph(*args, **kwargs):
        if not getattr(once, "synced", False):
            once.synced = True
            barrier.wait(timeout=10)
        return real_graph(*args, **kwargs)

    monkeypatch.setattr(breakdowns, "graph", synced_graph)

    real_stdout = sys.stdout
    try:
        run_threads(lambda _: sol.table())
    finally:
        leaked = sys.stdout is not real_stdout
        if leaked:
            sys.stdout = real_stdout  # don't swallow the rest of the session
    assert not leaked, "sys.stdout was not restored after concurrent tables"


def test_printing_table_backward_compat():
    """printing.table(sol) still works and returns a string."""
    x = Variable("x_st", "m", "free variable")
    c = Variable("c_st", 2.0, "m", "fixed variable")
    m = Model(x, [x >= c])
    sol = m.solve(verbosity=0)
    result = printing.table(sol)
    assert isinstance(result, str)
    assert len(result) > 0
