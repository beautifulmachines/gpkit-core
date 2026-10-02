"""Tests for Solution class"""

import gc
import json
import pickle
import sys
import threading

import numpy as np
import pytest

import gpkit
from gpkit import (
    Model,
    SignomialsEnabled,
    Var,
    Variable,
    VectorVariable,
    breakdowns,
    printing,
)
from gpkit.constraints.set import keyed_constraints
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


def test_solving_leaves_the_model_picklable():
    """A solve must not leave the terminal's stdout attached to the model.

    The solver's output is captured by swapping in a SolverLog, which holds the
    real stdout so it can echo as it goes.  The log is kept on the program
    afterwards, and the model keeps the program, so holding that stream past the
    capture made every solved model unpicklable -- and is why a Solution refers
    to its model weakly.
    """
    x = Variable("x_pkl")
    m = Model(x, [x >= 2])
    assert pickle.dumps(m)  # unsolved
    m.solve(verbosity=0)
    assert pickle.dumps(m)  # and solved
    assert m.program.solve_log.lines() is not None  # text still readable


def test_printing_table_backward_compat():
    """printing.table(sol) still works and returns a string."""
    x = Variable("x_st", "m", "free variable")
    c = Variable("c_st", 2.0, "m", "fixed variable")
    m = Model(x, [x >= c])
    sol = m.solve(verbosity=0)
    result = printing.table(sol)
    assert isinstance(result, str)
    assert len(result) > 0


class TestConstraintSensitivitiesByKey:
    """Per-constraint sensitivities a consumer outside the process can read.

    sens.constraints is keyed by live constraint objects, which is right in
    process and meaningless once serialized -- and is the solution's only handle
    on the objects themselves.  sens.constraints_by_key carries the same numbers
    under the key to_ir() and the report dict both use.
    """

    def test_keys_are_the_report_keys(self):
        "A sensitivity's key is the key the report gives that same constraint."
        m = _sens_model()
        sol = m.solve(verbosity=0)
        refs = {k.ref for k in sol.sens.constraints_by_key}
        report = m.report(sol, fmt="dict")
        entries = [c for g in report["constraint_groups"] for c in g["constraints"]]
        for child in report["children"]:
            entries += [c for g in child["constraint_groups"] for c in g["constraints"]]
        assert {e["key"] for e in entries} == refs

    def test_agrees_with_the_object_keyed_map(self):
        "Same numbers, reached by key instead of by object."
        m = _sens_model()
        sol = m.solve(verbosity=0)
        for key, c in keyed_constraints(m):
            assert sol.sens.constraints_by_key[key] == pytest.approx(
                sol.sens.constraints[c]
            )

    def test_is_json_serializable(self):
        "The point of the key: the numbers survive leaving the process."
        sol = _sens_model().solve(verbosity=0)
        json.dumps({k.ref: v for k, v in sol.sens.constraints_by_key.items()})

    def test_survives_the_model_being_discarded(self):
        "Solve-and-discard is the recommended concurrent pattern."

        def solve_and_discard():
            m = _sens_model()
            return m.solve(verbosity=0), len(list(m.flat()))

        sol, n_constraints = solve_and_discard()
        gc.collect()
        assert len(sol.sens.constraints_by_key) == n_constraints

    def test_duplicate_constraints_stay_distinct(self):
        "Two identical constraints are two constraints; position tells them apart."
        x = Variable("x_dup")
        m = Model(x, [x >= 1, x >= 1, x >= 2])
        sol = m.solve(verbosity=0)
        assert len(sol.sens.constraints_by_key) == 3


class _SensChild(Model):
    y = Var("-")

    def setup(self):
        return [self.y >= 2]


def _sens_model():
    "A model with a child, so ids span more than one report section."

    class _SensTop(Model):
        def setup(self):
            x = Variable("x_sens")
            self.child = _SensChild()
            return [x >= self.child.y, x <= 10, self.child]

    return _SensTop()
