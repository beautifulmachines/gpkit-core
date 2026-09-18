"Tests for the construction-context managers in gpkit.util.globals"

import asyncio
import json
import sys
import threading
import time

import pytest

import gpkit.util.globals as globals_module
from gpkit import Model, NamedVariables, SignomialsEnabled, Variable, Vectorize
from gpkit.examples import uav
from gpkit.nomials.math import SignomialInequality
from gpkit.solvers.cvxopt import optimize as cvxopt_optimize
from gpkit.tests.conftest import run_threads
from gpkit.util.globals import load_settings


def test_signomials_enabled_is_reentrant():
    "Exiting an inner SignomialsEnabled block must not disable the outer one."
    x = Variable("x")
    y = Variable("y")
    with SignomialsEnabled():
        with SignomialsEnabled():
            pass
        assert bool(SignomialsEnabled)
        constr = x >= 1 - y
        assert isinstance(constr, SignomialInequality)


def test_namedvariables_thread_isolation():
    """Each thread numbers model instances independently of other threads.

    Each `with NamedVariables("Box")` below is its own root build (nothing
    nests them), so each independently gets num 0 -- root-build numbering is
    reset per build, not accumulated across a thread's lifetime.
    """
    barrier = threading.Barrier(4)

    def build(_):
        barrier.wait()  # maximize interleaving across threads
        for _ in range(3):
            with NamedVariables("Box") as (lineage, _unused):
                assert lineage == (("Box", 0),)

    run_threads(build)


def test_vectorize_thread_isolation():
    "A Vectorize context in one thread is invisible to others."
    barrier = threading.Barrier(2)

    def worker(i):
        if i == 0:
            with Vectorize(3):
                barrier.wait(timeout=10)  # context is open; let observer look
                assert Vectorize.vectorization == (3,)
                barrier.wait(timeout=10)  # hold it open until observer is done
        else:
            barrier.wait(timeout=10)
            assert Vectorize.vectorization == ()
            barrier.wait(timeout=10)

    run_threads(worker, count=2)


def test_signomials_enabled_thread_isolation():
    "SignomialsEnabled in one thread does not enable signomials in others."
    barrier = threading.Barrier(2)

    def worker(i):
        if i == 0:
            with SignomialsEnabled():
                barrier.wait(timeout=10)
                barrier.wait(timeout=10)  # hold the context open
        else:
            barrier.wait(timeout=10)
            assert not bool(SignomialsEnabled)
            barrier.wait(timeout=10)

    run_threads(worker, count=2)


def test_concurrent_build_and_solve():
    "The tradespace scenario: concurrent per-thread builds + solves, no lock."
    results = {}
    barrier = threading.Barrier(4)

    def build_and_solve(tid):
        barrier.wait(timeout=30)
        model = uav.UAV()
        ir = json.dumps(model.to_ir(), sort_keys=True, default=str)
        results[tid] = (ir, float(model.solve(verbosity=0).cost))

    run_threads(build_and_solve)

    assert len(results) == 4
    assert len({ir for ir, _ in results.values()}) == 1, "IRs differ across threads"
    assert len({cost for _, cost in results.values()}) == 1


def test_concurrent_solves_dont_corrupt_stdout():
    """Overlapping solves must not race the sys.stdout save/restore.

    GP.solve() captures solver output by swapping sys.stdout out and back in;
    without a lock, one thread's restore can stomp a second thread's swapped-
    in SolverLog, leaving it permanently installed as sys.stdout. A slow
    solverfn widens the window so four threads reliably overlap.
    """

    def slow_optimize(prob, meq_idxs, **kwargs):
        time.sleep(0.05)
        return cvxopt_optimize(prob, meq_idxs, **kwargs)

    x = Variable("x")
    barrier = threading.Barrier(4)

    def worker(_):
        barrier.wait(timeout=10)
        Model(x, [x >= 1]).gp().solve(solver=slow_optimize, verbosity=1)

    real_stdout = sys.stdout
    try:
        run_threads(worker)
    finally:
        leaked = sys.stdout is not real_stdout
        if leaked:
            sys.stdout = real_stdout  # don't swallow the rest of the session
    assert not leaked, "sys.stdout was not restored after concurrent solves"


def test_load_settings_toml(tmp_path):
    "Settings load from TOML; default_solver derives from installed_solvers."
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text('installed_solvers = ["cvxopt", "mosek_conif"]\n')
    loaded = load_settings(path=str(settings_file), trybuild=False)
    assert loaded["installed_solvers"] == ["cvxopt", "mosek_conif"]
    assert loaded["default_solver"] == "cvxopt"


def test_load_settings_toml_explicit_default(tmp_path):
    "A default_solver set in the TOML file wins over the derived default."
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text(
        'installed_solvers = ["cvxopt", "mosek_conif"]\ndefault_solver = "mosek_conif"\n'
    )
    loaded = load_settings(path=str(settings_file), trybuild=False)
    assert loaded["default_solver"] == "mosek_conif"


def test_namedvariables_asyncio_isolation():
    """Concurrent asyncio tasks number models independently and don't leak
    counts back to the caller once they finish.

    asyncio.Task's copy_context() copies var->value bindings, not the
    underlying dict, so a ContextVar bound once (e.g. at import, via
    SequentialGeometricProgram's class-body NamedVariables("RelaxPCCP"))
    can end up shared across tasks -- unlike threading.Thread, which starts
    from a fresh Context.
    """

    async def build_task():
        with NamedVariables("Box") as (lineage, _unused):
            await asyncio.sleep(0)  # yield control, maximize interleaving
            return lineage

    async def main():
        return await asyncio.gather(*(build_task() for _ in range(4)))

    lineages = asyncio.run(main())
    assert lineages == [(("Box", 0),)] * 4, lineages

    # A synchronous build after the tasks finish must also start fresh.
    with NamedVariables("Box") as (lineage, _unused):
        assert lineage == (("Box", 0),)


def test_settings_lazy_load_is_thread_safe(monkeypatch):
    "Concurrent first access to `settings` must call load_settings only once."
    call_count = []

    def slow_load_settings(*_args, **_kwargs):
        call_count.append(1)
        time.sleep(0.05)  # widen the check-then-set race window
        return {"installed_solvers": [], "default_solver": ""}

    monkeypatch.setattr(globals_module, "load_settings", slow_load_settings)
    fresh_settings = globals_module._Settings()
    barrier = threading.Barrier(8)

    def access(_):
        barrier.wait(timeout=10)  # maximize interleaving across threads
        fresh_settings["default_solver"]

    run_threads(access, count=8)
    assert len(call_count) == 1, f"load_settings called {len(call_count)} times"


class Box(Model):
    "A minimal model, used to check root-build numbering behavior."

    def setup(self):
        x = Variable("x", "m")
        x_min = Variable("x_min", 1, "m")
        self.cost = x
        return [x >= x_min]


def test_root_build_determinism():
    "Two independent builds of the same class produce identical IR."
    m1, m2 = Box(), Box()
    assert m1.to_ir() == m2.to_ir()


def test_composed_lineage_collision_raises():
    "Composing two distinct, separately-built same-class instances errors."
    b1, b2 = Box(), Box()
    with pytest.raises(ValueError):
        Model(b1.cost, [b1, b2])
