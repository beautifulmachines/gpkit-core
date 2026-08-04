"Tests for the construction-context managers in gpkit.util.globals"

import asyncio
import json
import sys
import threading
import time

import gpkit.util.globals as globals_module
from gpkit import NamedVariables, SignomialsEnabled, Variable, Vectorize
from gpkit.examples import uav
from gpkit.nomials.math import SignomialInequality
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


def run_threads(target, count=4):
    "Run target(thread_index) in count threads; re-raise any thread's error."
    errors = []

    def wrapped(i):
        try:
            target(i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


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

    # GP.solve swaps sys.stdout process-globally (SolverLog); concurrent
    # solves can race the restore. Guard it here so a leaked SolverLog
    # can't swallow the rest of the pytest session's output.
    saved_stdout = sys.stdout
    try:
        run_threads(build_and_solve)
    finally:
        sys.stdout = saved_stdout

    assert len(results) == 4
    assert len({ir for ir, _ in results.values()}) == 1, "IRs differ across threads"
    assert len({cost for _, cost in results.values()}) == 1


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
    """Concurrent asyncio tasks must number models independently of each
    other, and must not leak counts back into the caller once they finish.

    Regression test for a shared-dict leak: `modelnums`/`namedvars` were
    bound into a ContextVar once (at first access, e.g. at import time via
    SequentialGeometricProgram's class-body `NamedVariables("RelaxPCCP")`),
    and every context that inherits that binding via `copy_context()`
    (which is how asyncio.Task construction works) shares the same dict
    object thereafter -- unlike `threading.Thread`, which starts from a
    fresh, uninherited Context and so never observed this.
    """

    async def build_task():
        with NamedVariables("Box") as (lineage, _unused):
            await asyncio.sleep(0)  # yield control, maximize interleaving
            return lineage

    async def main():
        return await asyncio.gather(*(build_task() for _ in range(4)))

    lineages = asyncio.run(main())
    assert lineages == [(("Box", 0),)] * 4, lineages

    # After the tasks finish, a synchronous build in the calling context
    # must also start fresh -- it must not inherit counts the tasks
    # accumulated in their (buggy, shared) dict.
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
