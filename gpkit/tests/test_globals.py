"Tests for the construction-context managers in gpkit.util.globals"

import threading

from gpkit import NamedVariables, SignomialsEnabled, Variable, Vectorize
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
    "Each thread numbers model instances independently of other threads."
    barrier = threading.Barrier(4)

    def build(_):
        barrier.wait()  # maximize interleaving across threads
        for expected_num in range(3):
            with NamedVariables("Box") as (lineage, _unused):
                assert lineage == (("Box", expected_num),)

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
