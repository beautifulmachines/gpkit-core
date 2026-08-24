"""Catalog smoke test: every registered model must build, solve, and match cost.

Public API (importable by other repos):
  load_catalog(start)          — load models list from catalog.toml nearest to `start`
  catalog_ids(models)          — generate pytest parametrize IDs
  run_catalog_test(entry)      — the shared test body
  run_catalog_snapshots(entry, start)
                               — write structure and solution snapshots for one
                                 entry into <start's dir>/snapshots/; drift shows
                                 up as a git diff, like docs/source/examples
"""

import importlib
import tomllib
from pathlib import Path

import numpy as np
import pytest

from gpkit.util.small_scripts import mag

# Expected (cost, rel_tol) for each gpkit-core catalog entry by id.
# External repos using run_catalog_test can put expected_cost/expected_cost_tol
# directly in their catalog.toml entries instead.
_EXPECTED_COSTS = {
    "box": (0.003674, 0.01),
    "water_tank": (1.293, 0.01),
    "beam": (0.7825, 0.001),
    "uav": (7105.40, 0.01),
    "wing": (35.64, 0.001),
    "bemt_hover": (43582.47, 0.01),
}


def _find_catalog(start):
    """Walk up from `start` to find catalog.toml."""
    p = Path(start).resolve().parent
    for _ in range(4):
        candidate = p / "catalog.toml"
        if candidate.exists():
            return candidate
        p = p.parent
    raise FileNotFoundError("catalog.toml not found in any parent directory")


def load_catalog(start):
    """Load models list from the catalog.toml nearest to `start`."""
    with open(_find_catalog(start), "rb") as f:
        return tomllib.load(f).get("models", [])


def catalog_ids(models):
    """Return pytest parametrize IDs for a list of catalog model entries."""
    return [m.get("id", f"{m['module']}:{m['class']}") for m in models]


def run_catalog_test(model_entry):
    """Each catalog entry must: import, build, and solve. Assert cost if provided."""
    mod = importlib.import_module(model_entry["module"])
    cls = getattr(mod, model_entry["class"])

    m = cls.default()

    assert m.cost is not None, (
        f"{cls.__name__} did not set self.cost. setup() must assign self.cost."
    )

    sol = m.solve(verbosity=0) if m.is_gp() else m.localsolve(verbosity=0)
    assert sol is not None
    for val in sol.primal.values():
        assert not np.isnan(np.atleast_1d(mag(val))).any(), (
            f"{cls.__name__}: NaN in solution"
        )

    # expected_cost in the catalog entry takes precedence (for external repos);
    # fall back to the built-in table for gpkit-core entries.
    if "expected_cost" in model_entry:
        expected = model_entry["expected_cost"]
        tol = model_entry.get("expected_cost_tol", 0.01)
    else:
        entry = _EXPECTED_COSTS.get(model_entry.get("id"))
        if entry is None:
            return
        expected, tol = entry

    assert mag(sol.cost) == pytest.approx(expected, rel=tol), (
        f"{cls.__name__} cost {mag(sol.cost):.6g} does not match "
        f"expected {expected} (rel tol {tol})"
    )


def structure_digest(model) -> str:
    """Line-per-item rendering of a model's variable names and constraints.

    Built from an unsolved report, so it holds no values and needs no solver:
    it changes only when naming or structure changes.  One line per item keeps
    diffs minimal.
    """
    lines: list[str] = []

    def walk(section):
        lines.append(f"[{section['lineage_path'] or section['title']}]")
        for v in section["free_variables"]:
            lines.append(f"  free  {v['name']}")
        for v in section["fixed_variables"]:
            lines.append(f"  fixed {v['name']}")
        for group in section["constraint_groups"]:
            label = f" ({group['label']})" if group["label"] else ""
            for c in group["constraints"]:
                lines.append(f"  cons{label}  {c}")
        for child in section["children"]:
            walk(child)

    walk(model.report(fmt="dict"))
    return "\n".join(lines) + "\n"


def _write_snapshot(path: Path, content: str):
    "Write content to path, creating parent dirs. Drift surfaces as a git diff."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_catalog_snapshots(model_entry, start):
    """Write structure and solution snapshots for one catalog entry.

    Snapshots land in <start's directory>/snapshots/.  Following the
    docs/source/examples convention, these are regenerated rather than
    asserted — a change shows up as a git diff to review, so deliberate
    improvements are as easy to land as regressions are to spot.
    """
    mod = importlib.import_module(model_entry["module"])
    cls = getattr(mod, model_entry["class"])
    entry_id = model_entry.get("id", cls.__name__)
    outdir = Path(start).resolve().parent / "snapshots"

    m = cls.default()
    _write_snapshot(outdir / f"{entry_id}.structure.txt", structure_digest(m))

    sol = m.solve(verbosity=0) if m.is_gp() else m.localsolve(verbosity=0)
    _write_snapshot(outdir / f"{entry_id}.solution.txt", sol.table())


try:
    _CATALOG = load_catalog(__file__)
except FileNotFoundError:
    _CATALOG = []


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_model(model_entry):
    """Each catalog entry must: import, build, and solve. Assert cost if provided."""
    run_catalog_test(model_entry)


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_snapshots(model_entry):
    """Regenerate each catalog entry's snapshots; drift shows as a git diff."""
    run_catalog_snapshots(model_entry, __file__)
