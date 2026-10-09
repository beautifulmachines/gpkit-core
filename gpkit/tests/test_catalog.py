"""Catalog smoke test: every registered model must build, solve, and match cost.

Public API (importable by other repos):
  load_catalog(start)          — load models list from catalog.toml nearest to `start`
  catalog_ids(models)          — generate pytest parametrize IDs
  catalog_params(models, xfail={id: reason})
                               — the same, as pytest.param values, xfailing the
                                 named ids (strictly, so a fix reports XPASS)
  run_catalog_test(entry)      — the shared test body
  run_catalog_snapshots(entry, start)
                               — write structure and solution snapshots for one
                                 entry into <start's dir>/snapshots/; drift shows
                                 up as a git diff, like docs/source/examples
  run_catalog_to_ir(entry)     — the IR document serializes and is self-consistent
  run_catalog_toml_roundtrip(entry)
                               — the model survives to_toml → load_toml and solves
                                 to the same cost
"""

import importlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from gpkit.toml import load_toml
from gpkit.toml._printer import to_toml
from gpkit.util.small_scripts import mag

# Expected (cost, rel_tol) for each gpkit-core catalog entry by id.
# External repos using run_catalog_test can put expected_cost/expected_cost_tol
# directly in their catalog.toml entries instead.
_EXPECTED_COSTS = {
    "box": (0.003674, 0.01),
    "water_tank": (1.293, 0.01),
    "beam": (0.7825, 0.001),
    "pipeline": (234504.0, 0.01),
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


def catalog_params(models, xfail=None):
    """Parametrize argvalues for `models`, xfailing the ids named in `xfail`.

    `xfail` maps a catalog id to the reason it is expected to fail, normally an
    open issue. Strict, so a fixed entry reports XPASS as a failure and the
    exemption has to be deleted rather than quietly outliving the bug.
    """
    xfail = xfail or {}
    return [
        pytest.param(
            model,
            id=entry_id,
            marks=(
                [pytest.mark.xfail(reason=xfail[entry_id], strict=True)]
                if entry_id in xfail
                else []
            ),
        )
        for model, entry_id in zip(models, catalog_ids(models), strict=True)
    ]


def run_catalog_test(model_entry):
    """Each catalog entry must: import, build, and solve. Assert cost if provided."""
    m = _catalog_model(model_entry)
    name = type(m).__name__

    assert m.cost is not None, (
        f"{name} did not set self.cost. setup() must assign self.cost."
    )

    sol = _solve(m)
    assert sol is not None
    for val in sol.primal.values():
        assert not np.isnan(np.atleast_1d(mag(val))).any(), f"{name}: NaN in solution"

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
        f"{name} cost {mag(sol.cost):.6g} does not match "
        f"expected {expected} (rel tol {tol})"
    )


def _catalog_model(model_entry):
    "The built model for a catalog entry."
    mod = importlib.import_module(model_entry["module"])
    return getattr(mod, model_entry["class"]).default()


def _solve(model):
    "Solve however the model's own math requires."
    return model.solve(verbosity=0) if model.is_gp() else model.localsolve(verbosity=0)


def run_catalog_to_ir(model_entry):
    """The IR document survives JSON and declares every variable it substitutes.

    The IR is export only, so what it owes a consumer is a complete and
    self-consistent document rather than a reconstructable one.
    """
    ir = _catalog_model(model_entry).to_ir()
    assert json.loads(json.dumps(ir)) == ir
    dangling = sorted(set(ir.get("substitutions", {})) - set(ir["variables"]))
    assert not dangling, f"substitutions name undeclared variables: {dangling}"

    def tree_indices(node):
        yield from node["constraint_indices"]
        for child in node["children"]:
            yield from tree_indices(child)

    # every constraint claimed by exactly one node, which is what lets a
    # consumer join model_tree against the flat constraints list
    assert sorted(tree_indices(ir["model_tree"])) == list(range(len(ir["constraints"])))


def run_catalog_toml_roundtrip(model_entry):
    """The model survives to_toml → load_toml and solves to the same cost.

    TOML is the format a model is stored in and loaded from, so this is the
    round-trip that has to hold -- it exercises both halves of the path a user
    or a dashboard takes to get a model back.
    """
    m = _catalog_model(model_entry)
    m2 = load_toml(to_toml(m))
    cost, cost2 = mag(_solve(m).cost), mag(_solve(m2).cost)
    assert cost2 == pytest.approx(cost, rel=1e-5)


def structure_digest(model) -> str:
    """Line-per-item rendering of a model's variable names and constraints.

    Built from an unsolved report, so it holds no values and needs no solver:
    it changes only when naming or structure changes.  One line per item keeps
    diffs minimal.
    """
    lines: list[str] = []

    def walk(section):
        lines.append(f"[{section['lineage_path'] or section['title']}]")
        for kind in ("free", "fixed"):
            for v in section[f"{kind}_variables"]:
                # source locates variables owned elsewhere, whose displayed
                # names are shortened against this section
                src = f"  [{v['source']}]" if v["source"] else ""
                lines.append(f"  {kind:5} {v['name']}{src}")
        for group in section["constraint_groups"]:
            label = f" ({group['label']})" if group["label"] else ""
            for c in group["constraints"]:
                lines.append(f"  cons{label}  {c['str']}")
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
    m = _catalog_model(model_entry)
    entry_id = model_entry.get("id", type(m).__name__)
    outdir = Path(start).resolve().parent / "snapshots"

    _write_snapshot(outdir / f"{entry_id}.structure.txt", structure_digest(m))
    _write_snapshot(outdir / f"{entry_id}.solution.txt", _solve(m).table())


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


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_to_ir(model_entry):
    run_catalog_to_ir(model_entry)


@pytest.mark.parametrize("model_entry", catalog_params(_CATALOG))
def test_catalog_toml_roundtrip(model_entry):
    run_catalog_toml_roundtrip(model_entry)
