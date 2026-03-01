"""Catalog-driven smoke test: every registered model must default() and solve.

Public API (importable by other repos):
  load_catalog(start)       — load models list from catalog.toml nearest to `start`
  catalog_ids(models)       — generate pytest parametrize IDs
  run_catalog_test(entry)   — the shared test body
"""

import importlib
import tomllib
from pathlib import Path

import pytest


def _find_catalog(start):
    """Walk up from `start` to find catalog.toml."""
    p = Path(start).resolve().parent
    for _ in range(3):
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
    return [f"{m['module']}:{m['class']}" for m in models]


def run_catalog_test(model_entry):
    """Each catalog entry must: import, default(), and solve without exception."""
    mod = importlib.import_module(model_entry["module"])
    cls = getattr(mod, model_entry["class"])

    m = cls.default()

    assert m.cost is not None, (
        f"{cls.__name__}.default() did not set self.cost. "
        "default() must return a model with cost assigned."
    )

    sol = m.solve(verbosity=0) if m.is_gp() else m.localsolve(verbosity=0)

    assert sol is not None


_CATALOG = load_catalog(__file__)


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_model(model_entry):
    """Each catalog entry must: import, default(), and solve without exception."""
    run_catalog_test(model_entry)
