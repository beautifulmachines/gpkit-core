"""Catalog-driven smoke test: every registered model must default() and solve."""

import importlib
import tomllib
from pathlib import Path

import pytest


def _find_catalog():
    """Walk up from this file to find catalog.toml."""
    p = Path(__file__).resolve().parent
    for _ in range(3):
        candidate = p / "catalog.toml"
        if candidate.exists():
            return candidate
        p = p.parent
    raise FileNotFoundError("catalog.toml not found in any parent directory")


def _load_catalog():
    catalog_path = _find_catalog()
    with open(catalog_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("models", [])


_CATALOG = _load_catalog()


def _catalog_ids(models):
    return [f"{m['module']}:{m['class']}" for m in models]


@pytest.mark.parametrize("model_entry", _CATALOG, ids=_catalog_ids(_CATALOG))
def test_catalog_model(model_entry):
    """Each catalog entry must: import, default(), and solve without exception."""
    from gpkit.exceptions import InvalidGPConstraint

    mod = importlib.import_module(model_entry["module"])
    cls = getattr(mod, model_entry["class"])

    m = cls.default()

    assert m.cost is not None, (
        f"{cls.__name__}.default() did not set self.cost. "
        "default() must return a model with cost assigned."
    )

    try:
        sol = m.solve(verbosity=0)
    except InvalidGPConstraint:
        sol = m.localsolve(verbosity=0)

    assert sol is not None
