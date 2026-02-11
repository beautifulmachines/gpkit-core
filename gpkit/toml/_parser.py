"TOML model loader: parse .toml files into gpkit Model objects."

import re
import tomllib
from pathlib import Path

import numpy as np

from ..model import Model
from ..nomials.variables import ArrayVariable, Variable
from ._expr import TomlExpressionError, parse_constraint, parse_objective


class TomlParseError(Exception):
    """Raised when a TOML model file has invalid structure or content."""


# ---------------------------------------------------------------------------
# Variable spec parsing
# ---------------------------------------------------------------------------

# Matches a leading number (int, float, or scientific) followed by units
_NUMERIC_PREFIX = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+(.*)")


def _parse_var_spec(raw):
    """Parse a variable declaration value from TOML.

    Returns (value, units, label) where any may be None.

    Forms:
        "m"                -> (None, "m", None)         free variable
        "200 m^2"          -> (200.0, "m^2", None)      parameter
        "-"                -> (None, None, None)         dimensionless free
        2                  -> (2, None, None)            dimensionless param
        3.14               -> (3.14, None, None)         dimensionless param
        ["m", "desc"]      -> (None, "m", "desc")        free with label
        ["200 m^2", "desc"]-> (200.0, "m^2", "desc")     param with label
        ["-", "desc"]      -> (None, None, "desc")        dimensionless + label
        [2, "desc"]        -> (2, None, "desc")           param + label
    """
    if isinstance(raw, list):
        if len(raw) != 2:
            raise TomlParseError(
                f"Array variable spec must have exactly 2 elements "
                f"[spec, description], got {len(raw)}: {raw!r}"
            )
        spec, label = raw
        if not isinstance(label, str):
            raise TomlParseError(
                "Variable description must be a string, got "
                f"{type(label).__name__}: {raw!r}"
            )
        value, units, _ = _parse_var_spec(spec)
        return value, units, label

    if isinstance(raw, (int, float)):
        return raw, None, None

    if isinstance(raw, str):
        if raw == "-":
            return None, None, None
        m = _NUMERIC_PREFIX.match(raw)
        if m:
            return float(m.group(1)), m.group(2), None
        return None, raw, None

    raise TomlParseError(f"Invalid variable spec: {raw!r}")


def _make_variable(name, value, units, label):
    """Create a gpkit Variable from parsed spec components."""
    descr = {"name": name}
    if value is not None:
        descr["value"] = value
    if units is not None:
        descr["units"] = units
    if label is not None:
        descr["label"] = label
    return Variable(**descr)


def _make_vector_variable(name, shape, value, units, label):
    """Create a gpkit VectorVariable (ArrayVariable) from parsed spec."""
    descr = {"name": name}
    if value is not None:
        descr["value"] = np.ones(shape) * value
    if units is not None:
        descr["units"] = units
    if label is not None:
        descr["label"] = label
    return ArrayVariable(shape, **descr)


# ---------------------------------------------------------------------------
# TOML document normalization
# ---------------------------------------------------------------------------


def _normalize_doc(doc):
    """Normalize single-model format to multi-model internal format.

    Converts:
        [model] + top-level [vars]/[vectors.*]/[dimensions]
    into:
        [models.<id>] with nested vars/vectors/dimensions
    """
    if "model" in doc and "models" in doc:
        raise TomlParseError("Cannot have both [model] and [models] sections")

    if "model" in doc:
        model_section = doc.pop("model")
        model_id = model_section.pop("id", "main")
        model_section["vars"] = doc.pop("vars", {})
        model_section["vectors"] = doc.pop("vectors", {})
        model_section["dimensions"] = doc.pop("dimensions", {})
        doc["models"] = {model_id: model_section}
    elif "models" not in doc:
        raise TomlParseError("TOML model must have a [model] or [models.*] section")

    return doc


# ---------------------------------------------------------------------------
# Model assembly helpers
# ---------------------------------------------------------------------------


def _resolve_dimensions(model_def, dimension_overrides):
    """Extract and validate dimension definitions."""
    dimensions = dict(model_def.get("dimensions", {}))
    if dimension_overrides:
        dimensions.update(dimension_overrides)
    for k, v in dimensions.items():
        if not isinstance(v, int):
            raise TomlParseError(
                f"Dimension '{k}' must be an integer, got {type(v).__name__}: {v!r}"
            )
    return dimensions


def _build_scalar_vars(vars_section):
    """Parse scalar variable declarations into namespace and substitutions."""
    namespace = {}
    substitutions = {}
    for var_name, raw in vars_section.items():
        value, units, label = _parse_var_spec(raw)
        var = _make_variable(var_name, value, units, label)
        namespace[var_name] = var
        if value is not None:
            substitutions[var] = value
    return namespace, substitutions


def _build_vector_vars(vectors_section, dimensions, namespace, substitutions):
    """Parse vector variable declarations into namespace and substitutions."""
    for size_key, vec_vars in vectors_section.items():
        if size_key in dimensions:
            shape = dimensions[size_key]
        else:
            try:
                shape = int(size_key)
            except ValueError:
                raise TomlParseError(
                    f"Vector section key '{size_key}' is not a dimension name "
                    f"or integer. Available dimensions: {sorted(dimensions)}"
                ) from None

        for var_name, raw in vec_vars.items():
            value, units, label = _parse_var_spec(raw)
            vec = _make_vector_variable(var_name, shape, value, units, label)
            namespace[var_name] = vec
            if value is not None:
                for elem in vec.flat:
                    substitutions[elem] = elem.key.value


def _build_model(model_id, model_def, dimension_overrides=None):
    """Build a gpkit Model from a normalized model definition dict."""
    dimensions = _resolve_dimensions(model_def, dimension_overrides)
    namespace, substitutions = _build_scalar_vars(model_def.get("vars", {}))
    _build_vector_vars(
        model_def.get("vectors", {}), dimensions, namespace, substitutions
    )
    namespace.update(dimensions)

    objective_str = model_def.get("objective")
    if objective_str is None:
        raise TomlParseError(f"Model '{model_id}' is missing an 'objective' field")
    try:
        cost = parse_objective(objective_str, namespace)
    except TomlExpressionError as exc:
        raise TomlParseError(
            f"Error in objective of model '{model_id}': {exc}"
        ) from exc

    constraints = []
    for i, cstr in enumerate(model_def.get("constraints", [])):
        try:
            constraints.append(parse_constraint(cstr, namespace))
        except TomlExpressionError as exc:
            raise TomlParseError(
                f"Error in constraint {i} of model '{model_id}': {exc}"
            ) from exc

    return Model(cost, constraints, substitutions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_toml(source, *, dimensions=None, substitutions=None):
    """Load a TOML model specification and return a gpkit Model.

    Parameters
    ----------
    source : str or Path
        Path to a .toml file, or a TOML string.
    dimensions : dict, optional
        Override dimension values (e.g. ``{"N": 10}``).
    substitutions : dict, optional
        Additional substitutions applied after model construction.

    Returns
    -------
    gpkit.Model
    """
    if isinstance(source, Path):
        with open(source, "rb") as f:
            doc = tomllib.load(f)
    elif isinstance(source, str) and "\n" not in source and Path(source).is_file():
        with open(source, "rb") as f:
            doc = tomllib.load(f)
    elif isinstance(source, str):
        doc = tomllib.loads(source)
    else:
        raise TomlParseError(
            f"source must be a file path or TOML string, got {type(source).__name__}"
        )

    doc = _normalize_doc(doc)

    # For single-model files, build the one model
    models_section = doc["models"]
    if len(models_section) == 1:
        model_id, model_def = next(iter(models_section.items()))
        model = _build_model(model_id, model_def, dimensions)
    else:
        raise TomlParseError("Multi-model files are not yet supported (Phase 2)")

    # Apply additional substitutions
    if substitutions:
        model.substitutions.update(substitutions)

    return model
