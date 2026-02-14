"TOML model loader: parse .toml files into gpkit Model objects."

import keyword
import re
import tomllib
from pathlib import Path

import numpy as np

from ..constraints.set import ConstraintSet
from ..globals import NamedVariables, Vectorize
from ..model import Model
from ..nomials.variables import ArrayVariable, VectorizableVariable
from ._expr import TomlExpressionError, _AmbiguousVar, parse_constraint, parse_objective


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


def _validate_var_name(name):
    """Validate that a variable name is a valid Python identifier."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise TomlParseError(f"Variable name '{name}' is not a valid Python identifier")


def _make_variable(name, value, units, label):
    """Create a gpkit Variable from parsed spec components.

    Uses VectorizableVariable so that variables declared inside a
    Vectorize context automatically become ArrayVariables.
    """
    _validate_var_name(name)
    descr = {"name": name}
    if value is not None:
        descr["value"] = value
    if units is not None:
        descr["units"] = units
    if label is not None:
        descr["label"] = label
    return VectorizableVariable(**descr)


def _make_vector_variable(name, shape, value, units, label):
    """Create a gpkit VectorVariable (ArrayVariable) from parsed spec."""
    _validate_var_name(name)
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


# Keys in a model section that are model metadata, not variable declarations.
_RESERVED_MODEL_KEYS = frozenset(
    {
        "constraints",
        "dimensions",
        "id",
        "objective",
        "submodels",
        "vars",
        "vectorize",
        "vectors",
    }
)


def _extract_model_vars(model_def):
    """Get variable declarations from a model definition.

    Supports both formats:
    - Sub-table: [models.wing.vars] → model_def["vars"]
    - Flat: non-reserved keys in model_def are variable declarations

    Returns a dict of {name: raw_spec}.
    """
    vars_dict = dict(model_def.get("vars", {}))
    for key, value in model_def.items():
        if key not in _RESERVED_MODEL_KEYS:
            if key in vars_dict:
                raise TomlParseError(
                    f"Variable '{key}' defined both as flat key "
                    f"and in [vars] sub-table"
                )
            vars_dict[key] = value
    return vars_dict


# ---------------------------------------------------------------------------
# Multi-model namespace proxies
# ---------------------------------------------------------------------------


class _ModelNamespace:
    """Proxy for model-qualified variable access (e.g., wing.S)."""

    _toml_namespace = True

    def __init__(self, model_id, vars_dict):
        self._model_id = model_id
        self._vars = vars_dict

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._vars:
            available = sorted(self._vars)
            raise AttributeError(
                f"Model '{self._model_id}' has no variable '{name}'. "
                f"Available: {', '.join(available)}"
            )
        return self._vars[name]


class _SubmodelsProxy:
    """Proxy for summing a variable across submodels (e.g., submodels.W)."""

    _toml_namespace = True

    def __init__(self, submodel_vars):
        self._submodel_vars = submodel_vars  # list of (model_id, vars_dict)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        matches = []
        for _, vs in self._submodel_vars:
            if name in vs:
                matches.append(vs[name])
        if not matches:
            model_ids = [mid for mid, _ in self._submodel_vars]
            raise AttributeError(
                f"No submodel defines variable '{name}'. "
                f"Submodels: {', '.join(model_ids)}"
            )
        result = matches[0]
        for m in matches[1:]:
            result = result + m
        return result


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

    constraints = _parse_constraints(
        model_def.get("constraints", []), namespace, model_id
    )
    return Model(cost, constraints, substitutions)


# ---------------------------------------------------------------------------
# Multi-model assembly
# ---------------------------------------------------------------------------


def _resolve_model_graph(models_section):
    """Find root model and return topological ordering of reachable models.

    The root is the unique model not referenced as a submodel by any other.
    Returns (root_id, ordered_ids) with leaves first, root last.
    """
    # Find models referenced as submodels
    referenced = set()
    for mdef in models_section.values():
        for sub_id in mdef.get("submodels", []):
            referenced.add(sub_id)

    # Root = models not referenced by anyone
    roots = [mid for mid in models_section if mid not in referenced]
    if len(roots) == 0:
        raise TomlParseError(
            "All models are referenced as submodels (circular dependency)"
        )
    if len(roots) > 1:
        raise TomlParseError(
            f"Multiple root models (not referenced as submodels): {roots}. "
            f"Exactly one root model expected."
        )
    root_id = roots[0]

    if "objective" not in models_section[root_id]:
        raise TomlParseError(f"Root model '{root_id}' must have an 'objective' field")

    # DFS for topological ordering + cycle/missing detection
    ordered = []
    visited = set()
    visiting = set()

    def dfs(model_id):
        if model_id in visiting:
            raise TomlParseError(f"Circular submodel dependency involving '{model_id}'")
        if model_id in visited:
            return
        if model_id not in models_section:
            raise TomlParseError(f"Submodel '{model_id}' referenced but not defined")
        visiting.add(model_id)
        for sub_id in models_section[model_id].get("submodels", []):
            dfs(sub_id)
        visiting.discard(model_id)
        visited.add(model_id)
        ordered.append(model_id)

    dfs(root_id)
    return root_id, ordered


def _resolve_vectorize(model_def, dimensions):
    """Resolve a model section's vectorize field to an integer or None.

    The value must be a string referencing a key in [dimensions].
    """
    vec = model_def.get("vectorize")
    if vec is None:
        return None
    if not isinstance(vec, str):
        raise TomlParseError(
            f"vectorize must be a dimension name (string), "
            f"got {type(vec).__name__}: {vec!r}"
        )
    if vec not in dimensions:
        raise TomlParseError(
            f"vectorize references unknown dimension '{vec}'. "
            f"Available: {sorted(dimensions)}"
        )
    return dimensions[vec]


def _build_merged_namespace(per_model_vars, all_dimensions):
    """Build merged namespace with bare names, model proxies, and ambiguity."""
    namespace = {}
    name_to_models = {}

    for model_id, vars_dict in per_model_vars.items():
        for name in vars_dict:
            name_to_models.setdefault(name, []).append(model_id)

    # Bare names: unique → variable, ambiguous → sentinel
    for name, model_ids in name_to_models.items():
        if len(model_ids) == 1:
            namespace[name] = per_model_vars[model_ids[0]][name]
        else:
            namespace[name] = _AmbiguousVar(name, model_ids)

    # Model namespace proxies for qualified access (wing.S)
    for model_id, vars_dict in per_model_vars.items():
        if model_id in name_to_models:
            raise TomlParseError(f"Model ID '{model_id}' collides with a variable name")
        namespace[model_id] = _ModelNamespace(model_id, vars_dict)

    namespace.update(all_dimensions)
    return namespace


def _build_multi_model(
    models_section, dimension_overrides=None
):  # pylint: disable=too-many-locals
    """Build a single Model from multiple model sections with structure."""
    from contextlib import nullcontext

    root_id, ordered_ids = _resolve_model_graph(models_section)

    # Collect dimensions from all models
    all_dimensions = {}
    for model_id in ordered_ids:
        dims = _resolve_dimensions(models_section[model_id], dimension_overrides)
        all_dimensions.update(dims)

    # Pass 1: Build all variables, record lineages
    per_model_vars = {}
    per_model_lineage = {}
    substitutions = {}

    for model_id in ordered_ids:
        model_def = models_section[model_id]
        vec_length = _resolve_vectorize(model_def, all_dimensions)
        vec_ctx = Vectorize(vec_length) if vec_length else nullcontext()

        with vec_ctx:
            with NamedVariables(model_id) as (lineage, _):
                ns, subs = _build_scalar_vars(_extract_model_vars(model_def))
                substitutions.update(subs)
                _build_vector_vars(
                    model_def.get("vectors", {}),
                    all_dimensions,
                    ns,
                    substitutions,
                )
                per_model_vars[model_id] = ns
                per_model_lineage[model_id] = lineage

    # Build merged namespace with ambiguity detection
    namespace = _build_merged_namespace(per_model_vars, all_dimensions)

    # Pass 2: Parse constraints using merged namespace
    root_constraints = []
    submodel_sets = {}

    for model_id in ordered_ids:
        model_def = models_section[model_id]

        # Model-specific namespace: inject submodels proxy if applicable
        constraint_ns = dict(namespace)
        sub_ids = model_def.get("submodels", [])
        if sub_ids:
            constraint_ns["submodels"] = _SubmodelsProxy(
                [(sid, per_model_vars[sid]) for sid in sub_ids]
            )

        constraints = _parse_constraints(
            model_def.get("constraints", []),
            constraint_ns,
            model_id,
        )

        if model_id == root_id:
            root_constraints = constraints
        else:
            cs = ConstraintSet(constraints)
            cs.lineage = per_model_lineage[model_id]
            cs.unique_varkeys = frozenset(
                v.key for v in per_model_vars[model_id].values() if hasattr(v, "key")
            )
            submodel_sets[model_id] = cs

    # Parse objective with root's namespace (including its submodels proxy)
    root_def = models_section[root_id]
    objective_ns = dict(namespace)
    root_sub_ids = root_def.get("submodels", [])
    if root_sub_ids:
        objective_ns["submodels"] = _SubmodelsProxy(
            [(sid, per_model_vars[sid]) for sid in root_sub_ids]
        )
    try:
        cost = parse_objective(root_def["objective"], objective_ns)
    except TomlExpressionError as exc:
        raise TomlParseError(f"Error in objective of model '{root_id}': {exc}") from exc

    all_constraints = root_constraints[:]
    for model_id in ordered_ids:
        if model_id != root_id and model_id in submodel_sets:
            all_constraints.append(submodel_sets[model_id])

    return Model(cost, all_constraints, substitutions)


def _parse_constraints(constraint_strs, namespace, model_id):
    """Parse a list of constraint strings against a namespace."""
    constraints = []
    for i, cstr in enumerate(constraint_strs):
        try:
            constraints.append(parse_constraint(cstr, namespace))
        except TomlExpressionError as exc:
            raise TomlParseError(
                f"Error in constraint {i} of model '{model_id}': {exc}"
            ) from exc
    return constraints


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

    models_section = doc["models"]
    if len(models_section) == 1:
        model_id, model_def = next(iter(models_section.items()))
        model = _build_model(model_id, model_def, dimensions)
    else:
        # For multi-model, merge top-level [dimensions] with user overrides
        all_dims = dict(doc.get("dimensions", {}))
        if dimensions:
            all_dims.update(dimensions)
        model = _build_multi_model(models_section, all_dims or None)

    # Apply additional substitutions
    if substitutions:
        model.substitutions.update(substitutions)

    return model
