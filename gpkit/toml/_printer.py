"Generate TOML model specs from gpkit Models or IR dicts."

import re
import types

from ..ast_nodes import ast_from_ir
from ..util.repr_conventions import _toml_format_number as _format_number

# ---------------------------------------------------------------------------
# Ref string → Python variable name
# ---------------------------------------------------------------------------

# IR ref formats (from VarKey.ref):
#   dimensionless scalar:  "x"
#   scalar with units:     "h|ft"
#   vector:                "d#3|ft"
#   vector element:        "d[0]#3|ft"
#   with lineage:          "Aircraft.Wing.S|ft²"
#   lineage + vector:      "wing0.d[0]#3|ft"
# We want just the bare name: "x", "h", "A_wall", "d", "d[0]"
_REF_STRIP = re.compile(r"(#\d+)?(\|.*)?$")


def _ref_to_name(ref):
    """Extract the Python variable name from an IR ref string."""
    bare = _REF_STRIP.sub("", ref)
    # Strip lineage prefix: "Aircraft0.Wing0.S" → "S"
    return bare.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# AST → plain expression string
# ---------------------------------------------------------------------------


def ast_to_expr(node, name_fn=_ref_to_name):
    """Convert a gpkit AST node to a plain expression string.

    Accepts VarNode, ConstNode, ExprNode (from gpkit's ast_nodes),
    IR dicts (from to_ir() JSON), or raw numbers.

    Produces TOML-compatible syntax: ``*`` for multiply, ``**`` for power.
    Dispatch lives on the AST node classes themselves (ASTNode.to_toml_expr,
    gpkit/ast_nodes.py) — this just normalizes non-node inputs (raw numbers,
    IR dicts) before delegating.

    name_fn : ref -> str, default _ref_to_name (bare name, no qualification).
    Multi-model emission passes a resolver that qualifies a VarNode as
    "Model.name" when it belongs to a different model than the one whose
    constraint/objective is currently being rendered (see
    _make_name_resolver) — otherwise a cross-model reference like
    self.engine.W would print as a bare "W" indistinguishable from a local
    variable of the same name.
    """
    # Raw numbers (e.g. exponents in pow, coefficients)
    if isinstance(node, (int, float)):
        return _format_number(node)

    # IR dict — reconstruct an AST node first, then dispatch on it below
    if isinstance(node, dict):
        node = ast_from_ir(node, _RefNameRegistry())

    if hasattr(node, "to_toml_expr"):
        return node.to_toml_expr(name_fn)

    # Numpy scalars etc.
    if hasattr(node, "__float__"):
        return _format_number(float(node))

    raise ValueError(f"Cannot render AST node: {type(node).__name__}: {node!r}")


class _RefNameRegistry(dict):
    """Minimal registry mapping IR var refs back to objects with a .ref attr.

    ast_from_ir expects a registry mapping ref → VarKey.  We only need the
    .ref attribute for rendering, so we use SimpleNamespace stubs.
    """

    def __missing__(self, ref):
        stub = types.SimpleNamespace(ref=ref)
        self[ref] = stub
        return stub


# ---------------------------------------------------------------------------
# Constraint → expression string
# ---------------------------------------------------------------------------

# NOTE: gpkit stores equality as "=" internally, but TOML/Python uses "==".
# This is a gpkit-core inconsistency we should eventually fix upstream
# (track in a separate issue). For now we map here.
_OPER_MAP = {"=": "=="}


def constraint_to_expr(constraint_ir, name_fn=_ref_to_name):
    """Convert an IR constraint dict to a TOML constraint string."""
    oper = constraint_ir["oper"]
    oper = _OPER_MAP.get(oper, oper)

    left = _nomial_ir_to_expr(constraint_ir["left"], name_fn)
    right = _nomial_ir_to_expr(constraint_ir["right"], name_fn)

    return f"{left} {oper} {right}"


def _nomial_ir_to_expr(nomial_ir, name_fn=_ref_to_name):
    """Render a nomial IR dict to an expression string.

    Uses the AST when available (for expressions built from operations).
    For leaf nodes (bare Variables and numeric constants), renders directly
    from the terms — these are the only cases that lack an AST.
    """
    ast = nomial_ir.get("ast")
    if ast is not None:
        return ast_to_expr(ast, name_fn)

    # Leaf cases only: bare Variable or numeric Monomial
    terms = nomial_ir["terms"]
    if len(terms) == 1:
        term = terms[0]
        coeff = term["coeff"]
        exps = term.get("exps", {})
        if not exps:
            # Pure numeric constant
            return _format_number(coeff)
        if coeff == 1.0 and len(exps) == 1:
            ref, exp = next(iter(exps.items()))
            if exp == 1:
                # Bare variable reference
                return name_fn(ref)

    raise ValueError(
        f"Nomial IR has no AST and is not a trivial leaf node "
        f"(type={nomial_ir.get('type')}). This likely indicates a gap "
        f"in gpkit's AST tracking."
    )


def _is_reciprocal(ast_dict):
    """Check if an IR AST dict represents 1/expr. Returns expr or None."""
    if not isinstance(ast_dict, dict):
        return None
    if ast_dict.get("node") != "expr" or ast_dict.get("op") != "div":
        return None
    children = ast_dict.get("children", [])
    if len(children) != 2:
        return None
    numerator = children[0]
    if isinstance(numerator, (int, float)) and numerator == 1:
        return children[1]
    if (
        isinstance(numerator, dict)
        and numerator.get("node") == "const"
        and numerator.get("value") == 1
    ):
        return children[1]
    return None


def _format_objective(cost_ir, name_fn=_ref_to_name):
    """Determine objective direction and expression string.

    Detects the 1/expr pattern and returns ("max", expr_str) instead of
    ("min", "1/expr") for more natural readability.
    """
    ast = cost_ir.get("ast")
    if ast is not None:
        inner = _is_reciprocal(ast)
        if inner is not None:
            return "max", ast_to_expr(inner, name_fn)
    return "min", _nomial_ir_to_expr(cost_ir, name_fn)


# ---------------------------------------------------------------------------
# IR/Model → TOML file
# ---------------------------------------------------------------------------


def _group_variables(variables):
    """Group IR variables into scalars and vector groups.

    Returns (scalar_vars, vector_groups) where scalar_vars is a dict of
    ref → info for non-vector variables, and vector_groups is a dict of
    veckey_ref → {name, units, label, shape, elements}.
    """
    scalar_vars = {}
    veckeys = {}
    elements = []

    for ref, info in variables.items():
        if info.get("idx") is not None:
            elements.append((ref, info))
        elif info.get("shape") is not None:
            veckeys[ref] = info
        else:
            scalar_vars[ref] = info

    vector_groups = {}
    veckey_by_name_shape = {
        (info["name"], tuple(info["shape"])): (ref, info)
        for ref, info in veckeys.items()
    }
    for ref, info in elements:
        key = (info["name"], tuple(info.get("shape", [])))
        vecref, vecinfo = veckey_by_name_shape[key]
        assert info["name"] == vecinfo["name"]
        assert info.get("units") == vecinfo.get("units")
        if vecref not in vector_groups:
            vector_groups[vecref] = {
                "name": vecinfo["name"],
                "units": vecinfo.get("units"),
                "label": vecinfo.get("label"),
                "shape": vecinfo["shape"],
                "elements": [],
            }
        vector_groups[vecref]["elements"].append((ref, info))

    return scalar_vars, vector_groups


def _assign_model_ids(tree):
    """Map each model_tree node (by id()) to a unique TOML section name.

    Usually just the node's class name. A class can be instantiated more
    than once in the tree — e.g. AircraftPerf appears once under each of
    Outbound, Return, and SprintCondition, gpkit's three named flight
    conditions in the UAV example — so a bare class name isn't always
    unique. When a class collides, every one of its instances is
    consistently disambiguated using just enough of the node's
    instance_id lineage to tell them apart, e.g. "Outbound_AircraftPerf",
    "Return_AircraftPerf", "SprintCondition_AircraftPerf".
    """
    nodes = []

    def collect(node):
        segs = (
            node["instance_id"].split(".") if node["instance_id"] else [node["class"]]
        )
        nodes.append((node, segs))
        for child in node.get("children", []):
            collect(child)

    collect(tree)

    class_counts = {}
    for _, segs in nodes:
        class_counts[segs[-1]] = class_counts.get(segs[-1], 0) + 1

    used = set()
    ids = {}
    for node, segs in nodes:
        if class_counts[segs[-1]] == 1:
            ids[id(node)] = segs[-1]
            used.add(segs[-1])

    # Every instance of a repeated class is disambiguated together (not just
    # the first collision found), so named siblings read consistently.
    for node, segs in nodes:
        if id(node) in ids:
            continue
        for k in range(2, len(segs) + 1):
            candidate = "_".join(segs[-k:])
            if candidate not in used:
                break
        else:
            # instance_id is always globally unique, so this is only reached
            # if underscore-joining happened to collide; the dotted path
            # (which can't collide the same way) is the guaranteed fallback.
            candidate = node["instance_id"].replace(".", "_")
        used.add(candidate)
        ids[id(node)] = candidate
    return ids


def _build_ref_to_model_id(tree, model_ids):
    """Map every variable ref to the model_id (from _assign_model_ids) of
    the node that owns it. node["variables"] is already scoped to exactly
    that node's own instance, so this is unambiguous even when several
    nodes share a class."""
    mapping = {}

    def walk(node):
        mid = model_ids[id(node)]
        for ref in node.get("variables", []):
            mapping[ref] = mid
        for child in node.get("children", []):
            walk(child)

    walk(tree)
    return mapping


def _make_name_resolver(ref_to_model_id, current_model_id):
    """Build a name_fn for ast_to_expr/constraint_to_expr: bare name for a
    variable owned by the model currently being emitted, "Model.name" for
    a reference to a variable owned by a different model (e.g. Aircraft's
    constraint referencing self.engine.W) — otherwise it would print as a
    bare "W" indistinguishable from a same-named local variable."""

    def resolve(ref):
        name = _ref_to_name(ref)
        owner = ref_to_model_id.get(ref)
        if owner is not None and owner != current_model_id:
            return f"{owner}.{name}"
        return name

    return resolve


def to_toml(source, path=None):
    """Generate a TOML model spec from a gpkit Model or IR dict.

    Parameters
    ----------
    source : Model or dict
        A gpkit Model (calls .to_ir()) or an IR dict.
    path : str or Path, optional
        If provided, writes the TOML string to this file.

    Returns
    -------
    str
        The generated TOML string.
    """
    if hasattr(source, "to_ir"):
        ir = source.to_ir()
    else:
        ir = source

    lines = []

    # --- name/description ---
    name = ir.get("name", "")
    if name:
        lines.append(f'name = "{name}"')
    desc = ir.get("description", "")
    if desc:
        lines.append(f'description = "{desc}"')
    if name or desc:
        lines.append("")

    # --- detect multi-model ---
    tree = ir.get("model_tree", {})
    if tree.get("children"):
        _emit_multi_model(ir, lines)
    else:
        _emit_single_model(ir, lines)

    lines.append("")
    return _emit_lines(lines, path)


def _emit_single_model(ir, lines):
    """Emit [vars] + [model] sections for a flat single-model IR."""
    variables = ir.get("variables", {})
    substitutions = ir.get("substitutions", {})
    scalar_vars, vector_groups = _group_variables(variables)

    if scalar_vars:
        lines.append("[vars]")
        for ref, info in scalar_vars.items():
            vname = info["name"]
            units = info.get("units")
            label = info.get("label")
            value = substitutions.get(ref)
            lines.append(_format_var_line(vname, value, units, label))
        lines.append("")

    if vector_groups:
        by_shape = {}
        for group in vector_groups.values():
            shape = group["shape"]
            if isinstance(shape, (list, tuple)):
                shape = shape[0] if len(shape) == 1 else tuple(shape)
            by_shape.setdefault(shape, []).append(group)

        for shape, groups in by_shape.items():
            lines.append(f"[vectors.{shape}]")
            for group in groups:
                vname = group["name"]
                units = group["units"]
                label = group["label"]
                value = None
                if group["elements"]:
                    first_ref = group["elements"][0][0]
                    value = substitutions.get(first_ref)
                lines.append(_format_var_line(vname, value, units, label))
            lines.append("")

    lines.append("[model]")
    cost_ir = ir.get("cost", {})
    direction, cost_str = _format_objective(cost_ir)
    lines.append(f'objective = "{direction}: {cost_str}"')

    constraints = ir.get("constraints", [])
    if constraints:
        lines.append("constraints = [")
        for c in constraints:
            cstr = constraint_to_expr(c)
            lines.append(f'  "{cstr}",')
        lines.append("]")


def _emit_multi_model(ir, lines):
    """Emit [models.*] sections from a multi-model IR."""
    tree = ir["model_tree"]
    variables = ir.get("variables", {})
    substitutions = ir.get("substitutions", {})
    constraints = ir.get("constraints", [])

    model_ids = _assign_model_ids(tree)
    ref_to_model_id = _build_ref_to_model_id(tree, model_ids)

    # Flatten tree into ordered list of (model_id, node, child_ids)
    nodes = []

    def flatten(node):
        child_ids = [model_ids[id(c)] for c in node.get("children", [])]
        nodes.append((model_ids[id(node)], node, child_ids))
        for child in node.get("children", []):
            flatten(child)

    flatten(tree)

    # Emit non-root models first, then root (so submodels are defined first)
    root_entry = nodes[0]
    for model_id, node, child_ids in nodes[1:]:
        _emit_model_section(
            model_id,
            node,
            child_ids,
            variables,
            substitutions,
            constraints,
            lines,
            ref_to_model_id,
            is_root=False,
            cost_ir=None,
        )
    _emit_model_section(
        root_entry[0],
        root_entry[1],
        root_entry[2],
        variables,
        substitutions,
        constraints,
        lines,
        ref_to_model_id,
        is_root=True,
        cost_ir=ir.get("cost", {}),
    )


def _emit_model_section(  # noqa: PLR0913, PLR0917
    model_id,
    node,
    child_ids,
    variables,
    substitutions,
    all_constraints,
    lines,
    ref_to_model_id,
    *,
    is_root,
    cost_ir,
):
    """Emit a single [models.X] section."""
    lines.append(f"[models.{model_id}]")
    name_fn = _make_name_resolver(ref_to_model_id, model_id)

    # Variables (flat format: vars as keys in model section). node["variables"]
    # is this node's own unique_varkeys refs — already scoped to this exact
    # instance, so distinct instances of the same class never mix vars.
    model_vars = {ref: variables[ref] for ref in node.get("variables", [])}
    scalar_vars, vector_groups = _group_variables(model_vars)

    for ref, info in scalar_vars.items():
        vname = info["name"]
        units = info.get("units")
        label = info.get("label")
        value = substitutions.get(ref)
        lines.append(_format_var_line(vname, value, units, label))

    # Vector variables as [models.X.vectors.N] sub-tables
    if vector_groups:
        by_shape = {}
        for group in vector_groups.values():
            shape = group["shape"]
            if isinstance(shape, (list, tuple)):
                shape = shape[0] if len(shape) == 1 else tuple(shape)
            by_shape.setdefault(shape, []).append(group)

        for shape, groups in by_shape.items():
            lines.append(f"[models.{model_id}.vectors.{shape}]")
            for group in groups:
                vname = group["name"]
                units = group["units"]
                label = group["label"]
                value = None
                if group["elements"]:
                    first_ref = group["elements"][0][0]
                    value = substitutions.get(first_ref)
                lines.append(_format_var_line(vname, value, units, label))

    # Objective (root only)
    if is_root and cost_ir:
        direction, cost_str = _format_objective(cost_ir, name_fn)
        lines.append(f'objective = "{direction}: {cost_str}"')

    # Submodels
    if child_ids:
        child_str = ", ".join(f'"{cid}"' for cid in child_ids)
        lines.append(f"submodels = [{child_str}]")

    # Constraints
    node_constraints = [all_constraints[i] for i in node.get("constraint_indices", [])]
    if node_constraints:
        lines.append("constraints = [")
        for c in node_constraints:
            cstr = constraint_to_expr(c, name_fn)
            lines.append(f'  "{cstr}",')
        lines.append("]")

    lines.append("")


def _emit_lines(lines, path):
    """Join lines, optionally write to path, and return the string."""
    result = "\n".join(lines)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(result)
    return result


def _format_var_line(name, value, units, label):
    """Format a single variable line for TOML output."""
    if value is not None and units is not None:
        spec = f"{_format_number(value)} {units}"
    elif value is not None:
        spec = value if isinstance(value, (int, float)) else str(value)
    elif units is not None:
        spec = units
    else:
        spec = "-"

    if label:
        if isinstance(spec, (int, float)):
            return f'{name} = [{spec}, "{label}"]'
        return f'{name} = ["{spec}", "{label}"]'
    if isinstance(spec, (int, float)):
        return f"{name} = {spec}"
    return f'{name} = "{spec}"'
