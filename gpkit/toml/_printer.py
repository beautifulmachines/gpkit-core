"Generate TOML model specs from gpkit Models or IR dicts."

import numpy as np

from ..ast_nodes import ast_from_ir
from ..display.render import TOML, render_ast, render_ir
from ..display.render import exact_number as _format_number
from ..ir import check_ir_version, shown_whole

# ---------------------------------------------------------------------------
# Variable entry → Python variable name
# ---------------------------------------------------------------------------


def _toml_name(entry):
    "A variable entry's name as a TOML expression writes it: d, or d[0]."
    idx = entry.get("idx")
    return entry["name"] + (f"[{','.join(map(str, idx))}]" if idx else "")


def _entry_names(variables):
    "name_fn(ref) for the model IR's `variables`: each by its bare name."
    return lambda ref: _toml_name(variables[ref])


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------

# A TOML basic string may hold no character below 0x20, nor 0x7f.  The named
# escapes come second, so they win over \uXXXX for the characters TOML names.
_ESCAPES = str.maketrans(
    {ordinal: f"\\u{ordinal:04x}" for ordinal in [*range(0x20), 0x7F]}
    | {"\\": r"\\", '"': r"\"", "\b": r"\b", "\t": r"\t"}
    | {"\n": r"\n", "\f": r"\f", "\r": r"\r"}
)


def _quote(text):
    """A TOML basic string holding exactly `text`.

    Every quoted string the printer emits goes through here, so a label
    carrying LaTeX or a quote cannot break the document.
    """
    return '"' + str(text).translate(_ESCAPES) + '"'


# ---------------------------------------------------------------------------
# AST → plain expression string
# ---------------------------------------------------------------------------


def ast_to_expr(node, name_fn):
    """An AST node, its IR dict, or a raw number, in TOML expression syntax.

    name_fn(ref) names each variable.  Multi-model emission passes a resolver
    that qualifies a variable as "Model.name" when another model owns it (see
    _make_name_resolver), so a cross-model reference like self.engine.W does
    not print as a bare "W".
    """
    return render_ast(ast_from_ir(node), TOML, name_fn)


# ---------------------------------------------------------------------------
# Constraint → expression string
# ---------------------------------------------------------------------------


def constraint_to_expr(constraint_ir, name_fn):
    "An IR constraint entry as a TOML constraint string."
    if "oper" not in constraint_ir:
        raise ValueError(
            f"A {constraint_ir['type']} constraint has no written form, so TOML"
            " cannot spell it."
        )
    return render_ir(constraint_ir, TOML, name_fn)


def _written_exprs(refs, constraints, name_fn):
    """The constraints at refs as TOML strings, each as it was written.

    A vector constraint is one line, unless a side was never written as one
    expression; then only its elements' math is known, and each is a line.
    """
    for ref in refs:
        c = constraints[ref]
        if shown_whole(c):
            yield constraint_to_expr(c, name_fn)
            continue
        for element in constraints.values():
            if element.get("veckey_ref") == ref:
                yield constraint_to_expr(element, name_fn)


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


def _format_objective(cost_ir, name_fn):
    """Determine objective direction and expression string.

    Detects the 1/expr pattern and returns ("max", expr_str) instead of
    ("min", "1/expr") for more natural readability.
    """
    ast = cost_ir.get("ast")
    if ast is not None:
        inner = _is_reciprocal(ast)
        if inner is not None:
            return "max", ast_to_expr(inner, name_fn)
    return "min", render_ir(cost_ir, TOML, name_fn)


# ---------------------------------------------------------------------------
# IR/Model → TOML file
# ---------------------------------------------------------------------------


def _group_variables(variables, all_variables=None):
    """Group IR variables into scalars and vector groups.

    Returns (scalar_vars, vector_groups) where scalar_vars is a dict of
    ref → info for non-vector variables, and vector_groups is a dict of
    veckey_ref → {name, units, label, shape, elements}.

    An element names its parent with `veckey_ref`; `all_variables` is where
    that ref is looked up, defaulting to `variables` itself.
    """
    all_variables = variables if all_variables is None else all_variables
    scalar_vars = {}
    vector_groups = {}

    for ref, info in variables.items():
        if info.get("idx") is not None:
            vecref = info["veckey_ref"]
            if vecref not in vector_groups:
                vecinfo = all_variables[vecref]
                vector_groups[vecref] = {
                    "name": vecinfo["name"],
                    "units": vecinfo.get("units"),
                    "label": vecinfo.get("label"),
                    "shape": vecinfo["shape"],
                    "elements": [],
                }
            vector_groups[vecref]["elements"].append((ref, info))
        elif info.get("shape") is None:
            scalar_vars[ref] = info
        # a vector parent needs no entry of its own: its elements carry it

    return scalar_vars, vector_groups


def _by_shape(vector_groups):
    "Vector groups keyed by the TOML [vectors.<shape>] sub-table they belong in."
    by_shape = {}
    for group in vector_groups.values():
        shape = group["shape"]
        if isinstance(shape, (list, tuple)):
            shape = shape[0] if len(shape) == 1 else tuple(shape)
        by_shape.setdefault(shape, []).append(group)
    return by_shape


def _format_vector_line(group, substitutions):
    """Format one vector's declaration line.

    A vector whose elements share one value declares that value; one whose
    elements differ declares them all, in the vector's own index order.
    """
    values = np.full(tuple(group["shape"]), None, dtype=object)
    for ref, info in group["elements"]:
        values[tuple(info["idx"])] = substitutions.get(ref)

    distinct = set(values.flat)
    if len(distinct) == 1:
        value = distinct.pop()  # one shared value, or None for a free vector
    elif None in distinct:
        raise ValueError(
            f"cannot write vector '{group['name']}' to TOML: it is partially "
            f"fixed, which the format cannot express"
        )
    else:
        value = values.tolist()
    return _format_var_line(group["name"], value, group["units"], group["label"])


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


def _make_name_resolver(variables, ref_to_model_id, current_model_id):
    """Build a name_fn for ast_to_expr/constraint_to_expr: bare name for a
    variable owned by the model currently being emitted, "Model.name" for
    a reference to a variable owned by a different model (e.g. Aircraft's
    constraint referencing self.engine.W) — otherwise it would print as a
    bare "W" indistinguishable from a same-named local variable."""

    def resolve(ref):
        name = _toml_name(variables[ref])
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
    ir = source.to_ir() if hasattr(source, "to_ir") else source
    check_ir_version(ir)

    lines = []

    # --- name/description ---
    name = ir.get("name", "")
    if name:
        lines.append(f"name = {_quote(name)}")
    desc = ir.get("description", "")
    if desc:
        lines.append(f"description = {_quote(desc)}")
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

    for shape, groups in _by_shape(vector_groups).items():
        lines.append(f"[vectors.{shape}]")
        for group in groups:
            lines.append(_format_vector_line(group, substitutions))
        lines.append("")

    lines.append("[model]")
    cost_ir = ir.get("cost", {})
    name_fn = _entry_names(variables)
    direction, cost_str = _format_objective(cost_ir, name_fn)
    lines.append(f"objective = {_quote(f'{direction}: {cost_str}')}")

    refs = ir["model_tree"]["constraints"]
    if refs:
        lines.append("constraints = [")
        for expr in _written_exprs(refs, ir["constraints"], name_fn):
            lines.append(f"  {_quote(expr)},")
        lines.append("]")


def _emit_multi_model(ir, lines):
    """Emit [models.*] sections from a multi-model IR."""
    tree = ir["model_tree"]
    variables = ir.get("variables", {})
    substitutions = ir.get("substitutions", {})
    constraints = ir["constraints"]

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
    name_fn = _make_name_resolver(variables, ref_to_model_id, model_id)

    # Variables (flat format: vars as keys in model section). node["variables"]
    # is this node's own own_varkeys refs — already scoped to this exact
    # instance, so distinct instances of the same class never mix vars.
    model_vars = {ref: variables[ref] for ref in node.get("variables", [])}
    scalar_vars, vector_groups = _group_variables(model_vars, variables)

    for ref, info in scalar_vars.items():
        vname = info["name"]
        units = info.get("units")
        label = info.get("label")
        value = substitutions.get(ref)
        lines.append(_format_var_line(vname, value, units, label))

    # Objective (root only)
    if is_root and cost_ir:
        direction, cost_str = _format_objective(cost_ir, name_fn)
        lines.append(f"objective = {_quote(f'{direction}: {cost_str}')}")

    # Submodels
    if child_ids:
        child_str = ", ".join(_quote(cid) for cid in child_ids)
        lines.append(f"submodels = [{child_str}]")

    # Constraints
    if node["constraints"]:
        lines.append("constraints = [")
        for expr in _written_exprs(node["constraints"], all_constraints, name_fn):
            lines.append(f"  {_quote(expr)},")
        lines.append("]")

    # Last, because a TOML key belongs to the most recently opened table: this
    # section's own keys have to precede its sub-tables.
    for shape, groups in _by_shape(vector_groups).items():
        lines.append(f"[models.{model_id}.vectors.{shape}]")
        for group in groups:
            lines.append(_format_vector_line(group, substitutions))

    lines.append("")


def _emit_lines(lines, path):
    """Join lines, optionally write to path, and return the string."""
    result = "\n".join(lines)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(result)
    return result


def _format_values(value):
    "Render a number, or a vector's elements as a bracketed list."
    if isinstance(value, list):
        return "[" + ", ".join(_format_values(v) for v in value) + "]"
    return _format_number(value)


def _format_var_line(name, value, units, label):
    """Format a single variable line for TOML output.

    `value` is a number, a list of them, or None for a free variable. A
    dimensionless scalar constant is the one spec that reads as a bare TOML
    number rather than a string.
    """
    if value is None:
        spec = _quote(units if units is not None else "-")
    elif units is not None:
        spec = _quote(f"{_format_values(value)} {units}")
    elif isinstance(value, list):
        spec = _quote(_format_values(value))
    else:
        spec = _format_number(value)

    if label:
        return f"{name} = [{spec}, {_quote(label)}]"
    return f"{name} = {spec}"
