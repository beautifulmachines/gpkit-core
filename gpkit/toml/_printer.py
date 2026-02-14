"Generate TOML model specs from gpkit Models or IR dicts."

import re
import sys
import types

from ..ast_nodes import ASTNode, ConstNode, ExprNode, VarNode, ast_from_ir

# ---------------------------------------------------------------------------
# Ref string → Python variable name
# ---------------------------------------------------------------------------

# IR ref formats (from VarKey.ref):
#   dimensionless scalar:  "x"
#   scalar with units:     "h|ft"
#   vector:                "d#3|ft"
#   vector element:        "d[0]#3|ft"
#   with lineage:          "Aircraft0.Wing0.S|ft²"
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


def _parenthesize(s, *, for_add=True, for_mul=True):
    """Wrap s in parens if it contains bare (un-parenthesized) operators."""
    depth = 0
    bare = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            bare.append(ch)
    bare_str = "".join(bare)
    has_add = " + " in bare_str or " - " in bare_str
    has_mul = "*" in bare_str or "/" in bare_str
    if (for_add and has_add) or (for_mul and has_mul):
        return f"({s})"
    return s


def _format_number(v):
    """Format a number for TOML expression output."""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.4g}"


def ast_to_expr(node):
    """Convert a gpkit AST node to a plain expression string.

    Accepts VarNode, ConstNode, ExprNode (from gpkit's ast_nodes),
    IR dicts (from to_ir() JSON), or raw numbers.

    Produces TOML-compatible syntax: ``*`` for multiply, ``**`` for power.
    """
    # Raw numbers (e.g. exponents in pow, coefficients)
    if isinstance(node, (int, float)):
        return _format_number(node)

    # IR dict — reconstruct AST nodes first, then render
    if isinstance(node, dict):
        reconstructed = ast_from_ir(node, _RefNameRegistry())
        return ast_to_expr(reconstructed)

    if isinstance(node, VarNode):
        return _ref_to_name(node.ref)

    if isinstance(node, ConstNode):
        return _format_number(node.value)

    if isinstance(node, ExprNode):
        return _render_op(node.op, node.children)

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


def _render_op(
    op, children
):  # pylint: disable=too-many-return-statements,too-many-branches
    """Render an AST operation to a plain expression string."""
    if op == "add":
        left = ast_to_expr(children[0])
        right = ast_to_expr(children[1])
        if right.startswith("-"):
            return f"{left} - {right[1:]}"
        return f"{left} + {right}"

    if op == "mul":
        left = _parenthesize(ast_to_expr(children[0]), for_mul=False)
        right = _parenthesize(ast_to_expr(children[1]), for_mul=False)
        if left == "1":
            return right
        if right == "1":
            return left
        return f"{left}*{right}"

    if op == "div":
        left = _parenthesize(ast_to_expr(children[0]), for_mul=False)
        right = _parenthesize(ast_to_expr(children[1]))
        if right == "1":
            return left
        return f"{left}/{right}"

    if op == "pow":
        left = _parenthesize(ast_to_expr(children[0]))
        exp = children[1]
        exp_str = ast_to_expr(exp) if isinstance(exp, ASTNode) else _format_number(exp)
        if left == "1":
            return "1"
        return f"{left}**{exp_str}"

    if op == "neg":
        val = _parenthesize(ast_to_expr(children[0]), for_mul=False)
        return f"-{val}"

    if op == "sum":
        val = _parenthesize(ast_to_expr(children[0]))
        return f"sum({val})"

    if op == "prod":
        val = _parenthesize(ast_to_expr(children[0]))
        return f"prod({val})"

    if op == "index":
        left = ast_to_expr(children[0])
        idx_str = _format_index(children[1])
        return f"{left}[{idx_str}]"

    raise ValueError(f"Unknown AST op: {op}")


def _format_index(idx):
    """Format an index/slice for display."""
    if isinstance(idx, slice):
        return _format_slice(idx)
    if isinstance(idx, tuple):
        return ",".join(
            _format_slice(el) if isinstance(el, slice) else str(el) for el in idx
        )
    return str(idx)


def _format_slice(s):
    """Format a slice object as a string."""
    start = s.start if s.start is not None else ""
    stop = s.stop if s.stop is not None and s.stop < sys.maxsize else ""
    step = f":{s.step}" if s.step is not None else ""
    return f"{start}:{stop}{step}"


# ---------------------------------------------------------------------------
# Constraint → expression string
# ---------------------------------------------------------------------------

# NOTE: gpkit stores equality as "=" internally, but TOML/Python uses "==".
# This is a gpkit-core inconsistency we should eventually fix upstream
# (track in a separate issue). For now we map here.
_OPER_MAP = {"=": "=="}


def constraint_to_expr(constraint_ir):
    """Convert an IR constraint dict to a TOML constraint string."""
    oper = constraint_ir["oper"]
    oper = _OPER_MAP.get(oper, oper)

    left = _nomial_ir_to_expr(constraint_ir["left"])
    right = _nomial_ir_to_expr(constraint_ir["right"])

    return f"{left} {oper} {right}"


def _nomial_ir_to_expr(nomial_ir):
    """Render a nomial IR dict to an expression string.

    Uses the AST when available (for expressions built from operations).
    For leaf nodes (bare Variables and numeric constants), renders directly
    from the terms — these are the only cases that lack an AST.
    """
    ast = nomial_ir.get("ast")
    if ast is not None:
        return ast_to_expr(ast)

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
                return _ref_to_name(ref)

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


def _format_objective(cost_ir):
    """Determine objective direction and expression string.

    Detects the 1/expr pattern and returns ("max", expr_str) instead of
    ("min", "1/expr") for more natural readability.
    """
    ast = cost_ir.get("ast")
    if ast is not None:
        inner = _is_reciprocal(ast)
        if inner is not None:
            return "max", ast_to_expr(inner)
    return "min", _nomial_ir_to_expr(cost_ir)


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


def _var_model_id(ref):
    """Extract model_id from a variable ref's lineage prefix.

    "wing0.S|ft²" → "wing", "aircraft0.W|lbf" → "aircraft".
    Returns None if no lineage prefix is present.
    """
    bare = _REF_STRIP.sub("", ref)
    if "." not in bare:
        return None
    prefix = bare.rsplit(".", 1)[0]
    return re.sub(r"\d+$", "", prefix)


def _collect_child_ids(tree):
    """Collect all model_ids from children (recursively)."""
    ids = set()
    for child in tree.get("children", []):
        ids.add(child["class"])
        ids.update(_collect_child_ids(child))
    return ids


def _root_model_id(tree, all_variables):
    """Determine the root model's ID from IR variables and model_tree.

    Uses lineage prefixes from variable refs, excluding child model_ids,
    to find the root's unique model_id. Falls back to "main".
    """
    child_ids = _collect_child_ids(tree)

    # Try: find a model_id that appears in variables but not in children
    all_model_ids = {_var_model_id(ref) for ref in all_variables} - {None}
    root_candidates = all_model_ids - child_ids
    if len(root_candidates) == 1:
        return root_candidates.pop()

    # Fallback: check root node's own variable refs
    for ref in tree.get("variables", []):
        mid = _var_model_id(ref)
        if mid and mid not in child_ids:
            return mid

    return "main"


def to_toml(
    source, path=None
):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
    result = "\n".join(lines)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(result)

    return result


def _emit_single_model(ir, lines):  # pylint: disable=too-many-locals
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
        for _, group in vector_groups.items():
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


def _emit_multi_model(ir, lines):  # pylint: disable=too-many-locals,too-many-branches
    """Emit [models.*] sections from a multi-model IR."""
    tree = ir["model_tree"]
    variables = ir.get("variables", {})
    substitutions = ir.get("substitutions", {})
    constraints = ir.get("constraints", [])

    # Group all variables by model_id (from lineage prefix)
    root_id = _root_model_id(tree, variables)
    vars_by_model = {}
    for ref, info in variables.items():
        mid = _var_model_id(ref) or root_id
        vars_by_model.setdefault(mid, {})[ref] = info

    # Flatten tree into ordered list of (model_id, node, child_ids)
    nodes = []

    def flatten(node):
        if node.get("instance_id"):
            model_id = node["class"]
        else:
            model_id = root_id
        child_ids = [c["class"] for c in node.get("children", [])]
        nodes.append((model_id, node, child_ids))
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
            vars_by_model,
            substitutions,
            constraints,
            lines,
            is_root=False,
            cost_ir=None,
        )
    _emit_model_section(
        root_entry[0],
        root_entry[1],
        root_entry[2],
        vars_by_model,
        substitutions,
        constraints,
        lines,
        is_root=True,
        cost_ir=ir.get("cost", {}),
    )


def _emit_model_section(
    model_id,
    node,
    child_ids,
    vars_by_model,
    substitutions,
    all_constraints,
    lines,
    *,
    is_root,
    cost_ir,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Emit a single [models.X] section."""
    lines.append(f"[models.{model_id}]")

    # Variables (flat format: vars as keys in model section)
    model_vars = vars_by_model.get(model_id, {})
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
        for _, group in vector_groups.items():
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
        direction, cost_str = _format_objective(cost_ir)
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
            cstr = constraint_to_expr(c)
            lines.append(f'  "{cstr}",')
        lines.append("]")

    lines.append("")


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
