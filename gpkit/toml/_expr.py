"Safe expression parser for TOML model specs."

import ast
import operator

# ---------------------------------------------------------------------------
# Whitelist of allowed AST node types
# ---------------------------------------------------------------------------
_ALLOWED_NODES = frozenset(
    {
        # structural
        ast.Expression,
        ast.Compare,
        # values
        ast.Constant,
        ast.Name,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,  # context for Name/Subscript in eval mode
        # arithmetic
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        # comparison operators
        ast.GtE,
        ast.LtE,
        ast.Eq,
    }
)

_BINOP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


class TomlExpressionError(Exception):
    """Raised when an expression string is invalid or unsafe."""


class _AmbiguousVar:  # pylint: disable=too-few-public-methods
    """Sentinel for variable names defined in multiple models."""

    def __init__(self, name, model_ids):
        self.name = name
        self.model_ids = model_ids


# ---------------------------------------------------------------------------
# AST validation (whitelist check)
# ---------------------------------------------------------------------------


def _validate_ast(tree):
    """Walk the AST and reject any node not in the whitelist."""
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            _reject(node)


def _reject(node):
    """Raise a clear error for a rejected AST node."""
    cls = type(node).__name__
    if isinstance(node, ast.Call):
        func = ""
        if isinstance(node.func, ast.Name):
            func = f" '{node.func.id}'"
        raise TomlExpressionError(
            f"Function calls{func} are not allowed in expressions"
        )
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        raise TomlExpressionError("Imports are not allowed in expressions")
    if isinstance(node, ast.Lambda):
        raise TomlExpressionError("Lambdas are not allowed in expressions")
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        raise TomlExpressionError("Comprehensions are not allowed in expressions")
    raise TomlExpressionError(f"Unsupported syntax ({cls}) in expression")


# ---------------------------------------------------------------------------
# Safe recursive evaluator
# ---------------------------------------------------------------------------


def _eval_constant(node):
    """Evaluate an AST Constant node."""
    if not isinstance(node.value, (int, float)):
        raise TomlExpressionError(f"Non-numeric constant: {node.value!r}")
    return node.value


def _eval_name(node, ns):
    """Look up a variable name in the namespace."""
    if node.id not in ns:
        available = sorted(k for k in ns if not k.startswith("_"))
        raise TomlExpressionError(
            f"Unknown variable '{node.id}'. " f"Available: {', '.join(available)}"
        )
    val = ns[node.id]
    if isinstance(val, _AmbiguousVar):
        raise TomlExpressionError(
            f"Variable '{val.name}' is defined in multiple models: "
            f"{', '.join(val.model_ids)}. "
            f"Use qualified access, e.g. {val.model_ids[0]}.{val.name}"
        )
    return val


def _eval_unary(node, ns):
    """Evaluate a unary operation (negation or positive)."""
    operand = _eval_node(node.operand, ns)
    if isinstance(node.op, ast.USub):
        return -operand
    assert isinstance(node.op, ast.UAdd)
    return +operand


def _eval_binop(node, ns):
    """Evaluate a binary operation (+, -, *, /, **)."""
    left = _eval_node(node.left, ns)
    right = _eval_node(node.right, ns)
    handler = _BINOP_MAP.get(type(node.op))
    if handler is None:
        raise TomlExpressionError(
            f"Unsupported binary operator: {type(node.op).__name__}"
        )
    return handler(left, right)


def _eval_attribute(node, ns):
    """Evaluate attribute access (e.g. wing.S, submodels.W).

    Only allowed on objects that opt in via ``_toml_namespace = True``.
    This prevents traversal of the Python object graph through
    gpkit internals, descriptors, or dunder attributes.
    """
    value = _eval_node(node.value, ns)
    attr = node.attr
    if not getattr(value, "_toml_namespace", False):
        raise TomlExpressionError(
            f"Attribute access (.{attr}) is only allowed on model "
            f"namespaces, not on {type(value).__name__}"
        )
    try:
        return getattr(value, attr)
    except AttributeError:
        raise TomlExpressionError(
            f"Cannot access '.{attr}' on '{type(value).__name__}'"
        ) from None


def _eval_node(node, ns):  # pylint: disable=too-many-return-statements
    """Recursively evaluate an AST node against *ns* (name → object)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ns)
    if isinstance(node, ast.Constant):
        return _eval_constant(node)
    if isinstance(node, ast.Name):
        return _eval_name(node, ns)
    if isinstance(node, ast.Attribute):
        return _eval_attribute(node, ns)
    if isinstance(node, ast.UnaryOp):
        return _eval_unary(node, ns)
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, ns)
    if isinstance(node, ast.Subscript):
        value = _eval_node(node.value, ns)
        idx = _eval_slice(node.slice, ns)
        return value[idx]
    raise TomlExpressionError(  # pragma: no cover
        f"Unhandled AST node: {type(node).__name__}"
    )


def _eval_slice(node, ns):
    """Evaluate a subscript slice/index expression."""
    if isinstance(node, ast.Slice):
        lo = _eval_node(node.lower, ns) if node.lower is not None else None
        hi = _eval_node(node.upper, ns) if node.upper is not None else None
        step = _eval_node(node.step, ns) if node.step is not None else None
        return slice(lo, hi, step)
    # Simple index (integer expression, variable name, etc.)
    return _eval_node(node, ns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def eval_expr(text, namespace):
    """Parse and safely evaluate an expression string.

    Parameters
    ----------
    text : str
        A math expression, e.g. ``"2*h*w + 2*h*d"``.
    namespace : dict
        Mapping of names to gpkit Variable objects, numbers, or ints.

    Returns
    -------
    The result of evaluating the expression using gpkit operator overloads.
    For pure arithmetic this is a Monomial/Posynomial; for a comparison
    it's a gpkit constraint.
    """
    text = text.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise TomlExpressionError(f"Syntax error in expression: {text!r}") from exc
    _validate_ast(tree)
    return _eval_node(tree, namespace)


def parse_constraint(text, namespace):
    """Parse a constraint string like ``"A >= 2*h*w"`` into a gpkit constraint.

    The expression must contain exactly one comparison operator
    (``>=``, ``<=``, or ``==``).  Returns a gpkit constraint object
    (PosynomialInequality, MonomialEquality, etc.).
    """
    text = text.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise TomlExpressionError(f"Syntax error in constraint: {text!r}") from exc
    _validate_ast(tree)

    body = tree.body
    if not isinstance(body, ast.Compare):
        raise TomlExpressionError(
            f"Expected a constraint (>=, <=, ==), got expression: {text}"
        )
    if len(body.ops) != 1:
        raise TomlExpressionError(f"Chained comparisons not supported: {text}")

    left = _eval_node(body.left, namespace)
    right = _eval_node(body.comparators[0], namespace)
    op = body.ops[0]

    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Eq):
        return left == right

    raise TomlExpressionError(  # pragma: no cover
        f"Unsupported comparison operator in: {text}"
    )


def parse_objective(text, namespace):
    """Parse an objective string like ``"min: 1/(h*w*d)"``.

    Returns the cost expression (a gpkit nomial).
    For ``"max: expr"``, returns ``1/expr``.
    """
    text = text.strip()
    if text.startswith("min:"):
        expr_str = text[4:].strip()
    elif text.startswith("max:"):
        expr_str = text[4:].strip()
    else:
        raise TomlExpressionError(
            f"Objective must start with 'min:' or 'max:', got: {text!r}"
        )

    cost = eval_expr(expr_str, namespace)
    if text.startswith("max:"):
        cost = 1 / cost
    return cost
