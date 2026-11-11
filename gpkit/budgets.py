"""Budget computation and display for GPKit solutions.

A budget traces how a top-level GP variable (e.g. total mass) decomposes
across subcomponents by following the model's budget constraints.  A budget
constraint is any constraint of the form ``m >= components…`` where the
variable being budgeted is the sole term on the greater-than side with
exponent +1.

Usage example::

    m = model.budget(solution, model.m_total)
    print(m.text())
    print(m.markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast_nodes import ExprNode, VarNode, variables_of, written_ast
from .constraints.set import constraint_senss
from .display.naming import DisplayScope
from .display.render import TEXT, names_of, render_ast
from .display.spelling import MUL
from .units import DimensionalityError, qty
from .varkey import VarKey

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_gt_lt(constraint):
    """Return (gt_side, lt_side) for a constraint, or (None, None)."""
    if not hasattr(constraint, "oper"):
        return None, None
    if constraint.oper == ">=":
        return constraint.left, constraint.right
    if constraint.oper == "<=":
        return constraint.right, constraint.left
    if constraint.oper == "=":
        return constraint.left, constraint.right
    return None, None


_FULL_PATH = DisplayScope(excluded={"units"})  # how a budget names its top variable


def _scope(parent_vk):
    "How a budget names its parent's terms: relative to the parent's model."
    return DisplayScope(anchor=parent_vk.lineage, excluded={"units"})


def _format_term_label(exp, coeff, scope):
    """Format a monomial term's exponent dict + coefficient as a string.

    Example: coeff=0.1, exp={m: 1, f: 1} → "0.1⋅m⋅f"
    """
    parts = []
    if abs(coeff - 1.0) > 1e-10:
        parts.append(f"{coeff:.4g}")
    for vk, pow_ in exp.items():
        name = scope.text(vk)
        parts.append(name if pow_ == 1 else f"{name}^{pow_:.4g}")
    return MUL.join(parts) if parts else f"{coeff:.4g}"


def _collect_sum_terms(node):
    """Flatten nested 'add' AST nodes into a list of per-addend subtrees."""
    if isinstance(node, ExprNode) and node.op == "add":
        result = []
        for child in node.children:
            result.extend(_collect_sum_terms(child))
        return result
    return [node]


def _vks_of_ast_term(node, variables):
    """Return frozenset of VarKeys referenced in a monomial AST term."""
    if isinstance(node, VarNode):
        return frozenset({variables[node.ref]})
    if isinstance(node, ExprNode) and node.op in ("mul", "div"):
        vks = set()
        for child in node.children:
            vks.update(_vks_of_ast_term(child, variables))
        return frozenset(vks)
    if isinstance(node, ExprNode) and node.op == "pow":
        return _vks_of_ast_term(node.children[0], variables)
    return frozenset()


def _ast_label_map(lt, scope):
    """Build label and is_var_node maps from lt.ast.

    Returns a pair ``(labels, is_var)`` where both are
    ``{frozenset(varkeys): value}`` dicts.  ``labels`` maps to the display
    string for each addend term; ``is_var`` maps to ``True`` when the AST
    term is a bare VarNode (user wrote the variable directly, no scaling)
    and ``False`` otherwise.  Both dicts are empty when lt has no AST.
    """
    variables = variables_of(lt)
    tree = written_ast(lt, variables) if lt.ast is not None else None
    if tree is None:
        return {}, {}
    names = names_of(variables, TEXT, scope)
    labels = {}
    is_var = {}
    for term in _collect_sum_terms(tree):
        vks = _vks_of_ast_term(term, variables)
        if vks:
            labels[vks] = render_ast(term, TEXT, names)
            is_var[vks] = isinstance(term, VarNode)
    return labels, is_var


def _physical_coeff(exp, coeff, hmap_units):
    """Return *coeff* with the unit-conversion factor divided out.

    NomialMap.__add__ embeds a ``float(other.units / self.units)`` factor
    into each coefficient when combining terms with different units.  This
    helper reverses that so the displayed coefficient matches what the user
    wrote in the original expression.

    Falls back to the raw *coeff* if units are absent or conversion fails.
    """
    if not hmap_units:
        return coeff
    try:
        natural = qty("dimensionless")
        for vk, pow_ in exp.items():
            if vk.units is not None:
                natural = natural * vk.units**pow_
        conversion = float(natural.to(hmap_units).magnitude)
        if abs(conversion) > 1e-15:
            return coeff / conversion
    except (DimensionalityError, AttributeError, TypeError, ValueError):
        pass
    return coeff


# ---------------------------------------------------------------------------
# Public API: constraint scanning
# ---------------------------------------------------------------------------


def find_budget_constraints(vk, csenss):
    """Find constraints where *vk* has exponent +1 as the sole gt variable.

    A budget constraint has the form ``vk >= sum_of_terms`` — *vk* alone on
    the greater-than side with exponent exactly 1.

    Parameters
    ----------
    vk : VarKey or Variable
        The variable to look for as the budget root.
    csenss : dict
        {constraint: sensitivity}, as constraint_senss(model, solution) joins
        them: the constraints to scan, with how much each one binds.

    Returns
    -------
    list of (constraint, lt_posynomial, abs_sensitivity)
        Sorted descending by sensitivity so the tightest match comes first.
    """
    vk = getattr(vk, "key", vk)
    matches = []
    for c, sens in csenss.items():
        gt, lt = _get_gt_lt(c)
        if gt is None or len(gt.hmap) != 1:
            continue
        (exp,) = gt.hmap
        if dict(exp) == {vk: 1}:
            matches.append((c, lt, abs(sens)))
    return sorted(matches, key=lambda x: -x[2])


def _drop_nonbinding(matches, threshold=1e-5):
    "Drop near-zero-sensitivity matches when at least one binding match remains."
    binding = [m for m in matches if m[2] > threshold]
    return binding if binding else matches


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BudgetNode:
    """A single line in a hierarchical budget breakdown.

    Attributes
    ----------
    label : str
        Short display name — either the variable's own name (without lineage)
        or a formatted expression string for compound terms.
    vk : VarKey or None
        The underlying VarKey for simple named-variable nodes; ``None`` for
        compound / expression nodes.
    value : float
        Numerical value in *units* (the node's own per-row units).
    fraction : float
        Fraction of its parent row's value (0–1).
    slack : float
        Relative slack of *this node's own* budget constraint (0.0 = tight).
        Only non-zero when the node has children derived from a sub-budget
        constraint that is not fully binding.
    units : str
        Units for *value* — the node's own declared units (e.g. ``"g"`` for a
        variable specified in grams even when the budget level is ``"kg"``).
        Compound and slack nodes use the budget level units.
    children : list[BudgetNode]
        Sub-decomposition of this node, populated when the node's variable
        has its own budget constraint.
    """

    label: str
    vk: VarKey | None
    value: float
    fraction: float
    slack: float
    units: str = ""
    children: list = field(default_factory=list)
    cbe_value: float = 0.0
    ga_value: float = 0.0
    has_growth: bool = False
    # Fraction of *value* peeled off as a growth allowance at this level
    # (excludes growth from descendants).  Drives the leaf cbe/ga split.
    peeled_frac: float = 0.0


@dataclass
class Budget:
    """Hierarchical budget for a single GP variable.

    Build via :func:`build_budget`; render via :meth:`text`, :meth:`markdown`,
    or :meth:`to_dict`.

    Attributes
    ----------
    top_vk : VarKey
    total : float
        Value of the top-level variable in *units*.
    units : str
        Display units string (e.g. ``"kg"``).
    children : list[BudgetNode]
        Direct decomposition of the top-level variable.
    """

    top_vk: VarKey
    total: float
    units: str
    children: list = field(default_factory=list)
    cbe_total: float = 0.0
    ga_total: float = 0.0
    has_growth: bool = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _show_growth_columns(self) -> bool:
        "True when any growth allowance is present in this budget."
        return abs(self.ga_total) > 1e-6 * max(abs(self.total), 1.0)

    def _top_node(self) -> BudgetNode:
        "The top variable as a row, so every level renders the same way."
        return BudgetNode(
            label=_FULL_PATH.text(self.top_vk),
            vk=self.top_vk,
            value=self.total,
            fraction=1.0,
            slack=0.0,
            units=self.units,
            children=self.children,
            cbe_value=self.cbe_total,
            ga_value=self.ga_total,
            has_growth=self.has_growth,
        )

    def text(self) -> str:
        """Return an aligned-column plain-text budget table.

        When the budget contains any growth allowance, two extra columns
        (Nominal and Growth) appear before the totals column.
        """
        top, show_growth = self._top_node(), self._show_growth_columns()
        rows = []
        if show_growth:  # titled columns; a title also sets its column's width
            rows.append((0, "", [*_value_titles(show_growth), "", ""]))
        for depth, node in _walk(top):
            label = node.label + _slack_note(node, "  (slack {:.1f}%)")
            rows.append((depth, label, _cells(node, show_growth)))
        lbl_w = max(2 * depth + len(label) for depth, label, _ in rows)
        widths = [max(map(len, col)) for col in zip(*(cells for _, _, cells in rows))]
        header = f"Budget  —  {top.label}"
        lines = [header, "-" * len(header)]
        for depth, label, cells in rows:
            pad = " " * (lbl_w - 2 * depth - len(label))
            values = "".join(f"  {cell:>{w}}" for cell, w in zip(cells, widths))
            lines.append(f"  {'  ' * depth}{label}{pad}{values}")
        return "\n".join(lines)

    def markdown(self) -> str:
        """Return a GitHub-flavored markdown budget table."""
        top, show_growth = self._top_node(), self._show_growth_columns()
        titles = _value_titles(show_growth)
        lines = [
            _md_row(["Component", *titles, "Units", "Fraction"]),
            _md_row(["---", *["---:"] * len(titles), ":---", "---:"]),
        ]
        for depth, node in _walk(top):
            cells = _cells(node, show_growth)
            if depth == 0:
                lines.append(_md_row(f"**{c}**" for c in [node.label, *cells]))
            else:
                indent = "&nbsp;" * (4 * (depth - 1))
                label = indent + node.label + _slack_note(node, " *(slack {:.1f}%)*")
                lines.append(_md_row([label, *cells]))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serializable nested dict. Every row, the top
        included, has the same keys."""
        return _node_to_dict(self._top_node())

    def __repr__(self) -> str:
        return self.text()


# ------------------------------------------------------------------
# Rendering helpers (module-level to avoid circular dataclass refs)
# ------------------------------------------------------------------


def _walk(node, depth=0):
    "(depth, node) for a node and every row under it, depth-first."
    yield depth, node
    for child in node.children:
        yield from _walk(child, depth + 1)


def _value_titles(show_growth):
    "Titles of the number columns."
    return ["Nominal", "Growth", "Total"] if show_growth else ["Value"]


def _cells(node, show_growth):
    "One row's columns after its label: the numbers, then units and fraction."
    numbers = [f"{node.value:.4g}"]
    if show_growth:
        growth = f"{node.ga_value:.4g}" if node.has_growth else ""
        numbers = [f"{node.cbe_value:.4g}", growth, *numbers]
    return [*numbers, f"[{node.units}]", f"{node.fraction * 100:.1f}%"]


def _slack_note(node, template):
    "The template filled with the row's slack percentage, if it has slack."
    return template.format(node.slack * 100) if node.slack > 1e-4 else ""


def _md_row(cells):
    return "| " + " | ".join(cells) + " |"


def _node_to_dict(node):
    d = {
        "label": node.label,
        "value": node.value,
        "cbe_value": node.cbe_value,
        "ga_value": node.ga_value,
        "has_growth": node.has_growth,
        "units": node.units,
        "fraction": node.fraction,
        "slack": node.slack,
        "children": [_node_to_dict(c) for c in node.children],
    }
    # compound terms and [slack] are not variables, so they have no ref
    if node.vk is not None:
        d["ref"] = node.vk.ref
    return d


# ---------------------------------------------------------------------------
# Budget building
# ---------------------------------------------------------------------------


@dataclass
class _BudgetCtx:
    """Shared context passed through recursive budget-building calls."""

    solution: object
    csenss: dict  # {constraint: sensitivity}, joined through the model
    display_units: str
    level_units: str = ""
    visited: frozenset = field(default_factory=frozenset)


@dataclass
class _TermData:
    """Per-monomial data assembled before building a BudgetNode."""

    exp: object
    phys_coeff: float
    term_val: float
    ast_label: str | None
    is_var_node: bool | None = None  # True iff AST term is a bare VarNode
    term_qty: object = None  # pint Quantity from solution[mon], or None on failure


def _make_term(mon, hmap_units, ast_labels, ast_is_var, ctx):
    """Build _TermData for one monomial term."""
    (exp,) = mon.hmap.keys()
    coeff = mon.hmap[exp]
    vks = frozenset(exp.keys())
    ast_label = ast_labels.get(vks)
    is_var_node = ast_is_var.get(vks)
    phys_coeff = (
        coeff if ast_label is not None else _physical_coeff(exp, coeff, hmap_units)
    )
    term_qty = None
    try:
        term_qty = ctx.solution[mon]
        term_val = float(term_qty.to(ctx.level_units).magnitude)
    except (DimensionalityError, KeyError, AttributeError, TypeError, ValueError):
        term_val = float("nan")
    return _TermData(
        exp=exp,
        phys_coeff=phys_coeff,
        term_val=term_val,
        ast_label=ast_label,
        is_var_node=is_var_node,
        term_qty=term_qty,
    )


def _attach_sub_budget(node, child_vk, ctx):
    "Recursively attach sub-budget children to *node* if a budget constraint exists."
    sub_matches = find_budget_constraints(child_vk, ctx.csenss)
    if len(sub_matches) > 1:
        sub_matches = _drop_nonbinding(sub_matches)
    if not sub_matches or len(sub_matches) > 1:
        # No bound, or multiple genuinely tight bounds — leaf, stop expansion.
        return
    sub_c, sub_lt = sub_matches[0][0], sub_matches[0][1]
    child_units = child_vk.unitrepr if child_vk.unitrepr != "-" else "dimensionless"
    sub_ctx = _BudgetCtx(
        solution=ctx.solution,
        csenss=ctx.csenss,
        display_units=ctx.display_units,
        level_units=child_units,
        visited=ctx.visited | {child_vk},
    )
    node.children, node.peeled_frac, node.slack = _build_children(
        child_vk, sub_lt, sub_c, sub_ctx
    )


def _process_term(top_vk, term, ctx):
    """Build a single BudgetNode for one term in a budget constraint's RHS.

    Bare-variable terms (``m``, ``m_min``) become recursable nodes — we try
    to expand a sub-budget below them.  Compound terms (``rho*A``, ``1/m``,
    ``2*m``) become non-recursing leaves; ``_is_redundant_compound_leaf``
    drops them when they're the only child of their parent.
    """
    scope = _scope(top_vk)
    is_var = _term_is_var(term)
    if not is_var:
        label = (
            term.ast_label
            if term.ast_label is not None
            else _format_term_label(term.exp, term.phys_coeff, scope)
        )
        return BudgetNode(
            label=label,
            vk=None,
            value=term.term_val,
            fraction=0.0,
            slack=0.0,
            units=ctx.level_units,
        )
    (child_vk,) = term.exp.keys()
    label = term.ast_label if term.ast_label is not None else scope.text(child_vk)
    # Render in the variable's own declared units when they differ from
    # the parent's level units.
    units = child_vk.unitrepr if child_vk.unitrepr != "-" else ctx.level_units
    if units != ctx.level_units and term.term_qty is not None:
        val = float(term.term_qty.to(units).magnitude)
    else:
        units = ctx.level_units
        val = term.term_val
    node = BudgetNode(
        label=label,
        vk=child_vk,
        value=val,
        fraction=0.0,
        slack=0.0,
        units=units,
    )
    if (
        child_vk not in ctx.visited
        and child_vk.units is not None
        and child_vk.units.is_compatible_with(ctx.level_units)
    ):
        _attach_sub_budget(node, child_vk, ctx)
    return node


def _term_is_var(term):
    "True when *term* is a bare variable reference."
    if term.ast_label is not None:
        # AST tracked the term — trust its var/expression classification.
        return term.is_var_node is True
    if len(term.exp) != 1:
        return False
    (exp_value,) = term.exp.values()
    return exp_value == 1 and abs(term.phys_coeff - 1.0) <= 1e-10


def _is_allowance_term(term, parent_vk):
    "True if *term* is the auto-generated growth allowance term of *parent_vk*."
    if parent_vk.growth is None or len(term.exp) != 1:
        return False
    (vk,) = term.exp
    if term.exp[vk] != 1:
        return False
    return vk.name == f"{parent_vk.name}_growth" and vk.lineage == parent_vk.lineage


def _is_redundant_compound_leaf(nodes):
    "True when *nodes* is a single compound (non-bare-var) row — drop it."
    return len(nodes) == 1 and nodes[0].vk is None


def _build_children(top_vk, lt, constraint, ctx):
    """Recursively build BudgetNode children from the lt side of a budget constraint.

    When *top_vk* declares a growth allowance, the auto-generated allowance
    term in its budget constraint is peeled off rather than rendered as a
    child row — its value flows into the parent's ``ga_value`` instead.

    Returns ``(nodes, peeled_frac, slack_frac)``.  ``slack_frac`` is the
    constraint's relative slack (0 when tight) so callers don't have to
    reverse-engineer it from child fractions.
    """
    total_val = float(ctx.solution[top_vk].to(ctx.level_units).magnitude)
    is_tight = abs(ctx.csenss.get(constraint, 0.0)) > 1e-5
    ast_labels, ast_is_var = _ast_label_map(lt, _scope(top_vk))
    nodes = []
    peeled_frac = 0.0
    for mon in lt.chop():
        term = _make_term(mon, lt.hmap.units, ast_labels, ast_is_var, ctx)
        if _is_allowance_term(term, top_vk):
            peeled_frac += term.term_val / total_val if total_val else 0.0
            continue
        node = _process_term(top_vk, term, ctx)
        node.fraction = term.term_val / total_val if total_val else 0.0
        nodes.append(node)
    slack_frac = 0.0
    if not is_tight and total_val:
        slack_frac = max(0.0, 1.0 - sum(n.fraction for n in nodes) - peeled_frac)
        if slack_frac > 1e-4:
            nodes.append(
                BudgetNode(
                    label="[slack]",
                    vk=None,
                    value=slack_frac * total_val,
                    fraction=slack_frac,
                    slack=0.0,
                    units=ctx.level_units,
                )
            )
        else:
            slack_frac = 0.0
    if _is_redundant_compound_leaf(nodes):
        # Compound monomial expression as a lone child — visually redundant
        # with the parent's value; drop it regardless of depth.
        return [], peeled_frac, slack_frac
    return nodes, peeled_frac, slack_frac


def _nominal_share(children, child_shares, peeled_frac):
    """The share of a row's value that is nominal (cbe): its children's shares
    weighted by their fractions, or for a leaf, what its own growth leaves.

    Shares roll up rather than values, because a child's value may be in its
    own units while its fraction is always of its parent."""
    if children:
        return sum(c.fraction * s for c, s in zip(children, child_shares))
    return 1.0 - peeled_frac


def _split_growth(node):
    """Set a row's cbe/ga split and ``has_growth``, post-order, and return its
    nominal share. ``has_growth`` propagates up from peeled allowances at this
    level or in any descendant."""
    child_shares = [_split_growth(child) for child in node.children]
    share = _nominal_share(node.children, child_shares, node.peeled_frac)
    node.cbe_value = node.value * share
    node.ga_value = node.value - node.cbe_value
    node.has_growth = node.peeled_frac > 0 or any(c.has_growth for c in node.children)
    return share


def _resolve_vk(var):
    """Resolve a Variable or VarKey to a VarKey."""
    if isinstance(var, VarKey):
        return var
    if hasattr(var, "key"):
        return var.key
    raise TypeError(
        f"Expected a Variable or VarKey; got {type(var).__name__!r}. "
        "Use model.get_var('path.to.var') to look up a variable by name."
    )


def _trim_to_depth(nodes, remaining_depth):
    "Remove children beyond *remaining_depth* levels; cbe/ga values are unchanged."
    if remaining_depth <= 0:
        for node in nodes:
            node.children = []
        return
    for node in nodes:
        _trim_to_depth(node.children, remaining_depth - 1)


def build_budget(model, solution, var, display_units=None, depth=float("inf")):
    """Build a :class:`Budget` for a variable by scanning the solved constraints.

    Parameters
    ----------
    model : Model
        The model solved, which holds the constraints a budget is read from.
    solution : Solution
        A solution of *model*.
    var : Variable or VarKey
        The top-level budget variable (e.g. ``model.m_total``).
    display_units : str, optional
        Units for all displayed values.  Defaults to the variable's own units.
    depth : int or float, optional
        Maximum display depth. ``depth=0`` returns only the top variable with
        no children. ``depth=1`` shows one level of children. Defaults to
        ``float('inf')`` (fully recursive). Growth allowances and cbe/ga totals
        are always computed from the full tree regardless of *depth*.

    Returns
    -------
    Budget
        A hierarchical budget object with ``.text()``, ``.markdown()``, and
        ``.to_dict()`` rendering methods.

    Raises
    ------
    TypeError
        If *var* is not a Variable or VarKey.
    ValueError
        If no budget constraint is found for the variable.
    """
    top_vk = _resolve_vk(var)

    if display_units is None:
        display_units = top_vk.unitrepr if top_vk.unitrepr != "-" else "dimensionless"

    csenss = constraint_senss(model, solution)
    matches = find_budget_constraints(top_vk, csenss)
    if not matches:
        raise ValueError(
            f"No budget constraint found for {top_vk!r}. "
            f"A budget constraint must have '{top_vk.name} >= ...' with "
            f"'{top_vk.name}' as the sole term on the greater-than side."
        )
    if len(matches) > 1:
        matches = _drop_nonbinding(matches)
    if len(matches) > 1:
        rhs_list = "\n  ".join(str(m[1]) for m in matches)
        raise ValueError(
            f"Cannot build a budget for {top_vk}: multiple budget constraints "
            f"are simultaneously tight (the decomposition is ambiguous). "
            f"Right-hand sides:\n  {rhs_list}"
        )

    constraint, lt, _ = matches[0]
    total_val = float(solution[top_vk].to(display_units).magnitude)

    if depth == 0:
        return Budget(
            top_vk=top_vk,
            total=total_val,
            units=display_units,
            children=[],
            cbe_total=total_val,
            ga_total=0.0,
        )

    ctx = _BudgetCtx(
        solution=solution,
        csenss=csenss,
        display_units=display_units,
        level_units=display_units,
        visited=frozenset({top_vk}),
    )
    children, peeled_frac, _ = _build_children(top_vk, lt, constraint, ctx)
    shares = [_split_growth(child) for child in children]
    cbe_total = total_val * _nominal_share(children, shares, peeled_frac)
    has_growth = peeled_frac > 0 or any(c.has_growth for c in children)
    _trim_to_depth(children, depth - 1)
    return Budget(
        top_vk=top_vk,
        total=total_val,
        units=display_units,
        children=children,
        cbe_total=cbe_total,
        ga_total=total_val - cbe_total,
        has_growth=has_growth,
    )
