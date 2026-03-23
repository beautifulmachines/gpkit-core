"""Budget computation and display for GPKit solutions.

A budget traces how a top-level GP variable (e.g. total mass) decomposes
across subcomponents by following the model's budget constraints.  A budget
constraint is any constraint of the form ``m >= components…`` where the
variable being budgeted is the sole term on the greater-than side with
exponent +1.

Usage example::

    m = solution.budget("m_total")   # uses meta["model"] stored at solve time
    print(m.text())
    print(m.markdown())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

from .ast_nodes import ExprNode, VarNode
from .constraints.set import flatiter
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


def _format_term_label(exp, coeff, strip_prefix=None):
    """Format a monomial term's exponent dict + coefficient as a string.

    Example: coeff=0.1, exp={m: 1, f: 1} → "0.1·m·f"

    If ``strip_prefix`` is given (a lineagestr of the parent model), that
    prefix is removed from variable names so only the relative path is shown.
    """
    parts = []
    if abs(coeff - 1.0) > 1e-10:
        parts.append(f"{coeff:.4g}")
    for vk, pow_ in exp.items():
        if strip_prefix:
            name = vk.str_without(["units", f":MAGIC:{strip_prefix}"])
        else:
            name = vk.str_without(["lineage"])
        parts.append(name if pow_ == 1 else f"{name}^{pow_:.4g}")
    return "·".join(parts) if parts else f"{coeff:.4g}"


def _collect_sum_terms(node):
    """Flatten nested 'add' AST nodes into a list of per-addend subtrees."""
    if isinstance(node, ExprNode) and node.op == "add":
        result = []
        for child in node.children:
            result.extend(_collect_sum_terms(child))
        return result
    return [node]


def _vks_of_ast_term(node):
    """Return frozenset of VarKeys referenced in a monomial AST term."""
    if isinstance(node, VarNode):
        return frozenset({node.varkey})
    if isinstance(node, ExprNode) and node.op in ("mul", "div"):
        vks = set()
        for child in node.children:
            vks.update(_vks_of_ast_term(child))
        return frozenset(vks)
    if isinstance(node, ExprNode) and node.op == "pow":
        return _vks_of_ast_term(node.children[0])
    return frozenset()


def _ast_label_map(lt, strip_prefix=None):
    """Build a {frozenset(varkeys): label_str} map from lt.ast.

    Returns an empty dict when lt has no AST (caller falls back to
    _physical_coeff).  The label is generated with units excluded and,
    when *strip_prefix* is given, the model's lineage prefix stripped so
    only the relative path is shown.
    """
    if lt.ast is None:
        return {}
    excluded = ["units"]
    if strip_prefix:
        excluded.append(f":MAGIC:{strip_prefix}")
    result = {}
    for term in _collect_sum_terms(lt.ast):
        vks = _vks_of_ast_term(term)
        if vks:
            result[vks] = term.str_without(excluded)
    return result


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
            if vk.unitrepr:
                natural = natural * qty(vk.unitrepr) ** pow_
        conversion = float(natural.to(hmap_units).magnitude)
        if abs(conversion) > 1e-15:
            return coeff / conversion
    except (DimensionalityError, AttributeError, TypeError, ValueError):
        pass
    return coeff


# ---------------------------------------------------------------------------
# Public API: constraint scanning
# ---------------------------------------------------------------------------


def find_budget_constraints(model, vk, solution):
    """Find constraints where *vk* has exponent +1 as the sole gt variable.

    A budget constraint has the form ``vk >= sum_of_terms`` — *vk* alone on
    the greater-than side with exponent exactly 1.

    Parameters
    ----------
    model : Model
        The model whose constraints are scanned (all levels via flatiter).
    vk : VarKey or Variable
        The variable to look for as the budget root.
    solution : Solution
        Used to look up constraint sensitivities.

    Returns
    -------
    list of (constraint, lt_posynomial, abs_sensitivity)
        Sorted descending by sensitivity so the tightest match comes first.
    """
    vk = getattr(vk, "key", vk)
    matches = []
    for c in flatiter(model):
        gt, lt = _get_gt_lt(c)
        if gt is None or len(gt.hmap) != 1:
            continue
        (exp,) = gt.hmap
        if dict(exp) == {vk: 1}:
            sens = abs(solution.sens.constraints.get(c, 0.0))
            matches.append((c, lt, sens))
    return sorted(matches, key=lambda x: -x[2])


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
        Numerical value in the budget's display units.
    fraction : float
        Fraction of the top-level budget total (0–1).
    slack : float
        Relative slack of *this node's own* budget constraint (0.0 = tight).
        Only non-zero when the node has children derived from a sub-budget
        constraint that is not fully binding.
    children : list[BudgetNode]
        Sub-decomposition of this node, populated when the node's variable
        has its own budget constraint.
    """

    label: str
    vk: Optional[VarKey]
    value: float
    fraction: float
    slack: float
    children: list = field(default_factory=list)


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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def text(self) -> str:
        """Return an aligned-column plain-text budget table."""
        top_label = _vk_display(self.top_vk, lineage=True)
        header = f"Budget [{self.units}]  —  {top_label}"

        # Collect all rows as (indent_depth, label, value_str, pct_str)
        rows = [(0, top_label, f"{self.total:.4g}", "100.0%")]
        _collect_text_rows(self.children, rows, depth=1)

        # Compute column widths (r = (depth, label, val_str, pct_str))
        lbl_w = max(r[0] * 2 + len(r[1]) for r in rows)
        val_w = max(len(r[2]) for r in rows)
        pct_w = max(len(r[3]) for r in rows)

        lines = [header, "-" * len(header)]
        for depth, label, val_str, pct_str in rows:
            indent = "  " * depth
            pad = lbl_w - depth * 2 - len(label)
            lines.append(
                f"  {indent}{label}{' ' * pad}  "
                f"{val_str:>{val_w}}  {pct_str:>{pct_w}}"
            )
        return "\n".join(lines)

    def markdown(self) -> str:
        """Return a GitHub-flavored markdown budget table."""
        top_label = _vk_display(self.top_vk, lineage=True)
        lines = [
            f"| Component | Value [{self.units}] | Fraction |",
            "| --- | ---: | ---: |",
            f"| **{top_label}** | **{self.total:.4g}** | **100.0%** |",
        ]
        _collect_md_rows(self.children, lines, depth=0)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serializable nested dict suitable for browser/Compass."""
        return {
            "variable": str(self.top_vk),
            "total": self.total,
            "units": self.units,
            "children": [_node_to_dict(n) for n in self.children],
        }

    def __repr__(self) -> str:
        return self.text()


# ------------------------------------------------------------------
# Rendering helpers (module-level to avoid circular dataclass refs)
# ------------------------------------------------------------------


def _vk_display(vk, lineage=False, strip_prefix=None):
    """Return a display string for a VarKey.

    With ``lineage=True`` returns the model-qualified path.  If
    ``strip_prefix`` is given (a lineagestr of the parent model), that
    prefix is removed so only the path *below* the parent is shown
    (e.g. ``"Wing.m"`` instead of ``"Aircraft.Wing.m"`` when the parent
    is ``"Aircraft"``).  With ``lineage=False`` returns just the variable
    name without any lineage.
    """
    if lineage:
        if strip_prefix:
            return vk.str_without(["units", f":MAGIC:{strip_prefix}"])
        return vk.str_without(["units"])
    return vk.str_without(["units", "lineage"])


def _collect_text_rows(nodes, rows, depth):
    for node in nodes:
        label = node.label
        if node.slack > 1e-4:
            label += f"  (slack {node.slack * 100:.1f}%)"
        rows.append((depth, label, f"{node.value:.4g}", f"{node.fraction * 100:.1f}%"))
        if node.children:
            _collect_text_rows(node.children, rows, depth + 1)


def _collect_md_rows(nodes, lines, depth):
    indent = "&nbsp;" * (4 * depth)
    for node in nodes:
        label = f"{indent}{node.label}"
        if node.slack > 1e-4:
            label += f" *(slack {node.slack * 100:.1f}%)*"
        lines.append(f"| {label} | {node.value:.4g} | {node.fraction * 100:.1f}% |")
        if node.children:
            _collect_md_rows(node.children, lines, depth + 1)


def _node_to_dict(node):
    return {
        "label": node.label,
        "value": node.value,
        "fraction": node.fraction,
        "slack": node.slack,
        "children": [_node_to_dict(c) for c in node.children],
    }


# ---------------------------------------------------------------------------
# Budget building
# ---------------------------------------------------------------------------


@dataclass
class _BudgetCtx:
    """Shared context passed through recursive budget-building calls."""

    solution: object
    model: object
    display_units: str


def _attach_sub_budget(node, child_vk, ctx, visited, term_val):
    "Recursively attach sub-budget children to *node* if a budget constraint exists."
    sub_matches = find_budget_constraints(ctx.model, child_vk, ctx.solution)
    if not sub_matches:
        return
    if len(sub_matches) > 1:
        warnings.warn(
            f"Multiple budget constraints for {child_vk}; "
            "using highest-sensitivity one.",
            stacklevel=5,
        )
    sub_c, sub_lt = sub_matches[0][0], sub_matches[0][1]
    child_units = child_vk.unitrepr or "dimensionless"
    node.children = _build_children(
        child_vk, sub_lt, sub_c, ctx, visited | {child_vk}, level_units=child_units
    )
    sub_sum = sum(c.value for c in node.children if c.label != "[slack]")
    try:
        child_val = float(ctx.solution[child_vk].to(child_units).magnitude)
    except (DimensionalityError, KeyError, AttributeError, TypeError, ValueError):
        child_val = None
    if child_val:
        node.slack = max(0.0, (child_val - sub_sum) / child_val)


def _process_term(
    top_vk, exp, phys_coeff, term_val, ast_label, ctx, visited, level_units
):
    """Build a single BudgetNode for one term in a budget constraint's RHS.

    Parameters
    ----------
    top_vk : VarKey
        The variable being decomposed (parent level).
    exp : HashVector
        Exponent dict for this monomial term.
    phys_coeff : float
        Physical coefficient with unit-conversion factors stripped.  Used only
        when *ast_label* is ``None`` (AST unavailable).
    term_val : float
        Pre-evaluated numerical value in *level_units*.
    ast_label : str or None
        Label derived from lt.ast (the symbolic pre-unit-conversion form).
        When present this is used directly; when absent we fall back to
        constructing a label from *phys_coeff*.
    ctx : _BudgetCtx
    visited : frozenset[VarKey]
    level_units : str

    Returns
    -------
    BudgetNode
    """
    is_self_ref = top_vk in exp
    free_in_term = {vk for vk in exp if vk not in ctx.solution.constants}

    parent_prefix = top_vk.lineagestr() if top_vk.lineage else None

    is_simple = (
        not is_self_ref
        and len(free_in_term) == 1
        and exp.get(next(iter(free_in_term))) == 1
    )

    if is_simple:
        child_vk = next(iter(free_in_term))
        if ast_label is not None:
            label = ast_label
        else:
            # Fallback when no AST: use physical coefficient for label decisions.
            has_constants = any(vk in ctx.solution.constants for vk in exp)
            if has_constants or abs(phys_coeff - 1.0) > 1e-10:
                label = _format_term_label(exp, phys_coeff, strip_prefix=parent_prefix)
            else:
                label = _vk_display(child_vk, lineage=True, strip_prefix=parent_prefix)
        node = BudgetNode(
            label=label,
            vk=child_vk,
            value=term_val,
            fraction=0.0,
            slack=0.0,
        )
        if (
            child_vk not in visited
            and child_vk.units is not None
            and child_vk.units.is_compatible_with(level_units)
        ):
            _attach_sub_budget(node, child_vk, ctx, visited, term_val)
        return node

    label = (
        ast_label
        if ast_label is not None
        else _format_term_label(exp, phys_coeff, strip_prefix=parent_prefix)
    )
    if is_self_ref:
        label += " [margin]"
    return BudgetNode(label=label, vk=None, value=term_val, fraction=0.0, slack=0.0)


def _build_children(top_vk, lt, constraint, ctx, visited, level_units=None):
    """Recursively build BudgetNode children from the lt side of a budget constraint.

    Parameters
    ----------
    top_vk : VarKey
        The variable being decomposed at this level.
    lt : Posynomial
        The right-hand side of the budget constraint (sum of components).
    constraint : SingleEquationConstraint
        The budget constraint (used for sensitivity check).
    ctx : _BudgetCtx
        Shared context (solution, model, display_units).
    visited : frozenset[VarKey]
        Guards against infinite recursion.
    level_units : str, optional
        Units for values at this recursion level.  Defaults to ctx.display_units
        (top-level call) but sub-levels pass the child variable's own units so
        that cross-dimensional recursion (e.g. mass → volume → area) works.

    Returns
    -------
    list[BudgetNode]
    """
    if level_units is None:
        level_units = ctx.display_units
    total_val = float(ctx.solution[top_vk].to(level_units).magnitude)
    is_tight = abs(ctx.solution.sens.constraints.get(constraint, 0.0)) > 1e-5

    parent_prefix = top_vk.lineagestr() if top_vk.lineage else None
    ast_labels = _ast_label_map(lt, strip_prefix=parent_prefix)
    hmap_units = lt.hmap.units

    nodes = []
    for mon in lt.chop():
        (exp,) = mon.hmap.keys()
        coeff = mon.hmap[exp]
        ast_label = ast_labels.get(frozenset(exp.keys()))
        phys_coeff = (
            coeff if ast_label is not None else _physical_coeff(exp, coeff, hmap_units)
        )
        try:
            term_qty = ctx.solution[mon]
            term_val = float(term_qty.to(level_units).magnitude)
        except (DimensionalityError, KeyError, AttributeError, TypeError, ValueError):
            term_val = float("nan")
        nodes.append(
            _process_term(
                top_vk, exp, phys_coeff, term_val, ast_label, ctx, visited, level_units
            )
        )

    for node in nodes:
        node.fraction = node.value / total_val if total_val else 0.0

    if not is_tight:
        children_sum = sum(n.value for n in nodes)
        slack_val = total_val - children_sum
        if total_val and abs(slack_val / total_val) > 1e-4:
            nodes.append(
                BudgetNode(
                    label="[slack]",
                    vk=None,
                    value=slack_val,
                    fraction=slack_val / total_val,
                    slack=0.0,
                )
            )

    return nodes


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


def build_budget(solution, model, var, display_units=None):
    """Build a :class:`Budget` for a variable by scanning the model's constraints.

    Parameters
    ----------
    solution : Solution
        A solved GPKit solution.
    model : Model
        The model to scan for budget constraints.
    var : Variable or VarKey
        The top-level budget variable (e.g. ``model.m_total``).
    display_units : str, optional
        Units for all displayed values.  Defaults to the variable's own units.

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
        display_units = top_vk.unitrepr or "dimensionless"

    matches = find_budget_constraints(model, top_vk, solution)
    if not matches:
        raise ValueError(
            f"No budget constraint found for {top_vk!r}. "
            f"A budget constraint must have '{top_vk.name} >= ...' with "
            f"'{top_vk.name}' as the sole term on the greater-than side."
        )
    if len(matches) > 1:
        warnings.warn(
            f"Multiple budget constraints found for {top_vk}; "
            "using highest-sensitivity one.",
            stacklevel=2,
        )

    constraint, lt, _ = matches[0]
    total_val = float(solution[top_vk].to(display_units).magnitude)
    ctx = _BudgetCtx(solution=solution, model=model, display_units=display_units)

    children = _build_children(top_vk, lt, constraint, ctx, visited={top_vk})

    return Budget(
        top_vk=top_vk, total=total_val, units=display_units, children=children
    )
