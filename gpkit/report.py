"""Report infrastructure: format-independent IR and rendering backends.

ReportSection is the single tree that all output formats render from.
build_report_ir() traverses the model tree once; renderers are pure
functions of the IR.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .constraints.tight import Tight
from .model import Model as _Model
from .nomials import Variable
from .printing import _format_aligned_columns
from .util.repr_conventions import unitstr
from .util.small_classes import Quantity
from .varkey import lineage_display_context
from .varmap import VarMap


@dataclass
class VarEntry:
    """A variable row in a report section."""

    name: str  # display name (from VarKey.name)
    latex: str  # LaTeX rendering (from VarKey.latex())
    value: Any  # float | ndarray | None — never a pint Quantity
    sensitivity: Optional[float]  # shadow price from solution, or None
    units: str  # unit string
    label: str  # human label (from VarKey.label or "")
    source: str = ""  # lineagestr() for cross-model referenced vars; "" for local

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return {
            "name": self.name,
            "latex": self.latex,
            "value": _serialize_value(self.value),
            "sensitivity": self.sensitivity,
            "units": self.units,
            "label": self.label,
            "source": self.source,
        }


@dataclass
class CGroup:
    """A named constraint group within a report section."""

    label: str  # "" for unnamed groups
    constraints: list  # raw constraint objects; to_dict() serializes via str()


@dataclass
class ReportSection:  # pylint: disable=too-many-instance-attributes
    """Format-independent intermediate representation for model reports.

    lineage_map is a rendering hint (VarKey→display-depth) that allows
    the text renderer to use section-local variable name abbreviations in
    constraints. It is not semantic data and is excluded from to_dict().
    """

    title: str
    description: str
    assumptions: list  # list of str
    free_variables: list  # list of VarEntry — optimized by the solver
    fixed_variables: list  # list of VarEntry — prescribed constants
    constraint_groups: list  # list of CGroup
    lineage_path: str = ""  # dotted path e.g. "Aircraft.Wing"
    magic_prefix: str = (
        ""  # model.lineagestr() — stripped from variable names in renderers
    )
    is_anonymous: bool = False  # True for bare Model(...) instances (no subclass name)
    children: list = field(default_factory=list)  # list of ReportSection
    lineage_map: dict = field(default_factory=dict)  # NOT in to_dict
    references: list = field(default_factory=list)  # list of str
    front_matter: str = ""  # raw text/MD prepended at root only
    toc: bool = (
        False  # insert TOC marker (supported by renderers that have a native facility)
    )
    objective_str: str = ""  # text representation of cost expression; "" if constant
    objective_latex: str = ""  # LaTeX representation of cost expression
    objective_label: str = ""  # variable label when expr is a single variable; else ""
    objective_value: Optional[float] = (
        None  # magnitude of attained cost; None if unsolved
    )
    objective_units: str = ""  # unit string for the cost expression
    objective_direction: str = "minimize"  # "minimize" or "maximize"

    def to_dict(self) -> dict:
        """JSON-serializable dict (for format='dict' and future API)."""
        return {
            "title": self.title,
            "lineage_path": self.lineage_path,
            "magic_prefix": self.magic_prefix,
            "objective_direction": self.objective_direction,
            "objective_label": self.objective_label,
            "is_anonymous": self.is_anonymous,
            "description": self.description,
            "assumptions": list(self.assumptions),
            "references": list(self.references),
            "front_matter": self.front_matter,
            "toc": self.toc,
            "objective_str": self.objective_str,
            "objective_latex": self.objective_latex,
            "objective_value": self.objective_value,
            "objective_units": self.objective_units,
            "free_variables": [v.to_dict() for v in self.free_variables],
            "fixed_variables": [v.to_dict() for v in self.fixed_variables],
            "constraint_groups": [
                {"label": cg.label, "constraints": [str(c) for c in cg.constraints]}
                for cg in self.constraint_groups
            ],
            "children": [c.to_dict() for c in self.children],
        }


# ── Value helpers ─────────────────────────────────────────────────────────────


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable.

    val must be a plain float, numpy array, or None — not a pint Quantity.
    """
    if val is None:
        return None
    if isinstance(val, Quantity):
        raise TypeError(
            f"_serialize_value received a pint Quantity ({val!r}); "
            "VarEntry.value must be a plain numeric"
        )
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def _value_units(vk, varmap: VarMap) -> Tuple[Any, str]:
    """Get (magnitude, unit_str) for vk from varmap.

    Both values come from the same pint Quantity returned by varmap.quantity(),
    guaranteeing they are consistent.  Raises KeyError if vk is not in varmap.
    """
    qty = varmap.quantity(vk)
    return qty.magnitude, unitstr(qty) or "-"


def _resolve_sensitivity(vk, solution=None):
    """Get sensitivity (shadow price) for a variable from solution.

    Returns float for scalar variables, numpy array for vector variables,
    or None if unavailable.
    """
    if solution is None:
        return None
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        sens_vars = solution.sens.variables
        if vk not in sens_vars:
            return None
        raw = sens_vars[vk]
        mag = getattr(raw, "magnitude", raw)
        arr = np.asarray(mag)
        return arr if arr.shape else float(arr)
    except (AttributeError, KeyError, TypeError):
        return None


# ── Constraint rendering ──────────────────────────────────────────────────────


def _render_constraint(c) -> str:
    """Render a constraint, stripping units but respecting active lineage context."""
    try:
        return c.str_without({"units"})
    except AttributeError:
        return str(c)


# ── Core builder helpers ──────────────────────────────────────────────────────


def _collect_constraint_varkeys(constraint_groups: List[CGroup]) -> set:
    """Collect all VarKeys appearing in local constraint groups.

    Note: ConstraintSet.constrained_varkeys() (set.py) does the same concept
    but operates over model.vks, which aggregates the entire subtree including
    child models. Here we only touch the leaf constraints already filtered by
    _build_constraint_groups, giving us just this level's own equations.
    ConstraintSet has unique_varkeys as a class attribute, so hasattr() returns
    True for all ConstraintSet/Model instances — _build_constraint_groups
    already excluded them, leaving only leaf ScalarSingleEquationConstraints
    whose .vks is exactly the variables in that one equation.
    """
    vkeys: set = set()
    for cg in constraint_groups:
        for c in cg.constraints:
            if hasattr(c, "vks"):
                vkeys.update(c.vks)
    return vkeys


def _make_var_entry(
    display_vk, excluded, get_val_units, solution, source=""
) -> VarEntry:
    "Build a VarEntry for display_vk using the provided value-lookup callable."
    value, units_str = get_val_units(display_vk)
    return VarEntry(
        name=display_vk.str_without(excluded),
        latex=display_vk.latex(excluded),
        value=value,
        sensitivity=_resolve_sensitivity(display_vk, solution=solution),
        units=units_str,
        label=display_vk.label or "",
        source=source,
    )


def _is_free_vk(display_vk, solution, model) -> bool:
    """Return True if display_vk is an optimized (free) variable.

    With a solution: free iff the key appears in solution.primal.
    Without a solution: free iff the key has no substitution in the model.
    """
    if solution is not None:
        return display_vk in solution.primal
    return display_vk not in model.substitutions


def _build_split_var_entries(
    model, solution, extra_vks=None
) -> Tuple[List[VarEntry], List[VarEntry]]:
    """Build (free_entries, fixed_entries) from model.unique_varkeys.

    free_entries  — Optimized Variables: solved by the optimizer (no sens shown).
    fixed_entries — Fixed Variables: prescribed constants with sensitivities.

    Local variables: names disambiguated within the section scope using
    _get_lineage_map(); model's own lineage stripped so only sub-model context
    is shown. Vector variables collapsed to a single VarEntry.

    Cross-model variables (extra_vks): full dotted name stored in
    VarEntry.source for display in brackets.
    """
    display_map: VarMap = (
        solution.variables if solution is not None else model.substitutions
    )

    def _get_value_units(vk):
        try:
            return _value_units(vk, display_map)
        except KeyError:
            return None, unitstr(vk) or "-"

    lineage_map = model._get_lineage_map()  # pylint: disable=protected-access
    excluded = {":MAGIC:" + model.lineagestr()} if model.lineagestr() else set()
    free_entries: List[VarEntry] = []
    fixed_entries: List[VarEntry] = []
    seen_veckeys: set = set()

    with lineage_display_context(lineage_map):
        for vk in sorted(model.unique_varkeys, key=lambda v: v.name):
            if vk.veckey is not None:
                if vk.veckey in seen_veckeys:
                    continue
                seen_veckeys.add(vk.veckey)
                display_vk = vk.veckey
            else:
                display_vk = vk
            entry = _make_var_entry(display_vk, excluded, _get_value_units, solution)
            if _is_free_vk(display_vk, solution, model):
                free_entries.append(entry)
            else:
                fixed_entries.append(entry)

    if extra_vks:
        owned_display = {(vk.veckey or vk) for vk in model.unique_varkeys}
        cross_seen: set = set()
        with lineage_display_context(lineage_map):
            for vk in sorted(extra_vks, key=lambda v: v.name):
                display_vk = vk.veckey if vk.veckey is not None else vk
                if display_vk in owned_display or display_vk in cross_seen:
                    continue
                cross_seen.add(display_vk)
                entry = _make_var_entry(
                    display_vk,
                    excluded,
                    _get_value_units,
                    solution,
                    source=display_vk.lineagestr(),
                )
                if _is_free_vk(display_vk, solution, model):
                    free_entries.append(entry)
                else:
                    fixed_entries.append(entry)

    return free_entries, fixed_entries


def _collect_leaf_constraints(container) -> list:
    """Recursively collect single-equation leaf constraints from a container.

    Unwraps Tight; skips other ConstraintSet subclasses and Models (identified
    by unique_varkeys); flattens lists and tuples.

    Known limitation: only Tight is unwrapped. Other transparent wrappers
    (Loose, Bounded, SignomialEquality, ConstraintsRelaxed*) are silently
    dropped — the isinstance(Tight) check is not a principled protocol.
    """
    result = []
    for item in container:
        if isinstance(item, Tight):
            result.extend(_collect_leaf_constraints(item))
        elif hasattr(item, "unique_varkeys"):
            pass  # skip child Models and other ConstraintSet subclasses
        elif isinstance(item, (list, tuple)):
            result.extend(_collect_leaf_constraints(item))
        else:
            result.append(item)
    return result


def _build_constraint_groups(model) -> List[CGroup]:
    """Build CGroup list from model.cgroups or a single unnamed group.

    CGroup.constraints holds only leaf (single-equation) constraints;
    Tight wrappers are unwrapped so every item accepts .latex(excluded,
    aligned=True). Raw objects are stored; renderers call str() or .latex().
    """
    if model.cgroups is not None:
        return [
            CGroup(
                label=label,
                constraints=_collect_leaf_constraints(
                    items if isinstance(items, list) else [items]
                ),
            )
            for label, items in model.cgroups.items()
        ]

    own = _collect_leaf_constraints(model)
    return [CGroup(label="", constraints=own)] if own else []


def _reciprocal_if_1_over_x(cost):
    """Return (True, 1/cost) if cost is a single monomial 1/expr, else (False, cost).

    Detects the pattern coeff=1, all-negative exponents — the 1/x form that
    GPs use to express maximization.  Mirrors the TOML printer's _is_reciprocal
    check but operates on the live cost object rather than the IR AST dict.
    """
    hmap = cost.hmap
    if len(hmap) != 1:
        return False, cost
    (exp,) = hmap.keys()
    coeff = hmap[exp]
    exp_dict = dict(exp)
    if abs(coeff - 1.0) < 1e-10 and exp_dict and all(v < 0 for v in exp_dict.values()):
        # Build the inner expression from VarKeys rather than computing 1/cost.
        # 1/cost encodes div(1, cost.ast) in its AST, causing str/latex to
        # render as 1/(1/x) instead of x.
        inner = None
        for vk, e in exp_dict.items():
            term = Variable(vk) if e == -1 else Variable(vk) ** (-e)
            inner = term if inner is None else inner * term
        return True, inner
    return False, cost


def _build_objective(model, solution) -> dict:
    """Return objective keyword args for ReportSection for the model's cost.

    All fields are empty/None when the cost has no variables (i.e. it is a
    trivial constant placeholder rather than a real optimization objective).
    Detects the 1/expr pattern and flips direction to "maximize".
    """
    if not model.cost.vks:
        return {
            "objective_str": "",
            "objective_latex": "",
            "objective_label": "",
            "objective_value": None,
            "objective_units": "",
            "objective_direction": "minimize",
        }
    is_recip, expr = _reciprocal_if_1_over_x(model.cost)
    cost_value = (
        (1.0 / float(solution.cost) if is_recip else float(solution.cost))
        if solution is not None
        else None
    )
    excluded = {"units", "lineage"}
    vks = list(expr.vks)
    label = vks[0].label if len(vks) == 1 else ""
    return {
        "objective_str": expr.str_without(excluded),
        "objective_latex": expr.latex(excluded),
        "objective_label": label or "",
        "objective_value": cost_value,
        "objective_units": unitstr(expr),
        "objective_direction": "maximize" if is_recip else "minimize",
    }


# ── Standard text blocks ──────────────────────────────────────────────────────


def _model_stats(model) -> dict:
    """Compute model-wide counts for use in report text blocks.

    Returns a dict with keys: n_free, n_constraints, objective_str,
    objective_latex.  All values are derived from the model without requiring
    a solved solution.
    """
    n_free = len(model.vks) - len(model.substitutions)
    # flat() returns a generator (flatiter), so use sum() rather than len().
    n_constraints = sum(1 for _ in model.flat())
    if model.cost.vks:
        is_recip, expr = _reciprocal_if_1_over_x(model.cost)
        excluded = {"units", "lineage"}
        vks = list(expr.vks)
        obj_latex = expr.latex(excluded)
        obj_label = vks[0].label if len(vks) == 1 else ""
        obj_direction = "maximize" if is_recip else "minimize"
    else:
        obj_latex = None
        obj_label = ""
        obj_direction = "minimize"
    return {
        "n_free": n_free,
        "n_constraints": n_constraints,
        "objective_latex": obj_latex,
        "objective_label": obj_label or "",
        "objective_direction": obj_direction,
    }


def feasibility_block(model) -> str:
    """Return a markdown explanation of feasibility and optimality for *model*.

    Fills in the number of free variables, constraints, and current objective
    expression.  Suitable for use as ``front_matter`` or a ``report_preamble``
    in a custom report.

    Example usage::

        from gpkit.report import feasibility_block, sensitivities_block

        class Aircraft(Model):
            ...

        m = Aircraft()
        sol = m.solve()
        print(m.report(sol, fmt="md",
                       front_matter=feasibility_block(m) + "\\n\\n"
                                    + sensitivities_block()))
    """
    ctx = _model_stats(model)
    if ctx["objective_latex"]:
        direction = ctx["objective_direction"]
        label_clause = f", {ctx['objective_label']}" if ctx["objective_label"] else ""
        obj_clause = (
            f" The objective is currently set to {direction}"
            f" ${ctx['objective_latex']}${label_clause}."
        )
    else:
        obj_clause = ""
    return (
        f"## Feasibility and Optimality\n\n"
        f"The model currently has {ctx['n_free']} free variables and "
        f"{ctx['n_constraints']} constraints. A design satisfying all "
        f"constraints is *feasible*; the set of all feasible designs is the "
        f"*feasible set*."
        f"{obj_clause} "
        f"The solver finds a globally optimal solution — the unique feasible "
        f"design that cannot be improved further — with a reliable, efficient "
        f"algorithm. This guarantee comes from the convex structure of the "
        f"problem, not from luck or tuning."
    )


SENSITIVITIES_BLOCK = (
    "## Sensitivities\n\n"
    "Each fixed constant in the model has a *sensitivity* — a number that "
    "tells you how much the objective would change if that parameter changed. "
    "Specifically, if a constant $c$ has sensitivity $s$, then increasing $c$ "
    "by 1% would worsen the objective by approximately $s$% (holding all other "
    "constants fixed and re-solving). A sensitivity of 1.5 means a 1% increase "
    "in that constant would worsen the objective by 1.5%; a sensitivity of −0.8 "
    "means a 1% increase would improve the objective by 0.8%. "
    "Sensitivities with large magnitude flag the parameters that matter most; "
    "near-zero sensitivities indicate parameters the design is insensitive to. "
    "These numbers come for free alongside every solve — no extra computation "
    "required."
)


def sensitivities_block() -> str:
    """Return a markdown explanation of GP dual solution / sensitivity information.

    This text is model-independent and can be included in any GP report.
    """
    return SENSITIVITIES_BLOCK


def objective_block(model, solution=None) -> str:
    """Return a markdown summary of the model's objective for use in a report.

    Shows the objective direction (minimize/maximize), the expression without
    lineage, the variable label when the expression is a single variable, and
    the attained value when *solution* is provided.

    Like :func:`feasibility_block` and :func:`sensitivities_block`, this
    returns a plain markdown string so it can be placed wherever the author
    wants — front matter, a subsection preamble, or anywhere else.

    Example::

        from gpkit.report import objective_block
        m.report(sol, fmt="md", front_matter=objective_block(m, sol))
    """
    if not model.cost.vks:
        return ""
    is_recip, expr = _reciprocal_if_1_over_x(model.cost)
    excluded = {"units", "lineage"}
    direction = "maximize" if is_recip else "minimize"
    latex = expr.latex(excluded)
    vks = list(expr.vks)
    label_clause = f", {vks[0].label}" if len(vks) == 1 and vks[0].label else ""
    lines = [f"**Objective:** {direction} ${latex}${label_clause}"]
    if solution is not None:
        cost_value = 1.0 / float(solution.cost) if is_recip else float(solution.cost)
        val_str = _fmt_value(cost_value)
        attained = f"{val_str} {unitstr(expr)}".rstrip()
        lines.append("")
        lines.append(f"**Attained:** {attained}")
    return "\n".join(lines)


# ── Core builder ─────────────────────────────────────────────────────────────


def build_report_ir(
    model,
    solution=None,
    _parent_path: str = "",
    front_matter: str = "",
    toc: bool = False,
) -> ReportSection:
    """Build a ReportSection tree from *model*.

    Parameters
    ----------
    model : Model
        Root model to report on.
    solution : Solution, optional
        If provided, variable entries include solved values and sensitivities.
    front_matter : str, optional
        Raw text/markdown prepended before the root section.  For the root
        model, caller-supplied *front_matter* is combined with the model's
        ``report_preamble()`` (if any).  For child models, only
        ``report_preamble()`` is used.
    toc : bool, optional
        If True, a table-of-contents marker is inserted by renderers that have
        a native TOC facility (e.g. ``[TOC]`` in Markdown).  Set only on the
        root ReportSection; not propagated to children.
    """
    is_anon = type(model) is _Model  # pylint: disable=unidiomatic-typecheck
    own_name = "" if is_anon else type(model).__name__
    if own_name:
        lineage_path = f"{_parent_path}.{own_name}" if _parent_path else own_name
    else:
        lineage_path = _parent_path  # transparent: inherit parent path
    lineage_map = model._get_lineage_map()  # pylint: disable=protected-access
    cgroups = _build_constraint_groups(model)
    free_vars, fixed_vars = _build_split_var_entries(
        model,
        solution,
        extra_vks=_collect_constraint_varkeys(cgroups) - model.unique_varkeys,
    )
    if is_anon:
        desc = {"summary": "", "assumptions": [], "references": []}
        section_fm = front_matter
    else:
        desc = type(model).description()
        preamble = type(model).report_preamble()
        if preamble and front_matter:
            section_fm = front_matter + "\n\n" + preamble
        elif preamble:
            section_fm = preamble
        else:
            section_fm = front_matter
    return ReportSection(
        title=own_name or "Model",
        description=desc["summary"],
        assumptions=desc["assumptions"],
        references=desc["references"],
        lineage_path=lineage_path,
        magic_prefix=model.lineagestr(),
        is_anonymous=is_anon,
        free_variables=free_vars,
        fixed_variables=fixed_vars,
        constraint_groups=cgroups,
        lineage_map=lineage_map,
        front_matter=section_fm,
        toc=toc,
        **_build_objective(model, solution),
        children=[
            build_report_ir(child, solution=solution, _parent_path=lineage_path)
            for child in model.submodels
        ],
    )


# ── Text renderer ────────────────────────────────────────────────────────────

_INDENT = "  "


def _fmt_value(val, precision: int = 4, vecn: int = 6, col_widths=()) -> str:
    """Format a variable value for display, handling scalars and arrays.

    val must be a plain float, numpy array, or None — not a pint Quantity.
    _build_var_entries() pre-extracts magnitudes; callers constructing VarEntry
    directly must do the same.

    col_widths is a per-column list of minimum widths so that element position i
    across all vector rows in the same table renders at the same width.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    if val is None:
        return "-"
    if isinstance(val, Quantity):
        raise TypeError(
            f"_fmt_value received a pint Quantity ({val!r}); "
            "extract .magnitude before storing in VarEntry.value"
        )
    mag = val
    try:
        arr = np.asarray(mag)
        if arr.shape:
            flat = arr.ravel()
            shown = [f"{x:.{precision}g}" for x in flat[:vecn]]
            body = "  ".join(
                s.ljust(col_widths[i]) if i < len(col_widths) else s
                for i, s in enumerate(shown)
            )
            dots = " ..." if flat.size > vecn else ""
            return f"[ {body}{dots} ]"
        return f"{float(arr):.{precision}g}"
    except (TypeError, ValueError):
        return str(mag)


_SENS_NEARZERO_TOL = 1e-7


def _fmt_sensitivity(sens, vecn: int = 6, col_widths=()) -> str:
    """Format a sensitivity value for display, handling scalars and arrays."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if sens is None:
        return "-"
    try:
        arr = np.asarray(sens)
        if arr.shape:
            flat = arr.ravel()
            shown = [
                "~0" if abs(x) < _SENS_NEARZERO_TOL else f"{x:+.2g}"
                for x in flat[:vecn]
            ]
            body = "  ".join(
                s.ljust(col_widths[i]) if i < len(col_widths) else s
                for i, s in enumerate(shown)
            )
            dots = " ..." if flat.size > vecn else ""
            return f"( {body}{dots} )"
        scalar = float(arr)
        return "~0" if abs(scalar) < _SENS_NEARZERO_TOL else f"{scalar:+.2g}"
    except (TypeError, ValueError):
        return str(sens)


def _compute_vec_col_widths(variables: list, precision: int, vecn: int) -> list:
    """Pre-scan vector values to compute per-column widths for alignment."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    col_widths: list = []
    for ve in variables:
        try:
            arr = np.asarray(ve.value)
            if arr.shape:
                elems = [f"{x:.{precision}g}" for x in arr.ravel()[:vecn]]
                while len(col_widths) < len(elems):
                    col_widths.append(0)
                for i, s in enumerate(elems):
                    col_widths[i] = max(col_widths[i], len(s))
        except (TypeError, ValueError):
            pass
    return col_widths


def _var_name_cell(ve: "VarEntry") -> str:
    "Name with inline source annotation for cross-model variables."
    if ve.source:
        return f"{ve.name} [{ve.source}]"
    return ve.name


def _text_var_rows(
    variables: list,
    include_sensitivity: bool = False,
    precision: int = 4,
    vecn: int = 6,
) -> list:
    """Column-aligned rows for a variable table"""
    col_widths = _compute_vec_col_widths(variables, precision, vecn)
    rows = []
    for ve in variables:
        row = [
            _var_name_cell(ve),
            _fmt_value(ve.value, precision=precision, vecn=vecn, col_widths=col_widths),
            ve.units,
        ]
        if include_sensitivity:
            row.append(_fmt_sensitivity(ve.sensitivity))
        row.append(ve.label)
        rows.append(row)
    align = "<<<<<" if include_sensitivity else "<<<<"
    return _format_aligned_columns(rows, align, "  ")


def _text_cgroup_lines(constraint_groups: list, pad: str, lineage_map: dict) -> list:
    "Build text lines for all constraint groups in a section."
    lines = []
    for cg in constraint_groups:
        group_header = f"Constraints ({cg.label})" if cg.label else "Constraints"
        lines.append(f"{pad}  {group_header}")
        if cg.constraints:
            for row_line in _text_constraint_rows(cg.constraints, lineage_map):
                lines.append(f"{pad}    {row_line}")
        lines.append("")
    return lines


def _text_constraint_rows(constraints: list, lineage_map: dict = None) -> list:
    """Build aligned rows for a constraint group in text output.

    lineage_map is the section's _get_lineage_map() result; activating it
    makes variable names in constraints use section-local abbreviations.
    """
    ctx = lineage_display_context(lineage_map or {})
    with ctx:
        c_rows = [_split_constraint_str(_render_constraint(c)) for c in constraints]
    return _format_aligned_columns(c_rows, "<<", "  ")


def _text_prose_lines(ir: "ReportSection", pad: str) -> list:
    """Return prose lines: description, assumptions, references, objective."""
    lines = []
    if ir.description:
        lines.append(f"{pad}  {ir.description}")
        lines.append("")
    if ir.assumptions:
        lines.append(f"{pad}  Assumptions:")
        for item in ir.assumptions:
            lines.append(f"{pad}    - {item}")
        lines.append("")
    if ir.references:
        lines.append(f"{pad}  References:")
        for item in ir.references:
            lines.append(f"{pad}    - {item}")
        lines.append("")
    return lines


def render_text(ir: "ReportSection", indent: int = 0) -> str:
    """Render a ReportSection tree as hierarchical plain text.

    Parameters
    ----------
    ir : ReportSection
    indent : int
        Current indentation level (number of 2-space indent units).

    Returns
    -------
    str
        Hierarchical text representation.
    """
    pad = _INDENT * indent
    lines: list = []

    if ir.front_matter:
        lines.append(ir.front_matter)
        lines.append("")

    if ir.is_anonymous:
        # Transparent wrapper: no header, children rendered at the same level
        child_indent = indent
    else:
        # Section header: full lineage path when available, else title
        lines.append(f"{pad}{ir.lineage_path or ir.title}")
        child_indent = indent + 1

    lines.extend(_text_prose_lines(ir, pad))

    # Constraint groups
    lines.extend(_text_cgroup_lines(ir.constraint_groups, pad, ir.lineage_map))

    # Optimized Variables table (primal — no sensitivity column)
    if ir.free_variables:
        lines.append(f"{pad}  Optimized Variables")
        for row_line in _text_var_rows(ir.free_variables):
            lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Fixed Variables table (constants — value | units | sensitivity | label)
    if ir.fixed_variables:
        lines.append(f"{pad}  Fixed Variables")
        for row_line in _text_var_rows(ir.fixed_variables, include_sensitivity=True):
            lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Children (recursive)
    for child in ir.children:
        lines.append(render_text(child, indent=child_indent))

    return "\n".join(lines)


def _split_constraint_str(c_str: str):
    """Split a constraint string into (lhs_with_op, rhs) for column alignment.

    Returns a two-element list [lhs_and_op, rhs] so that _format_aligned_columns
    can right-align the operator column.  Falls back gracefully if no operator
    is found.
    """
    for op in (">=", "<=", "==", "="):
        if op in c_str:
            idx = c_str.index(op)
            lhs = c_str[: idx + len(op)]
            rhs = c_str[idx + len(op) :].lstrip()
            return [lhs, rhs]
    return [c_str, ""]


# ── Markdown renderer ────────────────────────────────────────────────────────


def _md_escape(text: str) -> str:
    r"""Escape characters that have special meaning in markdown pipe tables."""
    for ch in ("\\", "|", "*", "_", "`", "~", "[", "]", "<", ">", "%"):
        text = text.replace(ch, "\\" + ch)
    return text


def _md_var_table(variables: list, include_sensitivity: bool = False) -> list:
    "Markdown pipe-table lines for a variable section."
    has_source = any(ve.source for ve in variables)
    cols = ["Variable"] + (["Source"] if has_source else []) + ["Value", "Units"]
    if include_sensitivity:
        cols.append("Sensitivity")
    cols.append("Label")
    sep = ["-" * max(len(c), 3) for c in cols]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(sep) + " |"]
    for ve in variables:
        name_cell = f"${ve.latex}$" if ve.latex else _md_escape(ve.name)
        cells = [name_cell]
        if has_source:
            cells.append(_md_escape(ve.source))
        cells += [_fmt_value(ve.value), ve.units]
        if include_sensitivity:
            cells.append(_fmt_sensitivity(ve.sensitivity))
        cells.append(_md_escape(ve.label))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _md_prose_lines(ir: "ReportSection") -> list:
    """Return markdown lines for description, assumptions, references, and objective."""
    lines = []
    if ir.description:
        lines.append(ir.description)
        lines.append("")
    if ir.assumptions:
        lines.append(f"**Assumptions:** {'; '.join(ir.assumptions)}")
        lines.append("")
    if ir.references:
        lines.append(f"**References:** {'; '.join(ir.references)}")
        lines.append("")
    return lines


def render_markdown(ir: "ReportSection", level: int = 1) -> str:
    """Render a ReportSection tree as Markdown.

    Parameters
    ----------
    ir : ReportSection
    level : int
        Current heading level (1 for top-level model, increments for children).

    Returns
    -------
    str
        Markdown representation.
    """
    hdr = "#" * min(level, 6)
    lines: list = []

    # Front matter and TOC marker (before first heading)
    if ir.front_matter:
        lines.append(ir.front_matter)
        lines.append("")
    if ir.toc:
        lines.append("[TOC]")
        lines.append("")

    if ir.is_anonymous:
        # Transparent wrapper: skip heading, children rendered at same level
        child_level = level
    else:
        # Heading: use full lineage path when available, else title
        lines.append(f"{hdr} {ir.lineage_path or ir.title}")
        lines.append("")
        child_level = level + 1

    lines.extend(_md_prose_lines(ir))

    # Constraint groups
    excluded = ("units", ":MAGIC:" + ir.magic_prefix) if ir.magic_prefix else ("units",)
    for cg in ir.constraint_groups:
        group_header = f"**Constraints: {cg.label}**" if cg.label else "**Constraints**"
        lines.append(group_header)
        lines.append("")
        with lineage_display_context(ir.lineage_map):
            lines.append("$$\\begin{aligned}")
            for c in cg.constraints:
                if hasattr(c, "latex"):
                    c_latex = c.latex(excluded, aligned=True)
                else:
                    c_latex = str(c)
                lines.append(f"{c_latex} \\\\")
            lines.append("\\end{aligned}$$")
            lines.append("")

    # Optimized Variables pipe table (no sensitivity column)
    if ir.free_variables:
        lines.append("**Optimized Variables**")
        lines.append("")
        lines.extend(_md_var_table(ir.free_variables))
        lines.append("")

    # Fixed Variables pipe table (value | units | sensitivity | label)
    if ir.fixed_variables:
        lines.append("**Fixed Variables**")
        lines.append("")
        lines.extend(_md_var_table(ir.fixed_variables, include_sensitivity=True))
        lines.append("")

    # Children (recursive)
    for child in ir.children:
        child_md = render_markdown(child, level=child_level)
        lines.append(child_md)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── Renderer dispatcher ───────────────────────────────────────────────────────


def render_report(ir: "ReportSection", fmt: str = "text") -> "dict | str":
    """Dispatch to the appropriate format renderer.

    Parameters
    ----------
    ir : ReportSection
        The root IR produced by build_report_ir().
    fmt : str
        Output format: "dict", "text", "md", or "latex".
    """
    if fmt == "dict":
        return ir.to_dict()
    if fmt == "text":
        return render_text(ir)
    if fmt == "md":
        return render_markdown(ir)
    raise ValueError(
        f"Format '{fmt}' not yet implemented. "
        "Available formats: 'dict', 'text', 'md', 'latex'."
    )
