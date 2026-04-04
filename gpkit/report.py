"""Report infrastructure: format-independent IR and rendering backends.

ReportSection is the single tree that all output formats render from.
build_report_ir() traverses the model tree once; renderers are pure
functions of the IR.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .util.small_scripts import try_str_without
from .varkey import lineage_display_context

# ── Column alignment helper ───────────────────────────────────────────────────


def _format_aligned_columns(
    rows: list,  # each row is a list of column strings
    col_alignments: str,  # '<' left, '>' right, one char per column
    col_sep: str = " ",
) -> list:
    """Align arbitrary columns with dynamic widths.

    Input: list of rows, where each row is a list of column strings.
    Output: list of formatted/aligned lines.

    Does NOT sort - expects pre-sorted input.
    """
    if not rows:
        return []
    ncols_set = set(len(r) for r in rows)
    if len(ncols_set) != 1:
        # Rows have different column counts — fall back to simple join
        return [col_sep.join(str(c) for c in row).rstrip() for row in rows]
    (ncols,) = ncols_set
    if col_alignments is None:
        col_alignments = "<" * ncols
    assert len(col_alignments) == ncols
    widths = [max(len(str(row[i])) for row in rows) for i in range(ncols)]
    formatted = []
    for row in rows:
        parts = [
            f"{str(cell):{align}{width}}"
            for cell, width, align in zip(row, widths, col_alignments)
        ]
        formatted.append(col_sep.join(parts).rstrip())
    return formatted


@dataclass
class VarEntry:
    """A variable row in a report section."""

    name: str  # display name (from VarKey.name)
    latex: str  # LaTeX rendering (from VarKey.latex())
    value: Any  # float | Quantity | None (resolved per D-17 priority)
    sensitivity: Optional[float]  # shadow price from solution, or None
    units: str  # unit string
    label: str  # human label (from VarKey.label or "")
    source: str = ""  # lineagestr() for cross-model referenced vars; "" for local


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
    variables: list  # list of VarEntry
    constraint_groups: list  # list of CGroup
    lineage_path: str = ""  # dotted path e.g. "Aircraft.Wing"
    magic_prefix: str = (
        ""  # model.lineagestr() — stripped from variable names in renderers
    )
    children: list = field(default_factory=list)  # list of ReportSection
    lineage_map: dict = field(default_factory=dict)  # NOT in to_dict

    def to_dict(self) -> dict:
        """JSON-serializable dict (for format='dict' and future API)."""
        return {
            "title": self.title,
            "lineage_path": self.lineage_path,
            "magic_prefix": self.magic_prefix,
            "description": self.description,
            "assumptions": list(self.assumptions),
            "variables": [
                {
                    "name": v.name,
                    "latex": v.latex,
                    "value": _serialize_value(v.value),
                    "sensitivity": v.sensitivity,
                    "units": v.units,
                    "label": v.label,
                    "source": v.source,
                }
                for v in self.variables
            ],
            "constraint_groups": [
                {"label": cg.label, "constraints": [str(c) for c in cg.constraints]}
                for cg in self.constraint_groups
            ],
            "children": [c.to_dict() for c in self.children],
        }


# ── Value helpers ─────────────────────────────────────────────────────────────


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def _resolve_var_value(vk, solution=None, substitutions=None, model=None):
    """Resolve a variable's display value following priority:
    solution > substitutions override > model.substitutions > vk.value
    """
    if solution is not None:
        try:
            return solution[vk]
        except (KeyError, TypeError):
            pass
    if substitutions is not None:
        if vk in substitutions:
            return substitutions[vk]
    if model is not None:
        subs = getattr(model, "substitutions", {})
        if vk in subs:
            return subs[vk]
    return getattr(vk, "value", None)


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


def _disambiguate_latex(display_vk, taken: set) -> str:
    """Return the minimal-depth latex for display_vk that doesn't appear in taken.

    Starts at depth=0 (base symbol only) and adds one more lineage component
    as a subscript per step until the name is unique. Local vars win because
    their names are pre-loaded into taken before this is called.
    """
    max_depth = len(getattr(display_vk, "lineage", ()) or ())
    for depth in range(0, max_depth + 1):
        with lineage_display_context({display_vk: depth}):
            candidate = display_vk.latex(("units",))
        if candidate not in taken:
            return candidate
    return display_vk.latex(("units",))  # exhausted — use full lineage


def _build_var_entries(
    model, solution, substitutions, extra_vks=None
) -> List[VarEntry]:
    """Build VarEntry list from model.unique_varkeys, then cross-model extras.

    Local variables: names disambiguated within the section scope using
    _get_lineage_map(); model's own lineage stripped so only sub-model context
    is shown. Vector variables collapsed to a single VarEntry.

    Cross-model variables (extra_vks): latex disambiguated against local names
    by progressively adding lineage subscript components until unique. Full
    dotted name stored in VarEntry.source for display in brackets.
    """
    lineage_map = model._get_lineage_map()  # pylint: disable=protected-access
    model_prefix = model.lineagestr()
    excluded = {":MAGIC:" + model_prefix} if model_prefix else set()
    entries = []
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
            entries.append(
                VarEntry(
                    name=display_vk.str_without(excluded),
                    latex=display_vk.latex(excluded),
                    value=_resolve_var_value(
                        display_vk,
                        solution=solution,
                        substitutions=substitutions,
                        model=model,
                    ),
                    sensitivity=_resolve_sensitivity(display_vk, solution=solution),
                    units=display_vk.unitrepr or "-",
                    label=display_vk.label or "",
                )
            )

    if extra_vks:
        owned_display = {(vk.veckey or vk) for vk in model.unique_varkeys}
        taken_latex = {ve.latex for ve in entries}
        cross_seen: set = set()
        for vk in sorted(extra_vks, key=lambda v: v.name):
            display_vk = vk.veckey if vk.veckey is not None else vk
            if display_vk in owned_display or display_vk in cross_seen:
                continue
            cross_seen.add(display_vk)
            vk_latex = _disambiguate_latex(display_vk, taken_latex)
            taken_latex.add(vk_latex)
            entries.append(
                VarEntry(
                    name=display_vk.str_without(()),
                    latex=vk_latex,
                    value=_resolve_var_value(
                        display_vk,
                        solution=solution,
                        substitutions=substitutions,
                        model=model,
                    ),
                    sensitivity=_resolve_sensitivity(display_vk, solution=solution),
                    units=display_vk.unitrepr or "-",
                    label=display_vk.label or "",
                    source=display_vk.lineagestr(),
                )
            )

    return entries


def _build_constraint_groups(model) -> List[CGroup]:
    """Build CGroup list from model.cgroups or a single unnamed group.

    Raw constraint objects are stored; renderers call str() or .latex() as
    appropriate. to_dict() serializes via str().
    """
    if model.cgroups is not None:
        return [
            CGroup(
                label=label,
                constraints=list(items if isinstance(items, list) else [items]),
            )
            for label, items in model.cgroups.items()
        ]
    own = []
    try:
        for item in model:
            # Skip child models (unique_varkeys is set only on Model instances).
            if not hasattr(item, "unique_varkeys"):
                own.append(item)
    except TypeError:
        pass
    return [CGroup(label="", constraints=own)] if own else []


# ── Core builder ─────────────────────────────────────────────────────────────


def build_report_ir(
    model,
    solution=None,
    substitutions: Optional[dict] = None,
    _parent_path: str = "",
) -> ReportSection:
    """Build a ReportSection tree from *model*.

    Parameters
    ----------
    model : Model
        Root model to report on.
    solution : Solution, optional
        If provided, variable entries include solved values and sensitivities.
    substitutions : dict, optional
        One-off value overrides without mutating model.substitutions.
    """
    own_name = type(model).__name__
    lineage_path = f"{_parent_path}.{own_name}" if _parent_path else own_name
    lineage_map = model._get_lineage_map()  # pylint: disable=protected-access
    cgroups = _build_constraint_groups(model)
    return ReportSection(
        title=own_name,
        description="[description]",
        assumptions=[],
        lineage_path=lineage_path,
        magic_prefix=model.lineagestr(),
        variables=_build_var_entries(
            model,
            solution,
            substitutions,
            extra_vks=_collect_constraint_varkeys(cgroups) - model.unique_varkeys,
        ),
        constraint_groups=cgroups,
        lineage_map=lineage_map,
        children=[
            build_report_ir(
                child,
                solution=solution,
                substitutions=substitutions,
                _parent_path=lineage_path,
            )
            for child in model.submodels
        ],
    )


# ── Text renderer ────────────────────────────────────────────────────────────

_INDENT = "  "


def _fmt_value(val, precision: int = 4, vecn: int = 6, col_widths=()) -> str:
    """Format a variable value for display, handling scalars and arrays.

    col_widths is a per-column list of minimum widths so that element position i
    across all vector rows in the same table renders at the same width.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    if val is None:
        return "-"
    mag = getattr(val, "magnitude", val)
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


def _fmt_sensitivity(sens, vecn: int = 6, col_widths=()) -> str:
    """Format a sensitivity value for display, handling scalars and arrays."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if sens is None:
        return "-"
    mag = getattr(sens, "magnitude", sens)
    try:
        arr = np.asarray(mag)
        if arr.shape:
            flat = arr.ravel()
            shown = [f"{x:+.2g}" for x in flat[:vecn]]
            body = "  ".join(
                s.ljust(col_widths[i]) if i < len(col_widths) else s
                for i, s in enumerate(shown)
            )
            dots = " ..." if flat.size > vecn else ""
            return f"( {body}{dots} )"
        return f"{float(arr):+.2g}"
    except (TypeError, ValueError):
        return str(mag)


def _text_var_rows(variables: list, precision: int = 4, vecn: int = 6) -> list:
    """Build column-aligned rows for variables section in text output.

    Computes a uniform vec_width across all vector entries so that element
    columns stay aligned when multiple vectors share the same table.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    # Pre-scan: compute per-column max element widths across all vector values.
    # col_widths[i] = max formatted width of element i across all vector rows.
    col_widths: list = []
    for ve in variables:
        mag = getattr(ve.value, "magnitude", ve.value)
        try:
            arr = np.asarray(mag)
            if arr.shape:
                elems = [f"{x:.{precision}g}" for x in arr.ravel()[:vecn]]
                while len(col_widths) < len(elems):
                    col_widths.append(0)
                for i, s in enumerate(elems):
                    col_widths[i] = max(col_widths[i], len(s))
        except (TypeError, ValueError):
            pass

    rows = []
    for ve in variables:
        rows.append(
            [
                ve.name,
                _fmt_value(
                    ve.value, precision=precision, vecn=vecn, col_widths=col_widths
                ),
                _fmt_sensitivity(ve.sensitivity, vecn=vecn, col_widths=col_widths),
                ve.units,
                ve.label,
            ]
        )
    return _format_aligned_columns(rows, "<<<<<", "  ")


def _text_constraint_rows(constraints: list, lineage_map: dict = None) -> list:
    """Build aligned rows for a constraint group in text output.

    lineage_map is the section's _get_lineage_map() result; activating it
    makes variable names in constraints use section-local abbreviations.
    """
    ctx = lineage_display_context(lineage_map or {})
    with ctx:
        c_rows = [_split_constraint_str(_render_constraint(c)) for c in constraints]
    return _format_aligned_columns(c_rows, "<<", "  ")


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

    # Section header: use lineage path for nested models (shows where in tree)
    header = ir.lineage_path if indent > 0 else ir.title
    lines.append(f"{pad}{header}")

    # Description
    if ir.description:
        lines.append(f"{pad}  {ir.description}")
        lines.append("")

    # Assumptions
    if ir.assumptions:
        lines.append(f"{pad}  Assumptions:")
        for assumption in ir.assumptions:
            lines.append(f"{pad}    - {assumption}")
        lines.append("")

    # Variables table
    if ir.variables:
        lines.append(f"{pad}  Variables")
        for row_line in _text_var_rows(ir.variables):
            lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Constraint groups
    for cg in ir.constraint_groups:
        group_header = f"Constraints ({cg.label})" if cg.label else "Constraints"
        lines.append(f"{pad}  {group_header}")
        if cg.constraints:
            for row_line in _text_constraint_rows(cg.constraints, ir.lineage_map):
                lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Children (recursive)
    for child in ir.children:
        lines.append(render_text(child, indent=indent + 1))

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


def _md_var_row(ve: "VarEntry") -> str:
    """Format one VarEntry as a markdown pipe-table row."""
    if ve.source:
        name_cell = f"${ve.latex}$ [{ve.source}]"
    else:
        name_cell = f"${ve.latex}$" if ve.latex else ve.name
    return (
        f"| {name_cell} | {_fmt_value(ve.value)}"
        f" | {_fmt_sensitivity(ve.sensitivity)} | {ve.units} | {ve.label} |"
    )


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

    # Heading
    lines.append(f"{hdr} {ir.title}")
    lines.append("")

    # Description
    if ir.description:
        lines.append(ir.description)
        lines.append("")

    # Assumptions
    if ir.assumptions:
        assumption_str = "; ".join(ir.assumptions)
        lines.append(f"**Assumptions:** {assumption_str}")
        lines.append("")

    # Variable pipe table
    if ir.variables:
        local_vars = [ve for ve in ir.variables if not ve.source]
        ref_vars = [ve for ve in ir.variables if ve.source]
        lines.append("| Variable | Value | Sensitivity | Units | Label |")
        lines.append("|----------|-------|-------------|-------|-------|")
        for ve in local_vars:
            lines.append(_md_var_row(ve))
        if ref_vars:
            lines.append("| **— referenced —** | | | | |")
            for ve in ref_vars:
                lines.append(_md_var_row(ve))
        lines.append("")

    # Constraint groups
    excluded = ("units", ":MAGIC:" + ir.magic_prefix) if ir.magic_prefix else ("units",)
    for cg in ir.constraint_groups:
        group_header = f"**Constraints: {cg.label}**" if cg.label else "**Constraints**"
        lines.append(group_header)
        lines.append("")
        with lineage_display_context(ir.lineage_map):
            for c in cg.constraints:
                c_latex = try_str_without(c, excluded, latex=True)
                lines.append(f"$${c_latex}$$")
                lines.append("")

    # Children (recursive)
    for child in ir.children:
        child_md = render_markdown(child, level=level + 1)
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
