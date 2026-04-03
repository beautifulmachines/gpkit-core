"""Report infrastructure: format-independent IR and rendering backends.

ReportSection is the single tree that all output formats render from.
build_report_ir() traverses the model tree once; renderers are pure
functions of the IR.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .varkey import lineage_display_context

# ── Print options ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PrintOptions:
    """Options controlling report/solution formatting."""

    precision: int = 3
    topn: int = 10
    vecn: int = 3
    vec_width: int = 0


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


@dataclass
class CGroup:
    """A named constraint group within a report section."""

    label: str  # "" for unnamed groups
    constraints: list  # raw constraint objects; to_dict() serializes via str()


@dataclass
class ReportSection:
    """Format-independent intermediate representation for model reports."""

    title: str
    description: str
    assumptions: list  # list of str
    variables: list  # list of VarEntry
    constraint_groups: list  # list of CGroup
    children: list = field(default_factory=list)  # list of ReportSection

    def to_dict(self) -> dict:
        """JSON-serializable dict (for format='dict' and future API)."""
        return {
            "title": self.title,
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


def _resolve_sensitivity(vk, solution=None) -> Optional[float]:
    """Get sensitivity (shadow price) for a variable from solution."""
    if solution is None:
        return None
    try:
        sens_vars = solution.sens.variables
        if vk in sens_vars:
            return float(sens_vars[vk])
        return None
    except (AttributeError, KeyError, TypeError):
        return None


# ── Constraint rendering ──────────────────────────────────────────────────────


def _render_constraint(c) -> str:
    """Render a single constraint as a string."""
    try:
        return str(c)
    except Exception:  # pylint: disable=broad-except
        return repr(c)


# ── Core builder helpers ──────────────────────────────────────────────────────


def _build_var_entries(model, solution, substitutions) -> List[VarEntry]:
    """Build VarEntry list from model.unique_varkeys.

    Variable names are disambiguated within the section scope using
    _get_lineage_map(), so siblings in this section are disambiguated but
    variables in other sections are irrelevant (e.g. m_cap stays m_cap even
    if wing.spar.cap.m and tail.spar.cap.m both exist in the full model).
    """
    lineage_map = model._get_lineage_map()
    entries = []
    with lineage_display_context(lineage_map):
        for vk in sorted(model.unique_varkeys, key=lambda v: v.name):
            entries.append(
                VarEntry(
                    name=vk.str_without(),
                    latex=(
                        vk.latex() if callable(getattr(vk, "latex", None)) else vk.name
                    ),
                    value=_resolve_var_value(
                        vk, solution=solution, substitutions=substitutions, model=model
                    ),
                    sensitivity=_resolve_sensitivity(vk, solution=solution),
                    units=vk.unitrepr or "-",
                    label=vk.label or "",
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
    desc = type(model).description()
    return ReportSection(
        title=type(model).__name__,
        description=desc.get("summary", ""),
        assumptions=list(desc.get("assumptions", [])),
        variables=_build_var_entries(model, solution, substitutions),
        constraint_groups=_build_constraint_groups(model),
        children=[
            build_report_ir(child, solution=solution, substitutions=substitutions)
            for child in model.submodels
        ],
    )


# ── Text renderer ────────────────────────────────────────────────────────────

_INDENT = "  "


def _fmt_value(val, precision: int = 3) -> str:
    """Format a variable value for display."""
    if val is None:
        return "-"
    try:
        f = float(val)
        return f"{f:.{precision}g}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_sensitivity(sens) -> str:
    """Format a sensitivity value for display."""
    if sens is None:
        return "-"
    try:
        return f"{float(sens):+.2g}"
    except (TypeError, ValueError):
        return str(sens)


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

    # Section header (model class name)
    lines.append(f"{pad}{ir.title}")

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
        rows = []
        for ve in ir.variables:
            rows.append(
                [
                    ve.name,
                    _fmt_value(ve.value),
                    _fmt_sensitivity(ve.sensitivity),
                    ve.units,
                    ve.label,
                ]
            )
        aligned = _format_aligned_columns(rows, "<<<<<", "  ")
        for row_line in aligned:
            lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Constraint groups
    for cg in ir.constraint_groups:
        group_header = f"Constraints ({cg.label})" if cg.label else "Constraints"
        lines.append(f"{pad}  {group_header}")
        if cg.constraints:
            # Build aligned (lhs, op, rhs) from constraint objects
            constraint_rows = []
            for c in cg.constraints:
                c_str = _render_constraint(c)
                # Try to split on standard operators for alignment
                # Each constraint str() typically has form "lhs >= rhs" or "lhs == rhs"
                split_result = _split_constraint_str(c_str)
                constraint_rows.append(split_result)
            aligned = _format_aligned_columns(constraint_rows, "<<", "  ")
            for row_line in aligned:
                lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Children (recursive)
    for child in ir.children:
        child_text = render_text(child, indent=indent + 1)
        lines.append(child_text)

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
        lines.append("| Variable | Value | Sensitivity | Units | Label |")
        lines.append("|----------|-------|-------------|-------|-------|")
        for ve in ir.variables:
            name_cell = f"${ve.latex}$" if ve.latex else ve.name
            value_cell = _fmt_value(ve.value)
            sens_cell = _fmt_sensitivity(ve.sensitivity)
            units_cell = ve.units
            label_cell = ve.label
            row = (
                f"| {name_cell} | {value_cell}"
                f" | {sens_cell} | {units_cell} | {label_cell} |"
            )
            lines.append(row)
        lines.append("")

    # Constraint groups
    for cg in ir.constraint_groups:
        group_header = f"**Constraints: {cg.label}**" if cg.label else "**Constraints**"
        lines.append(group_header)
        lines.append("")
        for c in cg.constraints:
            c_str = _render_constraint(c)
            lines.append(f"$${c_str}$$")
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
