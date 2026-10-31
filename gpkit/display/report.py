"""A report: a model's view rendered as a document, in text or markdown.

gpkit.model_view.model_view() builds the view, a tree of ModelView nodes;
these render it, and render_report() picks the format.  They read the view's
attributes only, so this module needs nothing from the core above it.
"""

import numpy as np

from ..units import Quantity
from .naming import DisplayScope
from .render import LATEX_ALIGNED, TEXT, render
from .tables import _format_aligned_columns, _md_escape, _md_table

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


def _var_name_cell(ve) -> str:
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


def _text_cgroup_lines(constraint_groups: list, pad: str, scope) -> list:
    "Build text lines for all constraint groups in a section."
    lines = []
    for cg in constraint_groups:
        group_header = f"Constraints ({cg.label})" if cg.label else "Constraints"
        lines.append(f"{pad}  {group_header}")
        if cg.constraints:
            for row_line in _text_constraint_rows(cg.constraints, scope):
                lines.append(f"{pad}    {row_line}")
        lines.append("")
    return lines


def _text_constraint_rows(constraints: list, scope=None) -> list:
    """Build aligned rows for a constraint group in text output.

    scope names variables as the enclosing section shows them; without one,
    names carry their full lineage.
    """
    scope = (scope or DisplayScope()).also_excluding("units")
    c_rows = [_split_constraint_str(render(c, TEXT, scope)) for c in constraints]
    return _format_aligned_columns(c_rows, "<<", "  ")


def _text_prose_lines(view, pad: str) -> list:
    """Return prose lines: description, assumptions, references, objective."""
    lines = []
    if view.description:
        lines.append(f"{pad}  {view.description}")
        lines.append("")
    if view.assumptions:
        lines.append(f"{pad}  Assumptions:")
        for item in view.assumptions:
            lines.append(f"{pad}    - {item}")
        lines.append("")
    if view.references:
        lines.append(f"{pad}  References:")
        for item in view.references:
            lines.append(f"{pad}    - {item}")
        lines.append("")
    return lines


def render_text(view, indent: int = 0) -> str:
    """Render a ModelView tree as hierarchical plain text.

    Parameters
    ----------
    view : ModelView
    indent : int
        Current indentation level (number of 2-space indent units).

    Returns
    -------
    str
        Hierarchical text representation.
    """
    pad = _INDENT * indent
    lines: list = []

    if view.preamble:
        lines.append(view.preamble)
        lines.append("")

    if view.is_anonymous:
        # Transparent wrapper: no header, children rendered at the same level
        child_indent = indent
    else:
        # Section header: full lineage path when available, else title
        lines.append(f"{pad}{view.lineage_path or view.title}")
        child_indent = indent + 1

    lines.extend(_text_prose_lines(view, pad))

    # Constraint groups
    lines.extend(_text_cgroup_lines(view.constraint_groups, pad, view.scope))

    # Optimized Variables table (primal — no sensitivity column)
    if view.free_variables:
        lines.append(f"{pad}  Optimized Variables")
        for row_line in _text_var_rows(view.free_variables):
            lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Fixed Variables table (constants — value | units | sensitivity | label)
    if view.fixed_variables:
        lines.append(f"{pad}  Fixed Variables")
        for row_line in _text_var_rows(view.fixed_variables, include_sensitivity=True):
            lines.append(f"{pad}    {row_line}")
        lines.append("")

    # Children (recursive)
    for child in view.children:
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


def _md_var_table(variables: list, include_sensitivity: bool = False) -> list:
    "Markdown pipe-table lines for a variable section."
    has_source = any(ve.source for ve in variables)
    cols = ["Variable"] + (["Source"] if has_source else []) + ["Value", "Units"]
    if include_sensitivity:
        cols.append("Sensitivity")
    cols.append("Label")
    rows = []
    for ve in variables:
        name_cell = f"${ve.latex}$" if ve.latex else _md_escape(ve.name)
        cells = [name_cell]
        if has_source:
            cells.append(_md_escape(ve.source))
        cells += [_fmt_value(ve.value), _md_escape(ve.units)]
        if include_sensitivity:
            cells.append(_fmt_sensitivity(ve.sensitivity))
        cells.append(_md_escape(ve.label))
        rows.append(cells)
    return _md_table(cols, rows)


def _md_prose_lines(view) -> list:
    """Return markdown lines for description, assumptions, references, and objective."""
    lines = []
    if view.description:
        lines.append(view.description)
        lines.append("")
    if view.assumptions:
        lines.append(f"**Assumptions:** {'; '.join(view.assumptions)}")
        lines.append("")
    if view.references:
        lines.append(f"**References:** {'; '.join(view.references)}")
        lines.append("")
    return lines


def render_markdown(view, level: int = 1) -> str:
    """Render a ModelView tree as Markdown.

    Parameters
    ----------
    view : ModelView
    level : int
        Current heading level (1 for top-level model, increments for children).

    Returns
    -------
    str
        Markdown representation.
    """
    hdr = "#" * min(level, 6)
    lines: list = []

    if view.preamble:
        lines.append(view.preamble)
        lines.append("")

    if view.is_anonymous:
        # Transparent wrapper: skip heading, children rendered at same level
        child_level = level
    else:
        # Heading: use full lineage path when available, else title
        lines.append(f"{hdr} {view.lineage_path or view.title}")
        lines.append("")
        child_level = level + 1

    lines.extend(_md_prose_lines(view))

    # Constraint groups
    scope = view.scope.also_excluding("units")
    for cg in view.constraint_groups:
        group_header = f"**Constraints: {cg.label}**" if cg.label else "**Constraints**"
        lines.append(group_header)
        lines.append("")
        lines.append("$$\\begin{aligned}")
        for c in cg.constraints:
            lines.append(f"{render(c, LATEX_ALIGNED, scope)} \\\\")
        lines.append("\\end{aligned}$$")
        lines.append("")

    # Optimized Variables pipe table (no sensitivity column)
    if view.free_variables:
        lines.append("**Optimized Variables**")
        lines.append("")
        lines.extend(_md_var_table(view.free_variables))
        lines.append("")

    # Fixed Variables pipe table (value | units | sensitivity | label)
    if view.fixed_variables:
        lines.append("**Fixed Variables**")
        lines.append("")
        lines.extend(_md_var_table(view.fixed_variables, include_sensitivity=True))
        lines.append("")

    # Children (recursive)
    for child in view.children:
        child_md = render_markdown(child, level=child_level)
        lines.append(child_md)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── Renderer dispatcher ───────────────────────────────────────────────────────


def render_report(
    view,
    fmt: str = "text",
    front_matter: str = "",
    toc: bool = False,
) -> "dict | str":
    """Dispatch to the appropriate format renderer.

    Parameters
    ----------
    view : ModelView
        The root view, as model_view() builds it.
    fmt : str
        Output format: "dict", "text", "md", or "latex".
    front_matter : str, optional
        Raw text prepended once, ahead of the root section. A model's own
        `report_preamble()` is separate, and renders per section.
    toc : bool, optional
        Insert a table-of-contents marker, where the format has one. Markdown
        does; plain text accepts the argument and drops it.
    """
    if fmt == "dict":
        return view.to_dict() | {"front_matter": front_matter, "toc": toc}
    if fmt == "text":
        body = render_text(view)
    elif fmt == "md":
        body = f"[TOC]\n\n{render_markdown(view)}" if toc else render_markdown(view)
    else:
        raise ValueError(
            f"Format '{fmt}' not yet implemented. "
            "Available formats: 'dict', 'text', 'md', 'latex'."
        )
    return f"{front_matter}\n\n{body}" if front_matter else body
