"printing functionality for gpkit objects"

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from .util.repr_conventions import unitstr


def table(
    obj: Any,
    tables: Tuple[str, ...] = (
        "cost",
        "freevariables",
        "constants",
        "sensitivities",
        "warnings",
    ),
    *,
    topn: int = 10,
    max_elems: int = 6,
    max_solutions: int = 8,
    latex: bool = False,
    **_,
) -> str:
    """Render a simple text table for a Solution or SolutionSequence."""
    if _looks_like_solution(obj):
        return _table_solution(obj, tables, topn=topn, max_elems=max_elems)
    if _looks_like_sequence_of_solutions(obj):
        return _table_sequence(
            obj, tables, topn=topn, max_elems=max_elems, max_solutions=max_solutions
        )
    raise TypeError("Expected a Solution or iterable of Solutions.")


# ---------------- helpers ----------------
def _looks_like_solution(x) -> bool:
    return hasattr(x, "cost") and hasattr(x, "primal")


def _looks_like_sequence_of_solutions(x) -> bool:
    try:
        it = iter(x)
    except TypeError:
        return False
    try:
        first = next(it)
    except StopIteration:
        return True
    return _looks_like_solution(first)


def _fmt_qty(q) -> tuple[str, str]:
    """Return (value, unit) tuple for separate formatting"""
    try:
        mag = float(getattr(q, "magnitude", q))
        unit_str = unitstr(q, into="[%s]", dimless="")
        return f"{mag:.4g}", unit_str
    except Exception:
        try:
            return f"{float(q):.4g}", ""
        except Exception:
            return str(q), ""


def _fmt_name(vk) -> str:
    try:
        return str(vk)
    except Exception:
        return repr(vk)


def _format_aligned_columns(
    rows: list[list[str]],  # each row is list of column strings
    col_alignments: str = "><<",  # '<' left, '>' right, one char per column
    col_sep: str = " : ",  # separator after first column
) -> list[str]:
    """Align arbitrary columns with dynamic widths.

    Input: list of rows, where each row is list of column strings
    Output: list of formatted/aligned lines

    Example:
        rows = [["x", "1.5", "[m]", "length"],
                ["force", "10.2", "[N]", "thrust"]]
        alignments = "><<<"  # right, left, left, left
        -> ["    x : 1.5  [m]  length",
            "force : 10.2 [N]  thrust"]

    Does NOT sort - expects pre-sorted input.
    """
    if not rows:
        return []

    ncols = len(rows[0])
    widths = [max(len(row[i]) for row in rows) for i in range(ncols)]

    formatted = []
    for row in rows:
        parts = []
        for i, (cell, width, align) in enumerate(zip(row, widths, col_alignments)):
            if align == "<":
                parts.append(f"{cell:<{width}}")
            else:
                parts.append(f"{cell:>{width}}")

        # Join with special separator after first column
        line = parts[0] + col_sep + " ".join(parts[1:])
        formatted.append(line.rstrip())

    return formatted


def _format_table_rows(rows) -> list[str]:
    """Format table rows with dynamic column widths (legacy method)"""
    if not rows:
        return []

    # Calculate max widths for name, value, and unit columns
    name_width = max(len(row[0]) for row in rows)
    val_width = max(len(row[1]) for row in rows)
    unit_width = max(len(row[2]) for row in rows)

    formatted_rows = []
    for name, value, unit, label in rows:
        line = f"{name:>{name_width}} : {value:<{val_width}} "
        line += f" {unit:<{unit_width}}"
        if label:
            line += (" " if unit_width else "") + f"{label}"
        formatted_rows.append(line.rstrip())

    return formatted_rows


def _get_unit(vk) -> str:
    "get the unit string from a varkey"
    try:
        return f"{vk.units:~}" if getattr(vk, "units", None) else ""
    except Exception:
        return ""


def _fmt_number(x) -> str:
    "default number to string conversion"
    try:
        return f"{float(x):.3g}"
    except Exception:
        return str(x)


def _fmt_array_preview(arr, unit: str = "", n: int = 6) -> tuple[str, str]:
    """Return (value, unit) tuple for separate formatting"""
    flat = np.asarray(arr).ravel()
    shown = flat[:n]
    body = "  ".join(_fmt_number(x) for x in shown)
    tail = " ..." if flat.size > n else ""
    value = f"[ {body}{tail} ]"
    unit_str = f"[{unit}]" if unit and unit != "dimensionless" else ""
    return value, unit_str


def _group_items_by_model(items):
    """Group VarMap items by model string
    Input: iterable of (VarKey, value) pairs
    Output: mapping model: iterable of (VarKey, value) pairs
    lineage is dropped in the output strings, since it's captured in the key
    """
    out = {}
    for key, val in items:
        mod = key.lineagestr()
        if mod not in out:
            out[mod] = []
        out[mod].append((key, val))
    return out


# ---------------- extractors ----------------
def _extract_variable_columns(key, val, vmap, max_elems):
    """Extract [name, value, unit, label] for variable tables."""
    name = key.str_without("lineage")
    unit = _get_unit(key)
    label = key.descr.get("label", "")

    if np.shape(val):
        value, unit_str = _fmt_array_preview(val, unit, n=max_elems)
    else:
        value, unit_str = _fmt_qty(vmap.quantity(key))

    return [name, value, unit_str, label]


def _extract_sensitivity_columns(key, val, vmap, max_elems):
    """Extract [name, value, label] for sensitivity tables (no units!)."""
    name = key.str_without("lineage")
    label = key.descr.get("label", "")

    if np.shape(val):
        value, _ = _fmt_array_preview(val, unit="", n=max_elems)
    else:
        value = f"{float(val):+.3g}"

    return [name, value, label]


def _extract_constraint_columns(constraint, sens_str, vmap=None, max_elems=6):
    """Extract [sens, constraint_str] for constraint tables."""
    # Import here to avoid circular imports
    from .util.small_scripts import try_str_without

    excluded = {"units", "lineage"}
    # Handle case where constraint might not have lineagestr method
    try:
        lineage_str = constraint.lineagestr()
    except AttributeError:
        lineage_str = ""

    constrstr = try_str_without(constraint, {":MAGIC:" + lineage_str}.union(excluded))
    if " at 0x" in constrstr:  # don't print memory addresses
        constrstr = constrstr[: constrstr.find(" at 0x")] + ">"

    return [sens_str, constrstr]


# ---------------- table formatters ----------------
def _format_model_group(
    items: list[tuple],  # (key, value) pairs for ONE model
    extractor,  # function(key, val, vmap, max_elems) -> list[str]
    vmap=None,  # may be needed by extractor
    *,
    max_elems: int = 6,
    sortkey=None,  # function((key, val)) -> sortable
    col_alignments: str = "><<<",  # alignment per column
) -> list[str]:
    """Process one model group: sort, extract columns, align.

    Returns list of formatted line strings (no model header).
    """
    # 1. Sort within this model group
    if sortkey:
        items = sorted(items, key=sortkey)

    # 2. Extract to column strings
    rows = [extractor(k, v, vmap, max_elems) for k, v in items]

    # 3. Align columns
    return _format_aligned_columns(rows, col_alignments)


def _format_variable_table(
    items_or_vmap,  # VarMap or iterable of (key, val)
    extractor,  # specific to table type
    *,
    vmap=None,  # for extractors that need it
    max_elems: int = 6,
    group_by_model: bool = True,
    sortkey=None,  # sort within each model
    col_alignments: str = "><<<",
) -> list[str]:
    """High-level table formatter with model grouping.

    1. Normalize input to items
    2. Group by model (outer loop)
    3. For each model: format group (sort, extract, align)
    4. Insert model headers
    """
    # Normalize to items
    if hasattr(items_or_vmap, "vector_parent_items"):
        vmap = items_or_vmap
        items = list(vmap.vector_parent_items())
    else:
        items = list(items_or_vmap)

    # Group by model
    if group_by_model:
        bymod = _group_items_by_model(items)
    else:
        bymod = {"": items}

    # Process each model group
    lines = []
    for modelname in sorted(bymod.keys()):
        model_items = bymod[modelname]

        # Format this model's group
        model_lines = _format_model_group(
            model_items,
            extractor,
            vmap,
            max_elems=max_elems,
            sortkey=sortkey,
            col_alignments=col_alignments,
        )

        # Add model header
        if modelname and model_lines:
            # Compute padding from first line of model_lines
            first_line = model_lines[0]
            colon_pos = first_line.find(":")
            if colon_pos > 0:
                pad = colon_pos
            else:
                pad = 10  # fallback
            lines += ["", f"{'|':>{pad}} {modelname}"]

        lines += model_lines

    return lines


# Legacy function removed - replaced by _format_variable_table


# ---------------- single solution ----------------
def _table_solution(solution, tables, *, topn: int, max_elems: int) -> str:
    lines: list[str] = []

    if "cost" in tables:
        lines += ["\nOptimal Cost", "------------", f"  {solution.cost:.4g}"]

    if "warnings" in tables:
        warns = (getattr(solution, "meta", None) or {}).get("warnings", {})
        if warns:
            lines += ["~" * 8, "WARNINGS", "~" * 8]
            for name, detail in warns.items():
                for tup in detail:
                    lines += [f"{name}", "-" * len(name), f"{tup[0]}"]
            lines += ["~" * 8]

    if "freevariables" in tables:
        lines += ["", "Free Variables", "--------------"]
        lines += _format_variable_table(
            solution.primal,
            extractor=_extract_variable_columns,
            sortkey=lambda x: str(x[0]),
            col_alignments="><<<",  # name, value, unit, label
        )

    if "constants" in tables:
        lines += ["", "Fixed Variables", "---------------"]
        lines += _format_variable_table(
            solution.constants,
            extractor=_extract_variable_columns,
            sortkey=lambda x: str(x[0]),
            col_alignments="><<<",  # name, value, unit, label
        )

    if "sensitivities" in tables:
        sens_vars = getattr(getattr(solution, "sens", None), "variables", None)
        if sens_vars is not None:
            # Pre-process: filter and prepare items
            items = []
            vpi = getattr(sens_vars, "vector_parent_items", None)
            iterable = (
                sens_vars.vector_parent_items()
                if callable(vpi)
                else getattr(sens_vars, "items", lambda: [])()
            )
            for vk, v in iterable:
                val = np.asarray(v, dtype=float)
                sabs = float(np.nanmax(np.abs(val))) if val.size else 0.0
                items.append((vk, sabs, v))

            # Sort by absolute value descending, filter by threshold
            items.sort(key=lambda t: -t[1])
            items = [(vk, raw) for vk, sabs, raw in items[:topn] if sabs >= 0.01]

            lines += ["", "Variable Sensitivities", "----------------------"]
            lines += _format_variable_table(
                items,
                extractor=_extract_sensitivity_columns,  # 3 columns, no units
                vmap=sens_vars,
                sortkey=None,  # already sorted
                col_alignments="<<<",  # name, value, label
                group_by_model=True,  # NEW FEATURE!
            )

    if "tightest constraints" in tables:
        # Pre-process: convert constraints to items with sensitivity strings
        items = []
        for constraint, sens in solution.sens.constraints.items():
            sens_str = f"{sens:+.3g}"
            items.append((constraint, sens_str))

        # Sort by sensitivity descending
        items.sort(key=lambda x: -abs(float(x[1])))
        items = items[:topn]  # top N

        lines += ["", "Most Sensitive Constraints", "-" * 26]
        lines += _format_variable_table(
            items,
            extractor=_extract_constraint_columns,  # 2 columns
            sortkey=None,  # already sorted
            col_alignments="><",  # sens right-aligned, constraint left
            group_by_model=True,  # NEW FEATURE!
        )

    if "slack constraints" in tables:
        maxsens = 1e-5
        # Pre-process: convert constraints to items with sensitivity strings
        items = []
        for constraint, sens in solution.sens.constraints.items():
            sens_str = f"{sens:+.3g}"
            items.append((constraint, sens_str))

        # Sort by sensitivity ascending, filter by threshold
        items.sort(key=lambda x: abs(float(x[1])))
        items = [item for item in items if abs(float(item[1])) <= maxsens]

        lines += ["", f"Insensitive Constraints (below {maxsens})", "-" * 37]
        if not items:
            lines += ["(none)", ""]
        else:
            lines += _format_variable_table(
                items,
                extractor=_extract_constraint_columns,  # 2 columns
                sortkey=None,  # already sorted
                col_alignments="><",  # sens right-aligned, constraint left
                group_by_model=True,  # NEW FEATURE!
            )

    return "\n".join(lines).lstrip()


# ---------------- sequence summary ----------------
def _table_sequence(
    seq: Sequence, tables, *, topn: int, max_elems: int, max_solutions: int
) -> str:
    sols = list(seq)
    n = len(sols)
    lines = ["\nSolution Sequence", "-----------------"]

    if n:
        costs = np.array([getattr(s, "cost", np.nan) for s in sols], dtype=float)
        lines.append(f"  count: {n}")
        lines.append(
            f"  cost: min {np.nanmin(costs):.6g}"
            f"  median {np.nanmedian(costs):.6g}"
            f"  max {np.nanmax(costs):.6g}"
        )

    # Append short per-solution summaries for the first few
    for i, s in enumerate(sols[:max_solutions], 1):
        lines += ["", f"--- Solution {i} ---"]
        lines.append(
            _table_solution(
                s, ("cost", "freevariables"), topn=topn, max_elems=max_elems
            )
        )

    if n > max_solutions:
        lines += ["", f"(… {n - max_solutions} more solutions omitted …)"]

    return "\n".join(lines).strip()
