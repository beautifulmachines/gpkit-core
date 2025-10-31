"printing functionality for gpkit objects"

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .util.repr_conventions import unitstr


@dataclass(frozen=True)
class PrintOptions:
    precision: int = 4
    topn: int | None = None  # truncation per-group
    vecn: int = 6  # max vector elements to print before ...
    empty: str | None = None  # print this (e.g. "(none)" for empty section


def table(
    obj: Any,  # Solution or SolutionSequence
    tables: Tuple[str, ...] = (
        # "cost",
        "warnings",
        "freevariables",
        "constants",
        "sensitivities",
        "tightest constraints",
    ),
    **options,
) -> str:
    """Render a simple text table for a Solution or SolutionSequence."""
    opt = PrintOptions(**options)
    if hasattr(obj, "cost") and hasattr(obj, "primal"):  # looks like Solution
        return _table_solution(obj, tables, opt)
    if _looks_like_sequence_of_solutions(obj):
        raise NotImplementedError
        # return _table_sequence(
        #     obj, tables, topn=topn, max_elems=max_elems, max_solutions=max_solutions
        # )
    raise TypeError("Expected a Solution or iterable of Solutions.")


# ---------------- helpers ----------------
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


def _format_aligned_columns(
    rows: list[list[str]],  # each row is list of column strings
    col_alignments: str,  # '<' left, '>' right, one char per column
    col_sep: str = " : ",  # separator after first column
) -> list[str]:
    """Align arbitrary columns with dynamic widths.

    Input: list of rows, where each row is list of column strings
    Output: list of formatted/aligned lines

    Does NOT sort - expects pre-sorted input.
    """
    (ncols,) = set(len(r) for r in rows) or (0,)
    assert len(col_alignments) >= ncols
    widths = [max(len(row[i]) for row in rows) for i in range(ncols)]

    formatted = []
    for row in rows:
        parts = [
            f"{cell:{align}{width}}"
            for cell, width, align in zip(row, widths, col_alignments)
        ]

        # Join with special separator after first column
        line = parts[0] + (col_sep if parts[1:] else "") + " ".join(parts[1:])
        formatted.append(line.rstrip())

    return formatted


def _fmt_item(key, val, n: int = 6) -> tuple[str, str]:
    "Return (value, unit) tuple for separate formatting. Gets unit from key"
    unit_str = unitstr(key, into="[%s]", dimless="")
    if np.shape(val):
        flat = np.asarray(val).ravel()
        shown = flat[:n]
        body = "  ".join(f"{x:.3g}" for x in shown)
        tail = " ..." if flat.size > n else ""
        return f"[ {body}{tail} ]", unit_str
    return f"{val:.4g} ", unit_str


def _group_items_by_model(items):
    """Group VarMap items by model string
    Input: iterable of (VarKey, value) pairs
    Output: mapping model: iterable of (VarKey, value) pairs
    lineage is dropped in the output strings, since it's captured in the key
    """
    out = {}
    for key, val in items:
        mod = key.lineagestr() if hasattr(key, "lineagestr") else ""
        if mod not in out:
            out[mod] = []
        out[mod].append((key, val))
    return out


# ---------------- extractors ----------------
def _extract_variable_columns(key, val, max_elems):
    """Extract [name, value, unit, label] for variable tables."""
    name = key.str_without("lineage")
    label = key.descr.get("label", "")
    value, unit_str = _fmt_item(key, val)
    return [name, value, unit_str, label]


def _extract_sensitivity_columns(key, val, max_elems):
    """Extract [name, value, label] for sensitivity tables (no units!)."""
    name = key.str_without("lineage")
    label = key.descr.get("label", "")

    if np.shape(val):
        value, _ = _fmt_item("", val, n=max_elems)
    else:
        value = f"{float(val):+.3g}"

    return [name, value, label]


def _extract_constraint_columns(constraint, sens_str, max_elems=6):
    """Extract [sens, constraint_str] for constraint tables."""
    excluded = {"units", "lineage"}

    constrstr = (
        constraint.str_without(excluded)
        if hasattr(constraint, "str_without")
        else str(constraint)
    )

    if sens_str == "" and constrstr == "(none)":
        return [constrstr]

    return [sens_str, constrstr]


def _extract_cost_columns(key, val, max_elems=6):
    """Extract [name, value, unit] for cost display."""
    name = str(key) if key else "cost"
    value, unit_str = _fmt_item(key, val)
    return [name, value, unit_str]


def _extract_warning_columns(warning_type, warning_detail, max_elems=6):
    """Extract [warning_type, details] for warning display."""
    return [f"{warning_type}:\n" + "\n".join(warning_detail)]


# ---------------- table formatters ----------------
def _format_model_group(
    items: list[tuple],  # (key, value) pairs for ONE model
    extractor,  # function(key, val, max_elems) -> list[str]
    options,
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
    rows = [extractor(k, v, options.vecn) for k, v in items]

    # 3. Align columns
    return _format_aligned_columns(rows, col_alignments)


def _format_section(
    items_or_vmap,  # VarMap or iterable of (key, val)
    extractor,  # specific to table type
    options,
    col_alignments: str,
    group_by_model: bool = True,
    sortkey=None,  # sort within each model
) -> list[str]:
    """High-level table formatter with model grouping.

    1. Normalize input to items
    2. Group by model (outer loop)
    3. For each model: format group (sort, extract, align)
    4. Insert model headers
    """
    # Normalize to items
    if hasattr(items_or_vmap, "vector_parent_items"):
        items = list(items_or_vmap.vector_parent_items())
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
            options,
            sortkey=sortkey,
            col_alignments=col_alignments,
        )

        # Add model header
        if modelname and model_lines:
            # Compute padding from first line of model_lines
            first_line = model_lines[0]
            colon_pos = first_line.rfind(":")
            if colon_pos > 0:
                pad = colon_pos
            else:
                pad = 10  # fallback
            lines += [f"{'|':>{pad + 1}} {modelname}"]

        lines += model_lines + [""]

    return lines[:-1]


# ---------------- section methods ----------------
def _section_cost(solution, options):
    """Section method for cost display."""
    return {
        "title": "Optimal Cost",
        "data": [("cost", solution.cost)],
        "extractor": _extract_cost_columns,
        "format_kwargs": {
            "col_alignments": "><<",  # name, value, unit
            "group_by_model": False,
        },
    }


def _section_warnings(solution, options):
    """Section method for warnings display."""
    warns = (getattr(solution, "meta", None) or {}).get("warnings", {})
    if not warns:
        return None

    # Convert warnings to items list
    items = [(name, [x[0] for x in detail]) for name, detail in warns.items() if detail]

    return {
        "title": "WARNINGS",
        "data": items,
        "extractor": _extract_warning_columns,
        "format_kwargs": {
            "col_alignments": "<",  # warning_type, details
            "group_by_model": False,  # warnings don't have model context
        },
    }


def _section_freevariables(solution, options):
    """Section method for free variables display."""
    return {
        "title": "Free Variables",
        "data": solution.primal,
        "extractor": _extract_variable_columns,
        "format_kwargs": {
            "col_alignments": "><<<",  # name, value, unit, label
            "sortkey": lambda x: str(x[0]),
            "group_by_model": True,
        },
    }


def _section_constants(solution, options):
    """Section method for constants display."""
    return {
        "title": "Fixed Variables",
        "data": solution.constants,
        "extractor": _extract_variable_columns,
        "format_kwargs": {
            "col_alignments": "><<<",  # name, value, unit, label
            "sortkey": lambda x: str(x[0]),
            "group_by_model": True,
        },
    }


def _section_sensitivities(solution, options):
    """Section method for sensitivities display."""
    sens_vars = solution.sens.variables

    # Pre-process: filter and prepare items (from original logic)
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
    items.sort(key=lambda t: (-t[1], str(t[0])))
    items = [(vk, raw) for vk, sabs, raw in items[: options.topn] if sabs >= 0.01]

    if not items:
        return None

    return {
        "title": "Variable Sensitivities",
        "data": items,
        "extractor": _extract_sensitivity_columns,
        "format_kwargs": {
            "sortkey": None,  # already sorted
            "col_alignments": "<<<",  # name, value, label
            "group_by_model": True,
        },
    }


def _section_tight_constraints(solution, options):
    """Section method for tightest constraints display."""
    # Pre-process: convert constraints to items with sensitivity strings
    items = []
    for constraint, sens in solution.sens.constraints.items():
        sens_str = f"{sens:+.3g}"
        items.append((constraint, sens_str))

    # Sort by sensitivity descending
    items.sort(key=lambda x: (-abs(float(x[1])), str(x[0])))
    items = [x for x in items if abs(float(x[1])) > 1e-2]  # slow, fix this

    if not items:
        return None

    return {
        "title": "Most Sensitive Constraints",
        "data": items,
        "extractor": _extract_constraint_columns,
        "format_kwargs": {
            "sortkey": None,  # already sorted
            "col_alignments": "><",  # sens right-aligned, constraint left
            "group_by_model": True,
        },
    }


def _section_slack_constraints(solution, options):
    """Section method for slack constraints display."""
    maxsens = 1e-5
    # Pre-process: convert constraints to items with sensitivity strings
    items = []
    for constraint, sens in solution.sens.constraints.items():
        sens_str = f"{sens:+.3g}"
        items.append((constraint, sens_str))

    # Sort by sensitivity ascending, filter by threshold
    items.sort(key=lambda x: abs(float(x[1])))
    items = [item for item in items if abs(float(item[1])) <= maxsens]

    if not items:
        return {
            "title": f"Insensitive Constraints (below {maxsens})",
            "data": [("(none)", "")],
            "extractor": _extract_constraint_columns,
            "format_kwargs": {
                "sortkey": None,
                "col_alignments": "><",
                "group_by_model": True,
            },
        }

    return {
        "title": f"Insensitive Constraints (below {maxsens})",
        "data": items,
        "extractor": _extract_constraint_columns,
        "format_kwargs": {
            "sortkey": None,  # already sorted
            "col_alignments": "><",  # sens right-aligned, constraint left
            "group_by_model": True,
        },
    }


# ---------------- dispatcher ----------------
SECTION_METHODS = {
    "cost": _section_cost,
    "warnings": _section_warnings,
    "freevariables": _section_freevariables,
    "constants": _section_constants,
    "sensitivities": _section_sensitivities,
    "tightest constraints": _section_tight_constraints,
    "slack constraints": _section_slack_constraints,
}


# ---------------- single solution ----------------
def _table_solution(sol, tables, options: PrintOptions) -> str:
    sections: list[str] = []

    for table_name in tables:
        if table_name not in SECTION_METHODS:
            continue

        section_method = SECTION_METHODS[table_name]
        section = section_method(sol, options)

        if not section:  # Skip empty sections
            continue

        # title
        lines = [section["title"], "-" * len(section["title"])]

        # Format table content
        table_lines = _format_section(
            section["data"],
            section["extractor"],
            options,
            **section["format_kwargs"],
        )
        lines += table_lines or ["(none)"]
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ---------------- sequence summary ----------------
# def _table_sequence(
#     seq: Sequence, tables, *, topn: int, max_elems: int, max_solutions: int
# ) -> str:
#     sols = list(seq)
#     n = len(sols)
#     lines = ["\nSolution Sequence", "-----------------"]
#
#     if n:
#         costs = np.array([getattr(s, "cost", np.nan) for s in sols], dtype=float)
#         lines.append(f"  count: {n}")
#         lines.append(
#             f"  cost: min {np.nanmin(costs):.6g}"
#             f"  median {np.nanmedian(costs):.6g}"
#             f"  max {np.nanmax(costs):.6g}"
#         )
#
#     # Append short per-solution summaries for the first few
#     for i, s in enumerate(sols[:max_solutions], 1):
#         lines += ["", f"--- Solution {i} ---"]
#         lines.append(
#             _table_solution(
#                 s, ("cost", "freevariables"), topn=topn, max_elems=max_elems
#             )
#         )
#
#     if n > max_solutions:
#         lines += ["", f"(… {n - max_solutions} more solutions omitted …)"]
#
#     return "\n".join(lines).strip()
