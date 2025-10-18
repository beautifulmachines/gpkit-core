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


def _format_table_rows(rows) -> list[str]:
    """Format table rows with dynamic column widths"""
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


def _fmt_items(vmap, max_elems: int, group_by_model=True, sortkey=None):
    "format (key, value) pairs in a VarMap, optionally grouping by model"
    bymod = (
        _group_items_by_model(vmap.vector_parent_items())
        if group_by_model
        else {"": items}
    )
    lines = []
    for modelname, items in sorted(bymod.items(), key=lambda x: x[0]):
        if modelname:
            lines += ["", f"  | {modelname}"]
        rows = []
        for vk, val in sorted(items, key=sortkey):
            # name = _fmt_name(vk)
            name = vk.str_without("lineage") if group_by_model else _fmt_name(vk)
            unit = _get_unit(vk)
            label = vk.descr.get("label", "")
            if np.shape(val):
                value, unit_str = _fmt_array_preview(val, unit, n=max_elems)
            else:
                value, unit_str = _fmt_qty(vmap.quantity(vk))
            rows.append((name, value, unit_str, label))
        lines += _format_table_rows(rows)
    return lines


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
        lines += _fmt_items(
            # sorted(solution.primal.vector_parent_items(), key=lambda x: str(x[0])),
            solution.primal,
            max_elems=max_elems,
            sortkey=lambda x: str(x[0]),
        )

    if "constants" in tables:
        lines += ["", "Fixed Variables", "---------------"]
        rows = []
        for vk, val in sorted(
            solution.constants.vector_parent_items(), key=lambda x: str(x[0])
        ):
            name = _fmt_name(vk)
            unit = _get_unit(vk)
            label = vk.descr.get("label", "")
            if np.shape(val):
                value, unit_str = _fmt_array_preview(val, unit, n=max_elems)
            else:
                value, unit_str = _fmt_qty(solution.constants.quantity(vk))
            rows.append((name, value, unit_str, label))
        lines += _format_table_rows(rows)

    if "sensitivities" in tables:
        sens_vars = getattr(getattr(solution, "sens", None), "variables", None)
        if sens_vars is not None:
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
            items.sort(key=lambda t: -t[1])
            lines += ["", "Variable Sensitivities", "----------------------"]
            rows = []
            for vk, sabs, raw in items[:topn]:
                if sabs < 0.01:  # temporary hack to mimic SolutionArray
                    break
                name = _fmt_name(vk)
                label = vk.descr.get("label", "")
                if np.shape(raw):
                    value, unit_str = _fmt_array_preview(raw, n=max_elems)
                else:
                    value = f"{float(raw):+.3g}"
                    unit_str = ""
                rows.append((name, value, unit_str, label))
            lines += _format_table_rows(rows)

    if "tightest constraints" in tables:
        lines += ["", "Most Sensitive Constraints", "-" * 26]
        for constraint, sens in sorted(
            solution.sens.constraints.items(), key=lambda x: -abs(x[1])
        ):
            if abs(sens) < 0.001:
                break
            lines += [f"  {sens:+.4g} : {constraint}"]

    return "\n".join(lines).strip()


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
