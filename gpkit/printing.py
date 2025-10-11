"printing functionality for gpkit objects"
from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np


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


def _fmt_qty(q) -> str:
    try:
        mag = float(getattr(q, "magnitude", q))
        unit = f"{q.units:~}" if hasattr(q, "units") else ""
        if unit in ("", "dimensionless"):
            return f"{mag:.3g}"
        return f"{mag:.3g} {unit}"
    except Exception:
        try:
            return f"{float(q):.3g}"
        except Exception:
            return str(q)


def _fmt_name(vk) -> str:
    try:
        return str(vk)
    except Exception:
        return repr(vk)


def _fmt_row(name: str, val: str) -> str:
    return f"  {name:<28} {val}"


def _get_unit(vk) -> str:
    try:
        return f"{vk.units:~}" if getattr(vk, "units", None) else ""
    except Exception:
        return ""


def _fmt_number(x) -> str:
    try:
        return f"{float(x):.3g}"
    except Exception:
        return str(x)


def _fmt_array_preview(arr, unit: str = "", n: int = 6) -> str:
    flat = np.asarray(arr).ravel()
    shown = flat[:n]
    body = "  ".join(_fmt_number(x) for x in shown)
    tail = " ..." if flat.size > n else ""
    if not unit or unit == "dimensionless":
        return f"[ {body}{tail} ]"
    return f"[ {body}{tail} ] {unit}"


# ---------------- single solution ----------------
def _table_solution(solution, tables, *, topn: int, max_elems: int) -> str:
    lines: list[str] = []

    if "cost" in tables:
        lines += ["\nOptimal Cost", "------------", f"  {solution.cost:.6g}"]

    if "freevariables" in tables:
        lines += ["", "Free Variables", "--------------"]
        for vk, val in solution.primal.vector_parent_items():
            name = _fmt_name(vk)
            unit = _get_unit(vk)
            if np.shape(val):
                lines.append(
                    _fmt_row(
                        f"{name}[{np.shape(val)}]",
                        _fmt_array_preview(val, unit, n=max_elems),
                    )
                )
            else:
                lines.append(_fmt_row(name, _fmt_qty(solution.primal.quantity(vk))))

    if "constants" in tables:
        lines += ["", "Constants", "---------"]
        for vk, val in solution.constants.vector_parent_items():
            name = _fmt_name(vk)
            if np.shape(val):
                lines.append(
                    _fmt_row(
                        f"{name}[{np.shape(val)}]",
                        _fmt_array_preview(val, _get_unit(vk), n=max_elems),
                    )
                )
            else:
                lines.append(_fmt_row(name, _fmt_qty(solution.constants.quantity(vk))))

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
            lines += ["", "Top Variable Sensitivities", "--------------------------"]
            for vk, _, raw in items[:topn]:
                if np.shape(raw):
                    lines.append(
                        _fmt_row(_fmt_name(vk), _fmt_array_preview(raw, n=max_elems))
                    )
                else:
                    lines.append(_fmt_row(_fmt_name(vk), f"{float(raw):+.3g}"))

    if "warnings" in tables:
        warns = (getattr(solution, "meta", None) or {}).get("warnings", [])
        if warns:
            lines += ["", "Warnings", "--------"]
            lines += [f"  - {w}" for w in warns]

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
            f"  cost: min {np.nanmin(costs):.6g}  median {np.nanmedian(costs):.6g}  max {np.nanmax(costs):.6g}"
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
