"printing functionality for gpkit objects"

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

from .util.repr_conventions import unitstr

Item = tuple[Any, Any]


@dataclass(frozen=True)
class PrintOptions:
    "container for printing options"

    precision: int = 4
    topn: int | None = None  # truncation per-group
    vecn: int = 6  # max vector elements to print before ...
    empty: str | None = None  # output (e.g. "(none)") for empty sections


# pylint: disable=missing-class-docstring
class SectionSpec:
    title: str = "Untitled Section"
    group_by_model = True
    sortkey = None
    align = None
    filterfun = None
    filter_reduce = staticmethod(any)
    col_sep = " "

    def __init__(self, options: PrintOptions):
        self.options = options

    def items_from(self, ctx):
        "Return iterable of items given SolContext. Item defs are section-specific"
        raise NotImplementedError

    def row_from(self, item):
        "Convert a section-specific 'item' to a row, i.e. List[str]"
        raise NotImplementedError

    def format(self, ctx) -> List[str]:
        "Output this section's lines given a solution or solution context"
        items = [item for item in self.items_from(ctx) if self._passes_filter(item)]
        if self.group_by_model:
            bymod = _group_items_by_model(items)
        else:
            bymod = {"": items}

        # process each model group
        lines = []
        for modelname, model_items in sorted(bymod.items()):

            # 1. sort
            if self.sortkey:
                model_items.sort(key=self.sortkey)

            # 2. extract rows
            rows = [self.row_from(item) for item in model_items]

            # 3. Align columns
            model_lines = _format_aligned_columns(rows, self.align, self.col_sep)

            # add model header
            if modelname and model_lines:
                lines += [f"{modelname}"]

            lines += model_lines + [""]

        if not lines:  # empty section
            if self.options.empty is not None:
                lines += [str(self.options.empty), ""]
            else:
                return lines

        # title
        title_lines = [self.title, "-" * len(self.title)]
        assert lines[-1] == ""
        return title_lines + lines[:-1]

    def _fmt_val(self, val, pm="") -> str:
        n = self.options.vecn
        if np.shape(val):
            flat = np.asarray(val).ravel()
            shown = flat[:n]
            body = "  ".join(f"{x:.3g}" for x in shown)
            dots = " ..." if flat.size > n else ""
            return f"[ {body}{dots} ]"
        return f"{val:{pm}.{self.options.precision}g}"

    def _passes_filter(self, item) -> bool:
        # pylint: disable=not-callable
        if self.filterfun is None:
            return True
        k, v = item
        if not np.shape(v):  # scalar case
            return bool(self.filterfun(item))
        flags = (bool(self.filterfun((k, vi))) for vi in v)
        return self.filter_reduce(flags)  # vector case


class Cost(SectionSpec):

    title = "Optimal Cost"

    def row_from(self, item):
        """Extract [name, value, unit] for cost display."""
        key, val = item
        name = key.str_without("units") if key else "cost"
        return [f"{name} :", self._fmt_val(val), _unitstr(key)]

    def items_from(self, ctx):
        return ctx.cost_items()


class Warnings(SectionSpec):
    title = "WARNINGS"

    def row_from(self, item):
        """Extract [warning_type, details] for warning display."""
        warning_type, warning_detail = item
        return [f"{warning_type}:\n" + "\n".join(warning_detail)]

    def items_from(self, ctx):
        return ctx.warning_items()


class FreeVariables(SectionSpec):
    title = "Free Variables"
    align = "><<<"
    sortkey = staticmethod(lambda x: str(x[0]))

    def items_from(self, ctx):
        return ctx.primal_items()

    def row_from(self, item):
        """Extract [name, value, unit, label] for variable tables."""
        key, val = item
        name = key.str_without("lineage")
        label = key.descr.get("label", "")
        return [f"{name} :", self._fmt_val(val), _unitstr(key), label]


class Constants(SectionSpec):
    title = "Fixed Variables"
    align = "><<<"
    sortkey = staticmethod(lambda x: str(x[0]))

    def items_from(self, ctx):
        return ctx.constant_items()

    def row_from(self, item):
        """Extract [name, value, unit, label] for variable tables."""
        key, val = item
        name = key.str_without("lineage")
        label = key.descr.get("label", "")
        return [f"{name} :", self._fmt_val(val), _unitstr(key), label]


class Sweeps(Constants):
    title = "Swept Variables"

    def items_from(self, ctx):
        return ctx.swept_items()


class Sensitivities(SectionSpec):
    title = "Variable Sensitivities"
    sortkey = staticmethod(lambda x: (-rounded_mag(x[1]), str(x[0])))
    filterfun = staticmethod(lambda x: rounded_mag(x[1]) >= 0.01)
    align = "><<"

    def items_from(self, ctx):
        return ctx.variable_sens_items()

    def row_from(self, item):
        """Extract [name, value, label] (no units!)."""
        key, val = item
        name = key.str_without("lineage")
        label = key.descr.get("label", "")
        value = self._fmt_val(val, pm="+")
        return [f"{name} :", value, label]


class Constraints(SectionSpec):
    sortkey = staticmethod(lambda x: (-rounded_mag(x[1]), str(x[0])))
    col_sep = " : "

    def row_from(self, item):
        """Extract [sens, constraint_str] for constraint tables."""
        constraint, sens = item
        constrstr = constraint.str_without({"units", "lineage"})
        valstr = self._fmt_val(sens, pm="+")
        return [valstr, constrstr]

    def items_from(self, ctx):
        return ctx.constraint_sens_items()


class TightConstraints(Constraints):
    title = "Most Sensitive Constraints"
    filterfun = staticmethod(lambda x: abs(x[1]) > 1e-2)


class SlackConstraints(Constraints):

    maxsens = 1e-5
    filter_reduce = staticmethod(all)

    @property
    def title(self):
        "custom title property with embedded maxsens"
        return f"Insensitive Constraints (below {self.maxsens})"

    @property
    def filterfun(self):
        "returns True if slack"
        return lambda x: abs(x[1]) <= self.maxsens


SECTION_SPECS = {
    "cost": Cost,
    "warnings": Warnings,
    "freevariables": FreeVariables,
    "constants": Constants,
    "sweeps": Sweeps,
    "sensitivities": Sensitivities,
    "tightest constraints": TightConstraints,
    "slack constraints": SlackConstraints,
}


@dataclass(frozen=True)
class SolutionContext:
    """Adapter that exposes a single Solution's printable items."""

    sol: Any

    def cost_items(self) -> Iterable[Item]:
        """Return the solution cost as a single keyed item."""
        return [(self.sol.meta["cost function"], self.sol.cost)]

    def warning_items(self) -> Iterable[tuple[str, list[str]]]:
        """Return flattened warning messages keyed by warning name."""
        warns = (getattr(self.sol, "meta", None) or {}).get("warnings", {})
        # printing.py currently flattens warning details into strings
        return [
            (name, [x[0] for x in detail]) for name, detail in warns.items() if detail
        ]

    def primal_items(self) -> Iterable[Item]:
        """Return primal variable values grouped by parent."""
        return self.sol.primal.vector_parent_items()

    def constant_items(self) -> Iterable[Item]:
        """Return constant values grouped by parent."""
        return self.sol.constants.vector_parent_items()

    def swept_items(self) -> Iterable[Item]:
        "Return nothing for single Solution case"
        return []

    def variable_sens_items(self) -> Iterable[Item]:
        """Return sensitivities with respect to variables."""
        return self.sol.sens.variables.vector_parent_items()

    def constraint_sens_items(self) -> Iterable[tuple[Any, Any]]:
        """Return sensitivities with respect to constraints."""
        return self.sol.sens.constraints.items()


@dataclass(frozen=True)
class SequenceContext:
    """Adapter that stacks printable items across a sequence of Solutions."""

    sols: Sequence[Any]  # sequence of Solution-like objects

    def _stack(self, get_items: Callable[[Any], Iterable[Item]]) -> list[Item]:
        """Strict stacking: keys (and their order) must match across all sols."""
        if not self.sols:
            return []

        first = list(get_items(self.sols[0]))
        keys0 = tuple(k for k, _ in first)
        cols = {k: [v] for k, v in first}  # k -> list of values, seeded with sol[0]

        for s in self.sols[1:]:
            items = list(get_items(s))
            if tuple(k for k, _ in items) != keys0:
                raise ValueError("SolutionSequence key mismatch")
            for k, v in items:
                cols[k].append(v)

        return [(k, np.asarray(cols[k])) for k in keys0]

    def _stack_scalar(self, key: Any, get_val: Callable[[Any], Any]) -> list[Item]:
        if not self.sols:
            return []
        return [(key, np.asarray([get_val(s) for s in self.sols]))]

    def _sweep_point(self, s: Any) -> dict[Any, Any]:
        return (getattr(s, "meta", None) or {}).get("sweep_point", {}) or {}

    # ---- methods used by SectionSpecs ----

    def cost_items(self) -> Iterable[Item]:
        """Return the cost stacked across all solutions."""
        key = self.sols[0].meta.get("cost function") if self.sols else None
        return self._stack_scalar(key, lambda s: s.cost)

    def warning_items(self) -> Iterable[tuple[str, list[str]]]:
        """Merge warnings from all solutions into a single mapping."""
        # merged: dict[str, list[str]] = {}
        counts = Counter()
        for s in self.sols:
            warns = (getattr(s, "meta", None) or {}).get("warnings", {})
            for name, detail in warns.items():
                for msg, *_ in detail:
                    counts[(name, msg)] += 1
        items = []
        n = len(self.sols)
        for (name, msg), c in counts.items():
            items.append((f"{name} - in {c} of {n} solutions:", [msg]))
        return items

    def primal_items(self) -> Iterable[Item]:
        """Stack primal variable values across solutions."""
        return self._stack(lambda s: s.primal.vector_parent_items())

    def swept_items(self) -> Iterable[Item]:
        """Stack swept parameters, enforcing identical sweep keys."""
        # Strict: sweep keys/order taken from first; must match for all.
        if not self.sols:
            return []
        keys0 = tuple(self._sweep_point(self.sols[0]).keys())
        return self._stack(lambda s: [(k, self._sweep_point(s)[k]) for k in keys0])

    def constant_items(self) -> Iterable[Item]:
        """Stack constants excluding any swept parameters."""
        swept = set(self._sweep_point(self.sols[0]).keys()) if self.sols else set()
        return self._stack(
            lambda s: [
                (k, v) for k, v in s.constants.vector_parent_items() if k not in swept
            ]
        )

    def variable_sens_items(self) -> Iterable[Item]:
        """Stack variable sensitivities across solutions."""
        return self._stack(lambda s: s.sens.variables.vector_parent_items())

    def constraint_sens_items(self) -> Iterable[Item]:
        """Stack constraint sensitivities across solutions."""
        return self._stack(lambda s: s.sens.constraints.items())


def table(
    obj: Any,  # Solution or SolutionSequence
    tables: Tuple[str, ...] = (
        "sweeps",
        "cost",
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
    if _looks_like_solution(obj):  # looks like Solution
        return _table_solution(obj, tables, opt)
    if _looks_like_sequence_of_solutions(obj):
        return _table_sequence(obj, tables, opt)
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


def _format_aligned_columns(
    rows: list[list[str]],  # each row is list of column strings
    col_alignments: str,  # '<' left, '>' right, one char per column
    col_sep: str = " ",  # separator between each column
) -> list[str]:
    """Align arbitrary columns with dynamic widths.

    Input: list of rows, where each row is list of column strings
    Output: list of formatted/aligned lines

    Does NOT sort - expects pre-sorted input.
    """
    (ncols,) = set(len(r) for r in rows) or (0,)
    if col_alignments is None:
        col_alignments = "<" * ncols
    assert len(col_alignments) == ncols
    widths = [max(len(row[i]) for row in rows) for i in range(ncols)]

    formatted = []
    for row in rows:
        parts = [
            f"{cell:{align}{width}}"
            for cell, width, align in zip(row, widths, col_alignments)
        ]

        line = col_sep.join(parts)
        formatted.append(line.rstrip())

    return formatted


def _unitstr(key) -> str:
    return unitstr(key, into="[%s]", dimless="")


def rounded_mag(val, nround=8):
    "get the magnitude of a (vector or scalar) for stable sorting purposes"
    # if np.shape(val) and np.isnan(val).all():
    #     raise ValueError("all-nan-slice")
    return round(np.nanmax(np.absolute(val)), nround)


def _group_items_by_model(items):
    """Group VarMap items by model string
    Input: iterable of (VarKey, value) pairs
    Output: mapping model_str: iterable of (VarKey, value) pairs
    """
    out = {}
    for key, val in items:
        mod = key.lineagestr() if hasattr(key, "lineagestr") else ""
        if mod not in out:
            out[mod] = []
        out[mod].append((key, val))
    return out


# ---------------- single solution ----------------
def _table_solution(sol, tables, options: PrintOptions) -> str:
    sections: list[str] = []

    for table_name in tables:
        section = SECTION_SPECS[table_name](options=options)
        ctx = SolutionContext(sol)
        sec_lines = section.format(ctx)
        if sec_lines:
            sections.append("\n".join(sec_lines))

    return "\n\n".join(sections)


# ---------------- sequence summary ----------------
def _table_sequence(seq, tables, options: PrintOptions) -> str:
    sections: list[str] = []

    for table_name in tables:
        section = SECTION_SPECS[table_name](options=options)
        ctx = SequenceContext(seq)
        sec_lines = section.format(ctx)
        if sec_lines:
            sections.append("\n".join(sec_lines))

    return "\n\n".join(sections)
