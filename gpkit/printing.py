"printing functionality for gpkit objects"

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

from .util.repr_conventions import unitstr

Item = tuple[Any, Any]


@dataclass(frozen=True)
class PrintOptions:
    "container for printing options"

    empty: str | None = None  # output (e.g. "(none)") for empty sections
    precision: int = 4
    topn: int | None = None  # truncation per-group
    vecn: int = 6  # max vector elements to print before ...
    vec_width: int | None = None  # None -> auto-align elements when applicable


@dataclass(frozen=True)
class ItemSource:
    "Attribute path to retrieve a Mapping holding Items"

    path: str


@dataclass(frozen=True)
class DiffPair:
    "Difference between new and old"

    new: Any
    old: Any

    def rel(self):
        "simple relative difference"
        return np.asarray(self.new) / np.asarray(self.old) - 1

    @property
    def shape(self):
        "shape is inferred from new (old must match or be scalar)"
        s = np.shape(self.new)
        if np.shape(self.old):
            assert np.shape(self.old) == s
        return s


# pylint: disable=missing-class-docstring
class SectionSpec:
    align = None
    align_vecs = True  # aligns vectors if all same length
    col_sep = " "
    filterfun = None
    filter_reduce = staticmethod(any)
    group_by_model = True
    pm = ""  # sign format prefix (e.g. '+' for sensitivities)
    sortkey = None
    source = None
    title: str = "Untitled Section"

    def __init__(self, options: PrintOptions):
        self.options = options

    def items_from(self, ctx):
        "Return iterable of items given SolContext. Item defs are section-specific"
        if self.source is None:
            raise NotImplementedError
        return ctx.items(self.source)

    def row_from(self, item):
        "Convert a section-specific 'item' to a row, i.e. List[str]"
        raise NotImplementedError

    def _auto_vecwidth_rowspec(self, items):
        "Return a copy of this with vec_width set automatically for items"
        if self.align_vecs and self.options.vec_width is None:
            lengths = set(np.shape(v) for _, v in items if np.shape(v))
            if len(lengths) == 1:
                width = self._max_val_width(items)
                newopt = replace(self.options, vec_width=width)
                return self.__class__(options=newopt)
        return self

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
            # auto-compute width and replace option, if required
            rowspec = self._auto_vecwidth_rowspec(model_items)
            # 2. extract rows
            rows = [rowspec.row_from(item) for item in model_items]
            # 3. Align columns
            model_lines = _format_aligned_columns(rows, self.align, self.col_sep)
            # add model header
            if modelname and model_lines:
                lines.append(f"{modelname}")
            lines.extend(model_lines)
            lines.append("")

        if not lines:  # empty section
            if self.options.empty is not None:
                lines += [str(self.options.empty), ""]
            else:
                return lines

        # title
        title_lines = [self.title, "-" * len(self.title)]
        assert lines[-1] == ""
        return title_lines + lines[:-1]

    def _fmt_val(self, val, suff="") -> str:
        n = self.options.vecn
        p = self.options.precision
        w = self.options.vec_width or 0
        if np.shape(val):
            flat = np.asarray(val).ravel()
            shown = flat[:n]
            body = "  ".join(f"{x:{self.pm}.{p-1}g}{suff}".replace("+nan", "nan").ljust(w) for x in shown)
            dots = " ..." if flat.size > n else ""
            return f"[ {body}{dots} ]"
        return f"{val:{self.pm}.{p}g}{suff}"

    def _passes_filter(self, item) -> bool:
        # pylint: disable=not-callable
        if self.filterfun is None:
            return True
        k, v = item
        if not np.shape(v):  # scalar case
            return bool(self.filterfun(item))
        arr = np.asarray(v).ravel()
        flags = (bool(self.filterfun((k, vi))) for vi in arr)
        return self.filter_reduce(flags)  # vector case

    def _width_array(self, v):
        "Hook for subclasses: array-like value used for width inference."
        return v

    def _max_val_width(self, items):
        "infer how wide the widest vector element will be"
        w = 0
        p = self.options.precision
        for _, v in items:
            arr = self._width_array(v)
            if not np.shape(arr):
                continue
            flat = np.asarray(arr).ravel()
            w = max(w, max(len(f"{el:{self.pm}.{p-1}g}") for el in flat))
        return w


class Cost(SectionSpec):
    title = "Optimal Cost"
    source = staticmethod(lambda sol: {sol.meta["cost function"]: sol.cost})

    def row_from(self, item):
        """Extract [name, value, unit] for cost display."""
        key, val = item
        name = key.str_without("units") if key else "cost"
        return [f"{name} :", self._fmt_val(val), _unitstr(key)]


class Warnings(SectionSpec):
    title = "WARNINGS"
    align_vecs = False

    def row_from(self, item):
        """Extract [warning_type, details] for warning display."""
        warning_type, warning_list = item
        return [f"{warning_type}:\n" + "\n".join(warning_list)]

    def items_from(self, ctx):
        return ctx.warning_items()


def _warnings_single(sol):
    "get the warning dict for a single solution, handling any special cases"

    def _special_case(name, payload) -> str:
        # refactor architecture to avoid these two special cases
        if "Unexpectedly Loose Constraints" in name:
            _rel_diff, loosevalues, c = payload
            lhs, op, rhs = loosevalues
            cstr = c.str_without({"units", "lineage"})
            return f"{lhs:.4g} {op} {rhs:.4g} : {cstr}"
        if "Unexpectedly Tight Constraints" in name:
            relax_sens, c = payload
            cstr = c.str_without({"units", "lineage"})
            return f"{relax_sens:+6.2g} : {cstr}"
        return ""

    warns = getattr(sol, "meta", {}).get("warnings", {})
    out = {}
    for name, detail in warns.items():
        if not detail:
            continue
        out[name] = [_special_case(name, pay) or msg for msg, pay in detail]
    return out


class FreeVariables(SectionSpec):
    title = "Free Variables"
    align = "><<<"
    sortkey = staticmethod(lambda x: str(x[0]))
    source = ItemSource("primal")

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
    source = ItemSource("constants")

    def row_from(self, item):
        """Extract [name, value, unit, label] for variable tables."""
        key, val = item
        name = key.str_without("lineage")
        label = key.descr.get("label", "")
        return [f"{name} :", self._fmt_val(val), _unitstr(key), label]


class Sweeps(Constants):
    title = "Swept Variables"
    source = staticmethod(lambda s: getattr(s, "meta", {}).get("sweep_point", {}))


class Sensitivities(SectionSpec):
    title = "Variable Sensitivities"
    sortkey = staticmethod(lambda x: (-rounded_mag(x[1]), str(x[0])))
    filterfun = staticmethod(lambda x: rounded_mag(x[1]) >= 0.01)
    align = "><<"
    pm = "+"
    source = ItemSource("sens.variables")

    def row_from(self, item):
        """Extract [name, value, label] (no units!)."""
        key, val = item
        name = key.str_without("lineage")
        label = key.descr.get("label", "")
        value = self._fmt_val(val)
        return [f"{name} :", value, label]


class Constraints(SectionSpec):
    sortkey = staticmethod(lambda x: (-rounded_mag(x[1]), str(x[0])))
    col_sep = " : "
    pm = "+"
    source = ItemSource("sens.constraints")

    def row_from(self, item):
        """Extract [sens, constraint_str] for constraint tables."""
        constraint, sens = item
        constrstr = constraint.str_without({"units", "lineage"})
        valstr = self._fmt_val(sens)
        return [valstr, constrstr]


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


class DiffSection(SectionSpec):

    def row_from(self, item):
        "still abstract at this level"
        raise NotImplementedError

    def _width_array(self, v):
        "Use relative change when inferring widths for diff-style sections."
        return v.rel()


class DiffCost(DiffSection):
    title = "Cost Change"
    source = staticmethod(Cost.source)
    pm = "+"

    def row_from(self, item):
        key, pair = item
        name = key.str_without("units") if key else "cost"
        u = unitstr(key, into="%s", dimless="")
        vec = np.shape(pair.rel())
        return [
            f"{name} :",
            f"{self._fmt_val(pair.rel() * 100, suff='%')}",
            f"({pair.new:.4g}{u} vs {pair.old:.4g}{u})" if not vec else "",
        ]


class DiffFreeVariables(DiffSection):
    title = "Free Variable Changes"
    source = staticmethod(FreeVariables.source)
    sortkey = staticmethod(
        lambda kv: (-rounded_mag(np.max(np.abs(kv[1].rel()))), str(kv[0]))
    )
    pm = "+"
    align = "><<"

    # filter out zero vals
    filterfun = staticmethod(lambda kv: np.any(kv[1].rel() != 0))

    def row_from(self, item):
        key, pair = item
        name = key.str_without("lineage")
        u = unitstr(key, into="%s", dimless="")
        label = key.descr.get("label", "")
        rel = pair.rel()
        diffstr = f"{self._fmt_val(rel * 100, suff='%')}"
        if not np.shape(rel):
            diffstr += f"  ({pair.new:.4g}{u} vs {pair.old:.4g}{u})"
        return [f"{name} :", diffstr, label]


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


DIFF_SECTION_SPECS = {
    "cost": DiffCost,
    "freevariables": DiffFreeVariables,
}


@dataclass(frozen=True)
class SolutionContext:
    """Adapter that exposes a single Solution's printable items."""

    sol: Any

    def items(self, source: [ItemSource, Callable]) -> Iterable[Item]:
        "Get the items associated with a particular attribute (source)"
        if isinstance(source, ItemSource):
            obj = _resolve_attrpath(self.sol, source.path)
            return getattr(obj, "vector_parent_items", obj.items)()
        return source(self.sol).items()

    def warning_items(self) -> Iterable[tuple[str, list[str]]]:
        """Return flattened warning messages keyed by warning name."""
        return _warnings_single(self.sol).items()


@dataclass(frozen=True)
class SequenceContext:
    """Adapter that stacks printable items across a sequence of Solutions."""

    sols: Sequence[Any]  # sequence of Solution-like objects

    def items(self, source: [ItemSource, Callable]) -> Iterable[Item]:
        "Items for a given attribute are stacked across self.sols"

        def _items_one_sol(sol):
            if isinstance(source, ItemSource):
                return _resolve_attrpath(sol, source.path).items()
            return source(sol).items()

        return self._stack(_items_one_sol)

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

    def warning_items(self) -> Iterable[tuple[str, list[str]]]:
        """Merge warnings from all solutions into a single mapping."""
        counts = defaultdict(Counter)
        n = len(self.sols)
        for s in self.sols:
            for name, warn_list in _warnings_single(s).items():
                for entry in warn_list:
                    counts[name][entry] += 1
        items = defaultdict(list)
        for name, cnt in counts.items():
            for w, c in cnt.items():
                items[name].append(f"{w}  (in {c} of {n} solutions)")
        return items.items()


@dataclass(frozen=True)
class DiffContext:
    "Adapter that provides (key, DiffPair(new, old)) items."

    new: Any  # SolutionContext or SequenceContext
    baseline: Any  # Solution-like

    def items(self, source: [ItemSource, Callable]) -> Iterable[Item]:
        "Items are (key, DiffPair)"
        new_items = list(self.new.items(source))
        old_items = dict(SolutionContext(self.baseline).items(source))
        return [(k, DiffPair(v, old_items.get(k))) for k, v in new_items]


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
    ctx = SolutionContext(obj) if _looks_like_solution(obj) else SequenceContext(obj)
    blocks: list[str] = []
    for table_name in tables:
        sec = SECTION_SPECS[table_name](options=opt)
        sec_lines = sec.format(ctx)
        if sec_lines:
            blocks.append("\n".join(sec_lines))
    return "\n\n".join(blocks)


def diff(obj, baseline, tables=("cost", "freevariables"), **options):
    "Render text tables of differences between obj and baseline"
    opt = PrintOptions(**options)
    new_ctx = (
        SolutionContext(obj) if _looks_like_solution(obj) else SequenceContext(obj)
    )
    ctx = DiffContext(new=new_ctx, baseline=baseline)

    blocks = []
    for name in tables:
        sec = DIFF_SECTION_SPECS[name](options=opt)
        lines = sec.format(ctx)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _looks_like_solution(x) -> bool:
    return hasattr(x, "cost") and hasattr(x, "primal")


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


def _resolve_attrpath(obj: Any, path: str) -> Any:
    """Resolve a dotted attribute path (e.g. 'sens.variables')."""
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def rounded_mag(val, nround=8):
    "get the magnitude of a (vector or scalar) for stable sorting purposes"
    return round(np.nanmax(np.absolute(val)), nround)


def _group_items_by_model(items):
    """Group VarMap items by model string
    Input: iterable of (VarKey, value) pairs
    Output: mapping model_str: iterable of (VarKey, value) pairs
    """
    out = defaultdict(list)
    for key, val in items:
        mod = key.lineagestr() if hasattr(key, "lineagestr") else ""
        out[mod].append((key, val))
    return out
