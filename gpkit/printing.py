"printing functionality for gpkit objects"

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .ir import diff_solutions
from .util.repr_conventions import unitstr
from .varmap import is_veckey

Item = tuple[Any, Any]


@dataclass(frozen=True, slots=True)
class PrintOptions:
    "container for printing options"

    empty: str | None = None  # output (e.g. "(none)") for empty sections
    precision: int = 4
    topn: int | None = None  # truncation per-group
    vecn: int = 6  # max vector elements to print before ...
    vec_width: int | None = None  # None -> auto-align elements when applicable
    condition_table: Any = None  # dict[str, Model] for ConditionTable section


@dataclass(frozen=True, slots=True)
class ItemSource:
    "Attribute path to retrieve a Mapping holding Items"

    path: str


@dataclass(frozen=True, slots=True)
class DiffRow:
    """One row of a diff: the comparison, plus both values for context.

    The comparison is read from ir.diff_solutions rather than recomputed here, so
    the table and sol.almost_equal agree by construction.  rel is nan where no
    ratio exists -- a quantity only one side has, or one that grew away from zero.
    """

    rel: Any
    new: Any
    old: Any
    old_units: str = ""  # the baseline's own units, which need not be the new ones

    @property
    def shape(self):
        "Drives the section machinery's scalar-vs-vector handling."
        return np.shape(self.rel)

    @property
    def no_ratio(self) -> bool:
        return not np.shape(self.rel) and np.isnan(self.rel)


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
            lengths = {np.shape(v) for _, v in items if np.shape(v)}
            if len(lengths) == 1:
                width = self._max_val_width(items)
                newopt = replace(self.options, vec_width=width)
                return self.__class__(options=newopt)
        return self

    def format(self, ctx) -> list[str]:
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

    def _fmt_one(self, x, p, suff="") -> str:
        "Format a single scalar element for vector display."
        return f"{x:{self.pm}.{p - 1}g}{suff}".replace("+nan", "nan")

    def _fmt_val(self, val, suff="") -> str:
        n = self.options.vecn
        p = self.options.precision
        w = self.options.vec_width or 0
        if np.shape(val):
            flat = np.asarray(val).ravel()
            shown = flat[:n]
            body = "  ".join(self._fmt_one(x, p, suff).ljust(w) for x in shown)
            dots = " ..." if flat.size > n else ""
            return f"[ {body}{dots} ]"
        return f"{val:{self.pm}.{p}g}{suff}"

    def _passes_filter(self, item) -> bool:
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
            w = max(w, *(len(f"{el:{self.pm}.{p - 1}g}") for el in flat))
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
    "a single solution's warning messages, by category"
    warns = getattr(sol, "meta", {}).get("warnings", {})
    return {name: [w["message"] for w in ws] for name, ws in warns.items() if ws}


class FreeVariables(SectionSpec):
    title = "Free Variables"
    align = "><<<"
    sortkey = staticmethod(lambda x: str(x[0]))
    source = ItemSource("primal")

    def row_from(self, item):
        """Extract [name, value, unit, label] for variable tables."""
        key, val = item
        name = key.str_without("lineage")
        label = key.label
        return [f"{name} :", self._fmt_val(val), _unitstr(key), label]


class Constants(SectionSpec):
    title = "Fixed Variables"
    align = "><<<<"  # name(R), value(L), unit(L), sens-in-parens(L), label(L)
    sortkey = staticmethod(lambda x: (-rounded_mag(x[1][1]), str(x[0])))
    nearzero_tol = 1e-7
    _fmt_unit_w = 0  # overridden per-group by SolutionSection._format_model_group
    _fmt_col_widths = (
        None  # overridden per-group by SolutionSection._format_model_group
    )

    def items_from(self, ctx):
        """Yield (varkey, (value, sensitivity)) for each fixed variable."""
        constants = list(ctx.items(ItemSource("constants")))
        sens_dict = dict(ctx.items(ItemSource("sens.variables")))
        for key, val in constants:
            sens = sens_dict.get(key, 0.0)
            yield (key, (val, sens))

    def _fmt_one_sens(self, x, p) -> str:
        """Format a single sensitivity element: ~0 if near-zero, else +x."""
        if abs(x) < self.nearzero_tol:
            return "~0"
        return f"{x:+.{p - 1}g}".replace("+nan", "nan")

    def _fmt_sens(self, sens) -> str:
        """Format scalar sensitivity as (+x) or (~0)."""
        if abs(sens) < self.nearzero_tol:
            return "(~0)"
        p = self.options.precision
        return f"({sens:+.{p}g})"

    def _fmt_vec_pair(self, flat_val, flat_sens):
        """Build aligned value/sensitivity vector strings.

        Each element position uses the wider of its value or sensitivity string,
        so value and sensitivity rows align vertically column by column.
        Returns (val_vec, sens_vec) bracket/paren strings.
        """
        n, p = self.options.vecn, self.options.precision
        col_widths = getattr(self, "_fmt_col_widths", None)
        val_strs = [self._fmt_one(x, p) for x in flat_val[:n]]
        sens_strs = [self._fmt_one_sens(x, p) for x in flat_sens[:n]]
        min_ws = col_widths or [0] * len(val_strs)
        widths = [
            max(len(v), len(s), mw) for v, s, mw in zip(val_strs, sens_strs, min_ws)
        ]
        dots = " ..." if flat_val.size > n else ""
        val_body = "  ".join(v.ljust(w) for v, w in zip(val_strs, widths))
        sens_body = "  ".join(s.ljust(w) for s, w in zip(sens_strs, widths))
        return f"[ {val_body}{dots} ]", f"( {sens_body}{dots} )"

    def _fmt_vector_item(self, key, val, sens, name_w) -> list[str]:
        """Format a vector constant as two lines: values then sensitivities below."""
        flat_val = np.asarray(val).ravel()
        flat_sens = (
            np.asarray(sens).ravel()
            if np.shape(sens)
            else np.full(flat_val.shape, float(sens))
        )
        val_vec, sens_vec = self._fmt_vec_pair(flat_val, flat_sens)
        name_col = f"{key.str_without('lineage'):>{name_w}} :"
        unit = _unitstr(key).ljust(self._fmt_unit_w)
        parts = [name_col, val_vec, unit, key.label or ""]
        line1 = self.col_sep.join(x for x in parts if x).rstrip()
        sens_col = f"{'sens':>{name_w + 2}}"
        return [line1, f"{sens_col}{self.col_sep}{sens_vec}"]

    def _format_model_group(self, model_items) -> list[str]:
        """Render one model group: scalars column-aligned, vectors as two lines."""
        scalar_items = [(k, v) for k, v in model_items if not np.shape(v[0])]
        vector_items = [(k, v) for k, v in model_items if np.shape(v[0])]
        lines = []
        if scalar_items:
            rows = [self.row_from(item) for item in scalar_items]
            lines.extend(_format_aligned_columns(rows, self.align, self.col_sep))
        if vector_items:
            name_w = max(
                *(len(k.str_without("lineage")) for k, _ in vector_items),
                len("sens") - 2,
            )
            for key, (val, sens) in vector_items:
                lines.extend(self._fmt_vector_item(key, val, sens, name_w))
        return lines

    def format(self, ctx) -> list[str]:
        """Override to render vector constants as two vertically-aligned lines."""
        items = [item for item in self.items_from(ctx) if self._passes_filter(item)]
        bymod = _group_items_by_model(items) if self.group_by_model else {"": items}
        lines = []
        for modelname, model_items in sorted(bymod.items()):
            if self.sortkey:
                model_items.sort(key=self.sortkey)
            model_lines = self._format_model_group(model_items)
            if modelname and model_lines:
                lines.append(modelname)
            lines.extend(model_lines)
            lines.append("")
        if not lines:
            if self.options.empty is not None:
                lines += [str(self.options.empty), ""]
            else:
                return lines
        title_lines = [self.title, "-" * len(self.title)]
        assert lines[-1] == ""
        return title_lines + lines[:-1]

    def row_from(self, item):
        """Return [name, value, unit, (sensitivity), label] row for scalars."""
        key, (val, sens) = item
        name = key.str_without("lineage")
        label = key.label
        return [
            f"{name} :",
            self._fmt_val(val),
            _unitstr(key),
            self._fmt_sens(sens),
            label or "",
        ]


class SolutionSection(Constants):
    """Combined free + fixed variables, grouped by submodel.

    Inherits Constants to reuse all vector formatting logic; extends it to
    also yield free variables (with a None sensitivity sentinel) so that a
    single aligned table can display both optimized outputs and fixed inputs.
    """

    title = "Solution"
    sortkey = None  # sorting is handled per sub-group in _format_model_group

    def items_from(self, ctx):
        """Yield (varkey, (value, sensitivity)) for all variables.

        Free variables use None as a sentinel for the sensitivity field.
        """
        for key, val in ctx.items(ItemSource("primal")):
            yield (key, (val, None))
        yield from super().items_from(ctx)  # fixed vars with sensitivities

    def _vec_col_widths(self, vec_items):
        """Per-column element widths (max over all rows) for aligned vector printing."""
        sizes = {np.asarray(v[0]).size for _, v in vec_items if np.shape(v[0])}
        if len(sizes) != 1:
            return None
        n = min(next(iter(sizes)), self.options.vecn)
        p = self.options.precision
        widths = [0] * n
        for _, (val, sens) in vec_items:
            flat_val = np.asarray(val).ravel()
            flat_sens = (
                np.asarray(sens).ravel()
                if np.shape(sens)
                else np.full(flat_val.shape, float(sens))
            )
            for j in range(min(flat_val.size, n)):
                widths[j] = max(
                    widths[j],
                    len(self._fmt_one(flat_val[j], p)),
                    len(self._fmt_one_sens(flat_sens[j], p)),
                )
        return widths

    def _fmt_free_vec(self, val, vw) -> str:
        """Format a free vector value with uniform element width vw."""
        n, p = self.options.vecn, self.options.precision
        flat = np.asarray(val).ravel()
        body = "  ".join(self._fmt_one(x, p).ljust(vw) for x in flat[:n])
        dots = " ..." if flat.size > n else ""
        return f"[ {body}{dots} ]"

    def _format_model_group(self, model_items) -> list[str]:
        """Free vars (alphabetical) then fixed vars (by magnitude)."""
        scalar_free = sorted(
            [(k, v) for k, v in model_items if v[1] is None and not np.shape(v[0])],
            key=lambda x: str(x[0]),
        )
        scalar_fixed = sorted(
            [(k, v) for k, v in model_items if v[1] is not None and not np.shape(v[0])],
            key=lambda x: (-rounded_mag(x[1][1]), str(x[0])),
        )
        vec_free = sorted(
            [(k, v) for k, v in model_items if v[1] is None and np.shape(v[0])],
            key=lambda x: str(x[0]),
        )
        vec_fixed = sorted(
            [(k, v) for k, v in model_items if v[1] is not None and np.shape(v[0])],
            key=lambda x: (-rounded_mag(x[1][1]), str(x[0])),
        )
        name_w = max((len(k.str_without("lineage")) for k, _ in model_items), default=0)
        vw = self._max_val_width([(k, v[0]) for k, v in vec_free])
        # Instance attrs communicate formatting to _fmt_vector_item/_fmt_vec_pair.
        self._fmt_unit_w = max(
            (len(_unitstr(k)) for k, _ in vec_free + vec_fixed), default=0
        )
        # Run all scalars together for shared column widths, split at the boundary.
        aligned = (
            _format_aligned_columns(
                [
                    [
                        f"{k.str_without('lineage'):>{name_w}} :",
                        self._fmt_val(v[0]),
                        _unitstr(k),
                        "" if v[1] is None else self._fmt_sens(v[1]),
                        k.label or "",
                    ]
                    for k, v in scalar_free + scalar_fixed
                ],
                self.align,
                self.col_sep,
            )
            if scalar_free or scalar_fixed
            else []
        )
        lines = list(aligned[: len(scalar_free)])
        for key, (val, _) in vec_free:
            lines.append(
                self.col_sep.join(
                    x
                    for x in [
                        f"{key.str_without('lineage'):>{name_w}} :",
                        self._fmt_free_vec(val, vw),
                        _unitstr(key).ljust(self._fmt_unit_w),
                        key.label or "",
                    ]
                    if x
                ).rstrip()
            )
        if scalar_fixed or vec_fixed:
            lines.append("...constants")
        lines.extend(aligned[len(scalar_free) :])
        self._fmt_col_widths = self._vec_col_widths(vec_fixed)
        for key, (val, sens) in vec_fixed:
            lines.extend(super()._fmt_vector_item(key, val, sens, name_w))
        return lines


class Sweeps(SectionSpec):
    title = "Swept Variables"
    align = "><<<"
    sortkey = staticmethod(lambda x: str(x[0]))
    source = staticmethod(lambda s: getattr(s, "meta", {}).get("sweep_point", {}))

    def row_from(self, item):
        """Extract [name, value, unit, label] for swept variable tables."""
        key, val = item
        name = key.str_without("lineage")
        label = key.label
        return [f"{name} :", self._fmt_val(val), _unitstr(key), label]


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


def _rel_magnitude(row) -> float:
    "Largest |rel|; a missing ratio outranks any number, being news in itself."
    rel = np.asarray(row.rel, dtype=float)
    if np.isnan(rel).any():
        return float("inf")
    return float(np.max(np.abs(rel))) if rel.size else 0.0


class DiffSection(SectionSpec):
    "A section whose numbers come from ir.diff_solutions."

    diff_section = None  # which part of that output supplies this section's rel
    sortkey = staticmethod(lambda kv: (-rounded_mag(_rel_magnitude(kv[1])), str(kv[0])))
    # an unchanged quantity says nothing; a missing ratio is itself worth a row
    filterfun = staticmethod(lambda kv: _rel_magnitude(kv[1]) != 0)

    def items_from(self, ctx):
        return ctx.diff_items(self.source, self.diff_section)

    def row_from(self, item):
        "still abstract at this level"
        raise NotImplementedError

    def _width_array(self, v):
        "Use relative change when inferring widths for diff-style sections."
        return v.rel

    def _pctstr(self, row) -> str:
        "The relative change, or what happened instead of one."
        if row.old is None:
            return "(new)"
        if row.new is None:
            return "(only in baseline)"
        if row.no_ratio:
            return ""  # a rewritten objective, say: the two values speak for it
        return self._fmt_val(np.asarray(row.rel) * 100, suff="%")

    def _bothstr(self, row, units) -> str:
        "Both values, each with its own units, which need not be the same ones."
        if row.new is None or row.old is None or np.shape(row.rel):
            return ""
        return f"({row.new:.4g}{units} vs {row.old:.4g}{row.old_units})"

    def _relstr(self, row, units) -> str:
        parts = (self._pctstr(row), self._bothstr(row, units))
        return "  ".join(p for p in parts if p)


class DiffCost(DiffSection):
    title = "Cost Change"
    source = staticmethod(Cost.source)
    diff_section = "cost"
    filterfun = None  # the headline number, reported even when it did not move
    pm = "+"

    def row_from(self, item):
        key, row = item
        name = key.str_without("units") if key else "cost"
        u = unitstr(key, into="%s", dimless="")
        return [f"{name} :", self._pctstr(row), self._bothstr(row, u)]


class DiffFreeVariables(DiffSection):
    title = "Free Variable Changes"
    source = staticmethod(FreeVariables.source)
    diff_section = "primal"
    pm = "+"
    align = "><<"

    def row_from(self, item):
        key, row = item
        u = unitstr(key, into="%s", dimless="")
        return [f"{key.str_without('lineage')} :", self._relstr(row, u), key.label]


class DiffConstants(DiffFreeVariables):
    title = "Constant Changes"
    source = ItemSource("constants")
    diff_section = "constants"


class DiffSensitivities(DiffSection):
    """Sensitivities moved by a difference, not a ratio.

    They are already log derivatives, so a percentage of one would be a
    percentage of a derivative -- the swing itself is the readable number.
    """

    title = "Sensitivity Changes"
    source = ItemSource("sens.variables")
    diff_section = "sensitivities"
    pm = "+"
    align = "><<"

    def row_from(self, item):
        key, row = item
        diffstr = self._fmt_val(row.rel)
        if not np.shape(row.rel):
            diffstr += f"  ({row.new:+.4g} vs {row.old:+.4g})"
        return [f"{key.str_without('lineage')} :", diffstr, key.label]


class ConditionTable(SectionSpec):
    """Side-by-side comparison of named operating conditions within a model.

    Rows are variables; columns are condition names.  A dash is shown when a
    variable is absent in a condition.  Variables whose values are identical
    across every condition they appear in are suppressed (they add no signal).

    Pass via sol.table(tables=["condition_table"], condition_table={...}).
    """

    title = "Condition Comparison"

    def row_from(self, item):
        raise NotImplementedError  # ConditionTable uses format() directly

    def _collect_condition_data(self, sol, conditions):
        col_data: dict[str, dict[str, tuple[float, str, str]]] = {}
        all_names: list[str] = []
        seen_names: set[str] = set()
        for cname, model in conditions.items():
            col_data[cname] = {}
            for vk in sorted(model.vks, key=lambda v: v.name):
                if vk in sol.primal:
                    val = sol.primal[vk]
                elif vk in sol.constants:
                    val = sol.constants[vk]
                else:
                    continue
                if np.shape(val):
                    continue
                col_data[cname][vk.name] = (float(val), _unitstr(vk), vk.label or "")
                if vk.name not in seen_names:
                    all_names.append(vk.name)
                    seen_names.add(vk.name)
        return col_data, all_names

    def _build_row(self, name, cnames, col_data, p):
        unitlabel = label = ""
        vals: list[str] = []
        raw_vals: list[float] = []
        for cname in cnames:
            entry = col_data[cname].get(name)
            if entry is not None:
                v, unitlabel, label = entry
                vals.append(f"{v:.{p - 1}g}")
                raw_vals.append(v)
            else:
                vals.append("—")
        if raw_vals and len(set(raw_vals)) == 1:
            return None  # suppress when identical across all conditions
        return [name, unitlabel] + vals + [label]

    def format(self, ctx) -> list[str]:
        conditions = self.options.condition_table
        if not conditions:
            return []
        col_data, all_names = self._collect_condition_data(ctx.sol, conditions)
        cnames = list(conditions.keys())
        rows = [["Variable", "Units"] + cnames + ["Label"]]
        for name in all_names:
            row = self._build_row(name, cnames, col_data, self.options.precision)
            if row is not None:
                rows.append(row)
        if len(rows) == 1:  # header only
            return []
        lines = [self.title, "-" * len(self.title), ""]
        lines.extend(_format_aligned_columns(rows, None, "  "))
        return lines


SECTION_SPECS = {
    "cost": Cost,
    "warnings": Warnings,
    "solution": SolutionSection,
    "freevariables": FreeVariables,  # kept: sgp.py backward compat
    "constants": Constants,  # kept: public API + base class
    "sweeps": Sweeps,
    "tightest constraints": TightConstraints,
    "slack constraints": SlackConstraints,
    "condition_table": ConditionTable,
}


DIFF_SECTION_SPECS = {
    "cost": DiffCost,
    "freevariables": DiffFreeVariables,
    "constants": DiffConstants,
    "sensitivities": DiffSensitivities,
}


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class DiffContext:
    """Adapter that provides (key, DiffRow) items.

    The numbers come from ir.diff_solutions over the two solutions' IR, and the
    live keys come from the solutions themselves -- refs to compare with, objects
    to name and label with.  A sequence is N one-to-one diffs with their ratios
    stacked, which enables sweep comparisons.
    """

    new: Any  # SolutionContext or SequenceContext
    baseline: Any  # Solution-like

    @property
    def _scenarios(self) -> list:
        "The solutions on the new side: one, or a sweep's many."
        return list(getattr(self.new, "sols", None) or [self.new.sol])

    def diff_items(self, source, section: str) -> Iterable[Item]:
        "Items are (key, DiffRow), one per key the new side shows."
        baseline_ir = self.baseline.to_ir()
        diffs = [diff_solutions(baseline_ir, s.to_ir()) for s in self._scenarios]
        rels = [_rel_by_scenario_ref(d, section) for d in diffs]

        new_items = list(self.new.items(source))
        old_items = dict(SolutionContext(self.baseline).items(source))
        varset = self.baseline.primal.varset
        # A solution has exactly one cost, so its value pairs by position -- keying
        # on the expression would fail the moment the objective is rewritten, which
        # a scenario may legitimately do, and nothing can be "dropped" either.
        is_cost = section == "cost"
        out = []
        for key, value in new_items:
            rel = _stack_rel(rels, key, varset, scalar=len(diffs) == 1)
            old_key, old = _baseline_entry(old_items, key, positional=is_cost)
            out.append((key, DiffRow(rel, value, old, _display_units(old_key))))
        if is_cost:
            return out
        shown = {key for key, _ in new_items}
        for key, value in old_items.items():  # dropped, not merely unchanged
            if key not in shown:
                out.append(
                    (key, DiffRow(float("nan"), None, value, _display_units(key)))
                )
        return out


def _display_units(key) -> str:
    "The units to print a baseline value in -- its own, not the scenario's."
    return "" if key is None else unitstr(key, into="%s", dimless="")


def _baseline_entry(old_items: dict, key, positional: bool):
    "The baseline's (key, value) for a scenario key, or (None, None) if it has none."
    if positional:
        return next(iter(old_items.items()), (None, None))
    return (key, old_items[key]) if key in old_items else (None, None)


def _rel_by_scenario_ref(diff: dict, section: str) -> dict:
    """One diff's comparisons, keyed by the ref the scenario uses.

    diff_solutions keys by the baseline's ref; the table walks the new side, so a
    redeclared unit has to be followed to the scenario's spelling.
    """
    if section == "cost":  # one scalar, under the name _stack_rel looks it up by
        return {"cost": diff["cost"].get("rel", float("nan"))}
    part = diff.get(section, {})
    if section == "sensitivities":
        part = part.get("variables", {})
    redeclared = part.get("units_redeclared", {})
    return {
        redeclared.get(ref, ref): value
        for ref, value in part.get("matched", {}).items()
    }


def _stack_rel(rels: list[dict], key, varset, scalar: bool):
    """This key's comparison, across however many scenarios there are.

    A vector shows one row per parent, so its elements' numbers are gathered back
    into the parent's shape -- the IR carries elements only, by design.
    """
    nan = float("nan")

    def one(rel: dict):
        # Only the cost's key is an expression rather than a VarKey, so it alone
        # has no ref; the diff files its comparison under "cost".
        ref = getattr(key, "ref", "cost")
        elements = varset.by_vec(key) if is_veckey(key) else None
        if elements is None or not elements.size:
            return rel.get(ref, nan)
        flat = [rel.get(getattr(vk, "ref", None), nan) for vk in elements.flat]
        return np.array(flat).reshape(elements.shape)

    stacked = [one(rel) for rel in rels]
    return stacked[0] if scalar else np.array(stacked)


def table(
    obj: Any,  # Solution or SolutionSequence
    tables: tuple[str, ...] = (
        "sweeps",
        "cost",
        "warnings",
        "solution",
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
    if not rows:
        return []
    (ncols,) = {len(r) for r in rows} or (0,)
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
    if np.isnan(val).all():
        return np.nan
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
