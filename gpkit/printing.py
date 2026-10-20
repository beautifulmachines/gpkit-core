"printing functionality for gpkit objects"

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np

from .constraints.set import keyed_constraints
from .display.naming import decorated
from .ir import diff_solutions
from .units import Quantity
from .util.repr_conventions import lineagestr, unitstr
from .util.small_scripts import try_str_without

Item = tuple[Any, Any]


@dataclass(frozen=True, slots=True)
class PrintOptions:
    "container for printing options"

    fmt: str = "text"  # or "md"
    empty: str | None = None  # output (e.g. "(none)") for empty sections
    precision: int = 4
    topn: int | None = None  # truncation per-group
    vecn: int = 6  # max vector elements to print before ...
    vec_width: int | None = None  # None -> auto-align elements when applicable
    condition_table: Any = None  # dict[str, Model] for ConditionTable section


class Display:
    """How a table shows what a solution document names only by ref.

    The documents carry the numbers; this carries what a reader sees for each
    ref -- name, units, label, model heading -- and each constraint's text.
    Variables it reads from entries in the model IR's `variables` format, so a
    table with a model and one without render them through the same code.
    """

    def __init__(self, entries, model=None):
        self.entries = entries
        self.has_model = model is not None
        # a solution holds the cost's value; only the model knows its expression
        self.cost_name = "cost" if model is None else model.cost.str_without("units")
        self._constraints = (
            {} if model is None else {k.ref: c for k, c in keyed_constraints(model)}
        )

    def name(self, ref) -> str:
        entry = self.entries[ref]
        return decorated(entry["name"], entry.get("idx"), entry.get("shape"))

    def units(self, ref) -> str:
        return _units(self.entries[ref].get("units", ""))

    def label(self, ref) -> str:
        return self.entries[ref].get("label", "")

    def heading(self, ref) -> str:
        "The model a row is shown under; none for what no model owns."
        if ref in self.entries:
            return lineagestr(tuple(map(tuple, self.entries[ref].get("lineage", ()))))
        constraint = self._constraints.get(ref)
        return constraint.lineagestr() if hasattr(constraint, "lineagestr") else ""

    def has_constraint(self, ref) -> bool:
        return ref in self._constraints

    def constraint(self, ref) -> str:
        text = try_str_without(self._constraints[ref], {"units", "lineage"})
        if " at 0x" in text:  # a custom constraint's default repr
            text = text[: text.find(" at 0x")] + ">"
        return text


def _primal(doc) -> dict:
    return {ref: entry["value"] for ref, entry in doc["primal"].items()}


def _constants(doc) -> dict:
    return {ref: entry["value"] for ref, entry in doc.get("constants", {}).items()}


def _variable_senss(doc) -> dict:
    return doc["sensitivities"]["variables"]


def _constraint_senss(doc) -> dict:
    return doc["sensitivities"]["constraints"]


@cache
def _units(spelling: str) -> str:
    "Units for display, from any spelling a document carries of them."
    return unitstr(Quantity(1, spelling), dimless="")


def _cost_units(doc) -> str:
    "The cost's units for display, from the document's own spelling of them."
    return _units(doc["cost"].get("units", ""))


def _variable_entries(solutions) -> dict:
    """Entries for the keys solutions hold and their vectors, as the model IR has.

    VarKey.to_ir is what the model IR's `variables` is built from, so a table
    without a model names its rows exactly as one with a model does.
    """
    keys = {}
    for sol in solutions:
        for vmap in (sol.primal, sol.constants, sol.sens.variables):
            for vk in vmap:
                keys[vk.ref] = vk
                if vk.veckey is not None:
                    keys[vk.veckey.ref] = vk.veckey
    return {ref: vk.to_ir() for ref, vk in keys.items()}


@dataclass(frozen=True, slots=True)
class DiffRow:
    """One row of a diff: the comparison, plus both values for context.

    The comparison is read from ir.diff_solutions rather than recomputed here, so
    the table and sol.almost_equal agree by construction.  rel is nan where no
    ratio exists -- a quantity only one side has, or one that grew away from zero.
    Each value carries its own units, which need not be the other's.
    """

    rel: Any
    new: Any
    old: Any
    units: str = ""
    old_units: str = ""

    @property
    def no_ratio(self) -> bool:
        return not np.shape(self.rel) and np.isnan(self.rel)


class SectionSpec:
    align = None
    align_vecs = True  # aligns vectors if all same length
    col_sep = " "
    columns: tuple[tuple[str, str], ...] = ()  # (header, text form) for each cell
    filterfun = None
    filter_reduce = staticmethod(any)
    group_by_model = True
    pm = ""  # sign format prefix (e.g. '+' for sensitivities)
    sortkey = None
    source = None
    title: str = "Untitled Section"

    def __init__(self, options: PrintOptions, display: Display):
        self.options = options
        self.display = display

    def items_from(self, ctx):
        "Return iterable of items given the Documents. Item defs are section-specific"
        if self.source is None:
            raise NotImplementedError
        return ctx.items(self.source)

    def cells(self, item) -> list[str]:
        "item's row, one cell per column; each format lays them out its own way"
        raise NotImplementedError

    def row_from(self, item):
        "The text row: each cell in its column's text form, an empty one left empty."
        cells = zip(self.columns, self.cells(item))
        return [form.format(cell) if cell else "" for (_, form), cell in cells]

    def _auto_vecwidth_rowspec(self, items):
        "Return a copy of this with vec_width set automatically for items"
        if self.align_vecs and self.options.vec_width is None:
            shapes = (np.shape(self._width_array(v)) for _, v in items)
            lengths = {shape for shape in shapes if shape}
            if len(lengths) == 1:
                width = self._max_val_width(items)
                newopt = replace(self.options, vec_width=width)
                return self.__class__(options=newopt, display=self.display)
        return self

    def _groups(self, ctx):
        "(heading, items) for each model, in heading order, the items sorted."
        items = [item for item in self.items_from(ctx) if self._passes_filter(item)]
        bymod = self._by_heading(items) if self.group_by_model else {"": items}
        for heading, model_items in sorted(bymod.items()):
            if self.sortkey:
                model_items.sort(key=self.sortkey)
            yield heading, model_items

    def markdown(self, ctx) -> list[str]:
        "This section in markdown: a pipe table under each model's heading."
        lines = []
        for heading, items in self._groups(ctx):
            if heading:
                lines += [f"**{_md_escape(heading)}**", ""]
            rows = [[_md_cell(cell) for cell in self.cells(i)] for i in items]
            lines += [*_md_table([h for h, _ in self.columns], rows), ""]
        return [f"### {self.title}", "", *lines[:-1]] if lines else []

    def format(self, ctx) -> list[str]:
        "Output this section's lines given the Documents"
        lines = []
        for modelname, model_items in self._groups(ctx):
            model_lines = self._format_model_group(model_items)
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

    def _format_model_group(self, model_items) -> list[str]:
        "One model's rows, aligned in columns."
        rowspec = self._auto_vecwidth_rowspec(model_items)
        rows = [rowspec.row_from(item) for item in model_items]
        return _format_aligned_columns(rows, self.align, self.col_sep)

    def _by_heading(self, items):
        "items grouped under the model each is shown in"
        out = defaultdict(list)
        for ref, val in items:
            out[self.display.heading(ref)].append((ref, val))
        return out

    def _name(self, ref) -> str:
        return self.display.name(ref)

    def _unitstr(self, ref) -> str:
        units = self.display.units(ref)
        return f"[{units}]" if units else ""

    def _fmt_one(self, x, p, suff="") -> str:
        "Format a single scalar element for vector display."
        return f"{x:{self.pm}.{p - 1}g}{suff}".replace("+nan", "nan")

    def _fmt_val(self, val, suff="", one=None) -> str:
        "val, a vector's elements each formatted by one, if given"
        n = self.options.vecn
        p = self.options.precision
        w = self.options.vec_width or 0
        if np.shape(val):
            one = one or (lambda x: self._fmt_one(x, p, suff))
            flat = np.asarray(val).ravel()
            shown = flat[:n]
            body = "  ".join(one(x).ljust(w) for x in shown)
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


VARIABLE, VALUE, UNITS = ("Variable", "{} :"), ("Value", "{}"), ("Units", "[{}]")
SENSITIVITY, LABEL = ("Sensitivity", "({})"), ("Label", "{}")


class Cost(SectionSpec):
    title = "Optimal Cost"
    columns = (("Cost", "{} :"), VALUE, UNITS)
    group_by_model = False  # the whole model's, though its name may match a ref

    def items_from(self, ctx):
        return [(self.display.cost_name, (ctx.cost(), ctx.cost_units()))]

    def cells(self, item):
        name, (val, units) = item
        return [name, self._fmt_val(val), units]

    def _width_array(self, v):
        return v[0]


class Warnings(SectionSpec):
    title = "WARNINGS"
    columns = (("Warning", "{}"),)
    align_vecs = False

    def cells(self, item):
        warning_type, warning_list = item
        return [f"{warning_type}:\n" + "\n".join(warning_list)]

    def items_from(self, ctx):
        return ctx.warning_items()


class FreeVariables(SectionSpec):
    title = "Free Variables"
    columns = (VARIABLE, VALUE, UNITS, LABEL)
    align = "><<<"
    source = staticmethod(_primal)

    def sortkey(self, item):
        return self._name(item[0])

    def cells(self, item):
        ref, val = item
        d = self.display
        return [d.name(ref), self._fmt_val(val), d.units(ref), d.label(ref)]


class Constants(SectionSpec):
    title = "Fixed Variables"
    columns = (VARIABLE, VALUE, UNITS, SENSITIVITY, LABEL)
    align = "><<<<"  # name(R), value(L), unit(L), sens-in-parens(L), label(L)
    nearzero_tol = 1e-7
    _fmt_unit_w = 0  # overridden per-group by SolutionSection._format_model_group
    _fmt_col_widths = (
        None  # overridden per-group by SolutionSection._format_model_group
    )

    def sortkey(self, item):
        return (-rounded_mag(item[1][1]), self._name(item[0]))

    def items_from(self, ctx):
        """Yield (ref, (value, sensitivity)) for each fixed variable."""
        sens_dict = dict(ctx.items(_variable_senss))
        for ref, val in ctx.items(_constants):
            yield (ref, (val, sens_dict.get(ref, 0.0)))

    def _fmt_one_sens(self, x, p) -> str:
        """Format a single sensitivity element: ~0 if near-zero, else +x."""
        if abs(x) < self.nearzero_tol:
            return "~0"
        return f"{x:+.{p - 1}g}".replace("+nan", "nan")

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

    def _fmt_vector_item(self, ref, val, sens, name_w) -> list[str]:
        """Format a vector constant as two lines: values then sensitivities below."""
        flat_val = np.asarray(val).ravel()
        flat_sens = (
            np.asarray(sens).ravel()
            if np.shape(sens)
            else np.full(flat_val.shape, float(sens))
        )
        val_vec, sens_vec = self._fmt_vec_pair(flat_val, flat_sens)
        name_col = f"{self._name(ref):>{name_w}} :"
        unit = self._unitstr(ref).ljust(self._fmt_unit_w)
        parts = [name_col, val_vec, unit, self.display.label(ref)]
        line1 = self.col_sep.join(x for x in parts if x).rstrip()
        sens_col = f"{'sens':>{name_w + 2}}"
        return [line1, f"{sens_col}{self.col_sep}{sens_vec}"]

    def _format_model_group(self, model_items) -> list[str]:
        """Render one model group: scalars column-aligned, vectors as two lines."""
        scalar_items = [(k, v) for k, v in model_items if not np.shape(v[0])]
        vector_items = [(k, v) for k, v in model_items if np.shape(v[0])]
        # one name width for the group, so every row's colon lines up
        name_w = max(len(self._name(k)) for k, _ in model_items)
        if vector_items:
            name_w = max(name_w, len("sens") - 2)
        rows = [self.row_from(item) for item in scalar_items]
        for row in rows:
            row[0] = row[0].rjust(name_w + 2)
        lines = _format_aligned_columns(rows, self.align, self.col_sep)
        for ref, (val, sens) in vector_items:
            lines.extend(self._fmt_vector_item(ref, val, sens, name_w))
        return lines

    def cells(self, item):
        "A vector's sensitivities share a cell, as its values do."
        ref, (val, sens) = item
        p = self.options.precision
        if sens is None:  # a free variable, in SolutionSection
            senstr = ""
        elif np.shape(val):
            sens = np.broadcast_to(sens, np.shape(val))
            senstr = self._fmt_val(sens, one=lambda x: self._fmt_one_sens(x, p))
        else:
            senstr = self._fmt_one_sens(sens, p + 1)
        d = self.display
        return [d.name(ref), self._fmt_val(val), d.units(ref), senstr, d.label(ref)]


class SolutionSection(Constants):
    """Combined free + fixed variables, grouped by submodel.

    Inherits Constants to reuse all vector formatting logic; extends it to
    also yield free variables (with a None sensitivity sentinel) so that a
    single aligned table can display both optimized outputs and fixed inputs.
    """

    title = "Solution"

    def sortkey(self, item):
        "free variables by name, then fixed ones by sensitivity"
        ref, (_, sens) = item
        return (0, self._name(ref)) if sens is None else (1, *super().sortkey(item))

    def items_from(self, ctx):
        """Yield (ref, (value, sensitivity)) for all variables.

        Free variables use None as a sentinel for the sensitivity field.
        """
        for ref, val in ctx.items(_primal):
            yield (ref, (val, None))
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

        def part(free, vector):  # in sortkey's order, as the items arrive
            return [
                (k, v)
                for k, v in model_items
                if (v[1] is None) == free and bool(np.shape(v[0])) == vector
            ]

        scalar_free, scalar_fixed = part(True, False), part(False, False)
        vec_free, vec_fixed = part(True, True), part(False, True)
        name_w = max((len(self._name(k)) for k, _ in model_items), default=0)
        vw = self._max_val_width([(k, v[0]) for k, v in vec_free])
        # Instance attrs communicate formatting to _fmt_vector_item/_fmt_vec_pair.
        self._fmt_unit_w = max(
            (len(self._unitstr(k)) for k, _ in vec_free + vec_fixed), default=0
        )
        # Run all scalars together for shared column widths, split at the boundary.
        rows = [self.row_from(item) for item in scalar_free + scalar_fixed]
        for row in rows:
            row[0] = row[0].rjust(name_w + 2)
        aligned = _format_aligned_columns(rows, self.align, self.col_sep)
        lines = list(aligned[: len(scalar_free)])
        for ref, (val, _) in vec_free:
            lines.append(
                self.col_sep.join(
                    x
                    for x in [
                        f"{self._name(ref):>{name_w}} :",
                        self._fmt_free_vec(val, vw),
                        self._unitstr(ref).ljust(self._fmt_unit_w),
                        self.display.label(ref),
                    ]
                    if x
                ).rstrip()
            )
        if scalar_fixed or vec_fixed:
            lines.append("...constants")
        lines.extend(aligned[len(scalar_free) :])
        self._fmt_col_widths = self._vec_col_widths(vec_fixed)
        for ref, (val, sens) in vec_fixed:
            lines.extend(super()._fmt_vector_item(ref, val, sens, name_w))
        return lines


class Sweeps(FreeVariables):
    title = "Swept Variables"

    def items_from(self, ctx):
        return ctx.sweep_items()


class Constraints(SectionSpec):
    columns = (("Sensitivity", "{}"), ("Constraint", "{}"))
    col_sep = " : "
    pm = "+"

    def sortkey(self, item):
        return (-rounded_mag(item[1]), self.display.constraint(item[0]))

    def items_from(self, ctx):
        "Each constraint with its sensitivity: the model has the constraints."
        if not self.display.has_model:
            raise ValueError(
                f"'{self.title}' shows constraints, which a solution holds only by"
                " key; render it with the model, as model.table(sol)."
            )
        return ctx.items(self._modeled)

    def _modeled(self, doc) -> dict:
        "The sensitivities of the constraints the model has."
        senss = _constraint_senss(doc)
        return {ref: s for ref, s in senss.items() if self.display.has_constraint(ref)}

    def cells(self, item):
        ref, sens = item
        return [self._fmt_val(sens), self.display.constraint(ref)]


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
    # an unchanged quantity says nothing; a missing ratio is itself worth a row
    filterfun = staticmethod(lambda kv: _rel_magnitude(kv[1]) != 0)

    def sortkey(self, item):
        return (-rounded_mag(_rel_magnitude(item[1])), self._name(item[0]))

    def items_from(self, ctx):
        return ctx.diff_items(self.source, self.diff_section)

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

    def _bothstr(self, row) -> str:
        "Both values, each with its own units, which need not be the same ones."
        if row.new is None or row.old is None or np.shape(row.rel):
            return ""
        return f"({row.new:.4g}{row.units} vs {row.old:.4g}{row.old_units})"

    def _relstr(self, row) -> str:
        parts = (self._pctstr(row), self._bothstr(row))
        return "  ".join(p for p in parts if p)


CHANGE = ("Change", "{}")


class DiffCost(DiffSection):
    title = "Cost Change"
    columns = (Cost.columns[0], CHANGE, ("Values", "{}"))
    filterfun = None  # the headline number, reported even when it did not move
    sortkey = None  # one row, named by no ref
    group_by_model = False
    pm = "+"

    def items_from(self, ctx):
        return [(self.display.cost_name, ctx.cost_diff())]

    def cells(self, item):
        name, row = item
        return [name, self._pctstr(row), self._bothstr(row)]


class DiffFreeVariables(DiffSection):
    title = "Free Variable Changes"
    columns = (VARIABLE, CHANGE, LABEL)
    source = staticmethod(_primal)
    diff_section = "primal"
    pm = "+"
    align = "><<"

    def cells(self, item):
        ref, row = item
        return [self._name(ref), self._relstr(row), self.display.label(ref)]


class DiffConstants(DiffFreeVariables):
    title = "Constant Changes"
    source = staticmethod(_constants)
    diff_section = "constants"


class DiffSensitivities(DiffSection):
    """Sensitivities moved by a difference, not a ratio.

    They are already log derivatives, so a percentage of one would be a
    percentage of a derivative -- the swing itself is the readable number.
    """

    title = "Sensitivity Changes"
    columns = (VARIABLE, CHANGE, LABEL)
    source = staticmethod(_variable_senss)
    diff_section = "sensitivities"
    pm = "+"
    align = "><<"

    def cells(self, item):
        ref, row = item
        if row.new is None or row.old is None:
            diffstr = self._pctstr(row)
        else:
            diffstr = self._fmt_val(row.rel)
            if not np.shape(row.rel):
                diffstr += f"  ({row.new:+.4g} vs {row.old:+.4g})"
        return [self._name(ref), diffstr, self.display.label(ref)]


class ConditionTable(SectionSpec):
    """Side-by-side comparison of named operating conditions within a model.

    Rows are variables; columns are condition names.  A dash is shown when a
    variable is absent in a condition.  Variables whose values are identical
    across every condition they appear in are suppressed (they add no signal).

    Pass via sol.table(tables=["condition_table"], condition_table={...}).
    """

    title = "Condition Comparison"

    def _collect_condition_data(self, doc, conditions):
        values = {**_primal(doc), **_constants(doc)}
        col_data: dict[str, dict[str, tuple[float, str, str]]] = {}
        all_names: list[str] = []
        seen_names: set[str] = set()
        for cname, model in conditions.items():
            col_data[cname] = {}
            refs = [vk.ref for vk in model.vks if vk.ref in values]
            for ref in sorted(refs, key=lambda r: self.display.entries[r]["name"]):
                name = self.display.entries[ref]["name"]
                entry = (values[ref], self._unitstr(ref), self.display.label(ref))
                col_data[cname][name] = entry
                if name not in seen_names:
                    all_names.append(name)
                    seen_names.add(name)
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

    def _rows(self, ctx) -> list[list[str]]:
        "The header, then a row per variable; none if no variable differs."
        conditions = self.options.condition_table
        if not conditions:
            return []
        (doc,) = ctx.docs
        col_data, all_names = self._collect_condition_data(doc, conditions)
        cnames = list(conditions.keys())
        rows = [["Variable", "Units"] + cnames + ["Label"]]
        for name in all_names:
            row = self._build_row(name, cnames, col_data, self.options.precision)
            if row is not None:
                rows.append(row)
        return rows if len(rows) > 1 else []

    def format(self, ctx) -> list[str]:
        if not (rows := self._rows(ctx)):
            return []
        lines = [self.title, "-" * len(self.title), ""]
        lines.extend(_format_aligned_columns(rows, None, "  "))
        return lines

    def markdown(self, ctx) -> list[str]:
        if not (rows := self._rows(ctx)):
            return []
        header, *rows = rows
        cells = [[_md_cell(cell) for cell in row] for row in rows]
        return [f"### {self.title}", "", *_md_table(header, cells)]


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
class Documents:
    """The solution IR documents a table reads its numbers from.

    One document, or a sweep's many: a single solution shows each vector as one
    row, while a sweep already spends its columns on solutions, so it shows each
    element as a row of its own.  A diff adds the baseline it compares against.
    """

    docs: list[dict]
    display: Display
    is_sweep: bool
    swept: tuple[str, ...] = ()  # refs of what a sweep varied
    baseline: dict | None = None
    tol: float = 0.0  # a comparison within it is no change, as almost_equal says

    def items(self, source) -> list[Item]:
        "(ref, value) for each row of source, which reads one document."
        return self._rows([source(doc) for doc in self.docs])

    def cost(self):
        return self._across([doc["cost"]["value"] for doc in self.docs])

    def cost_units(self) -> str:
        return _cost_units(self.docs[0])

    def sweep_items(self) -> list[Item]:
        "Each swept variable's values, which the solutions hold as constants."
        return self.items(lambda doc: {ref: _constants(doc)[ref] for ref in self.swept})

    def warning_items(self) -> list[tuple[str, list[str]]]:
        "Warning messages by category; a sweep's say how many solutions had each."
        if not self.is_sweep:
            (doc,) = self.docs
            return list(_warning_messages(doc).items())
        counts = defaultdict(Counter)
        n = len(self.docs)
        for doc in self.docs:
            for name, messages in _warning_messages(doc).items():
                counts[name].update(messages)
        return [
            (name, [f"{w}  (in {c} of {n} solutions)" for w, c in cnt.items()])
            for name, cnt in counts.items()
        ]

    def _rows(self, per_doc: list[dict]) -> list[Item]:
        if not self.is_sweep:
            (values,) = per_doc
            return list(self._grouped(values).items())
        return _stack(per_doc)

    def _grouped(self, values: dict) -> dict:
        """values as one document's rows, a vector's elements gathered into one.

        The IR carries elements only, so showing a vector as one row is this
        step; an element with no value is nan.  A sweep's columns are already
        its solutions, so there each element keeps a row of its own.
        """
        if self.is_sweep:
            return dict(values)
        out = {}
        for ref, value in values.items():
            entry = self.display.entries.get(ref, {})  # none for a constraint
            if "veckey_ref" not in entry:
                out[ref] = value
                continue
            row = out.setdefault(entry["veckey_ref"], np.full(entry["shape"], np.nan))
            row[tuple(entry["idx"])] = value
        return out

    def _across(self, values: list):
        "One value per document, as a sweep's array or a single solution's own."
        return np.array(values) if self.is_sweep else values[0]

    def _moved(self, rel: float) -> float:
        "rel, or 0 where it is within tol: a solver returns no number exactly."
        return 0.0 if abs(rel) <= self.tol else rel

    def cost_diff(self) -> DiffRow:
        "The cost's comparison, which pairs by position: a solution has one cost."
        rels = [diff_solutions(self.baseline, doc)["cost"] for doc in self.docs]
        return DiffRow(
            rel=self._across([self._moved(r.get("rel", float("nan"))) for r in rels]),
            new=self.cost(),
            old=self.baseline["cost"]["value"],
            units=self.cost_units(),
            old_units=_cost_units(self.baseline),
        )

    def diff_items(self, source, section: str) -> list[Item]:
        """(ref, DiffRow), one per row of either side, paired as the diff paired them.

        diff_solutions matches a redeclared unit's two refs; renaming the
        baseline's to the scenario's before grouping pairs their rows too.
        """
        diffs = [diff_solutions(self.baseline, doc) for doc in self.docs]
        parts = [_diff_part(diff, section) for diff in diffs]
        renamed = {
            o: n for p in parts for o, n in p.get("units_redeclared", {}).items()
        }
        spelled = {}  # each row's ref, to the baseline's ref for that row
        for old, new in renamed.items():
            spelled[new] = old
            if vec := self.display.entries[new].get("veckey_ref"):
                spelled[vec] = self.display.entries[old]["veckey_ref"]

        def rows(values):
            return self._grouped({renamed.get(r, r): v for r, v in values.items()})

        def row(ref, rel, new, old):
            units = self.display.units  # the baseline's own, not the scenario's
            return ref, DiffRow(rel, new, old, units(ref), units(spelled.get(ref, ref)))

        rels = [
            rows({r: self._moved(v) for r, v in p.get("matched", {}).items()})
            for p in parts
        ]
        old_rows = rows(source(self.baseline))
        nan = float("nan")
        out = [
            row(
                ref,
                self._across([r.get(ref, nan) for r in rels]),
                value,
                old_rows.pop(ref, None),
            )
            for ref, value in self.items(source)
        ]  # then what was dropped, not merely unchanged
        return out + [row(ref, nan, None, old) for ref, old in old_rows.items()]


def _warning_messages(doc) -> dict:
    "a solution document's warning messages, by category"
    warns = doc["meta"].get("warnings", {})
    return {name: [w["message"] for w in ws] for name, ws in warns.items() if ws}


def _stack(per_doc: list[dict]) -> list[Item]:
    """Strict stacking: keys (and their order) must match across all documents."""
    if not per_doc:
        return []
    keys0 = tuple(per_doc[0])
    for values in per_doc[1:]:
        if tuple(values) != keys0:
            raise ValueError("Sweep key mismatch")
    return [(k, np.array([values[k] for values in per_doc])) for k in keys0]


def _diff_part(diff: dict, section: str) -> dict:
    "The part of a diff_solutions output that compares one section."
    part = diff.get(section, {})
    return part.get("variables", {}) if section == "sensitivities" else part


def _documents(obj, model=None, baseline=None, tol=0.0) -> Documents:
    "The documents and display for a Solution or Sweep, against a baseline if any."
    is_sweep = not _looks_like_solution(obj)
    sols = list(obj) if is_sweep else [obj]
    # model.vks.to_ir() is the model IR's `variables`, without serializing every
    # constraint to get it -- which a custom constraint may not support
    entries = _variable_entries(sols) if model is None else model.vks.to_ir()
    if baseline is not None:  # which may be a solution of another model
        entries = {**_variable_entries([baseline]), **entries}
    display = Display(entries, model)
    return Documents(
        docs=[sol.to_ir() for sol in sols],
        display=display,
        is_sweep=is_sweep,
        swept=tuple(vk.ref for vk in getattr(obj, "swept", ())),
        baseline=None if baseline is None else baseline.to_ir(),
        tol=tol,
    )


def table(
    obj: Any,  # Solution or Sweep
    tables: tuple[str, ...] | None = None,
    *,
    model=None,
    **options,
) -> str:
    """Render a table for a Solution or Sweep, as text or as markdown ("md").

    By default it shows what obj has to show: a sweep's swept variables, and
    the constraints only when given the model, since a solution holds those
    only by key.
    """
    ctx = _documents(obj, model)
    if tables is None:
        tables = (
            *(("sweeps",) if ctx.is_sweep else ()),
            "cost",
            "warnings",
            "solution",
            *(("tightest constraints",) if model is not None else ()),
        )
    return _render(SECTION_SPECS, tables, ctx, PrintOptions(**options))


def diff(
    obj, baseline, tables=("cost", "freevariables"), *, model=None, tol=1e-6, **options
):
    """Render tables of differences between obj and baseline.

    The cost is named by its expression when given the model, else "cost".  A
    number moved only beyond tol -- relative for values, absolute for
    sensitivities -- which is what sol.almost_equal(baseline, tol) asks.
    """
    ctx = _documents(obj, model, baseline, tol)
    return _render(DIFF_SECTION_SPECS, tables, ctx, PrintOptions(**options))


def _render(specs, tables, ctx, opt) -> str:
    "The named sections, as text or markdown, a blank line between each."
    if opt.fmt not in ("text", "md"):
        raise ValueError(f"fmt is 'text' or 'md', not {opt.fmt!r}")
    blocks = []
    for name in tables:
        sec = specs[name](options=opt, display=ctx.display)
        lines = sec.format(ctx) if opt.fmt == "text" else sec.markdown(ctx)
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


def _md_escape(text: str) -> str:
    r"""Escape characters that have special meaning in markdown pipe tables."""
    for ch in ("\\", "|", "*", "_", "`", "~", "[", "]", "<", ">", "%", "^"):
        text = text.replace(ch, "\\" + ch)
    return text


def _md_cell(text: str) -> str:
    "text escaped for a pipe table, where a cell is one line"
    return _md_escape(text).replace("\n", "<br>")


def _md_table(header, rows) -> list[str]:
    "Pipe-table lines for cells already in markdown."

    def line(cells):
        return "| " + " | ".join(cells) + " |"

    return [line(header), line("-" * max(len(h), 3) for h in header), *map(line, rows)]


def rounded_mag(val, nround=8):
    "get the magnitude of a (vector or scalar) for stable sorting purposes"
    if np.isnan(val).all():
        return np.nan
    return round(np.nanmax(np.absolute(val)), nround)
