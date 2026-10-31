"""A view of the model tree: what each model holds, as a reader sees it.

model_view(model, solution=None) builds one ModelView per model in the tree:
its description, objective, free and fixed variable rows, and constraint
groups, with values and sensitivities when a solution is given.  Every output
format renders from it -- the report documents in display.report, and the dict
consumers read.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constraints.set import (
    constraint_varkeys,
    own_keyed_constraints,
    shown_as,
    solver_constraints,
)
from .display import DisplayScope
from .display.render import LATEX, TEXT, render
from .display.spelling import unitstr
from .nomials.constraints import ArrayConstraint
from .util.small_classes import Quantity
from .varinfo import VarKind, classify_variables
from .varmap import VarSet


@dataclass
class VarEntry:
    """A variable row in a model's view.

    `key` names the variable -- the veckey for a vector -- and is what joins a
    row to a solution or to to_ir()'s variables. The rest are renderings of it
    against this view's DisplayScope, display-layer fields that #366 may move
    out; prefer `key` over `name` for anything but display.
    """

    key: "VarKey"  # noqa: F821
    name: str  # display name, relative to this section's scope
    latex: str  # LaTeX rendering (from DisplayScope.latex)
    value: Any  # float | ndarray | None — never a pint Quantity
    sensitivity: float | None  # shadow price from solution, or None
    units: str  # unit string
    label: str  # human label (from VarKey.label or "")
    source: str = ""  # lineagestr() for cross-model referenced vars; "" for local

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return {
            "ref": self.key.ref,
            "name": self.name,
            "latex": self.latex,
            "value": _serialize_value(self.value),
            "sensitivity": self.sensitivity,
            "units": self.units,
            "label": self.label,
            "source": self.source,
        }


@dataclass
class CGroup:
    """A named constraint group within a model's view.

    keys name each constraint the way to_ir() and a solve's per-constraint
    sensitivities name it, so a consumer can join the three -- and the way a
    second build of the same model will name it, so two solves can be compared.
    """

    label: str  # "" for unnamed groups
    constraints: list  # raw constraint objects, shown through render()
    keys: list  # one ConstraintKey per constraint, same order


@dataclass
class ModelView:
    """One model in the tree, as a reader sees it; children are its submodels'.

    scope names variables the way this section shows them: anchored at the
    model, resolved against the variables the section displays. It is a
    rendering concern, not semantic data, and is excluded from to_dict().
    """

    title: str
    description: str
    assumptions: list  # list of str
    free_variables: list  # list of VarEntry — optimized by the solver
    fixed_variables: list  # list of VarEntry — prescribed constants
    constraint_groups: list  # list of CGroup
    lineage_path: str = ""  # dotted path e.g. "Aircraft.Wing"
    is_anonymous: bool = False  # True for bare Model(...) instances (no subclass name)
    children: list = field(default_factory=list)  # list of ModelView
    scope: DisplayScope = field(default_factory=DisplayScope)  # NOT in to_dict
    references: list = field(default_factory=list)  # list of str
    preamble: str = ""  # this model's own report_preamble(), rendered per section
    objective_str: str = ""  # text representation of cost expression; "" if constant
    objective_latex: str = ""  # LaTeX representation of cost expression
    objective_label: str = ""  # variable label when expr is a single variable; else ""
    objective_value: float | None = None  # magnitude of attained cost; None if unsolved
    objective_units: str = ""  # unit string for the cost expression
    objective_direction: str = "minimize"  # "minimize" or "maximize"

    def _render_constraint_groups(self) -> list:
        """Render constraint groups the same way the text path does."""
        scope = self.scope.also_excluding("units")
        return [
            {
                "label": cg.label,
                "constraints": [
                    _constraint_dict(c, key, scope)
                    for c, key in zip(cg.constraints, cg.keys, strict=True)
                ],
            }
            for cg in self.constraint_groups
        ]

    def to_dict(self) -> dict:
        """JSON-serializable dict (for format='dict' and future API)."""
        return {
            "title": self.title,
            "lineage_path": self.lineage_path,
            "objective_direction": self.objective_direction,
            "objective_label": self.objective_label,
            "is_anonymous": self.is_anonymous,
            "description": self.description,
            "assumptions": list(self.assumptions),
            "references": list(self.references),
            "preamble": self.preamble,
            "objective_str": self.objective_str,
            "objective_latex": self.objective_latex,
            "objective_value": self.objective_value,
            "objective_units": self.objective_units,
            "free_variables": [v.to_dict() for v in self.free_variables],
            "fixed_variables": [v.to_dict() for v in self.fixed_variables],
            "constraint_groups": self._render_constraint_groups(),
            "children": [c.to_dict() for c in self.children],
        }


# ── Value helpers ─────────────────────────────────────────────────────────────


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable.

    val must be a plain float, numpy array, or None — not a pint Quantity.
    """
    if val is None:
        return None
    if isinstance(val, Quantity):
        raise TypeError(
            f"_serialize_value received a pint Quantity ({val!r}); "
            "VarEntry.value must be a plain numeric"
        )
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


# ── Constraint rendering ──────────────────────────────────────────────────────


def _constraint_dict(c, key, scope) -> dict:
    """One constraint as JSON: how to print it, how to typeset it, what it is.

    str and latex name variables from the same scope, so the two forms of a
    constraint always agree.  key and type are what to_ir() records for the
    same constraint, so a consumer reads one name and one vocabulary from both.
    A vector constraint also names its elements, which are what a solution
    keys its sensitivities by.
    """
    out = {
        "str": render(c, TEXT, scope),
        "latex": render(c, LATEX, scope),
        "key": key.ref,
        "type": type(c).__name__,
    }
    if isinstance(c, ArrayConstraint):
        out["element_keys"] = [key.element(idx).ref for idx, _ in solver_constraints(c)]
    return out


# ── Core builder helpers ──────────────────────────────────────────────────────


def _entry_datum(values):
    "One VarInfo column as a VarEntry field: a scalar, an array, or None."
    if len(values) == 1:
        return values[0]
    if all(v is None for v in values):
        return None
    return np.array([np.nan if v is None else v for v in values])


def _make_var_entry(info, scope, source="") -> VarEntry:
    "Render a VarInfo against scope, keeping the key it was built from."
    return VarEntry(
        key=info.key,
        name=scope.text(info.key),
        latex=scope.latex(info.key),
        value=_entry_datum(info.values),
        sensitivity=_entry_datum(info.sensitivities),
        units=unitstr(info.key) or "-",
        label=info.key.label or "",
        source=source,
    )


def _build_split_var_entries(
    model, solution, scope, extra_vks=None
) -> tuple[list[VarEntry], list[VarEntry]]:
    """Build (free_entries, fixed_entries) from model.own_varkeys.

    free_entries  — Optimized Variables: solved by the optimizer (no sens shown).
    fixed_entries — Fixed Variables: prescribed constants with sensitivities.

    Names come from scope, so a variable this model owns shows only the context
    below it and one owned elsewhere shows its path relative to here. A vector is
    one entry, under its veckey.

    Cross-model variables (extra_vks) additionally store their absolute dotted
    name in VarEntry.source, for display in brackets.
    """
    free_entries: list[VarEntry] = []
    fixed_entries: list[VarEntry] = []

    def collect(vks):
        for info in classify_variables(vks, model.substitutions, solution):
            # Annotate exactly what the scope could not locate by name -- the
            # same keys legend() covers. A variable at or below this model shows
            # its path already, so a source column would only repeat it.
            source = "" if scope.owns(info.key) else info.key.lineagestr()
            entry = _make_var_entry(info, scope, source)
            if info.kind is VarKind.FREE:
                free_entries.append(entry)
            else:
                fixed_entries.append(entry)

    collect(model.own_varkeys)
    if extra_vks:
        owned = {(vk.veckey or vk) for vk in model.own_varkeys}
        collect(VarSet(vk for vk in extra_vks if (vk.veckey or vk) not in owned))

    return free_entries, fixed_entries


def _build_constraint_groups(model) -> list[CGroup]:
    """Build CGroup list from model.cgroups or a single unnamed group.

    CGroup.constraints holds the written (single-equation) constraints this
    model owns, in canonical order -- child models keep their own for their own
    sections, containers like Tight, Loose or SignomialEquality are walked
    through, and a vector constraint is one item (or its elements, if a side was
    never written as one expression: shown_as). Raw objects are stored; the
    renderers call gpkit.display.render on them.

    Keys come from own_keyed_constraints, the same call the IR names constraints
    with, so the two cannot name a constraint differently.
    """
    named = model.cgroups is not None
    groups = []
    for label, pairs in own_keyed_constraints(model):
        if pairs or named:  # an empty group is still a group if it was named
            rows = [row for key, c in pairs for row in shown_as(key, c)]
            groups.append(CGroup(label, [c for _, c in rows], [k for k, _ in rows]))
    return groups


def _build_objective(model, solution) -> dict:
    """Return objective keyword args for ModelView for the model's cost.

    All fields are empty/None when the cost has no variables (i.e. it is a
    trivial constant placeholder rather than a real optimization objective).
    Detects the 1/expr pattern and flips direction to "maximize".
    """
    if not model.cost.vks:
        return {
            "objective_str": "",
            "objective_latex": "",
            "objective_label": "",
            "objective_value": None,
            "objective_units": "",
            "objective_direction": "minimize",
        }
    obj = model.objective_info(solution=solution)
    excluded = {"units", "lineage"}
    return {
        "objective_str": render(obj.expr, TEXT, excluded),
        "objective_latex": render(obj.expr, LATEX, excluded),
        "objective_label": (obj.variable.label or "") if obj.variable else "",
        "objective_value": obj.value,
        "objective_units": obj.units,
        "objective_direction": obj.sense,
    }


# ── Standard text blocks ──────────────────────────────────────────────────────


def _model_stats(model) -> dict:
    """Compute model-wide counts for use in report text blocks.

    Returns a dict with keys: n_free, n_constraints, objective_str,
    objective_latex.  All values are derived from the model without requiring
    a solved solution.
    """
    n_free = len(model.vks) - len(model.substitutions)
    # flat() returns a generator (flatiter), so use sum() rather than len().
    n_constraints = sum(1 for _ in model.flat())
    if model.cost.vks:
        obj = model.objective_info()
        excluded = {"units", "lineage"}
        obj_latex = render(obj.expr, LATEX, excluded)
        obj_label = (obj.variable.label or "") if obj.variable else ""
        obj_direction = obj.sense
    else:
        obj_latex = None
        obj_label = ""
        obj_direction = "minimize"
    return {
        "n_free": n_free,
        "n_constraints": n_constraints,
        "objective_latex": obj_latex,
        "objective_label": obj_label or "",
        "objective_direction": obj_direction,
    }


# ── Core builder ─────────────────────────────────────────────────────────────


def model_view(
    model,
    solution=None,
    _parent_path: str = "",
) -> ModelView:
    """The ModelView tree of *model*.

    Parameters
    ----------
    model : Model
        Root model of the view.
    solution : Solution, optional
        If provided, variable entries include solved values and sensitivities.
    """
    is_anon = model.is_anonymous
    own_name = "" if is_anon else type(model).__name__
    lineage_path = model.lineagestr() if own_name else _parent_path
    cgroups = _build_constraint_groups(model)
    extra_vks = (
        constraint_varkeys(c for cg in cgroups for c in cg.constraints)
        - model.own_varkeys
    )
    # One set feeds both what the section shows and how much lineage each name
    # needs, so the two cannot disagree.
    # abbreviate: the variable tables below list every foreign key with its
    # owning model in VarEntry.source, so they are the legend short names need.
    scope = DisplayScope(
        anchor=model.lineage,
        shown=model.own_varkeys | extra_vks,
        abbreviate=True,
    )
    free_vars, fixed_vars = _build_split_var_entries(
        model, solution, scope, extra_vks=extra_vks
    )
    if is_anon:
        desc = {"summary": "", "assumptions": [], "references": []}
        preamble = ""
    else:
        desc = type(model).description()
        preamble = type(model).report_preamble()
    return ModelView(
        title=own_name or "Model",
        description=desc["summary"],
        assumptions=desc["assumptions"],
        references=desc["references"],
        lineage_path=lineage_path,
        is_anonymous=is_anon,
        free_variables=free_vars,
        fixed_variables=fixed_vars,
        constraint_groups=cgroups,
        scope=scope,
        preamble=preamble,
        **_build_objective(model, solution),
        children=[
            model_view(child, solution=solution, _parent_path=lineage_path)
            for child in model.submodels
        ],
    )
