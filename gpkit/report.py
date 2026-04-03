"""Report infrastructure: format-independent IR and rendering backends.

ReportSection is the single tree that all output formats render from.
build_report_ir() traverses the model tree once; renderers are pure
functions of the IR.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class VarEntry:
    """A variable row in a report section."""

    name: str  # display name (from VarKey.name)
    latex: str  # LaTeX rendering (from VarKey.latex())
    value: Any  # float | Quantity | None (resolved per D-17 priority)
    sensitivity: Optional[float]  # shadow price from solution, or None
    units: str  # unit string
    label: str  # human label (from VarKey.label or "")


@dataclass
class CGroup:
    """A named constraint group within a report section."""

    label: str  # "" for unnamed groups
    constraints: list  # list of constraint string representations


@dataclass
class ReportSection:
    """Format-independent intermediate representation for model reports."""

    title: str
    description: str
    assumptions: list  # list of str
    variables: list  # list of VarEntry
    constraint_groups: list  # list of CGroup
    children: list = field(default_factory=list)  # list of ReportSection

    def to_dict(self) -> dict:
        """JSON-serializable dict (for format='dict' and future API)."""
        return {
            "title": self.title,
            "description": self.description,
            "assumptions": list(self.assumptions),
            "variables": [
                {
                    "name": v.name,
                    "latex": v.latex,
                    "value": _serialize_value(v.value),
                    "sensitivity": v.sensitivity,
                    "units": v.units,
                    "label": v.label,
                }
                for v in self.variables
            ],
            "constraint_groups": [
                {"label": cg.label, "constraints": list(cg.constraints)}
                for cg in self.constraint_groups
            ],
            "children": [c.to_dict() for c in self.children],
        }


# ── Value helpers ─────────────────────────────────────────────────────────────


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def _resolve_var_value(vk, solution=None, substitutions=None, model=None):
    """Resolve a variable's display value following priority:
    solution > substitutions override > model.substitutions > vk.value
    """
    if solution is not None:
        try:
            return solution[vk]
        except (KeyError, TypeError):
            pass
    if substitutions is not None:
        if vk in substitutions:
            return substitutions[vk]
    if model is not None:
        subs = getattr(model, "substitutions", {})
        if vk in subs:
            return subs[vk]
    return getattr(vk, "value", None)


def _resolve_sensitivity(vk, solution=None) -> Optional[float]:
    """Get sensitivity (shadow price) for a variable from solution."""
    if solution is None:
        return None
    try:
        sens_vars = solution.sens.variables
        if vk in sens_vars:
            return float(sens_vars[vk])
        return None
    except (AttributeError, KeyError, TypeError):
        return None


# ── Constraint rendering ──────────────────────────────────────────────────────


def _render_constraint(c) -> str:
    """Render a single constraint as a string."""
    try:
        return str(c)
    except Exception:  # pylint: disable=broad-except
        return repr(c)


# ── Core builder helpers ──────────────────────────────────────────────────────


def _build_var_entries(model, solution, substitutions) -> List[VarEntry]:
    """Build VarEntry list from model.unique_varkeys."""
    entries = []
    for vk in sorted(model.unique_varkeys, key=lambda v: v.name):
        entries.append(
            VarEntry(
                name=vk.name,
                latex=vk.latex() if callable(getattr(vk, "latex", None)) else vk.name,
                value=_resolve_var_value(
                    vk, solution=solution, substitutions=substitutions, model=model
                ),
                sensitivity=_resolve_sensitivity(vk, solution=solution),
                units=vk.unitrepr or "-",
                label=vk.label or "",
            )
        )
    return entries


def _build_constraint_groups(model) -> List[CGroup]:
    """Build CGroup list from model.cgroups or a single unnamed group."""
    if model.cgroups is not None:
        return [
            CGroup(
                label=label,
                constraints=[
                    _render_constraint(c)
                    for c in (items if isinstance(items, list) else [items])
                ],
            )
            for label, items in model.cgroups.items()
        ]
    own = []
    try:
        for item in model:
            # Skip child models (unique_varkeys is set only on Model instances).
            if not hasattr(item, "unique_varkeys"):
                own.append(_render_constraint(item))
    except TypeError:
        pass
    return [CGroup(label="", constraints=own)] if own else []


# ── Core builder ─────────────────────────────────────────────────────────────


def build_report_ir(
    model,
    solution=None,
    substitutions: Optional[dict] = None,
) -> ReportSection:
    """Build a ReportSection tree from *model*.

    Parameters
    ----------
    model : Model
        Root model to report on.
    solution : Solution, optional
        If provided, variable entries include solved values and sensitivities.
    substitutions : dict, optional
        One-off value overrides without mutating model.substitutions.
    """
    desc = type(model).description()
    return ReportSection(
        title=type(model).__name__,
        description=desc.get("summary", ""),
        assumptions=list(desc.get("assumptions", [])),
        variables=_build_var_entries(model, solution, substitutions),
        constraint_groups=_build_constraint_groups(model),
        children=[
            build_report_ir(child, solution=solution, substitutions=substitutions)
            for child in model.submodels
        ],
    )


# ── Renderer dispatcher ───────────────────────────────────────────────────────


def render_report(ir: ReportSection, fmt: str = "text") -> "dict | str":
    """Dispatch to the appropriate format renderer.

    Parameters
    ----------
    ir : ReportSection
        The root IR produced by build_report_ir().
    fmt : str
        Output format: "dict", "text", "md", or "latex".
    """
    if fmt == "dict":
        return ir.to_dict()
    raise ValueError(
        f"Format '{fmt}' not yet implemented. "
        "Available formats: 'dict', 'text', 'md', 'latex'."
    )
