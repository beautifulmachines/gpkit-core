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
        Keys are VarKey objects.

    Returns
    -------
    ReportSection
        Root of the IR tree (children correspond to model.submodels).
    """
    title = type(model).__name__

    # Description metadata from Model.description() classmethod
    desc_meta = type(model).description()
    description = desc_meta.get("summary", "")
    assumptions = list(desc_meta.get("assumptions", []))

    # Build VarEntry list from model.unique_varkeys
    variables: List[VarEntry] = []
    for vk in sorted(model.unique_varkeys, key=lambda v: v.name):
        val = _resolve_var_value(
            vk, solution=solution, substitutions=substitutions, model=model
        )
        sens = _resolve_sensitivity(vk, solution=solution)
        units_str = vk.unitrepr or "-"
        latex_str = vk.latex() if callable(getattr(vk, "latex", None)) else vk.name
        variables.append(
            VarEntry(
                name=vk.name,
                latex=latex_str,
                value=val,
                sensitivity=sens,
                units=units_str,
                label=vk.label or "",
            )
        )

    # Build CGroup list from _cgroups or a single unnamed group
    constraint_groups: List[CGroup] = []
    cgroups = getattr(model, "_cgroups", None)
    if cgroups is not None:
        # Named groups from dict-returning setup()
        for label, items in cgroups.items():
            flat_items = items if isinstance(items, list) else [items]
            rendered = [_render_constraint(c) for c in flat_items]
            constraint_groups.append(CGroup(label=label, constraints=rendered))
    else:
        # Single unnamed group containing all model-level constraints
        # Use the model's own constraint list (direct, not flattened into children)
        own_constraints = []
        try:
            # CostedConstraintSet is a list; iterate top-level items
            for item in model:
                # Skip child models (they appear in children, not cgroups).
                # unique_varkeys is set only on Model instances.
                if hasattr(item, "unique_varkeys"):
                    continue
                own_constraints.append(_render_constraint(item))
        except TypeError:
            pass
        if own_constraints:
            constraint_groups.append(CGroup(label="", constraints=own_constraints))

    # Recurse into direct children (submodels)
    children = [
        build_report_ir(child, solution=solution, substitutions=substitutions)
        for child in model.submodels
    ]

    return ReportSection(
        title=title,
        description=description,
        assumptions=assumptions,
        variables=variables,
        constraint_groups=constraint_groups,
        children=children,
    )


# ── Renderer dispatcher ───────────────────────────────────────────────────────


def render_report(
    ir: ReportSection, format: str = "text"
):  # pylint: disable=redefined-builtin
    """Dispatch to the appropriate format renderer.

    Parameters
    ----------
    ir : ReportSection
        The root IR produced by build_report_ir().
    format : str
        Output format. Currently only "dict" is implemented.
        "text", "md", and "latex" backends will be added in Plans 03 and 04.

    Returns
    -------
    dict | str
        The rendered report in the requested format.
    """
    if format == "dict":
        return ir.to_dict()
    raise ValueError(
        f"Format '{format}' not yet implemented. "
        "Available formats: 'dict', 'text', 'md', 'latex'."
    )
