"Classes for representing solutions"

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from . import printing
from .breakdowns import bdtable_gen
from .budgets import build_budget
from .ir import IR_VERSION, diff_solutions, ir_units
from .units import Quantity
from .varkey import VarKey
from .varmap import VarMap, display_names


class SolveStatus(StrEnum):
    "Possible values of Solution.meta['status']. Members compare equal to their string value."

    OPTIMAL = "optimal"
    OPTIMAL_WITH_WARNINGS = "optimal_with_warnings"


@dataclass(frozen=True, slots=True)
class RawSolution:
    "Standardized raw data produced by a solver"

    x: Sequence
    nu: Sequence
    la: Sequence
    cost: float
    status: str
    meta: dict

    def __post_init__(self):
        self.validate()

    def validate(self):
        "Run basic validation"
        for vec, name in ((self.x, "x"), (self.nu, "nu"), (self.la, "la")):
            s = vec.shape
            if len(s) != 1:
                raise ValueError(f"Expected 1-D {name}; got shape {s}")
        _ = float(self.cost)
        assert len(self.nu) >= len(self.la)


@dataclass(frozen=True, slots=True)
class Sensitivities:
    "Container for a Solution's sensitivities"

    constraints: dict  # {constraint object: sensitivity} -- in-process only
    constraints_by_key: dict  # {ConstraintKey: sensitivity} -- the serializable form
    # cost: dict
    models: dict
    variables: VarMap
    variablerisk: VarMap  # only used for breakdowns

    def __getitem__(self, key: VarKey) -> float:
        return self.variables[key]

    def _key_of(self, constraint):
        """The ConstraintKey for a constraint object, or None if absent.

        The two constraint maps are one enumeration under two kinds of key, so
        position pairs them -- the only bridge a Solution has, holding no model.
        Goes away with the object-keyed map itself (#283).
        """
        return dict(zip(self.constraints, self.constraints_by_key)).get(constraint)


@dataclass(frozen=True, slots=True)
class MarginSolution:
    "Value and per-constant sensitivities for a MarginObjective"

    name: str
    value: float  # A* − B*
    plus_value: float  # A*
    minus_value: float  # B*
    units: str  # unit string from plus_var.key.units (may be empty)
    sensitivities: dict  # {VarKey: ∂(margin)/∂c} for each constant

    def to_ir(self) -> dict:
        "Serialize to a dict, with sensitivities keyed by VarKey.ref."
        return {
            "name": self.name,
            "value": float(self.value),
            "plus_value": float(self.plus_value),
            "minus_value": float(self.minus_value),
            "units": self.units,
            "sensitivities": {vk.ref: float(s) for vk, s in self.sensitivities.items()},
        }

    def table(self, cost_sens=None) -> str:
        """Format sensitivities, ordered by |GP cost sensitivity| when provided.

        Parameters
        ----------
        cost_sens : dict, optional
            {VarKey: ∂log(cost)/∂log(c)} from sol.sens.variables.  When given,
            rows are ordered by descending |cost_sens| — a dimensionless,
            unit-invariant ranking of each constant's influence.  When omitted,
            rows are ordered by the margin sensitivity value (most negative first).
        """
        u = f" {self.units}" if self.units else ""
        lines = [
            (
                f"\n{self.name}: {self.value:.4g}{u}"
                f"  (plus={self.plus_value:.4g}{u}, minus={self.minus_value:.4g}{u})"
            ),
        ]
        if not self.sensitivities:
            return "\n".join(lines)
        if cost_sens is not None:
            label = "by |GP sensitivity|"
            items = sorted(
                self.sensitivities.items(),
                key=lambda kv: (
                    -float(f"{abs(cost_sens.get(kv[0], 0)):.4g}"),
                    kv[0].ref,
                ),
            )
        else:
            label = "most negative first"
            items = sorted(
                self.sensitivities.items(),
                key=lambda kv: (float(f"{kv[1]:.4g}"), kv[0].ref),
            )
        lines.append(f"  ∂({self.name})/∂c [{label}]:")
        names = display_names([vk for vk, _ in items])
        for vk, s in items:
            if self.units and vk.units:
                c_units = vk.unitstr()
                c_fmt = f"({c_units})" if "/" in c_units else c_units
                su = f" {self.units}/{c_fmt}"
            else:
                su = u
            lines.append(f"    {names[vk]:20s}  {s:+.4g}{su}")
        return "\n".join(lines)


SUMMARY_TABLES = ("sweeps", "cost", "warnings", "solution")


def _value_ir(value, units: str) -> dict:
    "A number and the units it is expressed in, which are the declared ones."
    ir = {"value": float(value)}
    if units:
        ir["units"] = units
    return ir


def _varmap_ir(vmap: VarMap) -> dict:
    "Vector elements appear individually; Model.to_ir() holds the parent."
    # Units as the model IR spells them, so the two documents join cleanly.
    return {vk.ref: _value_ir(v, vk.to_ir().get("units", "")) for vk, v in vmap.items()}


def _subject_ir(sol, warning: dict) -> dict:
    "Name a warning's subject by ref, dropping the fields that are empty."
    ir = {}
    subject = warning["subject"]
    if subject is not None:
        key = getattr(subject, "key", None)
        ref = key.ref if isinstance(key, VarKey) else None
        if ref is None:
            constraint_key = sol.sens._key_of(subject)
            ref = constraint_key.ref if constraint_key is not None else None
        if ref is not None:
            ir["subject"] = ref
    if warning["value"] is not None:
        ir["value"] = float(warning["value"])
    return ir


@dataclass(frozen=True, slots=True)
class Solution:
    "A single GP solution, with mappings back to variables and constraints"

    cost: float
    primal: VarMap
    constants: VarMap
    sens: Sensitivities
    # program : GP
    meta: dict
    derived: "MarginSolution | None" = None

    @property
    def variables(self):
        "All variables: primal (free) + constants (substituted)"
        vmap = VarMap(self.primal)
        vmap.update(self.constants)
        return vmap

    def __getitem__(self, key: VarKey) -> "Quantity":
        if key in self.primal:
            return self.primal.quantity(key)
        if key in self.constants:
            return self.constants.quantity(key)
        if hasattr(key, "sub"):
            subbed = key.sub(self.variables, require_positive=False)
            # Use .cs rather than .c: a zero-valued sub produces a Signomial
            # (0 ≤ 0 triggers any_nonpositive_cs), which lacks .c but has .cs.
            (c,) = subbed.cs
            if isinstance(c, Quantity):
                return c
            return Quantity(c, key.units or "dimensionless")
        raise KeyError(f"no variable '{key}' found in the solution")

    def almost_equal(self, other, tol=1e-6):
        """Checks for almost-equality between two solutions.
        tol is treated as relative for primal; absolute for sensitivities
        """
        return not diff_solutions(other.to_ir(), self.to_ir(), tol)["changed"]

    def subinto(self, posy):
        "solution substituted into posy."
        for target_vmap in (self.primal, self.constants):
            if posy in target_vmap:
                return target_vmap.quantity(posy)

        if not hasattr(posy, "sub"):
            raise ValueError(f"no variable '{posy}' found in the solution")

        return posy.sub(self.variables, require_positive=False)

    def diff(self, baseline, **kwargs):
        "printable difference table between this and other"
        return printing.diff(self, baseline, **kwargs)

    def save(self, filename, **pickleargs):
        """Pickle the solution and save it to a file. Load again with e.g:
        >>> import pickle
        >>> with open("solution.pkl") as fil:
                sol = pickle.load(fil)
        """
        with open(filename, "wb") as fil:
            pickle.dump(self, fil, **pickleargs)

    def to_ir(self) -> dict:
        """Serialize this Solution to a JSON-serializable dict of results.

        Keyed by VarKey.ref and ConstraintKey.ref so it joins Model.to_ir() rather
        than restating it: structure stays in the model IR, values are in each
        variable's declared units, and anything empty is omitted.
        """
        ir = {
            "gpkit_ir_version": IR_VERSION,
            "cost": _value_ir(self.cost, ir_units(self.meta["cost function"])),
            "primal": _varmap_ir(self.primal),
            "sensitivities": {
                "variables": {
                    vk.ref: float(s) for vk, s in self.sens.variables.items()
                },
                "constraints": {
                    key.ref: float(s) for key, s in self.sens.constraints_by_key.items()
                },
                "models": {k: float(v) for k, v in self.sens.models.items()},
            },
            "meta": {
                "status": str(self.meta["status"]),
                "soltime": float(self.meta["soltime"]),
                "warnings": self._warnings_ir(),
            },
        }
        if self.constants:
            ir["constants"] = _varmap_ir(self.constants)
        if self.derived is not None:
            ir["derived"] = self.derived.to_ir()
        return ir

    def _warnings_ir(self) -> dict:
        "Warnings with each subject named by ref instead of held as an object."
        out = {}
        for category, warns in self.meta["warnings"].items():
            out[category] = [
                {"message": w["message"], **_subject_ir(self, w)} for w in warns
            ]
        return out

    def summary(self, **kwargs) -> str:
        "Print a summary table of this Solution"
        lines = self.cost_breakdown() + self.model_sens_breakdown() + [""]
        if self.derived is not None:
            lines.append(self.derived.table())
        table = printing.table(self, tables=SUMMARY_TABLES, **kwargs)
        return "\n".join(lines) + table

    def table(self, **kwargs) -> str:
        "Per legacy, prints breakdowns then Solution.table"
        lines = []
        if "tables" not in kwargs:  # don't add breakdowns if tables custom
            lines += self.cost_breakdown() + self.model_sens_breakdown() + [""]
            if self.derived is not None:
                lines.append(self.derived.table())
        return "\n".join(lines) + printing.table(self, **kwargs)

    def budget(self, var, display_units=None, depth=float("inf")):
        """Build and return a Budget breakdown for a variable.

        Parameters
        ----------
        var : Variable or VarKey
            The top-level budget variable (e.g. ``aircraft.m_total``).
        display_units : str, optional
            Units for all displayed values.  Defaults to the variable's units.
        depth : int or float, optional
            Maximum expansion depth. ``depth=0`` returns only the top variable
            with no children. ``depth=1`` expands one level of submodels.
            Defaults to ``float('inf')`` (fully recursive).

        Returns
        -------
        Budget
            Call ``.text()`` or ``print(sol.budget(...))`` to display.
        """
        return build_budget(self, var, display_units, depth=depth)

    def cost_breakdown(self) -> str:
        "printable visualization of cost breakdown"
        return bdtable_gen("cost")(self, set())

    def model_sens_breakdown(self) -> str:
        "printable visualization of model sensitivity breakdown"
        return bdtable_gen("model sensitivities")(self, set())


class SolutionSequence(list[Solution]):
    """
    Ordered collection of Solution objects all sharing same underlying model.
    """

    def __init__(self, iterable=()):
        super().__init__()
        for s in iterable:
            self.append(s)

    def append(self, sol: Solution) -> None:
        "Standard list append, with integrity check"
        super().append(sol)

    # ----------------------------------------------------------------
    # Convenience utilities (runtime helpers, minimal API)
    # ----------------------------------------------------------------
    def latest(self) -> Solution:
        """Return the most recent Solution."""
        return self[-1]

    def __repr__(self) -> str:
        if not self:
            return "SolutionSequence([])"
        return f"SolutionSequence(n={len(self)})"

    def plot(self, var):
        "Eventual plotting capability"
        raise NotImplementedError

    def diff(self, baseline, **kwargs):
        "printable difference table between this and other"
        return printing.diff(self, baseline, **kwargs)

    def save(self, filename, **pickleargs):
        "Pickle the SolutionSequence and save it to filename"
        with open(filename, "wb") as fil:
            pickle.dump(self, fil, **pickleargs)

    def table(self, **kwargs):
        "Per legacy, prints breakdowns then Solution.table"
        return printing.table(self, **kwargs)

    def summary(self, **kwargs):
        "Print a summary table"
        return printing.table(self, tables=SUMMARY_TABLES, **kwargs)
