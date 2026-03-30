"Classes for representing solutions"

import json
import pickle
import weakref
from dataclasses import dataclass
from typing import List, Sequence

from . import printing
from .breakdowns import bdtable_gen
from .budgets import build_budget
from .varkey import VarKey
from .varmap import VarMap


class _WeakModelRef:
    """Weakly references a model for use in solution.meta.

    Storing the full model object in meta would make solutions unpicklable
    (models hold a reference to the solved program which contains
    unpicklable file handles).  This wrapper holds a weakref instead, and
    when pickled it restores as an empty (dead) ref so that the solution
    remains serialisable.
    """

    __slots__ = ("_ref",)

    def __init__(self, model):
        self._ref = weakref.ref(model) if model is not None else None

    def __call__(self):
        "Return the referenced model, or None if it has been garbage-collected."
        return self._ref() if self._ref is not None else None

    def __reduce__(self):
        # Pickle as an empty ref — the model is not persisted across save/load.
        return (type(self), (None,))


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

    constraints: dict
    # cost: dict
    models: dict
    variables: VarMap
    variablerisk: VarMap  # only used for breakdowns

    def __getitem__(self, key: VarKey) -> float:
        return self.variables[key]

    @property
    def constants(self):
        "Sensitivity to each constant"
        raise NotImplementedError


SUMMARY_TABLES = ("sweeps", "cost", "warnings", "solution")


@dataclass(frozen=True, slots=True)
class Solution:
    "A single GP solution, with mappings back to variables and constraints"

    cost: float
    primal: VarMap
    constants: VarMap
    sens: Sensitivities
    # program : GP
    meta: dict

    @property
    def variables(self):
        "All variables: primal (free) + constants (substituted)"
        vmap = VarMap(self.primal)
        vmap.update(self.constants)
        return vmap

    def __getitem__(self, key: VarKey) -> float:
        if key in self.primal:
            return self.primal.quantity(key)
        if key in self.constants:
            return self.constants.quantity(key)
        if hasattr(key, "sub"):
            subbed = key.sub(self.variables, require_positive=False)
            # subbed should be a constant monomial
            assert getattr(subbed, "exp", {}) == {}
            return getattr(subbed, "c", subbed)
        raise KeyError(f"no variable '{key}' found in the solution")

    def almost_equal(self, other, tol=1e-6):
        """Checks for almost-equality between two solutions.
        tol is treated as relative for primal; absolute for sensitivities
        """
        if set(self.primal) != set(getattr(other, "primal", ())):
            return False
        if set(self.sens.variables) != set(other.sens.variables):
            return False
        for key in self.primal:
            reldiff = abs(self.primal[key] / other.primal[key] - 1)
            if reldiff > tol:
                return False
        for key in self.sens.variables:
            absdiff = abs(self.sens.variables[key] - other.sens.variables[key])
            if absdiff > tol:
                return False
        return True

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

    def savejson(self, filename):
        "Save primal variables to a json file"
        # only saving primal is legacy carryover -- eventually add more
        json_dict = {}
        for k, v in self.primal.items():
            val = list(v) if hasattr(v, "__len__") else v
            json_dict[str(k)] = {"v": val, "u": k.unitstr()}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_dict, f)

    def summary(self, **kwargs) -> str:
        "Print a summary table of this Solution"
        lines = self.cost_breakdown() + self.model_sens_breakdown() + [""]
        table = printing.table(self, tables=SUMMARY_TABLES, **kwargs)
        return "\n".join(lines) + table

    def table(self, **kwargs) -> str:
        "Per legacy, prints breakdowns then Solution.table"
        lines = []
        if "tables" not in kwargs:  # don't add breakdowns if tables custom
            lines += self.cost_breakdown() + self.model_sens_breakdown() + [""]
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
        model_ref = self.meta.get("model")
        model = model_ref() if callable(model_ref) else model_ref
        if model is None:
            raise ValueError(
                "No model in solution.meta. Ensure the solution was produced "
                "by a current gpkit solve (meta['model'] is set at solve time)."
            )
        return build_budget(self, model, var, display_units, depth=depth)

    def cost_breakdown(self) -> str:
        "printable visualization of cost breakdown"
        return bdtable_gen("cost")(self, set())

    def model_sens_breakdown(self) -> str:
        "printable visualization of model sensitivity breakdown"
        return bdtable_gen("model sensitivities")(self, set())


class SolutionSequence(List[Solution]):
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
