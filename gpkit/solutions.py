"Classes for representing solutions"

from dataclasses import dataclass
from typing import List, Sequence

from .solution_array import SolutionArray
from .varkey import VarKey
from .varmap import VarMap


@dataclass(frozen=True, slots=True)
class RawSolution:
    "Standardized raw data produced by a solver"

    x: Sequence
    nu: Sequence
    la: Sequence
    cost: float
    status: str
    meta: dict


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


@dataclass(frozen=True, slots=True)
class Solution:
    "A single GP solution, with mappings back to variables and constraints"

    cost: float
    primal: VarMap
    constants: VarMap
    sens: Sensitivities
    # program : GP
    meta: dict

    def __getitem__(self, key: VarKey) -> float:
        if key in self.primal:
            return self.primal.quantity(key)
        if key in self.constants:
            return self.constants.quantity(key)
        if hasattr(key, "sub"):
            variables = VarMap(self.primal)
            variables.update(self.constants)
            subbed = key.sub(variables, require_positive=False)
            # subbed should be a constant monomial
            assert getattr(subbed, "exp", {}) == {}
            return getattr(subbed, "c", subbed)
        raise KeyError(f"no variable '{key}' found in the solution")

    def almost_equal(self, other, **kwargs):
        "Checks for almost-equality between two solutions"
        return self.to_solution_array().almost_equal(other, **kwargs)

    def diff(self, *args, **kwargs):
        "Pass through to SolutionArray.diff"
        return self.to_solution_array().diff(*args, **kwargs)

    def save(self, *args, **kwargs):
        "Pass through to SolutionArray.save"
        self.to_solution_array().save(*args, **kwargs)

    def save_compressed(self, *args, **kwargs):
        "Pass through to SolutionArray.save_compressed"
        self.to_solution_array().save_compressed(*args, **kwargs)

    def summary(self, *args, **kwargs) -> str:
        "Pass through to SolutionArray.summary"
        return self.to_solution_array().summary(*args, **kwargs)

    def table(self, **kwargs) -> str:
        "Pass through to SolutionArray.table"
        return self.to_solution_array().table(**kwargs)

    def to_solution_array(self):
        "Convert this to a SolutionArray"
        variables = VarMap(self.primal)
        variables.update(self.constants)
        sol_array = SolutionArray(
            {
                "cost": self.cost,
                "cost function": self.meta["cost function"],
                "freevariables": self.primal,
                "constants": self.constants,
                "variables": variables,
                "soltime": self.meta["soltime"],
                "warnings": self.meta["warnings"],
                "sensitivities": {
                    "constraints": self.sens.constraints,
                    "variables": self.sens.variables,
                    "variablerisk": self.sens.variablerisk,
                    "models": self.sens.models,
                },
            }
        )
        sol_array.modelstr = self.meta.get("modelstr", None)
        return sol_array


class SolutionSequence(List[Solution]):
    """
    Ordered collection of Solution objects all sharing same underlying model.
    """

    def __init__(self, iterable=(), program=None):
        self.program = program  # may start as None, set on first append
        super().__init__()
        for s in iterable:
            self.append(s)

    def append(self, sol: Solution) -> None:
        "Standard list append, with integrity check"
        # if self.program is None:
        #    self.program = sol.program
        # elif sol.program is not self.program:
        #    raise ValueError("SolutionSequence elements must share the same program")
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
