"Classes for representing solutions"

from dataclasses import dataclass
from typing import List, Sequence

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
class Solution:
    "A single GP solution, with mappings back to variables and constraints"

    primal: VarMap
    dual: VarMap
    # program : GP

    def __getitem__(self, key: VarKey) -> float:
        return self.primal[key]


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
        if self.program is None:
            self.program = sol.program
        elif sol.program is not self.program:
            raise ValueError("SolutionSequence elements must share the same program")
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
