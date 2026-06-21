"MarginObjective: user-facing signed-difference objective with sensitivity tracking"

from dataclasses import dataclass

from .varkey import VarKey


@dataclass
class MarginObjective:
    """Specifies a maximize(plus_var − minus_var) user-facing objective.

    Attach as ``self.margin_objective = MarginObjective(...)`` in ``Model.setup()``.
    After solving, ``Solution.derived`` holds the value and per-constant sensitivities.
    plus_var and minus_var may be Variable or VarKey; they are normalized to VarKey.
    """

    name: str
    plus_var: VarKey
    minus_var: VarKey

    def __post_init__(self):
        if hasattr(self.plus_var, "key"):
            self.plus_var = self.plus_var.key
        if hasattr(self.minus_var, "key"):
            self.minus_var = self.minus_var.key
