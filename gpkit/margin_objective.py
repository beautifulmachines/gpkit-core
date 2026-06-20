"MarginObjective: user-facing signed-difference objective with sensitivity tracking"

from dataclasses import dataclass


@dataclass
class MarginObjective:
    """Specifies a maximize(plus_var − minus_var) user-facing objective.

    Attach as ``self.margin_objective = MarginObjective(...)`` in ``Model.setup()``.
    After solving, ``Solution.derived`` holds the value and per-constant sensitivities.
    """

    name: str
    plus_var: object  # Variable or VarKey (A)
    minus_var: object  # Variable or VarKey (B)
