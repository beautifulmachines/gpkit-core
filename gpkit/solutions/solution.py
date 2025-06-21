"""Core solution handling for geometric programs.

This module provides the Solution class, which represents a single solution to a geometric program.
The Solution class is designed to be simple, focused, and easily serializable.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

from ..keydict import KeyDict
from ..varkey import VarKey


@dataclass
class Solution:
    """A single solution to a geometric program.

    This class represents the core data and functionality for a single solution,
    providing a clean interface for accessing variable values and sensitivities.

    Attributes
    ----------
    cost : float
        The optimal cost value
    variables : KeyDict
        Dictionary mapping VarKeys to their optimal values
    sensitivities : Dict[str, Any]
        Dictionary containing sensitivity information
    modelstr : str
        String representation of the solved model
    """

    cost: float
    variables: KeyDict
    sensitivities: Dict[str, Any] = field(default_factory=dict)
    modelstr: str = ""

    def __post_init__(self):
        """Validate and process the solution data after initialization."""
        if not isinstance(self.variables, KeyDict):
            self.variables = KeyDict(self.variables)

    def __call__(self, key: Union[str, VarKey]) -> Any:
        """Get the value of a variable.

        Parameters
        ----------
        key : Union[str, VarKey]
            The variable to look up, either by name or VarKey

        Returns
        -------
        Any
            The value of the variable

        Raises
        ------
        KeyError
            If the variable is not found in the solution
        """
        return self.variables[key]

    def __getitem__(self, key: Union[str, VarKey]) -> Any:
        """Get the value of a variable using dictionary-style access.

        Parameters
        ----------
        key : Union[str, VarKey]
            The variable to look up, either by name or VarKey

        Returns
        -------
        Any
            The value of the variable

        Raises
        ------
        KeyError
            If the variable is not found in the solution
        """
        return self.variables[key]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the solution to a dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the solution
        """
        return {
            "cost": self.cost,
            "variables": dict(self.variables),
            "sensitivities": self.sensitivities,
            "modelstr": self.modelstr,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solution":
        """Create a Solution instance from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing solution data

        Returns
        -------
        Solution
            A new Solution instance
        """
        return cls(
            cost=data["cost"],
            variables=data["variables"],
            sensitivities=data.get("sensitivities", {}),
            modelstr=data.get("modelstr", ""),
        )

    def almost_equal(self, other: "Solution", reltol: float = 1e-3) -> bool:
        """Check if this solution is almost equal to another.

        Parameters
        ----------
        other : Solution
            The solution to compare against
        reltol : float, optional
            Relative tolerance for comparison, by default 1e-3

        Returns
        -------
        bool
            True if the solutions are almost equal
        """
        if not isinstance(other, Solution):
            return False

        if abs(self.cost - other.cost) / (abs(self.cost) + abs(other.cost)) > reltol:
            return False

        svars, ovars = self.variables, other.variables
        svks, ovks = set(svars), set(ovars)

        if svks != ovks:
            return False

        for key in svks:
            reldiff = np.max(abs(np.divide(svars[key], ovars[key]) - 1))
            if reldiff >= reltol:
                return False

        return True
