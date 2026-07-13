"GP and SP modeling package"

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("gpkit-core")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

import numpy as _np

from .ast_nodes import PiNode as _PiNode
from .constraints.costed import Objective
from .constraints.set import ConstraintSet
from .constraints.sigeq import SignomialEquality
from .margin_objective import MarginObjective
from .model import Model
from .nomials import (
    ArrayVariable,
    Monomial,
    NomialArray,
    Posynomial,
    Signomial,
    Variable,
    VectorVariable,
)
from .programs.gp import GeometricProgram
from .programs.sgp import SequentialGeometricProgram
from .units import DimensionalityError, units, ureg
from .util.build import build
from .util.globals import NamedVariables, SignomialsEnabled, Vectorize, settings
from .var import Var
from .varkey import VarKey
from .varmap import display_names

pi = Monomial(_np.pi, ast=_PiNode())
