"GP and SP modeling package"

__version__ = "0.3.5"

import numpy as _np

from .ast_nodes import PiNode as _PiNode
from .constraints.set import ConstraintSet
from .constraints.sigeq import SignomialEquality
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

pi = Monomial(_np.pi, ast=_PiNode())

if "just built!" in settings:  # pragma: no cover
    print(f"""
GPkit is now installed with solver(s) {settings['installed_solvers']}
To incorporate new solvers at a later date, run `gpkit.build()`.

If you encounter any bugs or issues using GPkit, please open a new issue at
https://github.com/beautifulmachines/gpkit-core/issues/new.

We hope you find the engineering-design models at
https://github.com/beautifulmachines/gpkit-models/ useful for your own applications.

Enjoy!
""")
