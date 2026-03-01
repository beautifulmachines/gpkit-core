"GP and SP modeling package"

__version__ = "0.2.2"

from .build import build
from .constraints.set import ConstraintSet
from .constraints.sigeq import SignomialEquality
from .globals import NamedVariables, SignomialsEnabled, Vectorize, settings
from .model import Model
from .nomials import ArrayVariable, Monomial, NomialArray, Posynomial, Signomial
from .nomials import VectorizableVariable as Variable  # Variable the user sees
from .nomials import VectorVariable
from .programs.gp import GeometricProgram
from .programs.sgp import SequentialGeometricProgram
from .units import DimensionalityError, units, ureg
from .var import Var
from .varkey import VarKey

if "just built!" in settings:  # pragma: no cover
    print(f"""
GPkit is now installed with solver(s) {settings['installed_solvers']}
To incorporate new solvers at a later date, run `gpkit.build()`.

If you encounter any bugs or issues using GPkit, please open a new issue at
https://github.com/beautifulmachines/gpkit-core/issues/new.

Finally, we hope you find our documentation (https://gpkit.readthedocs.io/) and
engineering-design models (https://github.com/beautifulmachines/gpkit-models/)
useful for your own applications.

Enjoy!
""")
