"Contains nomials, inequalities, and arrays"

from .array import NomialArray
from .core import Nomial
from .data import NomialData
from .map import NomialMap
from .math import (
    Monomial,
    MonomialEquality,
    Posynomial,
    PosynomialInequality,
    Signomial,
    SignomialInequality,
    SingleSignomialEquality,
)
from .variables import ArrayVariable, Variable, VectorizableVariable

VectorVariable = ArrayVariable
