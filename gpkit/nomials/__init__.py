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
    Shares,
    Signomial,
    SignomialInequality,
    SingleSignomialEquality,
    guess,
)
from .variables import ArrayVariable, Variable

VectorVariable = ArrayVariable
