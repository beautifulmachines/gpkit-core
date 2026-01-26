"""Miscellaneous small classes"""

from functools import reduce
from operator import xor
from typing import TypeAlias

import numpy as np
from scipy.sparse import csr_matrix

from ..units import Quantity

Strings: TypeAlias = str
Numbers: TypeAlias = int | float | np.number | Quantity


# pylint: disable=too-few-public-methods
class FixedScalarMeta(type):
    "Metaclass to implement instance checking for fixed scalars"

    def __instancecheck__(cls, obj):
        return getattr(obj, "hmap", None) and len(obj.hmap) == 1 and not obj.vks


# pylint: disable=too-few-public-methods
class FixedScalar(metaclass=FixedScalarMeta):
    "Instances of this class are scalar Nomials with no variables"


class Count:
    "Like python 2's itertools.count, for Python 3 compatibility."

    def __init__(self):
        self.count = -1

    def next(self):
        "Increment self.count and return it"
        self.count += 1
        return self.count


def matrix_converter(name):
    "Generates conversion function."

    def to_(self):  # used in tocoo, tocsc, etc below
        "Converts to another type of matrix."
        return getattr(self.tocsr(), "to" + name)()

    return to_


class CootMatrix:
    "A very simple sparse matrix representation."

    def __init__(self, row, col, data):
        self.row, self.col, self.data = row, col, data
        self.shape = [
            (max(self.row) + 1) if self.row else 0,
            (max(self.col) + 1) if self.col else 0,
        ]

    def __eq__(self, other):
        return (
            self.row == other.row
            and self.col == other.col
            and self.data == other.data
            and self.shape == other.shape
        )

    tocoo = matrix_converter("coo")
    tocsc = matrix_converter("csc")
    todia = matrix_converter("dia")
    todok = matrix_converter("dok")
    todense = matrix_converter("dense")

    def tocsr(self):
        "Converts to a Scipy sparse csr_matrix"
        return csr_matrix((self.data, (self.row, self.col)))

    def dot(self, arg):
        "Returns dot product with arg."
        return self.tocsr().dot(arg)


class SolverLog:
    "Adds a `write` method to list so it's file-like and can replace stdout."

    def __init__(self, output=None, *, verbosity=0):
        self.written = ""
        self.verbosity = verbosity
        self.output = output

    def write(self, writ):
        "Append and potentially write the new line."
        if writ[:2] == "b'":
            writ = writ[2:-1]
        self.written += writ
        if self.verbosity > 0:  # pragma: no cover
            self.output.write(writ)

    def lines(self):
        "Returns the lines presently written."
        return self.written.split("\n")

    def flush(self):
        "Dummy function for I/O api compatibility"


class HashVector(dict):
    """A simple, sparse, string-indexed vector. Inherits from dict.

    The HashVector class supports element-wise arithmetic:
    any undeclared variables are assumed to have a value of zero.

    Arguments
    ---------
    arg : iterable

    Example
    -------
    >>> x = gpkit.nomials.Monomial("x")
    >>> exp = gpkit.small_classes.HashVector({x: 2})
    """

    hashvalue = None

    def __hash__(self):
        "Allows HashVectors to be used as dictionary keys."
        if self.hashvalue is None:
            self.hashvalue = reduce(xor, map(hash, self.items()), 0)
        return self.hashvalue

    def copy(self):
        "Return a copy of this"
        hv = self.__class__(self)
        hv.hashvalue = self.hashvalue
        return hv

    def __pow__(self, other):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, Numbers):
            return self.__class__({k: v**other for (k, v) in self.items()})
        return NotImplemented

    def __mul__(self, other):
        """Accepts scalars and dicts. Returns with each value multiplied.

        If the other object inherits from dict, multiplication is element-wise
        and their key's intersection will form the new keys."""
        if isinstance(other, (Numbers, dict)):
            return self.__class__({k: v * other for (k, v) in self.items()})
        return NotImplemented

    def __add__(self, other):
        """Accepts scalars and dicts. Returns with each value added.

        If the other object inherits from dict, addition is element-wise
        and their key's union will form the new keys."""
        if isinstance(other, Numbers):
            return self.__class__({k: v + other for (k, v) in self.items()})
        if isinstance(other, dict):
            sums = self.copy()
            for key, value in other.items():
                if key in sums:
                    svalue = sums[key]
                    if value == -svalue:
                        del sums[key]  # remove zeros created by addition
                    else:
                        sums[key] = value + svalue
                else:
                    sums[key] = value
            sums.hashvalue = None
            return sums
        return NotImplemented

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __rmul__(self, other):
        return self * other


EMPTY_HV = HashVector()
