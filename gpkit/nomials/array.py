"""Module for creating NomialArray instances.

Example
-------
>>> x = gpkit.Monomial('x')
>>> px = gpkit.NomialArray([1, x, x**2])

"""

from functools import reduce
from operator import eq, ge, le, xor

import numpy as np

from ..ast_nodes import expr, to_ast
from ..util.repr_conventions import ReprMixin
from ..util.small_classes import EMPTY_HV, HashVector, Numbers
from ..util.small_scripts import try_str_without
from .constraints import ArrayConstraint
from .map import NomialMap
from .math import Signomial


@np.vectorize
def vec_recurse(element, function, *args, **kwargs):
    "Vectorizes function with particular args and kwargs"
    return function(element, *args, **kwargs)


def _elementwise(ufunc, op, left, right):
    "left op right, each element computed and the whole recorded in written order."
    return NomialArray(ufunc(left, right), ast=expr(op, (to_ast(left), to_ast(right))))


def array_constraint(symbol, func):
    "Return function which creates constraints of the given operator."
    vecfunc = np.vectorize(func)

    def wrapped_func(self, other):
        "Creates array constraint from vectorized operator."
        written = other  # what the constraint says, before broadcasting it
        if not isinstance(other, NomialArray):
            other = NomialArray(other)
        result = vecfunc(self, other)
        return ArrayConstraint(
            result, getattr(self, "key", self), symbol, getattr(written, "key", written)
        )

    return wrapped_func


class NomialArray(ReprMixin, np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Arguments
    ---------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.NomialArray([1, x, x**2])
    """

    def __mul__(self, other):
        return _elementwise(np.multiply, "mul", self, other)

    def __rmul__(self, other):
        return _elementwise(np.multiply, "mul", other, self)

    def __truediv__(self, other):
        return _elementwise(np.true_divide, "div", self, other)

    def __rtruediv__(self, other):
        return _elementwise(np.true_divide, "div", other, self)

    def __add__(self, other):
        if isinstance(other, Numbers) and other == 0:
            return self  # as Signomial does, so sum() records no leading 0
        return _elementwise(np.add, "add", self, other)

    def __radd__(self, other):
        if isinstance(other, Numbers) and other == 0:
            return self
        return _elementwise(np.add, "add", other, self)

    def __sub__(self, other):
        "Elementwise, so each element's __sub__ says whether subtracting is allowed."
        ast = expr("add", (to_ast(self), expr("neg", (to_ast(other),))))
        return NomialArray(np.subtract(self, other), ast=ast)

    def __rsub__(self, other):
        ast = expr("add", (to_ast(other), expr("neg", (to_ast(self),))))
        return NomialArray(np.subtract(other, self), ast=ast)

    # In-place ops rebind, as they do for scalars, instead of writing into the
    # array: it may be a VectorVariable's own, and the AST has to follow too.
    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __pow__(self, expo):
        result = np.ndarray.__pow__(self, expo)
        return NomialArray(result, ast=expr("pow", (to_ast(self), expo)))

    def __neg__(self):
        result = np.ndarray.__neg__(self)
        return NomialArray(result, ast=expr("neg", (to_ast(self),)))

    def __getitem__(self, idxs):
        out = np.ndarray.__getitem__(self, idxs)
        if not getattr(out, "shape", None):
            return out
        return NomialArray(out, ast=expr("index", (to_ast(self), idxs)))

    def str_without(self, excluded=()):
        "Returns string without certain fields (such as 'lineage')."
        if "ast" not in excluded and self.ast:
            return self.parse_ast(excluded)
        if "key" not in excluded and hasattr(self, "key"):
            return self.key.str_without(excluded)
        if not self.shape:
            return try_str_without(self.flatten()[0], excluded)

        joined = ", ".join(
            [
                try_str_without(np.ndarray.__getitem__(self, i), excluded)
                for i in range(self.shape[0])
            ]
        )
        return f"[{joined}]"

    def latex(self, excluded=()):
        "Returns latex representation without certain fields, as str_without does."
        if "ast" not in excluded and self.ast:
            return self.ast.latex(excluded)
        units = self.latex_unitstr() if "units" not in excluded else ""
        if hasattr(self, "key"):
            return self.key.latex(excluded) + units
        joined = ", ".join(
            try_str_without(el, excluded, latex=True) for el in self.flat
        )
        return rf"\left[{joined}\right]"

    def __hash__(self):
        return reduce(xor, map(hash, self.flat), 0)

    def __new__(cls, input_array, *, ast=None):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input is an already formed ndarray instance cast to our class type
        obj = np.asarray(input_array).view(cls)
        if ast is not None:
            obj.ast = ast
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."

    def __array_wrap__(self, out_arr, context=None, return_scalar=True):
        """Called by numpy ufuncs.
        Special case to avoid creation of 0-dimensional arrays
        See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if return_scalar and out_arr.ndim == 0:
            val = out_arr.item()
            return np.float(val) if isinstance(val, np.generic) else val
        return np.ndarray.__array_wrap__(self, out_arr, context, return_scalar)

    __eq__ = array_constraint("=", eq)
    __le__ = array_constraint("<=", le)
    __ge__ = array_constraint(">=", ge)

    def inner(self, other):
        "Returns the array and argument's inner product."
        return NomialArray(np.inner(self, other))

    def outer(self, other):
        "Returns the array and argument's outer product."
        return NomialArray(np.outer(self, other))

    def vectorize(self, function, *args, **kwargs):
        "Apply a function to each terminal constraint, returning the array"
        return vec_recurse(self, function, *args, **kwargs)

    def sub(self, subs, require_positive=True):
        "Substitutes into the array"
        return self.vectorize(lambda nom: nom.sub(subs, require_positive))

    def sum(self, *args, **kwargs):
        "Returns a sum. O(N) if no arguments are given."
        if not self.size:
            raise ValueError("cannot sum NomialArray of size 0")
        if args or kwargs or not self.shape:
            return np.ndarray.sum(self, *args, **kwargs)
        hmap = NomialMap()
        for p in self.flat:
            if not hmap and hasattr(p, "units"):
                hmap.units = p.units
            if hasattr(p, "hmap"):
                hmap += p.hmap
            else:
                if hasattr(p, "units"):
                    p = p.to(hmap.units).magnitude
                elif hmap.units and p and not np.isnan(p):
                    p /= float(hmap.units)
                hmap[EMPTY_HV] = p + hmap.get(EMPTY_HV, 0)
        return Signomial(hmap, ast=expr("sum", (to_ast(self),)))

    def prod(self, *args, **kwargs):
        "Returns a product. O(N) if no arguments and only contains monomials."
        if not self.size:
            raise ValueError("cannot prod NomialArray of size 0")
        if args or kwargs:
            return np.ndarray.prod(self, *args, **kwargs)
        c, exp = 1.0, HashVector()
        hmap = NomialMap()
        for m in self.flat:
            try:
                ((mexp, mc),) = m.hmap.items()
            except (AttributeError, ValueError):
                return np.ndarray.prod(self, *args, **kwargs)
            c *= mc
            exp += mexp
            if m.units:
                hmap.units = (hmap.units or 1) * m.units
        hmap[exp] = c
        return Signomial(hmap, ast=expr("prod", (to_ast(self),)))
