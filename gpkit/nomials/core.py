"The shared non-mathematical backbone of all Nomials"

from ..ast_nodes import ast_from_terms, to_ast
from ..display.render import LATEX, TEXT, render, varkey_names
from ..util.small_classes import FixedScalar, Numbers
from .data import NomialData


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"

    sub = None

    def _rendered(self, fmt, excluded, unit_literals):
        """This nomial in fmt: as written, or from its terms if it never was.

        The terms also stand in when the written form renders empty, which
        happens when it was only unit literals and the caller prints the units.
        """
        for tree in (to_ast(self), ast_from_terms(self.hmap.items())):
            if tree is None:
                continue
            names = varkey_names(tree, excluded, latex=fmt is LATEX)
            text = render(tree, fmt, names, unit_literals)
            if text:
                return text
        return ""

    def str_without(self, excluded=()):
        "String representation, excluding fields ('units', varkey attributes)"
        units = "" if "units" in excluded else self.unitstr(" [%s]")
        if hasattr(self, "key"):
            return self.key.str_without(excluded) + units
        # with a units suffix, unit literals would show the units twice
        return self._rendered(TEXT, excluded, unit_literals=not units) + units

    def latex(self, excluded=()):
        "Latex representation, parsing `excluded` just as .str_without does"
        if hasattr(self, "key"):
            return self.key.latex(excluded)
        units = ""
        if "units" not in excluded:
            units = self.unitstr(r"\mathrm{~\left[ %s \right]}", ":L~")
            units = units.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return self._rendered(LATEX, excluded, unit_literals=not units) + units

    @property
    def value(self):
        """Self, with values substituted for variables that have values

        Returns
        -------
        float, if no symbolic variables remain after substitution
        (Monomial, Posynomial, or Nomial), otherwise.
        """
        if isinstance(self, FixedScalar):
            return self.cs[0]
        p = self.sub({k: k.value for k in self.vks if k.value is not None})
        return p.cs[0] if isinstance(p, FixedScalar) else p

    def __eq__(self, other):
        "True if self and other are algebraically identical."
        if isinstance(other, Numbers):
            return isinstance(self, FixedScalar) and self.value == other
        return super().__eq__(other)

    __hash__ = NomialData.__hash__

    def __ne__(self, other):
        return not Nomial.__eq__(self, other)

    def __radd__(self, other):
        return self.__add__(other, rev=True)

    def __rmul__(self, other):
        return self.__mul__(other, rev=True)

    def prod(self):
        "Return self for compatibility with NomialArray"
        return self

    sum = prod
