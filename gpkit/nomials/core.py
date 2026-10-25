"The shared non-mathematical backbone of all Nomials"

from ..display.render import Rendered
from ..util.small_classes import FixedScalar, Numbers
from .data import NomialData


class Nomial(Rendered, NomialData):
    "Shared non-mathematical properties of all nomials"

    sub = None

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
