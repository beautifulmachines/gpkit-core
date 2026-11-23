"Implements the NomialMap class"

import numpy as np

from .. import units
from ..ir import ir_units
from ..units import DimensionalityError, qty
from ..util.small_classes import EMPTY_HV, FixedScalar, HashVector, Numbers, Strings
from .substitution import parse_subs

DIMLESS_QUANTITY = qty("dimensionless")


class NomialMap(HashVector):
    """Class for efficent algebraic represention of a nomial

    A NomialMap is a mapping between hashvectors representing exponents
    and their coefficients in a posynomial.

    For example,  {{x : 1}: 2.0, {y: 1}: 3.0} represents 2*x + 3*y, where
    x and y are VarKey objects.
    """

    units = None
    expmap = None  # used for monomial-mapping postsubstitution; see .mmap()
    csmap = None  # used for monomial-mapping postsubstitution; see .mmap()

    def to_ir(self):
        "Serialize this NomialMap to an IR dict."
        terms = []
        for exp_hv, coeff in self.items():
            term = {"coeff": float(coeff)}
            if exp_hv:  # non-empty exponent vector (not a constant)
                term["exps"] = {
                    vk.ref: int(x) if x == int(x) else float(x)
                    for vk, x in exp_hv.items()
                }
            terms.append(term)
        ir = {"terms": terms}
        if self.units:
            ir["units"] = ir_units(self)
        return ir

    def copy(self):
        "Return a copy of this"
        return self.__class__(self)

    def units_of_product(self, thing, thing2=None):
        "Sets units to those of `thing*thing2`. Ugly optimized code."
        if thing is None and thing2 is None:
            self.units = None
        elif hasattr(thing, "units"):
            if hasattr(thing2, "units"):
                self.units, dimless_convert = units.of_product(thing, thing2)
                if dimless_convert:
                    for key in self:
                        self[key] *= dimless_convert
            else:
                self.units = qty(thing.units)
        elif hasattr(thing2, "units"):
            self.units = qty(thing2.units)
        elif thing2 is None and isinstance(thing, Strings):
            self.units = qty(thing)
        else:
            self.units = None

    def to(self, to_units):
        "Returns a new NomialMap of the given units"
        sunits = self.units or DIMLESS_QUANTITY
        nm = self * sunits.to(to_units).magnitude  # note that * creates a copy
        nm.units_of_product(to_units)
        return nm

    def __add__(self, other):
        "Adds NomialMaps together"
        if self.units != other.units:
            try:
                other *= float(other.units / self.units)
            except (TypeError, AttributeError) as exc:  # one of those is None
                raise DimensionalityError(self.units, other.units) from exc
        hmap = HashVector.__add__(self, other)
        hmap.units = self.units
        return hmap

    def diff(self, varkey):
        "Differentiates a NomialMap with respect to a varkey"
        out = NomialMap()
        out.units_of_product(self.units, 1 / varkey.units if varkey.units else None)
        for exp in self:
            if varkey in exp:
                x = exp[varkey]
                c = self[exp] * x
                exp = exp.copy()
                if x == 1:
                    exp.hashvalue ^= hash((varkey, 1))
                    del exp[varkey]
                else:
                    exp.hashvalue ^= hash((varkey, x)) ^ hash((varkey, x - 1))
                    exp[varkey] = x - 1
                out[exp] = c
        return out

    def sub(self, substitutions, varkeys, parsedsubs=False):
        """Applies substitutions to a NomialMap

        Parameters
        ----------
        substitutions : (dict-like)
            list of substitutions to perform

        varkeys : (set-like)
            varkeys that are present in self
            (required argument so as to require efficient code)

        parsedsubs : bool
            flag if the substitutions have already been parsed
            to contain only keys in varkeys

        """
        if parsedsubs or not substitutions:
            fixed = substitutions
        else:
            fixed = parse_subs(varkeys, substitutions)
        if not fixed:
            if not self.expmap:
                self.expmap, self.csmap = {exp: exp for exp in self}, {}
            return self

        # Fold each term's fixed factors into its coefficient, e.g. with a fixed,
        # c * a**p * x**q becomes (c * a**p) * x**q
        cp = NomialMap()
        cp.units = self.units
        cp.expmap, cp.csmap = {}, {}
        cvals = {}
        for exp, c in self.items():
            new_exp = HashVector()
            for vk, x in exp.items():
                if vk not in fixed:
                    new_exp[vk] = x
                    continue
                if vk not in cvals:
                    cvals[vk] = numeric_value(vk, fixed[vk])
                c *= cvals[vk] ** float(x)  # not numpy's, where 0**-1 is inf
                if np.isnan(c):  # e.g. 0*inf, which isn't ours to decide
                    raise ValueError(
                        f"substituting {vk} = {cvals[vk]} is undefined (nan)"
                    )
            cp.expmap[exp], cp.csmap[exp] = new_exp, c
            cp[new_exp] = cp.get(new_exp, 0) + c  # merge terms with equal exps
        for exp in [exp for exp, c in cp.items() if c == 0]:
            del cp[exp]  # terms that vanished or cancelled
        if not cp:
            cp[EMPTY_HV] = 0.0
        return cp

    def mmap(self, orig):
        """Maps substituted monomials back to the original nomial

        self.expmap is the map from pre- to post-substitution exponents, and
            takes the form {original_exp: new_exp}

        self.csmap is the map from pre-substitution exponents to coefficients.

        m_from_ms is of the form {new_exp: [old_exps, ]}

        pmap is of the form [{orig_idx1: fraction1, orig_idx2: fraction2, }, ]
            where at the index corresponding to each new_exp is a dictionary
            mapping the indices corresponding to the old exps to their
            fraction of the post-substitution coefficient
        """
        pmap = [{} for _ in self]
        origexps = list(orig.keys())
        selfexps = list(self.keys())
        for orig_exp, self_exp in self.expmap.items():
            if not self.get(self_exp):  # can occur in tautological constraints
                continue  # after substitution
            fraction = self.csmap.get(orig_exp, orig[orig_exp]) / self[self_exp]
            orig_idx = origexps.index(orig_exp)
            pmap[selfexps.index(self_exp)][orig_idx] = fraction
        return pmap


def numeric_value(vk, cval: Numbers | FixedScalar) -> float:
    "The number cval stands for, in vk's units"
    if hasattr(cval, "hmap"):
        if any(cval.hmap.keys()):
            raise ValueError("Monomial substitutions are not supported.")
        (cval,) = cval.hmap.to(vk.units or DIMLESS_QUANTITY).values()
    elif hasattr(cval, "to"):
        cval = cval.to(vk.units or DIMLESS_QUANTITY).magnitude
    elif not isinstance(cval, (int, float, np.number)):
        raise ValueError("Monomial substitutions are not supported.")
    return float(cval)
