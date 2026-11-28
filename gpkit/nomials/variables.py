"Implement Variable and ArrayVariable classes"

from collections.abc import Iterable, Mapping, MutableMapping

import numpy as np

from ..util.globals import NamedVariables, Vectorize
from ..util.small_classes import HashVector, Numbers, Strings
from ..util.small_scripts import veclinkedfn
from ..varkey import VarKey
from .array import NomialArray
from .data import NomialData
from .map import NomialMap
from .math import Monomial


def addmodelstodescr(descr, addtonamedvars=None):
    "Add models to descr, optionally adding the second argument to NAMEDVARS"
    if NamedVariables.lineage:
        descr["lineage"] = NamedVariables.lineage
        if addtonamedvars:
            NamedVariables.namedvars[descr["lineage"]].append(addtonamedvars)


def _check_scalar_value(descr):
    "Refuse a value that is not one number, a quantity or a function."
    value = descr.get("value")
    if isinstance(value, Mapping):
        raise TypeError(
            f"a map of index to value fixes elements of a vector, but "
            f"'{descr.get('name')}' is a scalar"
        )
    if isinstance(value, Iterable) and not isinstance(value, (Numbers, Strings)):
        raise TypeError(
            f"a Variable's value is one number, not {value!r}; "
            "to sweep it, use Model.sweep"
        )


def _units(units):
    "A variable's units: a string, or None for dimensionless."
    if units is not None and not isinstance(units, Strings):
        raise TypeError(
            f"units must be a string, not {units!r}; "
            'value is keyword-only: Variable("x", "m", value=5)'
        )
    return units


def _fixed_elements(name, shape, values):
    "A vector's values from a map of index to value; the others stay free."
    arr = np.full(shape, None, dtype=object)
    seen = {}
    for idx, value in values.items():
        try:
            element = int(np.arange(shape[0])[idx])
        except IndexError:
            raise IndexError(
                f"index {idx} is out of range for '{name}', of length {shape[0]}"
            ) from None
        if element in seen:
            raise ValueError(
                f"indices {seen[element]} and {idx} both name element "
                f"{element} of '{name}'"
            )
        seen[element] = idx
        arr[element] = value
    return arr


def _vector_values(name, shape, values):
    "A vector's value as one entry per element, or a function, or None."
    if isinstance(values, Mapping):
        return _fixed_elements(name, shape, values)
    if values is None or callable(values):
        return values
    if Vectorize.vectorization:
        if isinstance(values, (list, tuple)):
            values = np.array(values)
        if hasattr(values, "shape"):
            values = np.broadcast_to(values, reversed(shape)).T
        else:  # one value for every element
            # keep its units for substitution to convert;
            # not np.full, which pint intercepts and strips units in
            filled = np.empty(shape, dtype=object)
            filled.fill(values)
            values = filled
    elif not hasattr(values, "shape"):
        values = np.array(values)
    if values.shape != shape:
        raise ValueError(
            f"value's shape {values.shape} is different from the vector's {shape}."
        )
    return values


class Variable(Monomial):
    """A described singlet Monomial: Variable(name, units, label, value=...).

    Without units (or with "-") it is dimensionless. value, if given, fixes it: a
    number, a quantity (stored in the variable's units) or a linked function.
    Other keywords are VarKey fields. Variable(varkey) rebuilds one from a key.
    """

    def __new__(cls, *args, **descr) -> "NomialArray | Variable":
        if args and isinstance(args[0], VarKey):
            return object.__new__(cls)
        if Vectorize.vectorization and "idx" not in descr:
            shape = descr.pop("shape", ())
            return ArrayVariable.__new__(ArrayVariable, shape, *args, **descr)
        return object.__new__(cls)

    def __init__(self, name, units=None, label="", *, value=None, **descr):
        if isinstance(name, VarKey):
            self.key = name
        else:
            descr.update(name=name, units=_units(units), label=label, value=value)
            _check_scalar_value(descr)
            addmodelstodescr(descr, addtonamedvars=self)
            self.key = VarKey(**descr)
        hmap = NomialMap({HashVector({self.key: 1}): 1.0})
        hmap.units = self.key.units
        original_class = type(self)
        Monomial.__init__(self, hmap)
        # NOTE: Signomial.__init__ mutates __class__; restore original
        self.__class__ = original_class

    __hash__ = NomialData.__hash__

    def __getitem__(self, idx):
        raise TypeError(
            f"{self!r} is a scalar Variable and cannot be indexed. "
            "Use Variable() inside a Vectorize context to create an array variable."
        )

    def to(self, units):
        "Create new Signomial converted to new units"
        return Monomial(self).to(units)

    def sub(self, *args, **kwargs):
        """Same as nomial substitution, but also allows single-argument calls

        Example
        -------
        x = Variable('x')
        assert x.sub(3) == Variable('x', value=3)
        """
        if len(args) == 1 and "val" not in kwargs:
            (arg,) = args
            if not isinstance(arg, MutableMapping):
                args = [{self: arg}]
        return Monomial.sub(self, *args, **kwargs)

    # Growth-allowance interface. Helpers live in .growth (same package);
    # imports are deferred to break the variables.py ↔ growth.py module-load
    # cycle. Returns a plain list so this method stays inside the nomials
    # layer (no constraints/ dependency); wrap with
    # GrowthAllowance.make_constraints for slack-warning diagnostics.
    def grown_from(self, expr):
        "Return the two bookkeeping constraints for self's growth allowance."
        from .growth import make_growth_constraints  # noqa: PLC0415

        return make_growth_constraints(self, expr)

    @property
    def growth(self):
        "The auto-generated allowance Variable sibling."
        from .growth import sibling_growth  # noqa: PLC0415

        return sibling_growth(self.key)

    @property
    def f_growth(self):
        "The auto-generated fraction Variable sibling."
        from .growth import sibling_fraction  # noqa: PLC0415

        return sibling_fraction(self.key)


class ArrayVariable(NomialArray):
    """A described vector of singlet Monomials:
    ArrayVariable(shape, name, units, label, value=...).

    value, if given, is one entry per element, a map of index to value that
    fixes only those elements, or a linked function. Each element is a Variable
    whose key carries its index.
    """

    def __new__(cls, shape, name, units=None, label="", *, value=None, **descr):
        if "idx" in descr:
            raise ValueError("the description field 'idx' is reserved")

        shape = (shape,) if isinstance(shape, Numbers) else tuple(shape)
        if Vectorize.vectorization:
            shape += Vectorize.vectorization

        descr.update(shape=shape, name=name, units=_units(units), label=label)
        values = _vector_values(name, shape, value)

        veckeydescr = descr.copy()
        addmodelstodescr(veckeydescr)
        if values is not None:
            veckeydescr["value"] = values
        veckey = VarKey(**veckeydescr)

        descr["veckey"] = veckey
        vl = np.empty(shape, dtype="object")
        it = np.nditer(vl, flags=["multi_index", "refs_ok"])
        while not it.finished:
            i = it.multi_index
            it.iternext()
            descr["idx"] = i
            if values is not None:
                if callable(values):  # a vector function
                    descr["value"] = veclinkedfn(values, i)
                else:
                    descr["value"] = values[i]
            vl[i] = Variable(**descr)

        obj = np.asarray(vl).view(NomialArray)
        obj.key = veckey
        obj.units = obj.key.units
        return obj
