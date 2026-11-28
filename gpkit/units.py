"wraps pint in gpkit monomials"

from typing import ClassVar

import pint

ureg = pint.UnitRegistry()
ureg.define("USD = [money] = $")
pint.set_application_registry(ureg)
Quantity = ureg.Quantity
DimensionalityError = pint.DimensionalityError
QTY_CACHE = {}


def qty(unit):
    "Returns a Quantity, caching the result for future retrievals"
    if unit not in QTY_CACHE:
        QTY_CACHE[unit] = unit if isinstance(unit, Quantity) else Quantity(1, unit)
    return QTY_CACHE[unit]


def refusal(key, value):
    "The error for a value that can't fix key"
    return ValueError(f"{key} must be fixed to a finite number, not {value!r}")


def to_magnitude(key, value):
    "value in key's units, if it was written with units: a Quantity or constant nomial"
    units = key.units or "dimensionless"
    try:
        if hasattr(value, "hmap"):
            if any(value.hmap.keys()):
                raise refusal(key, value)
            (value,) = value.hmap.to(units).values()
        elif isinstance(value, Quantity):
            value = value.to(units).magnitude
    except DimensionalityError as err:
        err.extra_msg += f" while fixing {key}"
        raise
    return value


class GPkitUnits:
    "Return Monomials instead of Quantitites"

    division_cache: ClassVar = {}
    multiplication_cache: ClassVar = {}
    monomial_cache: ClassVar = {}

    def __call__(self, unity):
        "Returns a unit Monomial, caching the result for future retrievals"
        from .nomials import Monomial  # noqa: PLC0415

        if unity not in self.monomial_cache:
            self.monomial_cache[unity] = Monomial(qty(unity))
        return self.monomial_cache[unity]

    __getattr__ = __call__

    def of_division(self, numerator, denominator):
        "Cached unit division. Requires Quantity inputs."
        if numerator.units is denominator.units:
            return 1
        key = (id(numerator.units), id(denominator.units))
        try:
            return self.division_cache[key]
        except KeyError:
            if numerator.units and denominator.units:
                conversion = numerator.units / denominator.units
            else:
                conversion = numerator.units or 1 / denominator.units
            try:
                self.division_cache[key] = float(conversion)
            except DimensionalityError as exc:
                raise DimensionalityError(numerator, denominator) from exc
        return self.division_cache[key]

    def of_product(self, thing1, thing2):
        "Cached unit division. Requires united inputs."
        key = (thing1.units, thing2.units)
        try:
            return self.multiplication_cache[key]
        except KeyError:
            mul_units = qty((thing1 * thing2).units)
            try:
                self.multiplication_cache[key] = (None, float(mul_units))
            except DimensionalityError:
                self.multiplication_cache[key] = (mul_units, None)
        return self.multiplication_cache[key]


units = GPkitUnits()
of_division = units.of_division
of_product = units.of_product
