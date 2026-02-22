"Implements the Var class-level variable descriptor"

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from .nomials.variables import Variable as _Variable

_RESERVED_NAMES = frozenset(
    {"cost", "lineage", "setup", "substitutions", "unique_varkeys", "vks"}
)


class Var:
    """Class-level variable declaration for Model subclasses.

    Usage
    -----
        class Wing(Model):
            W = Var("lbf", "wing weight")
            S = Var("ft^2", "wing area")
            A = Var("-", "aspect ratio", default=27)

            upper_unbounded = ("W",)
            lower_unbounded = ("S",)

            def setup(self):
                return [self.W >= self.S * self.rho]

    Notes
    -----
    - Lineage is correct because _create() runs inside NamedVariables context.
    - Reserved names (cost, lineage, setup, substitutions, unique_varkeys, vks)
      and names starting with '_var_' cannot be used.
    - For vector variables whose length comes from a setup() argument, use
      Vectorize inside setup() with VectorizableVariable() directly.
    - Flat (script-form) models should use VectorizableVariable() directly.
    - When a Model subclass is instantiated inside a Vectorize(N) context, all
      Var descriptors automatically become N-element vector variables.
    """

    def __init__(self, units: str, label: str = "", *, default=None, latex: str = None):
        self.units = units
        self.label = label
        self.default = default
        self.latex = latex
        self._name: str | None = None

    def __set_name__(self, owner, name):
        if name.startswith("_var_") or name in _RESERVED_NAMES:
            raise ValueError(
                f"Var name '{name}' is reserved. "
                f"Reserved names: {sorted(_RESERVED_NAMES)} "
                f"and any name starting with '_var_'."
            )
        self._name = name

    @overload
    def __get__(self, obj: None, objtype: type) -> "Var": ...

    @overload
    def __get__(self, obj: object, objtype: type) -> "_Variable": ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(f"_var_{self._name}")

    def _create(self, obj):
        """Create the Variable instance on obj, inside a NamedVariables context."""
        from .nomials.variables import VectorizableVariable

        key = f"_var_{self._name}"
        if key in obj.__dict__:
            raise RuntimeError(
                f"Variable '{self._name}' already initialized on {obj!r}. "
                "Possible duplicate Var declarations in the MRO."
            )
        args = [self._name]
        if self.default is not None:
            args.append(self.default)
        args.extend([self.units, self.label])
        v = VectorizableVariable(*args)
        obj.__dict__[key] = v
        return v
