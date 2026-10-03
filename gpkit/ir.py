"The IR documents produced by Model.to_ir() and Solution.to_ir()."

from .util.repr_conventions import unitstr

# Nothing validates this yet, and it has not tracked the schema changes made so
# far -- see the issue on versioning policy before relying on it.
IR_VERSION = "1.0"


def ir_units(obj) -> str:
    """The unit string an IR document carries, for anything bearing units.

    One spelling per unit however it was declared, and readable back by the unit
    registry.  Neither property holds of the alternatives: VarKey.unitrepr keeps
    the user's own spelling (m^2, m ** 2 and m*m all survive, by design, so TOML
    round-trips), and the display form substitutes a middle dot for products
    ("m⋅N") that no registry parses.
    """
    return unitstr(obj, "%s", ":~")
