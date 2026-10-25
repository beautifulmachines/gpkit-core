"Repository for representation standards"

import sys

from .small_classes import Quantity

if sys.platform[:3] == "win":  # pragma: no cover
    MUL = "*"
    PI_STR = "PI"
    UNICODE_EXPONENTS = False
    UNIT_FORMATTING = ":~"
else:  # pragma: no cover
    MUL = "\u22c5"  # DOT OPERATOR (U+22C5)
    PI_STR = "π"
    UNICODE_EXPONENTS = True
    UNIT_FORMATTING = ":P~"


def lineagestr(lineage, modelnums=True):
    "Returns properly formatted lineage string"
    if not isinstance(lineage, tuple):
        lineage = getattr(lineage, "lineage", None)
    return (
        ".".join(
            [f"{name}{num}" if (num and modelnums) else name for name, num in lineage]
        )
        if lineage
        else ""
    )


def also_excluding(excluded, *flags):
    """Add format flags to a render context.

    Accepts either a bare collection of flags or a DisplayScope, and returns
    the same kind, so a scope survives being narrowed mid-render.  Duck-typed
    rather than imported: DisplayScope is built on this module.
    """
    if hasattr(excluded, "also_excluding"):
        return excluded.also_excluding(*flags)
    return frozenset(excluded).union(flags)


def unitstr(units, into="%s", options=UNIT_FORMATTING, dimless=""):
    "Returns the string corresponding to an object's units."
    if hasattr(units, "units") and isinstance(units.units, Quantity):
        units = units.units
    if not isinstance(units, Quantity):
        return dimless
    if options == ":~" and "ohm" in str(units.units):  # pragma: no cover
        rawstr = str(units.units)  # otherwise it'll be a capital Omega
    else:
        rawstr = ("{%s}" % options).format(units.units)
    # pint's Pretty formatter used MIDDLE DOT (\u00b7) as its product
    # separator before pint 0.26, and switched to DOT OPERATOR (\u22c5,
    # our MUL) from 0.26 on. Normalize the old glyph so unit strings are
    # stable regardless of which pint version is installed.
    rawstr = rawstr.replace("\u00b7", MUL)
    units = rawstr.replace(" ", "").replace("dimensionless", dimless)
    return into % units if units else dimless


def latex_unitstr(units):
    "Returns latex unitstr"
    us = unitstr(units, r"~\mathrm{%s}", ":L~")
    utf = us.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
    return utf if utf != r"~\mathrm{-}" else ""


_GREEK = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "eta": r"\eta",
    "theta": r"\theta",
    "lambda": r"\lambda",
    "lamda": r"\lambda",  # Python keyword workaround
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "pi": r"\pi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "phi": r"\phi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
    "zeta": r"\zeta",
    "kappa": r"\kappa",
    "inf": r"\infty",
    "infty": r"\infty",
    # Uppercase
    "Gamma": r"\Gamma",
    "Delta": r"\Delta",
    "Theta": r"\Theta",
    "Lambda": r"\Lambda",
    "Xi": r"\Xi",
    "Pi": r"\Pi",
    "Sigma": r"\Sigma",
    "Phi": r"\Phi",
    "Psi": r"\Psi",
    "Omega": r"\Omega",
}


_LATEX_PREFIXES = {
    # trig / math functions
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "ln",
    "exp",
    # LaTeX accents
    "dot",
    "ddot",
    # operator-prefix Greeks
    "Delta",
    "Sigma",
}


def _latexify_sub_token(token: str) -> str:
    "Render a single subscript token as a LaTeX atom (Greek symbol or \\text{})."
    result = _GREEK.get(token)
    if result:
        return result
    return token if len(token) == 1 else r"\text{" + token + "}"


def extract_subscript(name: str):
    """If LaTeX string *name* ends with ``_{content}``, return ``(base, content)``.

    Uses brace counting so nested groups (e.g. ``\\text{foo}``) are handled
    correctly.  Returns ``None`` if no trailing ``_{...}`` subscript is found.
    """
    if not name.endswith("}"):
        return None
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        if name[i] == "}":
            depth += 1
        elif name[i] == "{":
            depth -= 1
            if depth == 0:
                if i >= 1 and name[i - 1] == "_":
                    return name[: i - 1], name[i + 1 : -1]
                return None
    return None


def _latexify_symbol(raw: str, bold: bool) -> str:
    "A name's base symbol: Greek, one italic letter, or a multi-letter name set upright."
    if raw in _GREEK:
        return r"\boldsymbol{%s}" % _GREEK[raw] if bold else _GREEK[raw]
    if len(raw) == 1:
        return r"\boldsymbol{%s}" % raw if bold else raw
    # \boldsymbol leaves \text regular, so an upright name bolds as \textbf
    return (r"\textbf{%s}" if bold else r"\text{%s}") % raw


def merge_subscript(name: str, sub: str) -> str:
    "Append sub to name's existing LaTeX subscript, or give it a new one."
    extracted = extract_subscript(name)
    if extracted is not None:
        base_name, existing = extracted
        return "%s_{%s,%s}" % (base_name, existing, sub)
    return "{%s}_{%s}" % (name, sub)


def latexify(name: str, bold: bool = False) -> str:
    """Convert a variable name to a LaTeX base string.

    - Pure Greek: 'rho' -> r'\\rho'
    - Underscore split: 'm_wet' -> r'm_{\\text{wet}}'
    - Greek + underscore: 'rho_inf' -> r'\\rho_{\\infty}'
    - Multi-part subscript: 'A_one_two' -> r'A_{\\text{one},\\text{two}}'
    - Math function + arg: 'tan_alpha' -> r'\\tan{\\alpha}'
    - Already escaped (starts with '\\'): return as-is

    *bold* sets the base symbol bold, as a vector's is (ISO 80000-2): operators,
    accents and subscripts stay regular.  A name written in LaTeX is not parsed,
    so it is bolded whole.
    """
    if name.startswith("\\"):  # already LaTeX — do not double-convert
        return r"\boldsymbol{%s}" % name if bold else name
    if "_" in name:
        parts = name.split("_", 1)
        if parts[0] in _LATEX_PREFIXES:
            arg = latexify(parts[1], bold)
            return "\\" + parts[0] + "{" + arg + "}"
        base = _latexify_symbol(parts[0], bold)
        if not parts[1]:  # trailing underscore (e.g. "lambda_") — strip it
            return base
        # Flatten all underscore-separated tokens into a single comma-separated
        # subscript.  This avoids nested structures like A_{b_{\text{c}}}:
        # each token is rendered independently as a Greek symbol or \text{}.
        sub = ",".join(_latexify_sub_token(t) for t in parts[1].split("_"))
        return base + "_{" + sub + "}"
    return _latexify_symbol(name, bold)


class ReprMixin:
    "This class combines various printing methods for easier adoption."

    lineagestr = lineagestr
    unitstr = unitstr
    latex_unitstr = latex_unitstr

    ast = None

    def __repr__(self):
        "Returns namespaced string."
        return f"gpkit.{self.__class__.__name__}({self})"

    def __str__(self):
        "Returns default string."
        return self.str_without()

    def _repr_latex_(self):
        "Returns default latex for automatic iPython Notebook rendering."
        return "$$" + self.latex() + "$$"
