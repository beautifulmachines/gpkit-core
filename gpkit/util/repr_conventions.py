"Repository for representation standards"

import re
import sys

import numpy as np

from .small_classes import Numbers, Quantity
from .small_scripts import try_str_without

INSIDE_PARENS = re.compile(r"\(.*\)")

if sys.platform[:3] == "win":  # pragma: no cover
    MUL = "*"
    PI_STR = "PI"
    UNICODE_EXPONENTS = False
    UNIT_FORMATTING = ":~"
else:  # pragma: no cover
    MUL = "·"
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


def latexify(name: str) -> str:
    """Convert a variable name to a LaTeX base string.

    - Pure Greek: 'rho' -> r'\\rho'
    - Underscore split: 'm_wet' -> r'm_{\\text{wet}}'
    - Greek + underscore: 'rho_inf' -> r'\\rho_{\\infty}'
    - Already escaped (starts with '\\'): return as-is
    """
    if name.startswith("\\"):
        return name  # already LaTeX — do not double-convert
    if "_" in name:
        parts = name.split("_", 1)
        base = _GREEK.get(parts[0], parts[0])
        if "_" in parts[1] or parts[1] in _GREEK:
            sub = latexify(parts[1])  # nested/Greek → already math, no \text{}
        else:
            sub = r"\text{" + parts[1] + "}"
        return base + "_{" + sub + "}"
    return _GREEK.get(name, name)


def strify(val, excluded):
    "Turns a value into as pretty a string as possible."
    if isinstance(val, Numbers):
        isqty = hasattr(val, "magnitude")
        if isqty:
            units = val
            val = val.magnitude
        if np.pi / 12 < val < 100 * np.pi and abs(12 * val / np.pi % 1) <= 1e-2:
            # val is in bounds and a clean multiple of PI!
            if val > 3.1:  # product of PI
                val = f"{val/np.pi:.3g}{PI_STR}"
                if val == f"1{PI_STR}":
                    val = PI_STR
            else:  # division of PI
                val = f"({PI_STR}/{np.pi/val:.3g})"
        else:
            val = f"{val:.3g}"
        if isqty:
            val += unitstr(units, " [%s]")
    else:
        val = try_str_without(val, excluded)
    return val


def parenthesize(string, addi=True, mult=True):
    "Parenthesizes a string if it needs it and isn't already."
    parensless = string if "(" not in string else INSIDE_PARENS.sub("", string)
    bare_addi = " + " in parensless or " - " in parensless
    bare_mult = "·" in parensless or "/" in parensless
    if parensless and (addi and bare_addi) or (mult and bare_mult):
        return f"({string})"
    return string


def _render_add(children, excluded):
    left = strify(children[0], excluded)
    right = strify(children[1], excluded)
    if right and right[0] == "-":
        return f"{left} - {right[1:]}"
    return f"{left} + {right}"


def _render_mul(children, excluded):
    left = parenthesize(strify(children[0], excluded), mult=False)
    right = parenthesize(strify(children[1], excluded), mult=False)
    if left in ("1", ""):
        return right
    if right in ("1", ""):
        return left
    return f"{left}{MUL}{right}"


def _render_div(children, excluded):
    left = parenthesize(strify(children[0], excluded), mult=False)
    right = parenthesize(strify(children[1], excluded))
    if right in ("1", ""):
        return left
    return f"{left}/{right}"


def _render_neg(children, excluded):
    val = parenthesize(strify(children[0], excluded), mult=False)
    return f"-{val}"


def _render_pow(children, excluded):
    left = parenthesize(strify(children[0], excluded))
    x = children[1]
    if left == "1":
        return "1"
    if (
        UNICODE_EXPONENTS
        and not getattr(x, "shape", None)
        and int(x) == x
        and 2 <= x <= 9
    ):
        x = int(x)
        if x in (2, 3):
            return f"{left}{chr(176 + x)}"
        return f"{left}{chr(8304 + x)}"
    return f"{left}^{x}"


def _render_prod(children, excluded):
    val = parenthesize(strify(children[0], excluded))
    return f"{val}.prod()"


def _render_sum(children, excluded):
    val = parenthesize(strify(children[0], excluded))
    return f"{val}.sum()"


def _fmt_slice(s):
    "Format a slice object as a string."
    start = s.start or ""
    stop = s.stop if s.stop and s.stop < sys.maxsize else ""
    step = f":{s.step}" if s.step is not None else ""
    return f"{start}:{stop}{step}"


def _render_index(children, excluded):
    left = parenthesize(strify(children[0], excluded))
    idx = children[1]
    if left[-3:] == "[:]":  # pure variable access
        left = left[:-3]
    if isinstance(idx, tuple):
        elstrs = []
        for el in idx:
            if isinstance(el, slice):
                elstrs.append(_fmt_slice(el))
            elif isinstance(el, Numbers):
                elstrs.append(str(el))
        idx = ",".join(elstrs)
    elif isinstance(idx, slice):
        idx = _fmt_slice(idx)
    return f"{left}[{idx}]"


_AST_RENDERERS = {
    "add": _render_add,
    "mul": _render_mul,
    "div": _render_div,
    "neg": _render_neg,
    "pow": _render_pow,
    "prod": _render_prod,
    "sum": _render_sum,
    "index": _render_index,
}


def _render_ast_node(node, excluded):
    "Renders an ExprNode as a string.  Called by ExprNode.str_without()."
    renderer = _AST_RENDERERS.get(node.op)
    if renderer is None:
        raise ValueError(f"Unknown AST op: {node.op}")
    return renderer(node.children, excluded)


# ── LaTeX AST renderers ─────────────────────────────────────────────────────


def _latex_strify(val, excluded):
    "Render an AST child as LaTeX (parallel to strify for text)."
    if hasattr(val, "latex"):
        return val.latex(excluded)
    if isinstance(val, Numbers):
        return f"{val:.4g}"
    return str(val)


def _is_compound(node):
    "True if node is an ExprNode (needs parens in some contexts)."
    return hasattr(node, "op")


def _latex_paren(latex_str, node):
    r"Wrap in \left(...\right) if node is a compound expression."
    if _is_compound(node):
        return rf"\left({latex_str}\right)"
    return latex_str


def _collect_frac_terms(node):
    """Walk a left-nested div chain, collecting numerator/denominator terms.

    Only recurses on the LEFT child (matching Python's left-associative /).
    Right children are appended to denominator. Explicit grouping like
    A/(B/C) is preserved as nested frac (right child is not flattened).
    """
    if _is_compound(node) and node.op == "div":
        num_terms, den_terms = _collect_frac_terms(node.children[0])
        den_terms.append(node.children[1])
        return num_terms, den_terms
    return [node], []


def _collect_mul_terms(node):
    "Walk a left-nested mul chain, collecting all factors."
    if _is_compound(node) and node.op == "mul":
        return _collect_mul_terms(node.children[0]) + _collect_mul_terms(
            node.children[1]
        )
    return [node]


def _latex_render_mul_term(term, excluded):
    "Render a single factor in a product, parenthesizing additions."
    s = _latex_strify(term, excluded)
    if _is_compound(term) and term.op in ("add", "neg"):
        return rf"\left({s}\right)"
    return s


def _latex_render_add(children, excluded):
    left = _latex_strify(children[0], excluded)
    right = _latex_strify(children[1], excluded)
    if right and right[0] == "-":
        return f"{left} - {right[1:]}"
    return f"{left} + {right}"


def _latex_render_mul(children, excluded):
    all_terms = []
    for child in children:
        all_terms.extend(_collect_mul_terms(child))
    parts = [s for t in all_terms if (s := _latex_render_mul_term(t, excluded))]
    return " ".join(parts) if parts else ""


def _latex_render_div(children, excluded):
    num_terms, den_terms = _collect_frac_terms(children[0])
    den_terms.append(children[1])
    num_parts = [s for t in num_terms if (s := _latex_render_mul_term(t, excluded))]
    num_str = " ".join(num_parts) if num_parts else "1"
    den_parts = [s for t in den_terms if (s := _latex_strify(t, excluded))]
    den_str = " ".join(den_parts) if den_parts else "1"
    return rf"\frac{{{num_str}}}{{{den_str}}}"


def _latex_render_pow(children, excluded):
    base_latex = _latex_strify(children[0], excluded)
    exp = children[1]
    if _is_compound(children[0]):
        base_latex = rf"\left({base_latex}\right)"
    if isinstance(exp, int) or (isinstance(exp, float) and exp == int(exp)):
        return f"{base_latex}^{{{int(exp)}}}"
    return f"{base_latex}^{{{exp:.4g}}}"


def _latex_render_neg(children, excluded):
    inner = _latex_strify(children[0], excluded)
    if _is_compound(children[0]) and children[0].op not in ("div",):
        inner = rf"\left({inner}\right)"
    return f"-{inner}"


def _latex_render_sum(children, excluded):
    val = _latex_strify(children[0], excluded)
    return rf"\operatorname{{sum}}\left({val}\right)"


def _latex_render_prod(children, excluded):
    val = _latex_strify(children[0], excluded)
    return rf"\operatorname{{prod}}\left({val}\right)"


def _latex_render_index(children, excluded):
    left = _latex_strify(children[0], excluded)
    idx = children[1]
    if isinstance(idx, tuple):
        elstrs = []
        for el in idx:
            if isinstance(el, slice):
                elstrs.append(_fmt_slice(el))
            elif isinstance(el, Numbers):
                elstrs.append(str(el))
        idx = ",".join(elstrs)
    elif isinstance(idx, slice):
        idx = _fmt_slice(idx)
    return f"{left}_{{{idx}}}"


_LATEX_AST_RENDERERS = {
    "add": _latex_render_add,
    "mul": _latex_render_mul,
    "div": _latex_render_div,
    "neg": _latex_render_neg,
    "pow": _latex_render_pow,
    "prod": _latex_render_prod,
    "sum": _latex_render_sum,
    "index": _latex_render_index,
}


def _render_ast_node_latex(node, excluded):
    "Renders an ExprNode as LaTeX.  Called by ExprNode.latex()."
    renderer = _LATEX_AST_RENDERERS.get(node.op)
    if renderer is None:
        raise ValueError(f"Unknown AST op: {node.op}")
    return renderer(node.children, excluded)


class ReprMixin:
    "This class combines various printing methods for easier adoption."

    lineagestr = lineagestr
    unitstr = unitstr
    latex_unitstr = latex_unitstr

    cached_strs = None
    ast = None

    def parse_ast(self, excluded=()):
        "Turns the AST of this object's construction into a faithful string"
        if self.ast is None:
            return self.str_without(excluded)  # pylint: disable=no-member
        excluded = frozenset({"units"}.union(excluded))
        if self.cached_strs is None:
            self.cached_strs = {}
        elif excluded in self.cached_strs:
            return self.cached_strs[excluded]
        aststr = self.ast.str_without(excluded)
        self.cached_strs[excluded] = aststr
        return aststr

    def __repr__(self):
        "Returns namespaced string."
        return f"gpkit.{self.__class__.__name__}({self})"

    def __str__(self):
        "Returns default string."
        return self.str_without()  # pylint: disable=no-member

    def _repr_latex_(self):
        "Returns default latex for automatic iPython Notebook rendering."
        return "$$" + self.latex() + "$$"  # pylint: disable=no-member
