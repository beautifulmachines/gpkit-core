"""Rendering nomials, arrays and constraints as text, LaTeX or TOML.

`render` takes a gpkit object; `render_ast` takes an expression tree, walks it,
and decides where parentheses go, from operator precedence.  A format (`TEXT`,
`LATEX`, `LATEX_ALIGNED`, `TOML`) supplies the spelling of each operator,
number and leaf.  `render_ast` names variables by the caller's `names(ref)`.
"""

import sys
from typing import ClassVar

import numpy as np

from ..ast_nodes import (
    ASTNode,
    ConstNode,
    ExprNode,
    PiNode,
    UnitsNode,
    ast_from_ir,
    ast_from_terms,
    variables_of,
    written_ast,
)
from ..units import ureg
from ..util.small_classes import Numbers
from ..varkey import VarKey
from .naming import DisplayScope
from .spelling import (
    MUL,
    PI_STR,
    UNICODE_EXPONENTS,
    extract_subscript,
    merge_subscript,
    unitstr,
)

# Operator precedence, loosest first.  A child is parenthesized when its
# precedence is below what its position needs.  Unary minus is below a product:
# x⋅(-y) needs parentheses, a leading -x⋅y does not.
SUM, NEG, PRODUCT, POWER, ATOM = range(1, 6)

_SUPERSCRIPTS = {2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"}

# A double represents every integer only up to 2**53, so beyond it an integer
# spelling would claim precision the value does not carry -- and TOML integers are
# 64-bit, which a longer literal would overflow.
_LARGEST_EXACT_INTEGRAL_FLOAT = 2**53


def exact_number(v):
    """A number spelled exactly, for a document that is read back by a parser.

    repr of a float is the shortest spelling that reads back as the same value.
    An integral value still prints as an integer, which costs no precision and
    keeps exponents reading as x**2.
    """
    if isinstance(v, int):
        return str(v)
    f = float(v)  # also normalizes numpy scalars, whose repr carries their type
    if f.is_integer() and abs(f) <= _LARGEST_EXACT_INTEGRAL_FLOAT:
        return str(int(f))
    return repr(f)


def _index_spec(idx):
    "An index or slice as Python writes it."
    if isinstance(idx, tuple):
        return ",".join(_index_spec(i) for i in idx)
    if idx is Ellipsis:
        return "..."
    if isinstance(idx, slice):
        start = "" if idx.start is None else idx.start
        stop = "" if idx.stop is None or idx.stop >= sys.maxsize else idx.stop
        step = "" if idx.step is None else f":{idx.step}"
        return f"{start}:{stop}{step}"
    return str(idx)


class Text:
    "gpkit's display syntax: ≥, ⋅, superscript exponents, π, [units]."

    relations: ClassVar = (
        {"<=": "≤", ">=": "≥", "=": "="}
        if UNICODE_EXPONENTS
        else {"<=": "<=", ">=": ">=", "=": "="}
    )
    product = MUL
    reduction_needs = ATOM  # x.sum() binds to an atom

    def relation(self, oper):
        return self.relations[oper]

    def number(self, v):
        "Four significant figures, shared by text and LaTeX."
        return f"{v:.4g}"

    def pi(self):
        return PI_STR

    def units(self, quantity):
        return unitstr(quantity, "[%s]")

    def group(self, text):
        return f"({text})"

    def div_slots(self, _ndens):
        "What a numerator and each denominator factor need to bind to."
        return NEG, POWER

    def div(self, num, dens):
        return "/".join([num, *dens])

    def pow(self, base, exponent):
        if isinstance(exponent, str):
            return f"{base}^{exponent}"
        if UNICODE_EXPONENTS and exponent == int(exponent) in _SUPERSCRIPTS:
            return base + _SUPERSCRIPTS[int(exponent)]
        return f"{base}^{self.number(exponent)}"

    def reduction(self, op, operand):
        return f"{operand}.{op}()"

    def index(self, base, spec):
        base = base.removesuffix("[:]")  # a whole vector, indexed
        return f"{base}[{spec}]"


class Latex(Text):
    "LaTeX math: \\frac, juxtaposed products, braced exponents and subscripts."

    relations: ClassVar = {"<=": r"\leq", ">=": r"\geq", "=": "="}
    product = " "
    reduction_needs = 0  # \operatorname{sum}\left(...\right) delimits its operand

    def number(self, v):
        text = super().number(v)
        if "e" in text:
            mantissa, exponent = text.split("e")
            return rf"{mantissa} \times 10^{{{int(exponent)}}}"
        return text

    def pi(self):
        return r"\pi"

    def units(self, quantity):
        latex = unitstr(quantity, "%s", ":L~")
        return latex.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")

    def group(self, text):
        return rf"\left({text}\right)"

    def div_slots(self, ndens):
        # \frac delimits both parts; only factors set side by side need grouping
        return 0, (0 if ndens == 1 else PRODUCT)

    def div(self, num, dens):
        return rf"\frac{{{num}}}{{{' '.join(dens)}}}"

    def pow(self, base, exponent):
        if not isinstance(exponent, str):
            exponent = (
                str(int(exponent))
                if exponent == int(exponent)
                else self.number(exponent)
            )
        return f"{base}^{{{exponent}}}"

    def reduction(self, op, operand):
        return rf"\operatorname{{{op}}}\left({operand}\right)"

    def index(self, base, spec):
        if extract_subscript(base) is not None:  # one subscript, not two
            return merge_subscript(base, spec)
        return f"{base}_{{{spec}}}"


class Toml(Text):
    "Python expression syntax with exact numbers, for TOML that is read back."

    relations: ClassVar = {"<=": "<=", ">=": ">=", "=": "=="}
    product = "*"
    reduction_needs = 0  # sum(...) delimits its operand

    def number(self, v):
        return exact_number(v)

    def pi(self):
        return exact_number(np.pi)

    def units(self, quantity):
        # Compact, so no spaces; single-quoted, inside a double-quoted TOML string
        return f"units('{quantity.units:~C}')"

    def pow(self, base, exponent):
        if not isinstance(exponent, str):
            exponent = self.number(exponent)
        return f"{base}**{exponent}"

    def reduction(self, op, operand):
        return f"{op}({operand})"

    def index(self, base, spec):
        return f"{base}[{spec}]"


class AlignedLatex(Latex):
    "LaTeX whose relations carry the & an align environment lines them up at."

    def relation(self, oper):
        return "&" + super().relation(oper)


TEXT, LATEX, LATEX_ALIGNED, TOML = Text(), Latex(), AlignedLatex(), Toml()


def render(obj, fmt, excluded=()):
    """A nomial, nomial array or constraint as fmt spells it.

    excluded is a DisplayScope, which names each variable, or a bare set of
    format flags ("units", "lineage", ...), which names each by its full path.
    Anything else, such as a custom constraint class, is shown as its str().
    """
    excluded = DisplayScope.of(excluded)
    if hasattr(obj, "oper"):  # a constraint as written
        sides = excluded.also_excluding("units")  # a relation shows no units
        left, right = (render(side, fmt, sides) for side in (obj.left, obj.right))
        return f"{left} {fmt.relation(obj.oper)} {right}"
    if isinstance(obj, VarKey):  # a vector constraint's side is its vector's key
        return (
            obj.latex(excluded) if isinstance(fmt, Latex) else obj.str_without(excluded)
        )
    if isinstance(obj, np.ndarray):
        return _array(obj, fmt, excluded)
    if hasattr(obj, "hmap"):
        return _nomial(obj, fmt, excluded)
    if isinstance(obj, Numbers):
        return fmt.number(obj)
    return str(obj)


class Rendered:
    "str() and notebook LaTeX through render, for nomials, arrays and constraints."

    def __str__(self):
        return render(self, TEXT)

    def _repr_latex_(self):
        return "$$" + render(self, LATEX) + "$$"


def render_ir(entry, fmt, names):
    """A constraint or nomial entry from an IR document, as fmt spells it.

    names(ref) names each variable.  A constraint with no written form has
    only its type to show.
    """
    if "oper" in entry:
        left, right = (render_ir(entry[side], fmt, names) for side in ("left", "right"))
        return f"{left} {fmt.relation(entry['oper'])} {right}"
    if "ast" in entry:
        return render_ast(ast_from_ir(entry["ast"]), fmt, names)
    if "terms" in entry:
        return render_ast(ast_from_terms(entry["terms"]), fmt, names)
    return f"<{entry['type']}>"


def render_ast(node, fmt, names, unit_literals=True):
    """A tree as fmt spells it, naming each variable by names(ref).

    unit_literals=False leaves units off constants, for a caller that prints
    the whole expression's units after it.
    """
    return _Render(fmt, names, unit_literals)(node)[0]


def names_of(variables, fmt, excluded):
    "names(ref) for {ref: VarKey}: each VarKey named as it names itself."
    if isinstance(fmt, Latex):
        return lambda ref: variables[ref].latex(excluded)
    return lambda ref: variables[ref].str_without(excluded)


def _nomial(nomial, fmt, excluded):
    "A nomial as written, or from its terms, then its units unless excluded."
    latex = isinstance(fmt, Latex)
    if hasattr(nomial, "key"):  # a variable, whose name LaTeX gives no units
        name = nomial.key.latex(excluded) if latex else nomial.key.str_without(excluded)
        return name if latex else name + _units_suffix(nomial, fmt, excluded)
    units = _units_suffix(nomial, fmt, excluded)
    variables = variables_of(nomial)
    names = names_of(variables, fmt, excluded)
    for tree in (
        written_ast(nomial, variables),
        ast_from_terms(nomial.hmap.to_ir()["terms"]),
    ):
        # the terms also stand in when the written form renders empty: it was
        # only unit literals, and the suffix carries those units
        text = "" if tree is None else render_ast(tree, fmt, names, not units)
        if text:
            return text + units
    return units


def _units_suffix(nomial, fmt, excluded):
    "A nomial's units after it, as fmt sets them, unless excluded."
    if "units" in excluded:
        return ""
    if isinstance(fmt, Latex):
        units = nomial.unitstr(r"\mathrm{~\left[ %s \right]}", ":L~")
        return units.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
    return nomial.unitstr(" [%s]")


def _array(array, fmt, excluded):
    "An array as written, as its vector's name, or element by element."
    latex = isinstance(fmt, Latex)
    if array.ast is not None:
        variables = variables_of(array)
        tree = written_ast(array, variables)
        if tree is not None:
            return render_ast(tree, fmt, names_of(variables, fmt, excluded))
    if hasattr(array, "key"):  # a vector variable
        if not latex:
            return array.key.str_without(excluded)
        units = array.latex_unitstr() if "units" not in excluded else ""
        return array.key.latex(excluded) + units
    if latex:
        joined = ", ".join(render(el, fmt, excluded) for el in array.flat)
        return rf"\left[{joined}\right]"
    if not array.shape:
        return render(array.flatten()[0], fmt, excluded)
    rows = (np.ndarray.__getitem__(array, i) for i in range(array.shape[0]))
    return "[" + ", ".join(render(row, fmt, excluded) for row in rows) + "]"


class _Render:
    "Walks a tree; each step returns its text and its precedence."

    def __init__(self, fmt, names, unit_literals):
        self.fmt, self.names, self.unit_literals = fmt, names, unit_literals

    def __call__(self, node):
        if isinstance(node, ExprNode):
            text, prec = getattr(self, "_" + node.op)(*node.children)
        else:
            text, prec = self._leaf(node)
        if text.startswith("-"):  # e.g. a negative number, or -x⋅y
            prec = min(prec, NEG)
        return text, prec

    def slot(self, node, needs):
        "node's text, parenthesized if its precedence is below needs."
        text, prec = self(node)
        return (self.fmt.group(text) if text and prec < needs else text), prec

    def _leaf(self, node):
        fmt = self.fmt
        if hasattr(node, "ref"):
            return self.names(node.ref), ATOM
        if isinstance(node, PiNode):
            return fmt.pi(), ATOM
        if isinstance(node, ConstNode):
            return fmt.number(node.value), ATOM
        if isinstance(node, UnitsNode):
            return (fmt.units(node.units) if self.unit_literals else ""), ATOM
        if hasattr(node, "magnitude"):  # a quantity: its number times its units
            units = UnitsNode(ureg.Quantity(1, node.units))
            return self._mul(node.magnitude, units)
        if isinstance(node, Numbers):
            return fmt.number(node), ATOM
        raise TypeError(f"cannot render {type(node).__name__}: {node!r}")

    def _add(self, left, right):
        lt, _ = self.slot(left, SUM)
        rt, _ = self.slot(right, SUM)
        if rt.startswith("-"):
            return f"{lt} - {rt[1:]}", SUM
        return f"{lt} + {rt}", SUM

    def _mul(self, left, right):
        # drop factors of 1, and unit literals the caller prints itself
        factors = [self.slot(left, NEG), self.slot(right, PRODUCT)]
        kept = [(t, p) for t, p in factors if t not in ("", "1")]
        if len(kept) == 1:
            return kept[0]
        if not kept:
            return ("1" if any(t == "1" for t, _ in factors) else ""), ATOM
        return self.fmt.product.join(t for t, _ in kept), PRODUCT

    def _div(self, left, right):
        num, dens = left, [right]
        while isinstance(num, ExprNode) and num.op == "div":  # x/y/z: one fraction
            num, dens = num.children[0], [num.children[1], *dens]
        num_needs, den_needs = self.fmt.div_slots(len(dens))
        nt, np_ = self.slot(num, num_needs)
        dts = [t for t, _ in (self.slot(d, den_needs) for d in dens)]
        dts = [t for t in dts if t not in ("", "1")]
        if not dts:
            return nt, np_
        return self.fmt.div(nt or "1", dts), PRODUCT

    def _neg(self, operand):
        text, _ = self.slot(operand, PRODUCT)
        return f"-{text}", NEG

    def _pow(self, base, exponent):
        bt, _ = self.slot(base, ATOM)
        if bt in ("", "1"):
            return bt, ATOM
        if isinstance(exponent, ASTNode):
            exponent, _ = self.slot(exponent, ATOM)
        return self.fmt.pow(bt, exponent), POWER

    def _reduction(self, op, operand):
        text, _ = self.slot(operand, self.fmt.reduction_needs)
        return self.fmt.reduction(op, text), ATOM

    def _sum(self, operand):
        return self._reduction("sum", operand)

    def _prod(self, operand):
        return self._reduction("prod", operand)

    def _index(self, base, idx):
        bt, _ = self.slot(base, ATOM)
        return self.fmt.index(bt, _index_spec(idx)), ATOM
