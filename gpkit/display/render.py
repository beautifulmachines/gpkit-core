"""Rendering an expression tree as text, LaTeX or TOML.

`render` walks the tree and decides where parentheses go, from operator
precedence.  A format (`TEXT`, `LATEX`, `TOML`) supplies the spelling of each
operator, number and leaf.  Variables are named by the caller's `names(ref)`.
"""

import sys
from typing import ClassVar

import numpy as np

from ..ast_nodes import ASTNode, ConstNode, ExprNode, PiNode, UnitsNode, varnodes
from ..units import ureg
from ..util.repr_conventions import (
    MUL,
    PI_STR,
    UNICODE_EXPONENTS,
    extract_subscript,
    merge_subscript,
    unitstr,
)
from ..util.small_classes import Numbers

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


TEXT, LATEX, TOML = Text(), Latex(), Toml()


def render(node, fmt, names, unit_literals=True):
    """node as fmt spells it, naming each variable by names(ref).

    unit_literals=False leaves units off constants, for a caller that prints
    the whole expression's units after it.
    """
    return _Render(fmt, names, unit_literals)(node)[0]


def varkey_names(tree, excluded, latex=False):
    "names(ref) for a live tree: each VarKey named as it names itself."
    keys = {node.ref: node.varkey for node in varnodes(tree)}
    if latex:
        return lambda ref: keys[ref].latex(excluded)
    return lambda ref: keys[ref].str_without(excluded)


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
