"Scripts for generating, solving and sweeping programs"

import warnings as pywarnings

import numpy as np
from adce import adnumber

from ..globals import SignomialsEnabled
from ..nomials.substitution import parse_linked, parse_subs
from ..util.small_classes import FixedScalar
from ..util.small_scripts import maybe_flatten
from ..varmap import VarMap


def evaluate_linked(constants, linked):
    # pylint: disable=too-many-branches
    "Evaluates the values and gradients of linked variables."
    kdc = VarMap({k: adnumber(maybe_flatten(v), k) for k, v in constants.items()})
    kdc_plain = None
    array_calulated = {}
    for key in constants:  # remove gradients from constants
        key.descr["gradients"] = None
    for v, f in linked.items():
        try:
            if v.veckey and v.veckey.vecfn:
                if v.veckey not in array_calulated:
                    with SignomialsEnabled():  # to allow use of gpkit.units
                        vecout = v.veckey.vecfn(kdc)
                    if not hasattr(vecout, "shape"):
                        vecout = np.array(vecout)
                    array_calulated[v.veckey] = vecout
                out = array_calulated[v.veckey][v.idx]
            else:
                with SignomialsEnabled():  # to allow use of gpkit.units
                    out = f(kdc)
            if isinstance(out, FixedScalar):  # to allow use of gpkit.units
                out = out.value
            if hasattr(out, "units"):
                out = out.to(v.units or "dimensionless").magnitude
            elif out != 0 and v.units:
                pywarnings.warn(
                    f"Linked function for {v} did not return a united value."
                    " Modifying it to do so (e.g. by using `()` instead of `[]`"
                    " to access variables) will reduce errors."
                )
            out = maybe_flatten(out)
            if not hasattr(out, "x"):
                constants[v] = out
                continue  # a new fixed variable, not a calculated one
            constants[v] = out.x
            v.descr["gradients"] = {
                adn.tag: grad for adn, grad in out.d().items() if adn.tag
            }
        except Exception as exception:  # pylint: disable=broad-except
            from ..globals import settings  # pylint: disable=import-outside-toplevel

            if settings.get("ad_errors_raise", None):
                raise
            if kdc_plain is None:
                kdc_plain = VarMap(constants)
            constants[v] = f(kdc_plain)
            v.descr["gradients"] = None
            print(
                "Warning: skipped auto-differentiation of linked variable"
                f" {v} because {exception!r} was raised. Set `gpkit.settings"
                '["ad_errors_raise"] = True` to raise such Exceptions'
                " directly.\n"
            )
            if (
                "Automatic differentiation not yet supported for <class "
                "'gpkit.nomials.math.Monomial'> objects"
            ) in str(exception):
                print(
                    "This particular warning may have come from using"
                    f" gpkit.units.* in the function for {v}; try using"
                    " gpkit.ureg.* or gpkit.units.*.units instead."
                )


def progify(program, return_attr=None):
    """Generates function that returns a program() and optionally an attribute.

    Arguments
    ---------
    program: NomialData
        Class to return, e.g. GeometricProgram or SequentialGeometricProgram
    return_attr: string
        attribute to return in addition to the program
    """

    def programfn(self, constants=None, **initargs):
        "Return program version of self"
        if not constants:
            constants = parse_subs(self.varkeys, self.substitutions)
            linked = parse_linked(self.varkeys, self.substitutions)
            if linked:
                evaluate_linked(constants, linked)
        prog = program(self.cost, self, constants, **initargs)
        prog.model = self  # NOTE SIDE EFFECTS
        if return_attr:
            return prog, getattr(prog, return_attr)
        return prog

    return programfn


def solvify(genfunction):
    "Returns function for making/solving/sweeping a program."

    def solvefn(self, solver=None, *, verbosity=1, **kwargs):
        """Forms a mathematical program and attempts to solve it.

        Arguments
        ---------
        solver : string or function (default None)
            If None, uses the default solver found in installation.
        verbosity : int (default 1)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to programs.
        **kwargs : Passed to solve and program init calls

        Returns
        -------
        sol : Solution

        Raises
        ------
        ValueError if the program is invalid.
        RuntimeWarning if an error occurs in solving or parsing the solution.
        """
        # NOTE SIDE EFFECTS: self.program and self.solution set below
        self.program, progsolve = genfunction(self, **kwargs)
        result = progsolve(solver, verbosity=verbosity, **kwargs)
        if kwargs.get("process_result", True):
            self.process_result(result)
        self.solution = result
        self.solution.meta["modelstr"] = str(self)
        return result

    return solvefn
