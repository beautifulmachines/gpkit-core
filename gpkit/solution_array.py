# pylint: disable=too-many-lines
"""Defines SolutionArray class"""

import sys
import warnings as pywarnings
from collections import defaultdict

import numpy as np

from .breakdowns import Breakdowns
from .nomials import NomialArray
from .units import Quantity
from .util.small_classes import DictOfLists, SolverLog


def bdtable_gen(key):
    "Generator for breakdown tablefns"

    def bdtable(self, _showvars, **_):
        "Cost breakdown plot"
        bds = Breakdowns(self)
        original_stdout = sys.stdout
        try:
            sys.stdout = SolverLog(original_stdout, verbosity=0)
            bds.plot(key)
        finally:
            lines = sys.stdout.lines()
            sys.stdout = original_stdout
        return lines

    return bdtable


def cast(function, val1, val2):
    "Relative difference between val1 and val2 (positive if val2 is larger)"
    with pywarnings.catch_warnings():  # skip those pesky divide-by-zeros
        pywarnings.simplefilter("ignore")
        if hasattr(val1, "shape") and hasattr(val2, "shape"):
            if val1.ndim == val2.ndim:
                return function(val1, val2)
            lessdim, dimmest = sorted([val1, val2], key=lambda v: v.ndim)
            dimdelta = dimmest.ndim - lessdim.ndim
            add_axes = (slice(None),) * lessdim.ndim + (np.newaxis,) * dimdelta
            if dimmest is val1:
                return function(dimmest, lessdim[add_axes])
            if dimmest is val2:
                return function(lessdim[add_axes], dimmest)
        return function(val1, val2)


class SolutionArray(DictOfLists):
    """A dictionary (of dictionaries) of lists, with convenience methods.

    Items
    -----
    cost : array
    variables: dict of arrays
    sensitivities: dict containing:
        monomials : array
        posynomials : array
        variables: dict of arrays
    localmodels : NomialArray
        Local power-law fits (small sensitivities are cut off)

    Example
    -------
    >>> import gpkit
    >>> import numpy as np
    >>> x = gpkit.Variable("x")
    >>> x_min = gpkit.Variable("x_{min}", 2)
    >>> sol = gpkit.Model(x, [x >= x_min]).solve(verbosity=0)
    >>>
    >>> # VALUES
    >>> values = [sol[x], sol.subinto(x), sol["variables"]["x"]]
    >>> assert all(np.array(values) == 2)
    >>>
    >>> # SENSITIVITIES
    >>> senss = [sol.sens(x_min), sol.sens(x_min)]
    >>> senss.append(sol["sensitivities"]["variables"]["x_{min}"])
    >>> assert all(np.array(senss) == 1)
    """

    modelstr = ""
    _name_collision_varkeys = None
    _lineageset = False

    def set_necessarylineage(self, clear=False):  # pylint: disable=too-many-branches
        "Returns the set of contained varkeys whose names are not unique"
        if self._name_collision_varkeys is None:
            self._name_collision_varkeys = {}
            varset = self["variables"].varset
            name_collisions = defaultdict(set)
            for key in varset:
                if len(varset.by_name(key.name)) == 1:  # unique
                    self._name_collision_varkeys[key] = 0
                else:
                    shortname = key.str_without(["lineage", "vec"])
                    if len(varset.by_name(shortname)) > 1:
                        name_collisions[shortname].add(key)
            for varkeys in name_collisions.values():
                min_namespaced = defaultdict(set)
                for vk in varkeys:
                    *_, mineage = vk.lineagestr().split(".")
                    min_namespaced[(mineage, 1)].add(vk)
                while any(len(vks) > 1 for vks in min_namespaced.values()):
                    for key, vks in list(min_namespaced.items()):
                        if len(vks) <= 1:
                            continue
                        del min_namespaced[key]
                        mineage, idx = key
                        idx += 1
                        for vk in vks:
                            lineages = vk.lineagestr().split(".")
                            submineage = lineages[-idx] + "." + mineage
                            min_namespaced[(submineage, idx)].add(vk)
                for (_, idx), vks in min_namespaced.items():
                    (vk,) = vks
                    self._name_collision_varkeys[vk] = idx
        if clear:
            self._lineageset = False
            for vk in self._name_collision_varkeys:
                del vk.descr["necessarylineage"]
        else:
            self._lineageset = True
            for vk, idx in self._name_collision_varkeys.items():
                vk.descr["necessarylineage"] = idx

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1
        except KeyError:
            return 0

    def __call__(self, posy):
        posy_subbed = self.subinto(posy)
        return getattr(posy_subbed, "c", posy_subbed)

    def almost_equal(self, other, reltol=1e-3):
        "Checks for almost-equality between two solutions"
        if isinstance(other, SolutionArray):
            svars, ovars = self["variables"], other["variables"]
        else:
            svars, ovars = self["freevariables"], other.primal
        svks, ovks = set(svars), set(ovars)
        if svks != ovks:
            return False
        for key in svks:
            reldiff = np.max(abs(cast(np.divide, svars[key], ovars[key]) - 1))
            if reldiff >= reltol:
                return False
        return True

    def subinto(self, posy):
        "Returns NomialArray of each solution substituted into posy."
        if posy in self["variables"]:
            (clean_key, val) = self["variables"].item(posy)
            return Quantity(val, clean_key.units or "dimensionless")

        if not hasattr(posy, "sub"):
            raise ValueError(f"no variable '{posy}' found in the solution")

        if len(self) > 1:
            return NomialArray(
                [self.atindex(i).subinto(posy) for i in range(len(self))]
            )

        return posy.sub(self["variables"], require_positive=False)

    def _parse_showvars(self, showvars):
        showvars_out = set()
        for k in showvars:
            key, _ = self["variables"].item(k)
            key = getattr(key, "veckey", None) or key
            showvars_out.add(key)
        return showvars_out
