"Implements ConstraintSet"

from collections import defaultdict
from itertools import chain

import numpy as np

from ..nomials import NomialArray, Variable
from ..util.repr_conventions import ReprMixin
from ..util.small_scripts import try_str_without
from ..varmap import VarMap, VarSet


# pylint: disable=fixme
def add_meq_bounds(bounded, meq_bounded):  # TODO: collapse with GP version?
    "Iterates through meq_bounds until convergence"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound in list(meq_bounded):
            if bound in bounded:  # bound already exists
                del meq_bounded[bound]
                continue
            for condition in meq_bounded[bound]:
                if condition.issubset(bounded):  # bound's condition is met
                    del meq_bounded[bound]
                    bounded.add(bound)
                    still_alive = True
                    break


def flatiter(iterable, yield_if_hasattr=None):
    "Yields contained constraints, optionally including constraintsets."
    if isinstance(iterable, dict):
        iterable = iterable.values()
    for constraint in iterable:
        if not hasattr(constraint, "__iter__") or (
            yield_if_hasattr and hasattr(constraint, yield_if_hasattr)
        ):
            yield constraint
        else:
            try:  # numpy array
                yield from constraint.flat
            except TypeError:  # ConstrainSet
                yield from constraint.flat(yield_if_hasattr)
            except AttributeError:  # probably a list or dict
                yield from flatiter(constraint, yield_if_hasattr)


class ConstraintSet(list, ReprMixin):  # pylint: disable=too-many-instance-attributes
    "Recursive container for ConstraintSets and Inequalities"

    unique_varkeys, idxlookup = frozenset(), {}
    _name_collision_varkeys = None
    _varkeys = None
    _lineageset = False

    def __init__(
        self, constraints, substitutions=None, *, bonusvks=None
    ):  # pylint: disable=too-many-branches,too-many-statements
        if isinstance(constraints, dict):
            keys, constraints = constraints.keys(), constraints.values()
            self.idxlookup = {k: i for i, k in enumerate(keys)}
        elif isinstance(constraints, ConstraintSet):
            constraints = [constraints]  # put it one level down
        list.__init__(self, constraints)
        self.vks = VarSet(self.unique_varkeys)
        self.substitutions = VarMap(
            {k: k.value for k in self.unique_varkeys if "value" in k.descr}
        )
        self.substitutions.register_keys(self.vks)
        self.bounded, self.meq_bounded = set(), defaultdict(set)
        for i, constraint in enumerate(self):
            if hasattr(constraint, "vks"):
                self._update(constraint)
            elif not (
                hasattr(constraint, "as_hmapslt1") or hasattr(constraint, "as_gpconstr")
            ):
                try:
                    for subconstraint in flatiter(constraint, "vks"):
                        self._update(subconstraint)
                except (TypeError, AttributeError) as e:
                    raise badelement(self, i, constraint) from e
            elif isinstance(constraint, ConstraintSet):
                raise badelement(
                    self, i, constraint, " It had not yet been initialized!"
                )
        if bonusvks:
            self.vks.update(bonusvks)
        if substitutions:
            self.substitutions.update(substitutions)
        for key in self.vks:
            if key not in self.substitutions:
                continue
            self.bounded.add((key, "upper"))
            self.bounded.add((key, "lower"))
            if key.value is not None and not key.constant:
                del key.descr["value"]
                if key.veckey and key.veckey.value is not None:
                    del key.veckey.descr["value"]
        add_meq_bounds(self.bounded, self.meq_bounded)

    def _update(self, constraint):
        "Update parameters with a given constraint"
        self.vks.update(constraint.vks)
        self.substitutions.register_keys(constraint.vks)
        if hasattr(constraint, "substitutions"):
            self.substitutions.update(constraint.substitutions)
        else:
            self.substitutions.update(
                {k: k.value for k in constraint.vks if "value" in k.descr}
            )
        self.bounded.update(constraint.bounded)
        for bound, solutionset in constraint.meq_bounded.items():
            self.meq_bounded[bound].update(solutionset)

    def __getitem__(self, key):
        if key in self.idxlookup:
            key = self.idxlookup[key]
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return self._choosevar(key, self.varkeys.keys(key))

    def _choosevar(self, key, variables):
        if not variables:
            raise KeyError(key)
        firstvar, *othervars = variables
        veckey = firstvar.key.veckey
        if veckey is None or any(v.key.veckey != veckey for v in othervars):
            if not othervars:
                return Variable(firstvar)
            raise ValueError(
                f"multiple variables are called '{key}'; show them"
                f" with `.varkeys.keys({key})`"
            )

        arr = NomialArray(np.full(veckey.shape, np.nan, dtype="object"))
        for v in variables:
            arr[v.key.idx] = Variable(v)
        arr.key = veckey
        return arr

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if self._varkeys is None:
            self._varkeys = VarSet(self.vks)
        return self._varkeys

    def constrained_varkeys(self):
        "Return all varkeys in non-ConstraintSet constraints"
        return self.vks - self.unique_varkeys

    flat = flatiter

    def as_hmapslt1(self, subs):
        "Yields hmaps<=1 from self.flat()"
        yield from chain(
            *(
                c.as_hmapslt1(subs)
                for c in flatiter(self, yield_if_hasattr="as_hmapslt1")
            )
        )

    def process_result(self, result):
        """Does arbitrary computation / manipulation of a program's result

        There's no guarantee what order different constraints will process
        results in, so any changes made to the program's result should be
        careful not to step on other constraint's toes.

        Potential Uses
        --------------
          - check that an inequality was tight
          - add values computed from solved variables

        """
        for constraint in self.flat(yield_if_hasattr="process_result"):
            if hasattr(constraint, "process_result"):
                constraint.process_result(result)
        evalfn_vars = {
            v.veckey or v
            for v in self.unique_varkeys
            if v.evalfn and v not in result.primal and v not in result.constants
        }
        for v in evalfn_vars:
            val = v.evalfn(result.primal)
            result.primal[v] = val

    def __repr__(self):
        "Returns namespaced string."
        if not self:
            return f"<gpkit.{self.__class__.__name__} object>"
        return (
            f"<gpkit.{self.__class__.__name__} object containing "
            f"{len(self)} top-level constraint(s) and "
            f"{len(self.varkeys)} variable(s)>"
        )

    def set_necessarylineage(self, clear=False):  # pylint: disable=too-many-branches
        "Returns the set of contained varkeys whose names are not unique"
        if self._name_collision_varkeys is None:
            self._name_collision_varkeys = {}
            name_collisions = defaultdict(set)
            for key in self.varkeys:
                if hasattr(key, "key"):
                    if key.veckey and all(
                        k.veckey == key.veckey for k in self.varkeys.keys(key.name)
                    ):
                        self._name_collision_varkeys[key] = 0
                        self._name_collision_varkeys[key.veckey] = 0
                    elif len(self.varkeys.keys(key.name)) == 1:
                        self._name_collision_varkeys[key] = 0
                    else:
                        shortname = key.str_without(["lineage", "vec"])
                        if len(self.varkeys.keys(shortname)) > 1:
                            name_collisions[shortname].add(key)
                else:
                    raise ValueError(f"unexpected key {key} has no key attribute")
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

    def lines_without(self, excluded):
        "Lines representation of a ConstraintSet."
        excluded = frozenset(excluded)
        root, rootlines = "root" not in excluded, []
        if root:
            excluded = {"root"}.union(excluded)
            self.set_necessarylineage()
            if hasattr(self, "_rootlines"):
                rootlines = self._rootlines(excluded)  # pylint: disable=no-member
        lines = recursively_line(self, excluded)
        indent = " " if root or getattr(self, "lineage", None) else ""
        if root:
            self.set_necessarylineage(clear=True)
        return rootlines + [(indent + line).rstrip() for line in lines]

    def str_without(self, excluded=("units",)):
        "String representation of a ConstraintSet."
        return "\n".join(self.lines_without(excluded))

    def latex(self, excluded=("units",)):
        "LaTeX representation of a ConstraintSet."
        lines = []
        root = "root" not in excluded
        if root:
            excluded += ("root",)
            lines.append("\\begin{array}{ll} \\text{}")
            if hasattr(self, "_rootlatex"):
                lines.append(self._rootlatex(excluded))  # pylint: disable=no-member
        for constraint in self:
            cstr = try_str_without(constraint, excluded, latex=True)
            if cstr[:6] != "    & ":  # require indentation
                cstr = "    & " + cstr + " \\\\"
            lines.append(cstr)
        if root:
            lines.append("\\end{array}")
        return "\n".join(lines)


def build_model_tree(model, ir_variables):
    """Build model_tree structure from a Model's constraint hierarchy.

    Walks the constraint tree in the same depth-first order as
    collect_flat_constraints_ir, tracking which flat constraint indices
    belong to which model node.

    Parameters
    ----------
    model : ConstraintSet
        The top-level constraint set (typically a Model).
    ir_variables : dict
        The variables dict from the IR document (var_ref -> ir_dict).

    Returns
    -------
    dict
        model_tree with class, instance_id, variables, constraint_indices,
        and children for each model node.
    """
    counter = [0]  # mutable flat constraint index
    all_claimed_vars = set()  # vars claimed by any node

    def _walk(cset):
        "Build tree node for a model/sub-model ConstraintSet."
        lineage = getattr(cset, "lineage", None) or ()
        if lineage:
            class_name, _ = lineage[-1]
            instance_id = ".".join(f"{n}{i}" for n, i in lineage)
            # e.g. "Aircraft0.Wing0"
        else:
            class_name = type(cset).__name__
            instance_id = ""

        constraint_indices = []
        children = []

        _collect(cset, constraint_indices, children)

        owned_vars = sorted(
            vk.var_ref for vk in getattr(cset, "unique_varkeys", frozenset())
        )
        all_claimed_vars.update(owned_vars)

        return {
            "class": class_name,
            "instance_id": instance_id,
            "variables": owned_vars,
            "constraint_indices": constraint_indices,
            "children": children,
        }

    def _collect(iterable, constraint_indices, children):
        """Walk items, mirroring flatiter's traversal order."""
        if isinstance(iterable, dict):
            iterable = iterable.values()

        for item in iterable:
            if isinstance(item, ConstraintSet) and getattr(item, "lineage", None):
                # Sub-model: create child node
                children.append(_walk(item))
            elif not hasattr(item, "__iter__"):
                # Leaf constraint (non-iterable)
                constraint_indices.append(counter[0])
                counter[0] += 1
            else:
                # Iterable: numpy array, list, ArrayConstraint, or
                # ConstraintSet without lineage
                try:
                    flat_items = item.flat
                    if callable(flat_items):
                        # ConstraintSet.flat is flatiter (a bound method);
                        # recurse into the ConstraintSet's items directly
                        _collect(item, constraint_indices, children)
                    else:
                        # numpy flatiter: process each element
                        _collect(flat_items, constraint_indices, children)
                except AttributeError:
                    # list, dict, ArrayConstraint, etc.
                    _collect(item, constraint_indices, children)

    tree = _walk(model)

    # Assign unclaimed variables to the root node (handles flat models
    # without setup() where unique_varkeys is empty)
    unclaimed = sorted(ref for ref in ir_variables if ref not in all_claimed_vars)
    if unclaimed:
        tree["variables"] = sorted(set(tree["variables"]) | set(unclaimed))

    return tree


def recursively_line(iterable, excluded):
    "Generates lines in a recursive tree-like fashion, the better to indent."
    named_constraints = {}
    if isinstance(iterable, dict):
        keys, iterable = iterable.keys(), iterable.values()
        named_constraints = dict(enumerate(keys))
    elif hasattr(iterable, "idxlookup"):
        named_constraints = {i: k for k, i in iterable.idxlookup.items()}
    lines = []
    for i, constraint in enumerate(iterable):
        if hasattr(constraint, "lines_without"):
            clines = constraint.lines_without(excluded)
        elif not hasattr(constraint, "__iter__"):
            clines = try_str_without(constraint, excluded).split("\n")
        elif iterable is constraint:
            clines = ["(constraint contained itself)"]
        else:
            clines = recursively_line(constraint, excluded)
        if getattr(constraint, "lineage", None) and isinstance(
            constraint, ConstraintSet
        ):
            name, num = constraint.lineage[-1]
            if not any(clines):
                clines = [" " + "(no constraints)"]  # named model indent
            if lines:
                lines.append("")
            lines.append(name if not num else name + str(num))
        elif "constraint names" not in excluded and i in named_constraints:
            lines.append(f'"{named_constraints[i]}":')
            clines = ["  " + line for line in clines]  # named constraint indent
        lines.extend(clines)
    return lines


def badelement(cns, i, constraint, cause=""):
    "Identify the bad element and raise a ValueError"
    cause = (
        cause
        if not isinstance(constraint, bool)
        else (" Did the constraint list contain an accidental equality?")
    )
    if len(cns) == 1:
        loc = "the only constraint"
    elif i == 0:
        loc = f"at the start, before {cns[i+1]}"
    elif i == len(cns) - 1:
        loc = f"at the end, after {cns[i-1]}"
    else:
        loc = f"between {cns[i-1]} and {cns[i+1]}"
    return ValueError(
        f"Invalid ConstraintSet element '{constraint!r}' "
        f"{type(constraint)} was {loc}.{cause}"
    )
