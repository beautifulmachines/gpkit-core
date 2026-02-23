"Implements Model"

import json
from pathlib import Path
from time import time

import numpy as np

from .constraints.costed import CostedConstraintSet
from .constraints.set import build_model_tree, flatiter
from .exceptions import Infeasible, InvalidGPConstraint
from .globals import NamedVariables
from .nomials import Monomial
from .nomials.math import constraint_from_ir, nomial_from_ir
from .programs.gp import GeometricProgram
from .programs.prog_factories import progify, solvify
from .programs.sgp import SequentialGeometricProgram
from .solutions import SolutionSequence
from .tools.autosweep import autosweep_1d
from .var import Var
from .varkey import VarKey
from .varmap import VarMap


class Model(CostedConstraintSet):
    """Symbolic representation of an optimization problem.

    The Model class is used both directly to create models with constants and
    sweeps, and indirectly inherited to create custom model classes.

    Arguments
    ---------
    cost : Posynomial (optional)
        Defaults to `Monomial(1)`.

    constraints : ConstraintSet or list of constraints (optional)
        Defaults to an empty list.

    substitutions : dict (optional)
        This dictionary will be substituted into the problem before solving,
        and also allows the declaration of sweeps and linked sweeps.

    Attributes with side effects
    ----------------------------
    `program` is set during a solve
    `solution` is set at the end of a solve
    """

    program = None
    solution = None
    computed = None  # dict of {VarKey: fn(solution) -> value} for post-solve

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._own_var_fields = tuple(
            v for v in cls.__dict__.values() if isinstance(v, Var)
        )

    def __init__(self, cost=None, constraints=None, *args, **kwargs):
        # pylint: disable=keyword-arg-before-vararg
        setup_vars = None
        substitutions = kwargs.pop("substitutions", None)  # reserved keyword
        if hasattr(self, "setup"):
            self.cost = None
            # lineage holds the (name, num) environment a model was created in,
            # including its own (name, num), and those of models above it
            with NamedVariables(self.__class__.__name__) as (self.lineage, setup_vars):
                # instantiate Var descriptors before setup() so self.x works
                seen = set()
                for klass in type(self).__mro__:
                    for var in getattr(klass, "_own_var_fields", ()):
                        if var._name not in seen:
                            seen.add(var._name)
                            var._create(self)
                args = (
                    tuple(arg for arg in [cost, constraints] if arg is not None) + args
                )
                cs = self.setup(*args, **kwargs)  # pylint: disable=no-member
                if isinstance(cs, tuple) and len(cs) == 2 and isinstance(cs[1], dict):
                    constraints, substitutions = cs
                else:
                    constraints = cs
            cost = self.cost
        elif args and not substitutions:
            # backwards compatibility: substitutions as third argument
            (substitutions,) = args

        cost = cost or Monomial(1)
        constraints = constraints or []
        if setup_vars:
            # add all the vars created in .setup to the Model's varkeys
            # even if they aren't used in any constraints
            self.unique_varkeys = frozenset(v.key for v in setup_vars)
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        self.computed = {}  # {VarKey: fn(solution)} for post-solve computation

    def process_result(self, result):
        "Evaluate computed variables and add to result.primal"
        super().process_result(result)
        for var, fn in self.computed.items():
            key = getattr(var, "key", var)
            result.primal[key] = fn(result)

    def to_ir(self):
        "Serialize this Model to a complete IR document dict."
        # Collect all variables (including veckeys for vector variables)
        all_vks = set(self.vks)
        variables = {}
        for vk in sorted(all_vks, key=lambda v: v.ref):
            variables[vk.ref] = vk.to_ir()
            if vk.veckey and vk.veckey.ref not in variables:
                variables[vk.veckey.ref] = vk.veckey.to_ir()

        # Serialize cost
        cost_ir = self.cost.to_ir()

        # Collect flat constraint list
        constraints_ir = [c.to_ir() for c in flatiter(self)]

        # Serialize substitutions (skip callables)
        subs_ir = {}
        for vk, val in self.substitutions.items():
            if callable(val):
                continue
            subs_ir[vk.ref] = float(val)

        ir = {
            "gpkit_ir_version": "1.0",
            "variables": variables,
            "cost": cost_ir,
            "constraints": constraints_ir,
        }
        if subs_ir:
            ir["substitutions"] = subs_ir

        # Phase 5: structural metadata for nested/composable models
        ir["model_tree"] = build_model_tree(self)

        return ir

    @classmethod
    def from_ir(cls, ir_doc):
        """Reconstruct a solvable Model from an IR document dict.

        Parameters
        ----------
        ir_doc : dict
            Complete IR document with variables, cost, constraints, and
            optional substitutions.

        Returns
        -------
        Model
            A flat Model (no nested sub-models) that can be solved.
        """
        # 1. Reconstruct var_registry
        var_registry = {}
        for ref, vk_ir in ir_doc["variables"].items():
            vk = VarKey.from_ir(vk_ir)
            var_registry[ref] = vk

        # 2. Reconstruct cost
        cost = nomial_from_ir(ir_doc["cost"], var_registry)

        # 3. Reconstruct constraints
        constraints = [
            constraint_from_ir(c_ir, var_registry) for c_ir in ir_doc["constraints"]
        ]

        # 4. Reconstruct substitutions
        subs = None
        if "substitutions" in ir_doc:
            subs = {}
            for ref, val in ir_doc["substitutions"].items():
                if ref in var_registry:
                    subs[var_registry[ref]] = val

        return cls(cost, constraints, substitutions=subs)

    def save(self, path):
        """Write this Model's IR to a JSON file."""
        Path(path).write_text(json.dumps(self.to_ir(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Load a Model from a JSON IR file."""
        ir_doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_ir(ir_doc)

    gp = progify(GeometricProgram)
    solve = solvify(progify(GeometricProgram, "solve"))

    sp = progify(SequentialGeometricProgram)
    localsolve = solvify(progify(SequentialGeometricProgram, "localsolve"))

    def sweep(self, sweepvals, skipfailures=False, **solveargs):
        "Sweeps {var: values} in-sync across one dim. Returns SolutionSequence"
        return self._sweep(self.solve, sweepvals, skipfailures, **solveargs)

    def localsweep(self, sweepvals, skipfailures=False, **solveargs):
        "Sweeps {var: values} in-sync across one dim. Returns SolutionSequence"
        return self._sweep(self.localsolve, sweepvals, skipfailures, **solveargs)

    def _validate_sweep(self, sweepvals):
        "Validate and return {VarKey: iterable} mapping"
        # this logic might eventually live in VarMap
        for var, vals in sweepvals.items():
            keys = self.varkeys.keys(var)
            for key in keys:
                if not key.shape:
                    continue
                # next line raises a ValueError if size mismatch
                np.broadcast(vals, np.empty(key.shape))
        return sweepvals

    def _sweep(self, solvefn, sweepvals, skipfailures, **solveargs):
        "Runs sweep using solvefn (either self.solve or self.localsolve)"
        sols = SolutionSequence()
        self._validate_sweep(sweepvals)
        lengths = {len(vals) for vals in sweepvals.values()}
        if len(lengths) != 1:
            raise ValueError(f"sweepvals has mismatched lengths {lengths}")
        oldsubs = self.substitutions
        tic = time()
        for i in range(lengths.pop()):
            sweepsubs = {var: vals[i] for var, vals in sweepvals.items()}
            self.substitutions = {**oldsubs, **sweepsubs}
            try:
                sols.append(solvefn(**solveargs))
                sp = VarMap()
                sp.register_keys(self.varkeys)
                sp.update(sweepsubs)
                sols[-1].meta["sweep_point"] = sp
            except Infeasible as err:
                if not skipfailures:
                    raise RuntimeWarning(
                        f"Solve {i} was infeasible. To continue sweeping after"
                        "failures, pass skipfailures=True to Model.sweep."
                    ) from err
        self.substitutions = oldsubs
        if not sols:
            raise RuntimeWarning("All solves were infeasible.")

        if solveargs.get("verbosity", 1) > 0:
            print(f"Sweeping took {time() - tic:.3g} seconds.")

        sols.modelstr = str(self)
        return sols

    def autosweep(self, sweeps, tol=0.01, samplepoints=100, **solveargs):
        """Autosweeps {var: (start, end)} pairs in sweeps to tol.

        Returns swept and sampled solutions.
        The original simplex tree can be accessed at sol.bst
        """
        sols = []
        for sweepvar, sweepvals in sweeps.items():
            sweepvar = self[sweepvar].key
            start, end = sweepvals
            bst = autosweep_1d(self, tol, sweepvar, [start, end], **solveargs)
            sols.append(bst.sample_at(np.linspace(start, end, samplepoints)))
        return sols if len(sols) > 1 else sols[0]

    # pylint: disable=import-outside-toplevel
    def debug(self, solver=None, verbosity=1, **solveargs):
        "Attempts to diagnose infeasible models."
        from .constraints.bounded import Bounded
        from .constraints.relax import ConstantsRelaxed, ConstraintsRelaxed

        sol = None
        solveargs["solver"] = solver
        solveargs["verbosity"] = verbosity - 1
        solveargs["process_result"] = False

        bounded = Bounded(self)
        tants = ConstantsRelaxed(bounded)
        if tants.relaxvars.size:
            feas = Model(tants.relaxvars.prod() ** 30 * self.cost, tants)
        else:
            feas = Model(self.cost, bounded)

        try:
            try:
                sol = feas.solve(**solveargs)
            except InvalidGPConstraint:
                sol = feas.sp(use_pccp=False).localsolve(**solveargs)
            # limited results processing
            bounded.check_boundaries(sol)
            tants.check_relaxed(sol)
        except Infeasible:
            if verbosity:
                print(
                    "<DEBUG> Model is not feasible with relaxed constants"
                    " and bounded variables."
                )
            traints = ConstraintsRelaxed(self)
            feas = Model(traints.relaxvars.prod() ** 30 * self.cost, traints)
            try:
                try:
                    sol = feas.solve(**solveargs)
                except InvalidGPConstraint:
                    sol = feas.sp(use_pccp=False).localsolve(**solveargs)
                # limited results processing
                traints.check_relaxed(sol)
            except Infeasible:
                print("<DEBUG> Model is not feasible with bounded constraints.")
        if sol and verbosity:
            warnings = sol.table(tables=["warnings"]).split("\n")[2:]
            if warnings and warnings != ["(none)"]:
                print("<DEBUG> Model is feasible with these modifications:")
                print("\n" + "\n".join(warnings) + "\n")
            else:
                print(
                    "<DEBUG> Model seems feasible without modification,"
                    " or only needs relaxations of less than 1%."
                    " Check the returned solution for details."
                )
        return sol
