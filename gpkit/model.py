"Implements Model"

import json
from pathlib import Path
from time import time

import numpy as np

from .constraints.costed import CostedConstraintSet
from .constraints.set import build_model_tree, flatiter
from .exceptions import (
    AmbiguousVariable,
    Infeasible,
    InvalidGPConstraint,
    VariableNotFound,
)
from .nomials import Monomial, Variable
from .nomials.map import DIMLESS_QUANTITY
from .nomials.math import constraint_from_ir, nomial_from_ir
from .programs.gp import GeometricProgram
from .programs.prog_factories import progify, solvify
from .programs.sgp import SequentialGeometricProgram
from .solutions import SolutionSequence
from .tools.autosweep import autosweep_1d
from .util.globals import NamedVariables, Vectorize
from .var import Var
from .varkey import VarKey
from .varmap import VarMap


class Model(CostedConstraintSet):  # pylint: disable=too-many-instance-attributes
    # Model carries GP state (cost, lineage, unique_varkeys, computed),
    # tree state (_children, _child_attrs), and report state (cgroups,
    # vectorized_block). Tracked in github.com/beautifulmachines/gpkit-core/issues/172.
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

    @classmethod
    def default(cls):
        """Return a ready-to-solve instance (assumes cost set; all Vars bounded)."""
        return cls()

    def is_gp(self):
        """Return True if the model contains no signomial constraints."""
        return not any(hasattr(cs, "as_gpconstr") for cs in self.flat())

    def __init__(self, cost=None, constraints=None, *args, **kwargs):
        # pylint: disable=keyword-arg-before-vararg
        setup_vars = None
        substitutions = kwargs.pop("substitutions", None)  # reserved keyword
        # True if created inside a Vectorize context — report uses this to
        # collapse N same-type sibling sections into one labeled section.
        self.vectorized_block = bool(Vectorize.vectorization)
        # Initialize _children and _child_attrs unconditionally so that flat
        # Model(cost, constraints) calls also have the attribute (empty list).
        self._children = []
        self._child_attrs = {}

        # Collect direct child Model instances from a constraints structure.
        # Recurse into lists and dicts but NOT into arbitrary ConstraintSet
        # instances — only direct Model instances count.
        def _scan_for_children(items):
            if isinstance(items, Model):
                if items not in self._children:
                    self._children.append(items)
            elif isinstance(items, dict):
                for item in items.values():
                    _scan_for_children(item)
            elif isinstance(items, list):
                for item in items:
                    _scan_for_children(item)

        if hasattr(self, "setup"):
            self.cost = None
            # lineage holds the (name, num) environment a model was created in,
            # including its own (name, num), and those of models above it
            with NamedVariables(self.__class__.__name__) as (self.lineage, setup_vars):
                self._instantiate_var_descriptors()
                args = (
                    tuple(arg for arg in [cost, constraints] if arg is not None) + args
                )
                cs = self.setup(*args, **kwargs)  # pylint: disable=no-member
                if isinstance(cs, tuple) and len(cs) == 2 and isinstance(cs[1], dict):
                    constraints, substitutions = cs
                else:
                    constraints = cs
            cost = self.cost
            # Named constraint groups from dict setup(); None for list setup().
            # Use `is None` checks — empty dict is a valid groups map.
            if isinstance(constraints, dict):
                self.cgroups = dict(constraints)
            else:
                self.cgroups = None
            _scan_for_children(constraints)
            # Map attribute names to child models (for get_var() path resolution)
            for attr_name, val in list(self.__dict__.items()):
                if isinstance(val, Model) and val in self._children:
                    self._child_attrs[attr_name] = val
        else:
            if args and not substitutions:
                # backwards compatibility: substitutions as third argument
                (substitutions,) = args
            self.cgroups = None
            _scan_for_children(constraints or [])

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

    def _instantiate_var_descriptors(self):
        """Create Var descriptor fields on self before setup() runs."""
        seen = set()
        for klass in type(self).__mro__:
            for var in getattr(klass, "_own_var_fields", ()):
                if var.name not in seen:
                    seen.add(var.name)
                    var.create(self)

    @property
    def submodels(self):
        """Direct child models in setup() definition order."""
        return list(self._children)

    def walk(self):
        """Yield all descendant models depth-first."""
        for child in self._children:
            yield child
            yield from child.walk()

    @classmethod
    def description(cls):
        """Return model description metadata.

        Returns a dict with keys: summary (str), assumptions (list[str]),
        references (list[str]).  Override in subclasses to provide structured
        descriptions.  The base implementation falls back to the class docstring
        for the summary field.
        """
        return {
            "summary": (cls.__doc__ or "").strip(),
            "assumptions": list(getattr(cls, "assumptions", [])),
            "references": list(getattr(cls, "references", [])),
        }

    def get_var(self, path: str):
        """Resolve a dotted attribute path to a Variable object.

        Parameters
        ----------
        path : str
            Dotted path using setup() attribute names, e.g. "wing.S" or "S".
            The first segment is an attribute name set via self.wing = Wing()
            in setup(). The last segment is a variable name.

        Returns
        -------
        Variable
            The Variable object at the resolved path.

        Raises
        ------
        VariableNotFound
            If no child matches the first segment, or no variable matches the leaf.
        AmbiguousVariable
            If multiple variables match the leaf name in this model's unique_varkeys.
        """
        parts = path.split(".")
        if len(parts) == 1:
            # Leaf: resolve in this model's own unique_varkeys only
            name = parts[0]
            matches = self.varkeys.by_name(name) & self.unique_varkeys
            if not matches:
                # VectorVariable fallback: varkeys.by_name(name) returns the
                # veckey (not element keys), so the intersection with
                # unique_varkeys (which contains only element keys) is empty.
                # Collect all element VarKeys whose veckey shares the name.
                vec_element_matches = {
                    vk
                    for vk in self.unique_varkeys
                    if getattr(getattr(vk, "veckey", None), "name", None) == name
                }
                if vec_element_matches:
                    # Delegate to _choosevar which handles NomialArray assembly.
                    # Use vec_element_matches (scoped to unique_varkeys) rather
                    # than varkeys.keys(name) which would include child models.
                    return self._choosevar(name, list(vec_element_matches))
            if not matches:
                cls = self.__class__.__name__
                raise VariableNotFound(
                    f"No variable '{name}' in {cls}. "
                    f"Variables: {sorted(vk.name for vk in self.unique_varkeys)}"
                )
            if len(matches) > 1:
                cls = self.__class__.__name__
                raise AmbiguousVariable(
                    f"'{name}' is ambiguous in {cls}: "
                    f"{sorted(vk.str_without() for vk in matches)}"
                )
            return Variable(next(iter(matches)))
        # Dotted path: first segment is child attribute name
        head, rest = parts[0], ".".join(parts[1:])
        if head not in self._child_attrs:
            cls = self.__class__.__name__
            available = sorted(self._child_attrs.keys())
            raise VariableNotFound(
                f"No child attribute '{head}' in {cls}. "
                f"Available children: {available}"
            )
        return self._child_attrs[head].get_var(rest)

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
            if hasattr(
                val, "hmap"
            ):  # constant Monomial or future GPkitUnit wrapper (#156)
                assert not any(
                    val.hmap.keys()
                ), "substitution value should not contain variables"
                (val,) = val.hmap.to(vk.units or DIMLESS_QUANTITY).values()
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

    @classmethod
    def report_preamble(cls) -> str:
        """Return optional markdown prose prepended before this model's report section.

        Override in Model subclasses to attach section-specific context to a
        specific part of a hierarchical report.  The returned string is placed
        before the section heading (and before the model's description/variables
        tables).

        Example::

            class Wing(Model):
                @classmethod
                def report_preamble(cls):
                    return "This section covers aerodynamic and structural wing sizing."

        For preambles that depend on model-specific counts (free variables,
        constraints, objective), use :func:`gpkit.report.feasibility_block`
        and :func:`gpkit.report.sensitivities_block` instead, and pass the
        result via the *front_matter* argument of :meth:`report`.
        """
        return ""

    def report(self, solution=None, fmt="text", front_matter="", toc=False):
        """Build a hierarchical report for this model.

        Parameters
        ----------
        solution : Solution, optional
            If provided, variable tables include solved values and sensitivities.
        fmt : str
            Output format: "dict", "text", "md", or "latex".
        front_matter : str, optional
            Raw text/markdown prepended before the entire report (before the
            root model's heading).  Combine with
            :func:`gpkit.report.feasibility_block` and
            :func:`gpkit.report.sensitivities_block` to add standard GP
            explanatory text::

                from gpkit.report import feasibility_block, sensitivities_block
                m.report(sol, fmt="md",
                         front_matter=feasibility_block(m) + "\\n\\n"
                                      + sensitivities_block())
        toc : bool, optional
            If True, a table-of-contents marker is inserted (Markdown only).
        """
        # pylint: disable=import-outside-toplevel
        from .report import build_report_ir, render_report

        ir = build_report_ir(
            self, solution=solution, front_matter=front_matter, toc=toc
        )
        return render_report(ir, fmt=fmt)

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
            if isinstance(sweepvar, str):
                sweepvar = self.get_var(sweepvar).key
            elif hasattr(sweepvar, "key"):
                sweepvar = sweepvar.key
            # else: sweepvar is already a VarKey
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
