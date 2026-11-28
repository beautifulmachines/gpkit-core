"""Implement the SequentialGeometricProgram class"""

import warnings as pywarnings
from time import time

import numpy as np

from ..exceptions import (
    Infeasible,
    InvalidGPConstraint,
    InvalidPosynomial,
    InvalidSGPConstraint,
    UnnecessarySGP,
)
from ..nomials import Posynomial, PosynomialInequality, Variable
from ..util.globals import NamedVariables
from ..util.small_scripts import appendsolwarning
from ..varmap import VarMap
from .gp import GeometricProgram

EPS = 1e-6  # 1 +/- this is used in a few relative differences


def split_constraints(model, substitutions):
    "model's constraints: (GP-compatible after substitution, signomial)"
    gpconstraints, sgpconstraints = [], []
    for cs in model.flat():
        try:
            if not hasattr(cs, "as_hmapslt1"):
                raise InvalidGPConstraint(cs)
            if not isinstance(cs, PosynomialInequality):
                cs.as_hmapslt1(substitutions)  # gp-compatible?
            gpconstraints.append(cs)
        except InvalidGPConstraint as exc:
            if not hasattr(cs, "as_gpconstr"):
                raise InvalidSGPConstraint(cs) from exc
            sgpconstraints.append(cs)
    return gpconstraints, sgpconstraints


def approximation_point(vks, substitutions, x0=None):
    """The values known when approximating: substitutions, then x0. A signomial
    constraint's shares are measured here only where these values collapse its
    greater-than side to a monomial; elsewhere its guess stands."""
    point = VarMap(substitutions)
    point.register_keys(vks)  # for string access and so forth
    point.update(x0 or {})
    return point


def approximate(sgpconstraints, point):
    "Each signomial constraint's GP stand-in at point, each linked to the other"
    approximated = []
    for cs in sgpconstraints:
        gpc = cs.as_gpconstr(point)
        gpc.generated_by, cs.generated = cs, gpc
        approximated.append(gpc)
    return approximated


def local_gp(cost, model, substitutions, *, x0=None, **kwargs):
    """The GP that stands in for model's signomial constraints: each one's
    greater-than side condensed at its shares (measured at x0 where x0
    determines them, else guessed)."""
    gpconstraints, sgpconstraints = split_constraints(model, substitutions)
    point = approximation_point(model.vks, substitutions, x0)
    unshared = [
        cs
        for cs in sgpconstraints
        if not hasattr(cs, "gt_shares") or cs.gt_shares(point) is None
    ]
    if unshared:
        raise InvalidGPConstraint(
            "no guess, and x0 does not set the shares, of: "
            + "; ".join(str(cs) for cs in unshared)
        )
    approximated = approximate(sgpconstraints, point)
    return GeometricProgram(cost, approximated + gpconstraints, substitutions, **kwargs)


class SequentialGeometricProgram:
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Posynomial
        Objective to minimize when solving
    constraints : list of Constraint or SignomialConstraint objects
        Constraints to maintain when solving (implicitly Signomials <= 1)
    verbosity : int (optional)
        Currently has no effect: SequentialGeometricPrograms don't know
        anything new after being created, unlike GeometricPrograms.

    Attributes with side effects
    ----------------------------
    `gps` is set during a solve
    `result` is set at the end of a solve

    Examples
    --------
    >>> gp = gpkit.geometric_program.SequentialGeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x - y/x,  # <= 1, implicitly
                            y/10  # <= 1
                        ])
    >>> gp.solve()
    """

    gps = solver_outs = _results = result = model = None
    with NamedVariables("RelaxPCCP"):
        slack = Variable("C", "-")

    def __init__(
        self, cost, model, substitutions, *, use_pccp=None, pccp_penalty=2e2, **kwargs
    ):
        self.cost = cost
        self.pccp_penalty = pccp_penalty
        if cost.any_nonpositive_cs:
            raise InvalidPosynomial("""an SGP's cost must be Posynomial

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable.""")
        gpconstraints, self.sgpconstraints = split_constraints(model, substitutions)
        # Fully guessed, the first GP is the modeler's own conservative design:
        # if it is infeasible, say so rather than slacken it away
        if use_pccp is None:
            use_pccp = not all(
                getattr(cs, "guess", None) is not None for cs in self.sgpconstraints
            )
        if not use_pccp:
            self.slack = 1
            self.gpconstraints = gpconstraints
        else:
            self.gpconstraints = [self.slack >= 1, *gpconstraints]
        if not self.sgpconstraints:
            raise UnnecessarySGP("""Model valid as a Geometric Program.

SequentialGeometricPrograms should only be created with Models containing
Signomial Constraints, since Models without Signomials have global
solutions and can be solved with 'Model.solve()'.""")
        self._gpcost = cost * self.slack**pccp_penalty
        self.vks, self.substitutions, self.gpkwargs = model.vks, substitutions, kwargs
        self._gp = self.gp()  # the first GP

    def localsolve(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        solver=None,
        *,
        verbosity=1,
        x0=None,
        reltol=1e-4,
        iteration_limit=50,
        err_on_relax=True,
        **solveargs,
    ):
        """Locally solves a SequentialGeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solve time and number of iterations.
            Each GP is created and solved with verbosity one less than this, so
            if greater than 1, prints solver name and time for each GP.
        x0 : dict (optional)
            Initial location to approximate signomials about.
        reltol : float
            Iteration ends when this is greater than the distance between two
            consecutive solve's objective values.
        iteration_limit : int
            Maximum GP iterations allowed.
        **solveargs :
            Passed to solver function.

        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        self.gps, self.solver_outs, self._results = [], [], []
        starttime = time()
        if verbosity > 0:
            print("Starting a sequence of GP solves")
            print(f" for {len(self.sgpconstraints)} signomial constraints,")
            print(f"  {len(self._gp.vars)} free variables")
            print(f"  and {len(self._gp.data.k)} posynomial inequalities.")
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.gps) > iteration_limit:
                raise Infeasible(
                    f"Unsolved after {len(self.gps)} iterations. "
                    "Check `m.program.results`; if they're converging, try "
                    "`.localsolve(..., iteration_limit=NEWLIMIT)`."
                )
            gp = self.gp(x0)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            if verbosity > 1:
                print(f"\nGP Solve {len(self.gps)}")
            if verbosity > 2:
                print("===============")
            solver_out = gp.solve(
                solver, verbosity=verbosity - 1, gen_result=False, **solveargs
            )
            self.solver_outs.append(solver_out)
            cost = float(solver_out.cost)
            x0 = dict(zip(gp.vars, np.exp(solver_out.x)))
            if verbosity > 2:
                result = gp.generate_result(solver_out, verbosity=verbosity - 3)
                self._results.append(result)
                # future: restore showvars filtering for SGP convergence display
                # showvars was removed from PrintOptions; restore for this use case
                vartable = result.table(tables=["freevariables"])
                vartable = "\n" + vartable.replace("Free", "SGP", 1)
                print(vartable)
            elif verbosity > 1:
                print(f"Solved cost was {cost:.4g}.")
            if prevcost is None:
                continue
            rel_improvement = (prevcost - cost) / (prevcost + cost)
            if cost / prevcost >= 1 + 10 * EPS:
                pywarnings.warn(
                    "SGP not convergent: Cost rose by "
                    f"{100 * (cost - prevcost) / prevcost:.2g}% "
                    f"({prevcost:.6g} to {cost:.6g}) on GP solve {len(self.gps)}. "
                    "Details can be found in `m.program.results` "
                    "or by solving at a higher verbosity. Note convergence "
                    "is not guaranteed for models with SignomialEqualities."
                )
                rel_improvement = cost = None
        # solved successfully!
        # Wire model so generate_result can find margin_objective.
        # Long-term fix: pass margin_obj explicitly (see issue #143).
        gp.model = self.model
        self.result = gp.generate_result(solver_out, verbosity=verbosity - 3)
        self.result.meta["soltime"] = time() - starttime
        if verbosity > 1:
            print()
        if verbosity > 0:
            print(
                f"Solving took {self.result.meta['soltime']:.3g} seconds and "
                f"{len(self.gps)} GP solves."
            )
        if hasattr(self.slack, "key"):
            excess_slack = self.result.primal[self.slack.key] - 1
            if excess_slack > EPS:
                msg = (
                    "Final PCCP solution let non-GP constraints slacken by"
                    f" {(100 * excess_slack):.2g}%."
                )
                expl = (
                    msg + " Calling .localsolve(pccp_penalty=...) with a higher"
                    f" `pccp_penalty` (it was {self.pccp_penalty:.3g} this time)"
                    " will reduce slack if the model is solvable with less. "
                    "To verify that the slack is needed, generate an SGP with"
                    " `use_pccp=False` and start it from this model's"
                    "  solution: e.g. `m.localsolve(use_pccp=False, x0="
                    'm.solution["variables"])`.'
                )
                if err_on_relax:
                    raise Infeasible(expl)
                appendsolwarning(
                    msg,
                    self.result,
                    "Slack Non-GP Constraints",
                    value=1 + excess_slack,
                )
                if verbosity > -1:
                    print(expl)
            del self.result.primal[self.slack.key]
            assert hasattr(self.result, "sens")  # not true for MIGP
            del self.result.sens.variables[self.slack.key]
            del self.result.sens.variablerisk[self.slack.key]
        return self.result

    @property
    def results(self):
        "Creates and caches results from the raw solver_outs"
        if not self._results:
            self._results = [
                gp.generate_result(s_o, dual_check=False)
                for gp, s_o in zip(self.gps, self.solver_outs)
            ]
        return self._results

    def gp(self, x0=None):
        "The GP approximating this SGP about x0, built fresh"
        point = approximation_point(self.vks, self.substitutions, x0)
        approximated = approximate(self.sgpconstraints, point)
        if hasattr(self.slack, "key"):  # PCCP: each may slacken by C
            approximated = [
                self._slackened(Posynomial(hmaplt1), gpc.generated_by)
                for gpc in approximated
                for hmaplt1 in gpc.as_hmapslt1({})
            ]
        return GeometricProgram(
            self._gpcost,
            approximated + self.gpconstraints,
            self.substitutions,
            **self.gpkwargs,
        )

    def _slackened(self, posy, sgpconstraint):
        "posy <= C, standing in for sgpconstraint"
        constraint = posy <= self.slack
        constraint.generated_by = sgpconstraint
        return constraint
