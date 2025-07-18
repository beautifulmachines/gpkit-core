"""Implement the SequentialGeometricProgram class"""

import warnings as pywarnings
from collections import defaultdict
from time import time

import numpy as np

from ..exceptions import (
    Infeasible,
    InvalidGPConstraint,
    InvalidPosynomial,
    InvalidSGPConstraint,
    UnnecessarySGP,
)
from ..globals import NamedVariables
from ..nomials import Posynomial, PosynomialInequality, Variable
from ..util.small_scripts import appendsolwarning, initsolwarning
from ..varmap import VarMap
from .gp import GeometricProgram

EPS = 1e-6  # 1 +/- this is used in a few relative differences


# pylint: disable=too-many-instance-attributes
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
        slack = Variable("C")

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self, cost, model, substitutions, *, use_pccp=True, pccp_penalty=2e2, **kwargs
    ):
        self.cost = cost
        self.pccp_penalty = pccp_penalty
        if cost.any_nonpositive_cs:
            raise InvalidPosynomial(
                """an SGP's cost must be Posynomial

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable."""
            )
        self.gpconstraints, self.sgpconstraints = [], []
        if not use_pccp:
            self.slack = 1
        else:
            self.gpconstraints.append(self.slack >= 1)
        cost *= self.slack**pccp_penalty
        self.approxconstraints = []
        self.sgpvks = set()
        x0 = VarMap(substitutions)
        x0.register_keys(model.vks)  # for string access and so forth
        for cs in model.flat():
            try:
                if not hasattr(cs, "as_hmapslt1"):
                    raise InvalidGPConstraint(cs)
                if not isinstance(cs, PosynomialInequality):
                    cs.as_hmapslt1(substitutions)  # gp-compatible?
                self.gpconstraints.append(cs)
            except InvalidGPConstraint as exc:
                if not hasattr(cs, "as_gpconstr"):
                    raise InvalidSGPConstraint(cs) from exc
                self.sgpconstraints.append(cs)
                for hmaplt1 in cs.as_gpconstr(x0).as_hmapslt1({}):
                    constraint = Posynomial(hmaplt1) <= self.slack
                    constraint.generated_by = cs
                    self.approxconstraints.append(constraint)
                    self.sgpvks.update(constraint.vks)
        if not self.sgpconstraints:
            raise UnnecessarySGP(
                """Model valid as a Geometric Program.

SequentialGeometricPrograms should only be created with Models containing
Signomial Constraints, since Models without Signomials have global
solutions and can be solved with 'Model.solve()'."""
            )
        self._gp = GeometricProgram(
            cost, self.approxconstraints + self.gpconstraints, substitutions, **kwargs
        )
        self._gp.x0 = x0
        self.a_idxs = defaultdict(list)
        last_cost_mon = self._gp.k[0]
        first_gp_mon = sum(self._gp.k[: 1 + len(self.approxconstraints)])
        for row_idx, m_idx in enumerate(self._gp.A.row):
            if last_cost_mon <= m_idx <= first_gp_mon:
                self.a_idxs[self._gp.p_idxs[m_idx]].append(row_idx)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=too-many-arguments
    def localsolve(
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
        mutategp: boolean
            Prescribes whether to mutate the previously generated GP
            or to create a new GP with every solve.
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
            print(f" for {len(self.sgpvks)} free variables")
            print(f"  in {len(self.sgpconstraints)} locally-GP constraints")
            print(f"  and for {len(self._gp.varlocs)} free variables")
            print(f"       in {len(self._gp.k)} posynomial inequalities.")
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.gps) > iteration_limit:
                raise Infeasible(
                    f"Unsolved after {len(self.gps)} iterations. "
                    "Check `m.program.results`; if they're converging, try "
                    "`.localsolve(..., iteration_limit=NEWLIMIT)`."
                )
            gp = self.gp(x0, cleanx0=len(self.gps) >= 1)  # clean the first x0
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            if verbosity > 1:
                print(f"\nGP Solve {len(self.gps)}")
            if verbosity > 2:
                print("===============")
            solver_out = gp.solve(
                solver, verbosity=verbosity - 1, gen_result=False, **solveargs
            )
            self.solver_outs.append(solver_out)
            cost = float(solver_out["objective"])
            x0 = dict(zip(gp.varlocs, np.exp(solver_out["primal"])))
            if verbosity > 2:
                result = gp.generate_result(solver_out, verbosity=verbosity - 3)
                self._results.append(result)
                vartable = result.table(self.sgpvks, tables=["freevariables"])
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
                    f"{100 * (cost - prevcost) / prevcost:.2g}%% "
                    f"({prevcost:.6g} to {cost:.6g}) on GP solve {len(self.gps)}. "
                    "Details can be found in `m.program.results` "
                    "or by solving at a higher verbosity. Note convergence "
                    "is not guaranteed for models with SignomialEqualities."
                )
                rel_improvement = cost = None
        # solved successfully!
        self.result = gp.generate_result(solver_out, verbosity=verbosity - 3)
        self.result["soltime"] = time() - starttime
        if verbosity > 1:
            print()
        if verbosity > 0:
            print(
                f"Solving took {self.result['soltime']:.3g} seconds and "
                f"{len(self.gps)} GP solves."
            )
        if hasattr(self.slack, "key"):
            initsolwarning(self.result, "Slack Non-GP Constraints")
            excess_slack = (
                self.result["variables"][self.slack.key] - 1
            )  # pylint: disable=no-member
            if excess_slack > EPS:
                msg = (
                    "Final PCCP solution let non-GP constraints slacken by"
                    f" {(100 * excess_slack):.2g}%%."
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
                    msg, (1 + excess_slack), self.result, "Slack Non-GP Constraints"
                )
                if verbosity > -1:
                    print(expl)
            self.result["cost function"] = self.cost
            del self.result["freevariables"][
                self.slack.key
            ]  # pylint: disable=no-member
            del self.result["variables"][self.slack.key]  # pylint: disable=no-member
            if "sensitivities" in self.result:  # not true for MIGP
                del self.result["sensitivities"]["variables"][
                    self.slack.key
                ]  # pylint: disable=no-member
                del self.result["sensitivities"]["variablerisk"][
                    self.slack.key
                ]  # pylint: disable=no-member
                slcon = self.gpconstraints[0]
                slconsenss = self.result["sensitivities"]["constraints"][slcon]
                del self.result["sensitivities"]["constraints"][slcon]
                # pylint: disable=fixme
                # TODO: create constraint in RelaxPCCP namespace
                self.result["sensitivities"]["models"][""] -= slconsenss
                if not self.result["sensitivities"]["models"][""]:
                    del self.result["sensitivities"]["models"][""]
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

    def _update_a_matrix(self, m_idx, hmap, a_idxs):
        """Update A matrix entries for a given monomial.

        Arguments
        ---------
        m_idx : int
            Starting index for this monomial
        hmap : dict
            Monomial mapping
        a_idxs : list
            List of A matrix indices that can be modified
        """
        for i, (exp, c) in enumerate(hmap.items()):
            self._gp.exps[m_idx + i] = exp
            self._gp.cs[m_idx + i] = c
            for var, x in exp.items():
                try:  # modify a particular A entry
                    row_idx = a_idxs.pop()
                    self._gp.A.row[row_idx] = m_idx + i
                    self._gp.A.col[row_idx] = self._gp.varidxs[var]
                    self._gp.A.data[row_idx] = x
                except IndexError:  # numbers of exps increased
                    a_idxs.append(len(self._gp.A.row))
                    self._gp.A.row.append(m_idx + i)
                    self._gp.A.col.append(self._gp.varidxs[var])
                    self._gp.A.data.append(x)
            for row_idx in a_idxs:  # number of exps decreased
                self._gp.A.row[row_idx] = 0  # zero out this entry
                self._gp.A.col[row_idx] = 0
                self._gp.A.data[row_idx] = 0

    def gp(self, x0=None, *, cleanx0=False):
        "Update self._gp for x0 and return it."
        if not x0:
            return self._gp  # return last generated
        if not cleanx0:
            cleanedx0 = VarMap()
            cleanedx0.register_keys(self._gp.varkeys)
            cleanedx0.update(x0)
            x0 = cleanedx0
        self._gp.x0.update({vk: x0[vk] for vk in self.sgpvks if vk in x0})
        p_idx = 0
        for sgpc in self.sgpconstraints:
            for hmaplt1 in sgpc.as_gpconstr(self._gp.x0).as_hmapslt1({}):
                approxc = self.approxconstraints[p_idx]
                approxc.left = self.slack
                approxc.right.hmap = hmaplt1
                approxc.unsubbed = [Posynomial(hmaplt1) / self.slack]
                p_idx += 1  # p_idx=0 is the cost; sp constraints are after it
                (hmap,) = approxc.as_hmapslt1(self._gp.substitutions)
                self._gp.hmaps[p_idx] = hmap
                m_idx = self._gp.m_idxs[p_idx].start
                a_idxs = list(self.a_idxs[p_idx])  # A's entries we can modify
                self._update_a_matrix(m_idx, hmap, a_idxs)
        return self._gp
