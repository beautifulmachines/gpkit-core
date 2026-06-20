# pylint: disable=fixme,import-outside-toplevel
"""Implement the GeometricProgram class"""

import sys
import warnings as pywarnings
from collections import defaultdict
from dataclasses import dataclass
from time import time
from typing import Sequence

import numpy as np

from ..constraints.set import ConstraintSet
from ..exceptions import (
    DualInfeasible,
    Infeasible,
    InvalidLicense,
    InvalidPosynomial,
    PrimalInfeasible,
    UnboundedGP,
    UnknownInfeasible,
)
from ..nomials.map import NomialMap
from ..solutions import MarginSolution, Sensitivities, Solution, _WeakModelRef
from ..util.repr_conventions import lineagestr
from ..util.small_classes import CootMatrix, FixedScalar, Numbers, SolverLog
from ..util.small_scripts import appendsolwarning, initsolwarning
from ..varmap import VarMap

DEFAULT_SOLVER_KWARGS = {"cvxopt": {"kktsolver": "ldl"}}
SOLUTION_TOL = {"cvxopt": 1e-3, "mosek_cli": 1e-4, "mosek_conif": 1e-3}


# pylint: disable=too-few-public-methods
class MonoEqualityIndexes:
    "Class to hold MonoEqualityIndexes"

    def __init__(self):
        self.all = set()
        self.first_half = set()


def _get_solver(solver, kwargs):
    """Get the solverfn and solvername associated with solver"""
    if solver is None:
        from ..util.globals import settings

        try:
            solver = settings["default_solver"]
        except KeyError as err:
            raise ValueError(
                "No default solver was set during build, so"
                " solvers must be manually specified."
            ) from err
    if solver == "cvxopt":
        from ..solvers.cvxopt import optimize
    elif solver == "mosek_cli":
        from ..solvers.mosek_cli import optimize_generator

        optimize = optimize_generator(**kwargs)
    elif solver == "mosek_conif":
        from ..solvers.mosek_conif import optimize
    elif hasattr(solver, "__call__"):
        solver, optimize = solver.__name__, solver
    else:
        raise ValueError(f"Unknown solver '{solver}'.")
    return solver, optimize


@dataclass(frozen=True, slots=True)
class CompiledGP:
    """
    Immutable numerical snapshot of a GP ready for a solver.

    Parameters
    ----------
    c : (n_mon,) coefficients of each monomial
    A : (n_mon, n_var) exponents of each monomial
    m_idxs (n_posy,): row indices of each posynomial's monomials
    """

    c: Sequence
    A: CootMatrix
    m_idxs: Sequence[range]

    @classmethod
    def from_hmaps(cls, hmaps, varcols):
        "Generates nomial and solve data (A, p_idxs) from posynomials."
        m_idxs, c = [], []
        m_idx = 0
        row, col, data = [], [], []
        for hmap in hmaps:
            m_idxs.append(range(m_idx, m_idx + len(hmap)))
            c.extend(hmap.values())
            for exp in hmap:
                if not exp:  # space out A matrix with constants for mosek
                    assert m_idx < len(hmap)  # constants only expected in cost
                    row.append(m_idx)
                    col.append(0)
                    data.append(0)
                else:
                    row.extend([m_idx] * len(exp))
                    col.extend([varcols[var] for var in exp])
                    data.extend(exp.values())
                m_idx += 1
        return cls(c=c, A=CootMatrix(row, col, data), m_idxs=m_idxs)

    def __post_init__(self):
        self.validate()

    def validate(self):
        "simple validation checks"
        nrow, _ = self.A.shape
        if len(self.c) != nrow:
            raise ValueError(f"c has {len(self.c)} rows but A has {nrow}.")
        last_stop = 0
        for rng in self.m_idxs:
            if not isinstance(rng, range):
                raise TypeError("m_idxs must contain range objects")
            if rng.start != last_stop:
                raise ValueError("Posynomial rows must be contiguous in A")
            last_stop = rng.stop
        if last_stop != nrow:
            raise ValueError(f"m_idxs maps {last_stop} rows but A has {nrow}.")

    @property
    def k(self):
        "length of each posynomial"
        return tuple(len(p) for p in self.m_idxs)

    @property
    def p_idxs(self):
        "posynomial index of each monomial"
        return np.repeat(range(len(self.m_idxs)), self.k)

    def compute_z(self, x):
        "z values for a given primal solution"
        return np.log(self.c) + self.A.dot(x)

    def check_solution(self, rawsol, tol, abstol=1e-20):
        """Run checks to mathematically confirm solution solves this GP

        Arguments
        ---------
        rawsol: RawSolution
            solution returned by solver

        Raises
        ------
        Infeasible if any problems are found
        """
        A = self.A.tocsr()

        def almost_equal(num1, num2):
            "local almost equal test"
            return (
                num1 == num2
                or abs((num1 - num2) / (num1 + num2)) < tol
                or abs(num1 - num2) < abstol
            )

        # check primal sol #
        primal_exp_vals = self.c * np.exp(A.dot(rawsol.x))  # c*e^Ax
        if not almost_equal(primal_exp_vals[self.m_idxs[0]].sum(), rawsol.cost):
            raise Infeasible(
                f"Primal computed cost {primal_exp_vals[self.m_idxs[0]].sum()} "
                f"did not match solver-returned cost {rawsol.cost}."
            )
        for mi in self.m_idxs[1:]:
            if primal_exp_vals[mi].sum() > 1 + tol:
                raise Infeasible(
                    "Primal solution violates constraint: "
                    f"{primal_exp_vals[mi].sum()} is greater than 1"
                )
        # check dual sol #
        # if self.integersolve:
        #     return
        # note: follows dual formulation in section 3.1 of
        # http://web.mit.edu/~whoburg/www/papers/hoburg_phd_thesis.pdf
        if not almost_equal(rawsol.nu[self.m_idxs[0]].sum(), 1):
            raise Infeasible(
                "Dual variables associated with objective sum"
                f" to {rawsol.nu[self.m_idxs[0]].sum()}, not 1"
            )
        if any(rawsol.nu < 0):
            minnu = min(rawsol.nu)
            if minnu < -tol / 1000:
                raise Infeasible(
                    f"Dual solution has negative entries as large as {minnu}."
                )
        if any(np.abs(A.T.dot(rawsol.nu)) > tol):
            raise Infeasible("Dual: nu^T * A did not vanish.")
        b = np.log(self.c)
        dual_cost = sum(
            rawsol.nu[mi].dot(b[mi] - np.log(rawsol.nu[mi] / rawsol.la[i]))
            for i, mi in enumerate(self.m_idxs)
            if rawsol.la[i]
        )
        if not almost_equal(np.exp(dual_cost), rawsol.cost):
            raise Infeasible(
                f"Dual cost {np.exp(dual_cost)} differs from primal cost {rawsol.cost}"
            )

    def compute_la(self, nu):
        "compute lambda from nu"
        assert np.shape(nu) == (len(self.c),)
        return np.array([sum(nu[mi]) for mi in self.m_idxs])

    def compute_nu(self, la, x):
        "compute nu from lambda and primal solution x"
        assert np.shape(la) == (len(self.m_idxs),)
        z = self.compute_z(x)
        nu_by_posy = [
            la[p_i] * np.exp(z[m_is]) / sum(np.exp(z[m_is]))
            for p_i, m_is in enumerate(self.m_idxs)
        ]
        return np.hstack(nu_by_posy)


class GeometricProgram:
    # pylint: disable=too-many-instance-attributes
    """Standard mathematical representation of a GP.

    Attributes with side effects
    ----------------------------
    `solver_out` and `solve_log` are set during a solve
    `result` is set at the end of a solve if solution status is optimal

    Examples
    --------
    >>> gp = gpkit.constraints.gp.GeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            x >= 1,
                        ], {})
    >>> gp.solve()
    """
    _result = solve_log = solver_out = model = None
    choicevaridxs = integersolve = None

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cost,
        constraints,
        substitutions,
        *,
        checkbounds=True,
        linked_derivs=None,
        **_,
    ):
        self.cost, self.substitutions = cost, substitutions
        self.linked_derivs = linked_derivs or {}
        for key, sub in self.substitutions.items():
            if isinstance(sub, FixedScalar):
                sub = sub.value
                if hasattr(sub, "units"):
                    sub = sub.to(key.units or "dimensionless").magnitude
                self.substitutions[key] = sub
            if not isinstance(sub, (Numbers, np.ndarray)):
                raise TypeError(
                    f"substitution {{{key}: {sub}}} has invalid value type {type(sub)}."
                )
        cost_hmap = cost.hmap.sub(self.substitutions, cost.vks)
        if any(c <= 0 for c in cost_hmap.values()):
            raise InvalidPosynomial("a GP's cost must be Posynomial")
        hmapgen = ConstraintSet.as_hmapslt1(constraints, self.substitutions)
        self.hmaps = [cost_hmap] + list(hmapgen)
        self.gen()  # Generate various maps into the posy- and monomials
        if checkbounds:
            self.check_bounds(err_on_missing_bounds=True)

    def check_bounds(self, err_on_missing_bounds=False):
        "Checks if any variables are unbounded, through equality constraints."
        missingbounds = {}
        if not self.vars:  # temporary band-aid for problems with no variables
            return missingbounds
        A = self.data.A.tocsc()
        nrow, ncol = A.shape
        ineq_idxs = [i for i in range(nrow) if i not in self.meq_idxs.all]
        A = A[ineq_idxs, :]  # only take credit for inequalities, not ==
        for j in range(ncol):
            istart, iend = A.indptr[j], A.indptr[j + 1]
            if not np.any(A.data[istart:iend] > 0):
                missingbounds[(self.vars[j], "upper")] = "."
            if not np.any(A.data[istart:iend] < 0):
                missingbounds[(self.vars[j], "lower")] = "."
        if not missingbounds:
            return {}  # all bounds found in inequalities
        meq_bounds = gen_meq_bounds(missingbounds, self.exps, self.meq_idxs)
        fulfill_meq_bounds(missingbounds, meq_bounds)
        if missingbounds and err_on_missing_bounds:
            raise UnboundedGP(
                "\n\n".join(
                    f"{v} has no {b} bound{x}" for (v, b), x in missingbounds.items()
                )
            )
        return missingbounds

    def gen(self):
        "compile this program and set meq_idxs"
        variables = set()
        self.meq_idxs = MonoEqualityIndexes()
        self.exps = []
        m_idx = 0
        for hmap in self.hmaps:
            if getattr(hmap, "from_meq", False):
                self.meq_idxs.all.add(m_idx)
                if len(self.meq_idxs.all) > 2 * len(self.meq_idxs.first_half):
                    self.meq_idxs.first_half.add(m_idx)
            self.exps.extend(hmap)
            for exp in hmap:
                variables.update(exp)
                m_idx += 1
        self.varcols = {vk: i for i, vk in enumerate(variables)}
        self.vars = tuple(variables)
        self.choicevaridxs = {vk: i for i, vk in enumerate(variables) if vk.choices}
        self.data = CompiledGP.from_hmaps(self.hmaps, self.varcols)

    # pylint: disable=too-many-locals,too-many-branches
    def solve(self, solver=None, *, verbosity=1, gen_result=True, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses a solver found during installation.
            If "mosek_conif", "mosek_cli", or "cvxopt", uses that solver.
            If a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (default 1)
            If greater than 0, prints solver name and solve time.
        gen_result : bool (default True)
            If True, makes a Solution from solver output.
        **kwargs :
            Passed to solver constructor and solver function.


        Returns
        -------
        Solution (or RawSolution if gen_result is False)
        """
        solvername, solverfn = _get_solver(solver, kwargs)
        if verbosity > 0:
            print(f"Using solver '{solvername}'")
            print(f" for {len(self.vars)} free variables")
            print(f"  in {len(self.data.k)} posynomial inequalities.")

        solverargs = DEFAULT_SOLVER_KWARGS.get(solvername, {})
        solverargs.update(kwargs)
        if self.choicevaridxs and solvername == "mosek_conif":
            solverargs["choicevaridxs"] = self.choicevaridxs
            self.integersolve = True
        starttime = time()
        solver_out, infeasibility, original_stdout = {}, None, sys.stdout
        try:
            sys.stdout = SolverLog(original_stdout, verbosity=verbosity - 2)
            solver_out = solverfn(self.data, meq_idxs=self.meq_idxs, **solverargs)
            solver_out.meta["soltime"] = time() - starttime
            if verbosity > 0:
                print(f"Solving took {solver_out.meta['soltime']:.3g} seconds.")
        except Infeasible as e:
            infeasibility = e
        except InvalidLicense as e:
            raise InvalidLicense(
                f'license for solver "{solvername}" is invalid.'
            ) from e
        except Exception as e:
            raise UnknownInfeasible("Something unexpected went wrong.") from e
        finally:
            self.solve_log = sys.stdout
            sys.stdout = original_stdout
            self.solver_out = solver_out

        if infeasibility:
            if isinstance(infeasibility, PrimalInfeasible):
                msg = (
                    "The model had no feasible points; relaxing some"
                    " constraints or constants will probably fix this."
                )
            elif isinstance(infeasibility, DualInfeasible):
                msg = (
                    "The model ran to an infinitely low cost"
                    " (or was otherwise dual infeasible); bounding"
                    " the right variables will probably fix this."
                )
            elif isinstance(infeasibility, UnknownInfeasible):
                msg = (
                    "Solver failed for an unknown reason. Relaxing"
                    " constraints/constants, bounding variables, or"
                    " using a different solver might fix it."
                )
            else:
                raise ValueError("Unexpected infeasibility {infeasibility}")
            soltime = time() - starttime
            if verbosity > 0 and soltime < 1 and self.model:
                print(
                    msg + "\nSince the model solved in less than a second,"
                    " let's run `.debug()` to analyze what happened.\n"
                )
                return self.model.debug(solver=solver)
            # else, raise a clarifying error
            msg += (
                " Running `.debug()` or increasing verbosity may pinpoint"
                " the trouble."
            )
            raise infeasibility.__class__(msg) from infeasibility

        if not gen_result:
            return solver_out
        # else, generate a Solution object
        self._result = self.generate_result(solver_out, verbosity=verbosity - 2)
        return self.result

    @property
    def result(self):
        "Creates and caches a result from the raw solver_out"
        if not self._result:
            self._result = self.generate_result(self.solver_out)
        return self._result

    def generate_result(self, solver_out, *, verbosity=0, dual_check=True):
        "Generates a Solution object and checks it."
        if verbosity > 0:
            soltime = solver_out.meta["soltime"]
            tic = time()
        # result packing #
        result = self._compile_result(solver_out)  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            rpackpct = (time() - tic) / soltime * 100
            print(f"Result packing took {rpackpct:.2g}%% of solve time.")
            tic = time()
        # solution checking #
        try:
            tol = SOLUTION_TOL.get(solver_out.meta["solver"], 1e-5)
            self.data.check_solution(solver_out, tol)
        except Infeasible as chkerror:
            msg = str(chkerror)
            if not ("Dual" in msg and not dual_check):
                initsolwarning(result, "Solution Inconsistency")
                appendsolwarning(msg, None, result, "Solution Inconsistency")
                if verbosity > -4:
                    print(f"Solution check warning: {msg}")
        if verbosity > 0:
            print(
                f"Solution checking took "
                f"{((time() - tic) / soltime * 100):.2g}% of solve time."
            )
        return result

    def _calculate_sensitivities(self, la, nu, varvals):
        """Calculate sensitivities for variables and constraints.

        Returns
        -------
        tuple
            (cost_senss, gpv_ss, absv_ss, m_senss)
        """
        nu_by_posy = [nu[mi] for mi in self.data.m_idxs]
        cost_senss = sum(
            nu_i * exp for (nu_i, exp) in zip(nu_by_posy[0], self.cost.hmap)
        )
        gpv_ss = cost_senss.copy()
        m_senss = defaultdict(float)
        constraint_senss = {}
        absv_ss = {vk: abs(x) for vk, x in cost_senss.items()}

        for las, nus, c in zip(la[1:], nu_by_posy[1:], self.hmaps[1:]):
            while getattr(c, "parent", None) is not None:
                if not isinstance(c, NomialMap):
                    c.parent.child = c
                c = c.parent  # parents get their sens_from_dual used...
            v_ss, c_senss = c.sens_from_dual(las, nus, varvals)
            for vk, x in v_ss.items():
                gpv_ss[vk] = x + gpv_ss.get(vk, 0)
                absv_ss[vk] = abs(x) + absv_ss.get(vk, 0)
            while getattr(c, "generated_by", None):
                c.generated_by.generated = c
                c = c.generated_by  # ...while generated_bys are just labels
            constraint_senss[c] = c_senss
            m_senss[lineagestr(c)] += abs(c_senss)

        # Handle linked sensitivities
        for v in list(v for v in gpv_ss if self.linked_derivs.get(v)):
            dlogcost_dlogv = gpv_ss.pop(v)
            dlogcost_dlogabsv = absv_ss.pop(v)
            val = np.array(self.substitutions[v])
            for c, dv_dc in self.linked_derivs[v].items():
                with pywarnings.catch_warnings():  # skip pesky divide-by-zeros
                    pywarnings.simplefilter("ignore")
                    dlogv_dlogc = dv_dc * self.substitutions[c] / val
                    gpv_ss[c] = gpv_ss.get(c, 0) + dlogcost_dlogv * dlogv_dlogc
                    absv_ss[c] = absv_ss.get(c, 0) + abs(
                        dlogcost_dlogabsv * dlogv_dlogc
                    )
                if v in cost_senss:
                    if c in self.cost.vks:  # TODO: seems unnecessary
                        dlogcost_dlogv = cost_senss.pop(v)
                        before = cost_senss.get(c, 0)
                        cost_senss[c] = before + dlogcost_dlogv * dlogv_dlogc

        return cost_senss, gpv_ss, absv_ss, m_senss, constraint_senss

    def _compute_variable_sensitivities(self, nu, primal_weights):
        """Compute ∂(Σ w_i x_i)/∂log(c_m) via one linear solve.

        Parameters
        ----------
        nu : array
            Full dual variable vector (length = total monomials, including cost).
        primal_weights : dict
            {VarKey: float} — coefficient of each free variable in the quantity
            of interest.  E.g. {A_key: A_val, B_key: -B_val} for margin A − B.

        Returns
        -------
        (v_prime, qnu) : (np.ndarray, np.ndarray)
            v_prime — solution to A_c^T v' = w, one entry per tight constraint
            qnu     — sensitivity weight per constraint monomial:
                      qnu[k] = (v'[i] / lambda_i) * nu_constr[k],
                      where i is the tight-constraint index for monomial k and
                      lambda_i = sum(nu_constr[monomials of constraint i]).

        Notes
        -----
        A_c (n_tight × n_vars) is the matrix of nu-weighted average exponents,
        one row per tight constraint: A_c[i,:] = nu_i @ A_i / lambda_i.
        Slack constraints (lambda_i ≈ 0) are skipped; they contribute nothing.
        """
        obj_end = self.data.m_idxs[0].stop
        A_constr = self.data.A.tocsr()[obj_end:, :]  # sparse (n_constr_mono, n_vars)
        nu_constr = np.asarray(nu[obj_end:], dtype=float)
        n_vars = len(self.vars)

        # Build A_c: one dense row per tight constraint.
        # A_c[i,:] = nu_i @ A_i / lambda_i  (nu-weighted average of exponents).
        # Relative threshold: skip constraints whose dual weight is negligible compared
        # to the overall dual magnitude (cvxopt solver tolerance ≈ 1e-5).
        nu_scale = float(nu_constr.sum()) + 1.0

        a_c_rows = []
        tight_slices = (
            []
        )  # (q_start, q_end, lambda_i, constr_i) for each tight constraint
        for constr_i, m_idx in enumerate(self.data.m_idxs[1:]):
            q_start = m_idx.start - obj_end
            q_end = m_idx.stop - obj_end
            nu_i = nu_constr[q_start:q_end]
            lambda_i = float(nu_i.sum())
            if lambda_i < 1e-5 * nu_scale:
                continue
            # A_constr[q_start:q_end, :].T @ nu_i  →  dense (n_vars,)
            a_c_row = np.asarray(A_constr[q_start:q_end, :].T.dot(nu_i)).ravel()
            a_c_rows.append(a_c_row / lambda_i)
            tight_slices.append((q_start, q_end, lambda_i, constr_i))

        n_constr = len(self.data.m_idxs) - 1
        qnu = np.zeros(len(nu_constr))
        constraint_v = np.zeros(n_constr)
        if not a_c_rows:
            return np.zeros(0), qnu, constraint_v

        A_c = np.vstack(a_c_rows)  # (n_tight × n_vars) dense

        w = np.zeros(n_vars)
        for vk, coeff in primal_weights.items():
            if vk in self.varcols:
                w[self.varcols[vk]] = coeff

        # Solve A_c^T v' = w: (n_vars equations, n_tight unknowns)
        v_prime, _, _, _ = np.linalg.lstsq(A_c.T, w, rcond=None)

        # For each tight constraint i, broadcast v'_i / lambda_i across its
        # constituent monomials, weighted by nu_constr.  Also record v'_i by
        # constraint index so callers can apply const_mmap corrections.
        for j, (q_start, q_end, lambda_i, constr_i) in enumerate(tight_slices):
            qnu[q_start:q_end] = (v_prime[j] / lambda_i) * nu_constr[q_start:q_end]
            constraint_v[constr_i] = v_prime[j]

        return v_prime, qnu, constraint_v

    def _accumulate_posy_const_sens(self, const_senss, hmap_qnu, c):
        """Accumulate ∂(A−B)/∂log(c_m) contributions from one PosynomialInequality.

        The inner loops (pmap entries × exponent dict entries) are O(1) and
        O(2-5) per monomial respectively in typical engineering models.
        """
        if not hasattr(c, "pmap"):
            raise RuntimeError(
                f"Constraint {c!r} is missing pmap.  "
                "_compute_margin_sensitivity must be called before "
                "_calculate_sensitivities (which deletes pmap).  "
                "See issue #200."
            )
        presub_exps = list(c.unsubbed[0].hmap)
        for k, qnu_j in enumerate(hmap_qnu):
            for presub_idx, fraction in c.pmap[k].items():
                for vk, exp in presub_exps[presub_idx].items():
                    if vk not in self.varcols:  # constant VarKey
                        const_senss[vk] -= qnu_j * fraction * exp

    def _compute_margin_sensitivity(self, nu, varvals, margin_obj):
        """Compute ∂(A−B)/∂log(c) for every constant c via one linear solve.

        Must be called BEFORE _calculate_sensitivities() because pmap is deleted
        there.  See GitHub issue #200 for the pmap mutation discussion.

        Parameters
        ----------
        nu : array
            Full dual variable vector from the solver.
        varvals : VarMap
            Combined free-variable primal values and substituted constants.
        margin_obj : MarginObjective
            Specifies plus_var (A) and minus_var (B).

        Returns
        -------
        MarginSolution
        """
        A_vk = getattr(margin_obj.plus_var, "key", margin_obj.plus_var)
        B_vk = getattr(margin_obj.minus_var, "key", margin_obj.minus_var)
        A_val = float(varvals.get(A_vk, 0))
        B_val = float(varvals.get(B_vk, 0))

        primal_weights = {}
        if A_vk in self.varcols:
            primal_weights[A_vk] = A_val
        if B_vk in self.varcols:
            primal_weights[B_vk] = -B_val

        _, qnu, constraint_v = self._compute_variable_sensitivities(nu, primal_weights)

        # Accumulate ∂(A−B)/∂log(c_m) = −Σ_j qnu_j * e_{jm} for each constant c_m.
        # Slack constraints self-eliminate: complementarity gives nu_k = 0 for
        # inactive constraints, so their qnu is zero and they contribute nothing.
        obj_end = self.data.m_idxs[0].stop
        const_senss = defaultdict(float)
        meq_unsubbed_idx = defaultdict(
            int
        )  # which unsubbed to use per MonomialEquality

        for i, hmap in enumerate(self.hmaps[1:]):
            c = hmap
            while getattr(c, "parent", None) is not None:
                c = c.parent

            q_start = self.data.m_idxs[i + 1].start - obj_end
            q_end = self.data.m_idxs[i + 1].stop - obj_end
            hmap_qnu = qnu[q_start:q_end]

            if getattr(hmap, "from_meq", False):
                # MonomialEquality emits two hmaps sharing one parent object.
                # Track which unsubbed index belongs to each (first=0, second=1).
                parent_id = id(c)
                unsubbed_idx = meq_unsubbed_idx[parent_id]
                meq_unsubbed_idx[parent_id] += 1
                (presub_exp,) = c.unsubbed[unsubbed_idx].hmap
                qnu_j = hmap_qnu[0]
                for vk, exp in presub_exp.items():
                    if vk not in self.varcols:
                        const_senss[vk] -= qnu_j * exp
            else:
                self._accumulate_posy_const_sens(const_senss, hmap_qnu, c)
                # const_mmap: constants absorbed into the constraint RHS shift the
                # effective tight-condition when they're perturbed.  Apply the same
                # correction that sens_from_dual uses, but scaled by v'_i instead
                # of nu[i] (mirrors the pmap → qnu2 transform for the free-var terms).
                if hasattr(c, "const_mmap"):
                    v_i = constraint_v[i]
                    scale = (1 - c.const_coeff) / c.const_coeff
                    presub_exps = list(c.unsubbed[0].hmap)
                    for const_presub_idx, pct in c.const_mmap.items():
                        for vk, exp in presub_exps[const_presub_idx].items():
                            if vk not in self.varcols:
                                const_senss[vk] -= v_i * pct * scale * exp

        # Direct contributions when A or B are constants (no linear system needed)
        if A_vk not in self.varcols:
            const_senss[A_vk] += A_val
        if B_vk not in self.varcols:
            const_senss[B_vk] -= B_val

        # Chain rule for linked constants (same pattern as _calculate_sensitivities).
        # pywarnings suppresses RuntimeWarning from 0-division when val == 0.
        for v in list(vk for vk in const_senss if self.linked_derivs.get(vk)):
            d_margin_d_logv = const_senss.pop(v)
            val = np.array(self.substitutions[v])
            for linked_c, dv_dc in self.linked_derivs[v].items():
                with pywarnings.catch_warnings():
                    pywarnings.simplefilter("ignore")
                    dlogv_dlogc = dv_dc * self.substitutions[linked_c] / val
                    const_senss[linked_c] += d_margin_d_logv * dlogv_dlogc

        units = A_vk.unitstr() if A_vk.units else ""
        return MarginSolution(
            name=margin_obj.name,
            value=A_val - B_val,
            plus_value=A_val,
            minus_value=B_val,
            units=units,
            sensitivities=dict(const_senss),
        )

    def _compile_result(self, solver_out):
        primal = solver_out.x
        if len(self.vars) != len(primal):
            raise RuntimeWarning("The primal solution was not returned.")
        varvals = VarMap(zip(self.vars, np.exp(primal)))
        varvals.update(self.substitutions)

        warnings = {}
        if self.integersolve or self.choicevaridxs:
            warnings.update(self._handle_choicevars(solver_out))

        # Compute margin sensitivity BEFORE _calculate_sensitivities(), which deletes
        # pmap on each constraint.  See issue #200 for the pmap mutation discussion.
        margin_obj = getattr(self.model, "margin_objective", None)
        derived = None
        if margin_obj is not None:
            derived = self._compute_margin_sensitivity(
                solver_out.nu, varvals, margin_obj
            )

        _, gpv_ss, absv_ss, m_senss, constraint_senss = self._calculate_sensitivities(
            solver_out.la, solver_out.nu, varvals
        )

        result = Solution(
            cost=float(solver_out.cost),
            primal=VarMap(zip(self.vars, np.exp(primal))),
            constants=VarMap(self.substitutions),
            sens=Sensitivities(
                constraints=constraint_senss,
                models=dict(m_senss),
                variables=VarMap(gpv_ss),
                variablerisk=VarMap(absv_ss),
            ),
            meta={"soltime": solver_out.meta["soltime"], "warnings": warnings},
            derived=derived,
        )
        result.meta["cost function"] = self.cost
        result.meta["model"] = _WeakModelRef(self.model)
        return result

    def _handle_choicevars(self, solver_out):
        "This is essentially archived code, until it can be tested with mosek"
        warnings = {}

        if self.integersolve:
            warnings["No Dual Solution"] = [
                (
                    "This model has the discretized choice variables"
                    f" {sorted(self.choicevaridxs.keys())} and hence no dual"
                    " solution. You can fix those variables to their optimal"
                    " values and get sensitivities to the resulting"
                    " continuous problem by updating your model's"
                    " substitions with `sol['choicevariables']`.",
                    self.choicevaridxs,
                )
            ]

        if self.choicevaridxs:
            warnings["Freed Choice Variables"] = [
                (
                    "This model has the discretized choice variables"
                    f" {sorted(self.choicevaridxs.keys())}, but since the "
                    f"'{solver_out.meta['solver']}' solver doesn't support "
                    "discretization they were treated as continuous variables.",
                    self.choicevaridxs,
                )
            ]  # TODO: choicevaridxs seems unnecessary

        return warnings


def gen_meq_bounds(
    missingbounds, exps, meq_idxs
):  # pylint: disable=too-many-locals,too-many-branches
    "Generate conditional monomial equality bounds"
    meq_bounds = defaultdict(set)
    for i in meq_idxs.first_half:
        p_upper, p_lower, n_upper, n_lower = set(), set(), set(), set()
        for key, x in exps[i].items():
            if (key, "upper") in missingbounds:
                if x > 0:
                    p_upper.add((key, "upper"))
                else:
                    n_upper.add((key, "upper"))
            if (key, "lower") in missingbounds:
                if x > 0:
                    p_lower.add((key, "lower"))
                else:
                    n_lower.add((key, "lower"))
        # (consider x*y/z == 1)
        # for a var (e.g. x) to be upper bounded by this monomial equality,
        #   - vars of the same sign (y) must be lower bounded
        #   - AND vars of the opposite sign (z) must be upper bounded
        p_ub = n_lb = frozenset(n_upper).union(p_lower)
        n_ub = p_lb = frozenset(p_upper).union(n_lower)
        for keys, ub in ((p_upper, p_ub), (n_upper, n_ub)):
            for key, _ in keys:
                needed = ub.difference([(key, "lower")])
                if needed:
                    meq_bounds[(key, "upper")].add(needed)
                else:
                    del missingbounds[(key, "upper")]
        for keys, lb in ((p_lower, p_lb), (n_lower, n_lb)):
            for key, _ in keys:
                needed = lb.difference([(key, "upper")])
                if needed:
                    meq_bounds[(key, "lower")].add(needed)
                else:
                    del missingbounds[(key, "lower")]
    return meq_bounds


def fulfill_meq_bounds(missingbounds, meq_bounds):
    "Bounds variables with monomial equalities"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound in set(meq_bounds):
            if bound not in missingbounds:
                del meq_bounds[bound]
                continue
            for condition in meq_bounds[bound]:
                if not any(bound in missingbounds for bound in condition):
                    del meq_bounds[bound]
                    del missingbounds[bound]
                    still_alive = True
                    break
    for var, bound in meq_bounds:
        boundstr = ", but would gain it from any of these sets: "
        for condition in list(meq_bounds[(var, bound)]):
            meq_bounds[(var, bound)].remove(condition)
            newcond = condition.intersection(missingbounds)
            if newcond and not any(
                c.issubset(newcond) for c in meq_bounds[(var, bound)]
            ):
                meq_bounds[(var, bound)].add(newcond)
        boundstr += " or ".join(
            str(list(condition)) for condition in meq_bounds[(var, bound)]
        )
        missingbounds[(var, bound)] = boundstr
