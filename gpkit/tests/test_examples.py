"""Unit testing of tests in docs/source/examples"""

import json
import os
import pickle

import numpy as np
import pytest

from gpkit import Model, Variable, settings, ureg
from gpkit.constraints.loose import Loose
from gpkit.exceptions import (
    DualInfeasible,
    PrimalInfeasible,
    UnboundedGP,
    UnknownInfeasible,
)
from gpkit.tests.conftest import assert_logtol
from gpkit.util.small_classes import Quantity
from gpkit.util.small_scripts import mag


# pylint: disable=too-many-public-methods
class TestExamples:
    """
    To test a new example, add a function called `test_$EXAMPLENAME`, where
    $EXAMPLENAME is the name of your example in docs/source/examples without
    the file extension.

    This function should accept two arguments (e.g. 'self' and 'example').
    The imported example script will be passed to the second: anything that
    was a global variable (e.g, "sol") in the original script is available
    as an attribute (e.g., "example.sol")

    If you don't want to perform any checks on the example besides making
    sure it runs, just put "pass" as the function's body, e.g.:

          def test_dummy_example(self, example):
              pass

    But it's good practice to ensure the example's solution as well, e.g.:

          def test_dummy_example(self, example):
              assert example.sol["cost"] == pytest.approx(3.121)
    """

    # skip test breakdowns -- failing due to pint errors in old pkl files
    # def test_breakdowns(self, example):
    #     pass

    def test_issue_1513(self, example):
        assert example.sol[0].cost == pytest.approx(20.0, rel=1e-2)

    def test_issue_1522(self, example):
        assert example.sol.cost == pytest.approx(3.0, rel=1e-2)

    def test_autosweep(self, example):
        bst1, tol1 = example.bst1, example.tol1
        bst2, tol2 = example.bst2, example.tol2

        w_ = np.linspace(1, 10, 100)
        for bst in [bst1, example.bst1_loaded]:
            sol1 = bst.sample_at(w_)
            assert_logtol(sol1("w"), w_)
            assert_logtol(sol1("A"), w_**2 + 1, tol1)
            assert_logtol(sol1["cost"], (w_**2 + 1) ** 2, tol1)
            assert Quantity(1.0, sol1("A").units) == Quantity(1.0, ureg.m) ** 2

        ndig = -int(np.log10(tol2))
        assert bst2.cost_at("cost", 3) == pytest.approx(1.0, abs=10 ** (-ndig))
        # before corner
        a_bc = np.linspace(1, 3, 50)
        sol_bc = bst2.sample_at(a_bc)
        assert_logtol(sol_bc("A"), (a_bc / 3) ** 0.5, tol2)
        assert_logtol(sol_bc["cost"], a_bc / 3, tol2)
        # after corner
        a_ac = np.linspace(3, 10, 50)
        sol_ac = bst2.sample_at(a_ac)
        assert_logtol(sol_ac("A"), (a_ac / 3) ** 2, tol2)
        assert_logtol(sol_ac["cost"], (a_ac / 3) ** 4, tol2)
        os.remove("autosweep.pkl")

    def test_treemap(self, example):
        # treemap.py is a visualization-only example; no model solve, no cost to assert
        pass

    def test_checking_result_changes(self, example):
        sol = example.sol
        assert sol.cost == pytest.approx(0.48, abs=0.01)
        os.remove("last_verified.sol")

    def test_evaluated_fixed_variables(self, example):
        t_night = example.t_night
        assert example.sol[t_night] / ureg.hr == pytest.approx(12)
        expected = [16, 12, 8]
        actual = [sol[t_night] for sol in example.sols]
        for exp, act in zip(expected, actual):
            # odd floating point round off issues when running with pytest
            assert act / ureg.hr == pytest.approx(exp)

    def test_evaluated_free_variables(self, example):
        x2 = example.x2
        sol = example.sol
        assert abs(sol[x2] - 4) <= 1e-4

    def test_external_constraint(self, example):
        pass  # external_constraint.py defines a class only; no solve, no cost to assert

    def test_migp(self, example):
        solx = [sol[example.x] for sol in example.sols]
        if settings["default_solver"] == "mosek_conif":
            assert_logtol(solx, [1] * 3 + [2] * 6 + [3] * 2)
        else:
            num = example.num
            assert_logtol(solx, np.sqrt([sol[num] for sol in example.sols]))

    def test_external_function(self, example):
        external_code = example.external_code
        assert external_code(0) == 0

    def test_external_sp(self, example):
        m = example.m
        sol = m.localsolve(verbosity=0)
        assert sol.cost == pytest.approx(0.707, abs=0.001)

    def test_freeing_fixed_variables(self, example):
        x = example.x
        y = Variable("y", 3)
        m = Model(x, [x >= 1 + y, y >= 1])
        sol = m.solve(verbosity=0)
        assert abs(sol.cost - 4) <= 1e-4
        assert y in sol.constants

        del m.substitutions["y"]
        sol = m.solve(verbosity=0)
        assert abs(sol.cost - 2) <= 1e-4
        assert y in sol.primal

    def test_gettingstarted(self, example):
        assert example.sol.cost == pytest.approx(1.414, rel=1e-2)

    def test_loose_constraintsets(self, example):
        m = example.m
        sol = m.solve(verbosity=0)
        assert sol.cost == pytest.approx(2, abs=0.001)

    def test_sub_multi_values(self, example):
        x = example.x
        y = example.y
        z = example.z
        p = example.p
        assert all(p.sub({x: 1, "y": 2}) == 2 * z)
        assert all(p.sub({x: 1, y: 2, "z": [1, 2]}) == z.sub({z: [2, 4]}))

    def test_substitutions(self, example):
        x = example.x
        p = example.p
        assert p.sub({x: 3}) == 9
        assert p.sub({x.key: 3}) == 9
        assert p.sub({"x": 3}) == 9

    def test_tight_constraintsets(self, example):
        m = example.m
        sol = m.solve(verbosity=0)
        assert sol.cost == pytest.approx(2, abs=0.01)

    def test_vectorization(self, example):
        x = example.x
        y = example.y
        z = example.z
        assert y.shape == (5, 3)
        assert x.shape == (2, 5, 3)
        assert z.shape == (7, 3)

    def test_model_var_access(self, example):
        model = example.PS
        _ = model["E"]
        # "m" exists on PowerSystem and its submodels; getitem prefers
        # the model's own variable when exactly one match is in unique_varkeys
        own_m = model["m"]
        assert own_m.key == model.m.key
        # all variables named "m" are still accessible via varkeys
        all_m = model.varkeys.keys("m")
        assert len(all_m) == 3  # PowerSystem.m, Battery.m, Motor.m

    @pytest.mark.skip(reason="pint units error - needs investigation")
    def test_plot_sweep1d(self, example):
        pass

    def test_performance_modeling(self, example):
        m = Model(example.M.cost, Loose(example.M), example.M.substitutions)
        sol = m.solve(verbosity=0)
        assert sol.cost == pytest.approx(2.1963, rel=1e-3)
        sol.table()
        sol.save("solution.pkl")
        sol.table()
        with open("solution.pkl", "rb") as fil:
            sol_loaded = pickle.load(fil)
        assert "Free Variables" in sol_loaded.table()
        os.remove("solution.pkl")

        sweepsol = m.sweep({example.AC.fuse.W: (50, 100, 150)}, verbosity=0)
        sweepsol.table()
        sweepsol.save("sweepsolution.pkl")
        assert "Swept Variables" in sweepsol.table()
        with open("sweepsolution.pkl", "rb") as fil:
            sol_loaded = pickle.load(fil)
        assert "Swept Variables" in sol_loaded.table()
        os.remove("sweepsolution.pkl")

        # testing savejson
        sol.savejson("solution.json")
        json_dict = {}
        with open("solution.json", "r", encoding="utf-8") as rf:
            json_dict = json.load(rf)
        os.remove("solution.json")
        for var in sol.primal:
            assert np.all(json_dict[str(var.key)]["v"] == sol.primal[var.key])
            assert json_dict[str(var.key)]["u"] == var.unitstr()

    def test_sp_to_gp_sweep(self, example):
        sol = example.sol
        assert sol[0].cost == pytest.approx(4628.21, abs=0.01)
        assert sol[1].cost == pytest.approx(6226.60, abs=0.01)
        assert sol[2].cost == pytest.approx(7362.77, abs=0.01)

    def test_boundschecking(self, example):  # pragma: no cover
        if "mosek_cli" in settings["default_solver"]:
            with pytest.raises(UnknownInfeasible):
                example.gp.solve(verbosity=0)
        else:
            example.gp.solve(verbosity=0)  # mosek_conif and cvxopt solve it

    def test_vectorize(self, example):
        sol = example.m.solve(verbosity=0)
        assert sol.cost == pytest.approx(2.0, rel=1e-2)

    def test_primal_infeasible_ex1(self, example):
        primal_or_unknown = PrimalInfeasible
        if "cvxopt" in settings["default_solver"]:  # pragma: no cover
            primal_or_unknown = UnknownInfeasible
        with pytest.raises(primal_or_unknown):
            example.m.solve(verbosity=0)

    def test_primal_infeasible_ex2(self, example):
        primal_or_unknown = PrimalInfeasible
        if "cvxopt" in settings["default_solver"]:  # pragma: no cover
            primal_or_unknown = UnknownInfeasible
        with pytest.raises(primal_or_unknown):
            example.m.solve(verbosity=0)

    def test_debug(self, example):
        dual_or_primal = DualInfeasible
        if "mosek_conif" == settings["default_solver"]:  # pragma: no cover
            dual_or_primal = PrimalInfeasible
        with pytest.raises(UnboundedGP):
            example.m.gp()
        with pytest.raises(dual_or_primal):
            gp = example.m.gp(checkbounds=False)
            gp.solve(verbosity=0)

        primal_or_unknown = PrimalInfeasible
        if "cvxopt" == settings["default_solver"]:  # pragma: no cover
            primal_or_unknown = UnknownInfeasible
        with pytest.raises(primal_or_unknown):
            example.m2.solve(verbosity=0)

        with pytest.raises(UnboundedGP):
            example.m3.gp()
        with pytest.raises(DualInfeasible):
            gp3 = example.m3.gp(checkbounds=False)
            gp3.solve(verbosity=0)

    def test_simple_sp(self, example):
        assert example.sol.cost == pytest.approx(0.9, rel=1e-2)

    def test_simple_box(self, example):
        sol = example.m.solve(verbosity=0)
        assert sol.cost == pytest.approx(0.003674, rel=1e-2)

    def test_x_greaterthan_1(self, example):
        assert example.sol.cost == pytest.approx(1.0, rel=1e-2)

    def test_beam(self, example):
        assert not np.isnan(example.sol["w"]).any()
        assert example.sol.cost == pytest.approx(1.6214, rel=1e-3)

    def test_water_tank(self, example):
        assert example.sol.cost == pytest.approx(1.293, rel=1e-2)

    def test_sin_approx_example(self, example):
        sol = example.m.solve(verbosity=0)
        assert sol.cost == pytest.approx(0.785, rel=1e-2)

    def test_simpleflight(self, example):
        assert example.sol.almost_equal(example.sol_loaded)
        for sol in [example.sol, example.sol_loaded]:
            freevarcheck = {
                "A": 8.46,
                "C_D": 0.0206,
                "C_f": 0.0036,
                "C_L": 0.499,
                "Re": 3.68e06,
                "S": 16.4,
                "W": 7.34e03,
                "V": 38.2,
                "W_w": 2.40e03,
            }
            # sensitivity values from p. 34 of W. Hoburg's thesis
            senscheck = {
                r"(\frac{S}{S_{wet}})": 0.4300,
                "e": -0.4785,
                "V_{min}": -0.3691,
                "k": 0.4300,
                r"\mu": 0.0860,
                "(CDA0)": 0.0915,
                "C_{L,max}": -0.1845,
                r"\tau": -0.2903,
                "N_{ult}": 0.2903,
                "W_0": 1.0107,
                r"\rho": -0.2275,
            }
            for key, val in freevarcheck.items():
                sol_rat = mag(sol.primal[key]) / val
                assert abs(1 - sol_rat) < 1e-2
            for key, val in senscheck.items():
                sol_rat = sol.sens.variables[key] / val
                assert abs(1 - sol_rat) < 1e-2
        os.remove("solution.pkl")
        os.remove("referencesplot.json")
        os.remove("referencesplot.html")

    def test_relaxation(self, example):
        sol1 = example.mr1.solve(verbosity=0)
        sol2 = example.mr2.solve(verbosity=0)
        sol3 = example.mr3.solve(verbosity=0)
        assert sol1.cost == pytest.approx(1.414, rel=1e-2)
        assert sol2.cost == pytest.approx(2.0, rel=1e-2)
        assert sol3.cost == pytest.approx(2.0, rel=1e-2)

    def test_unbounded(self, example):
        assert example.sol.cost == pytest.approx(1e-30, rel=1e-2)

    def test_bemt_hover(self, example):
        # Minimum induced power for a 5-bin BEMT rotor at 1e4 N vehicle weight
        assert example.sol.cost == pytest.approx(43582.47, rel=1e-2)

    def test_gp_textbook(self, example):
        # Five classic textbook GP problems; assert all five costs
        assert example.sol1.cost == pytest.approx(100.0, rel=1e-2)  # BoxTransport
        assert example.sol2.cost == pytest.approx(8.0, rel=1e-2)  # FencePlot
        assert example.sol3.cost == pytest.approx(0.1925, rel=1e-2)  # BeamCrossSection
        assert example.sol4.cost == pytest.approx(13.5, rel=1e-2)  # BoxFromSheet
        assert example.sol5.cost == pytest.approx(0.009006, rel=1e-2)  # WorkSleep

    def test_fuel_burn(self, example):
        # Multi-point aircraft fuel burn minimization (total W_fuel, 3 conditions)
        assert example.sol.cost == pytest.approx(7561.15, rel=1e-2)
