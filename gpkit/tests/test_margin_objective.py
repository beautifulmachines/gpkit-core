"Tests for MarginObjective and MarginSolution"

from gpkit import MarginObjective, Model, Variable
from gpkit.examples.growth_allowance import GrowthAllowance

# ---------------------------------------------------------------------------
# Simple models used across multiple tests
# ---------------------------------------------------------------------------


class SimpleMarginModel(Model):
    """B >= c * A_allow, cost = B/A, margin = A_allow - B."""

    def setup(self):
        a = Variable("A_allow", 100.0, "kg", "Allowable mass")
        b = Variable("B_mass", "kg", "System mass")
        c = Variable("c_frac", 0.8, "", "Mass fraction")
        self.cost = b / a
        self.margin_objective = MarginObjective(
            name="mass margin",
            plus_var=a,
            minus_var=b,
        )
        return [b >= c * a]


class BothFreeModel(Model):
    """A and B both free, each bounded by separate constants."""

    def setup(self):
        a = Variable("A", "kg", "allowable")
        b = Variable("B", "kg", "actual")
        a_max = Variable("A_max", 120.0, "kg", "A upper bound")
        b_min = Variable("B_min", 50.0, "kg", "B lower bound")
        alpha = Variable("alpha", 1.5, "", "A–B ratio floor")
        self.cost = b / a
        self.margin_objective = MarginObjective("gap", plus_var=a, minus_var=b)
        return [a <= a_max, b >= b_min, a >= alpha * b]


class ConstMapModel(Model):
    """Constraint with an additive constant term → triggers const_mmap.

    P_prop + P_avionics <= P_max, margin = P_max - P_prop.
    After substitution P_avionics and P_max are both constants, so the
    pure-constant monomial P_avionics/P_max is moved to const_mmap.
    Cost P_max/P_prop (budget per propulsion) drives P_prop to its upper bound.
    """

    def setup(self):
        p_prop = Variable("P_prop", "W", "propulsion power")
        p_avionics = Variable("P_avionics", 50.0, "W", "avionics power (fixed)")
        p_max = Variable("P_max", 200.0, "W", "total power limit")
        self.cost = p_max / p_prop  # minimize budget/propulsion → p_prop at upper bound
        self.margin_objective = MarginObjective(
            "power margin",
            plus_var=p_max,
            minus_var=p_prop,
        )
        return [p_prop + p_avionics <= p_max]


# ---------------------------------------------------------------------------
# Helper: solve with perturbed constant for finite-difference validation
# ---------------------------------------------------------------------------


def _fd_margin_sens(model_cls, const_name, eps=0.005):
    """Return finite-difference ∂(margin)/∂c for constant `const_name`."""
    c0 = None

    def margin_at(factor):
        nonlocal c0
        m = model_cls()
        for vk in m.substitutions:
            if vk.name == const_name:
                c0 = float(m.substitutions[vk])
                m.substitutions[vk] = m.substitutions[vk] * factor
                break
        return m.solve(verbosity=0).derived.value

    fd_log = (margin_at(1 + eps) - margin_at(1 - eps)) / (2 * eps)
    return fd_log / c0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_margin_value():
    """A* = 100, B* = 80 → margin = 20 kg."""
    sol = SimpleMarginModel().solve(verbosity=0)
    assert sol.derived is not None
    assert abs(sol.derived.value - 20.0) < 1e-4
    assert abs(sol.derived.plus_value - 100.0) < 1e-4
    assert abs(sol.derived.minus_value - 80.0) < 1e-4


def test_simple_margin_sensitivities_analytical():
    """Analytical check of ∂margin/∂c for SimpleMarginModel.

    B* = c_frac * A_allow = 80, margin = A_allow - B* = 20.
    ∂(margin)/∂A_allow: log-sens = +20 kg, divided by A_allow=100 kg → +0.2.
    ∂(margin)/∂c_frac:  log-sens = -80 kg, divided by c_frac=0.8 → -100 kg.
    """
    sol = SimpleMarginModel().solve(verbosity=0)
    senss = {vk.name: v for vk, v in sol.derived.sensitivities.items()}
    assert abs(senss["A_allow"] - 0.2) < 1e-6
    assert abs(senss["c_frac"] - (-100.0)) < 1e-4


def test_simple_margin_sensitivities_fd():
    """Finite-difference cross-check for SimpleMarginModel."""
    sol = SimpleMarginModel().solve(verbosity=0)
    for vk, sens in sol.derived.sensitivities.items():
        fd = _fd_margin_sens(SimpleMarginModel, vk.name)
        assert (
            abs(sens - fd) / max(abs(sens), 1e-10) < 1e-3
        ), f"FD check failed for {vk.name}: computed={sens:.4g}, fd={fd:.4g}"


def test_both_free_model_fd():
    """Finite-difference cross-check for BothFreeModel."""
    sol = BothFreeModel().solve(verbosity=0)
    assert sol.derived is not None
    for vk, sens in sol.derived.sensitivities.items():
        fd = _fd_margin_sens(BothFreeModel, vk.name)
        # Use absolute tolerance for near-zero sensitivities (slack constraints)
        tol = max(1e-3 * max(abs(sens), abs(fd)), 1e-4)
        assert (
            abs(sens - fd) < tol
        ), f"FD check failed for {vk.name}: computed={sens:.4g}, fd={fd:.4g}"


def test_const_mmap_fd():
    """Finite-difference check for const_mmap case (P_prop + P_avionics <= P_max).

    P_avionics appears only in the const_mmap term after substitution.
    Verify that sensitivities are correct (the const_mmap term self-eliminates
    from the linear system but the primal values capture its effect correctly).
    """
    sol = ConstMapModel().solve(verbosity=0)
    assert sol.derived is not None
    # P_prop* = P_max - P_avionics = 150; margin = P_max - P_prop* = 50
    assert abs(sol.derived.value - 50.0) < 1e-4
    for vk, sens in sol.derived.sensitivities.items():
        fd = _fd_margin_sens(ConstMapModel, vk.name)
        tol = max(1e-3 * max(abs(sens), abs(fd)), 1e-4)
        assert (
            abs(sens - fd) < tol
        ), f"FD check failed for {vk.name}: computed={sens:.4g}, fd={fd:.4g}"


def test_no_margin_objective():
    """Models without margin_objective have sol.derived is None."""

    class Plain(Model):
        """Minimal model with no margin_objective."""

        def setup(self):
            x = Variable("x")
            self.cost = x
            return [x >= 1]

    sol = Plain().solve(verbosity=0)
    assert sol.derived is None


def test_growth_allowance_with_margin():
    """GrowthAllowance example (with MarginObjective) solves and has valid derived."""
    model = GrowthAllowance()
    sol = model.solve(verbosity=0)
    assert sol.derived is not None
    assert sol.derived.name == "mass margin"
    assert sol.derived.value > 0  # margin = budget - mass > 0


def test_growth_allowance_margin_fd():
    """Finite-difference cross-check for GrowthAllowance margin sensitivities."""

    def stable_key(vk):
        "Instance-stable identifier: strip root model instance counter."
        return (vk.name, vk.lineage[1:])

    def margin_at_factor(src_vk, factor):
        key = stable_key(src_vk)
        m = GrowthAllowance()
        for vk in m.substitutions:
            if stable_key(vk) == key:
                m.substitutions[vk] = m.substitutions[vk] * factor
                break
        return m.solve(verbosity=0).derived.value

    def c0_for(src_vk):
        key = stable_key(src_vk)
        m = GrowthAllowance()
        for vk in m.substitutions:
            if stable_key(vk) == key:
                return float(m.substitutions[vk])
        return 1.0

    sol = GrowthAllowance().solve(verbosity=0)
    eps = 0.005
    for vk, sens in sol.derived.sensitivities.items():
        c0 = c0_for(vk)
        fd_log = (margin_at_factor(vk, 1 + eps) - margin_at_factor(vk, 1 - eps)) / (
            2 * eps
        )
        fd = fd_log / c0
        assert (
            abs(sens - fd) / max(abs(sens), 1e-10) < 1e-2
        ), f"FD check failed for {vk.name} ({vk.ref}): computed={sens:.4g}, fd={fd:.4g}"


def test_to_ir_includes_margin_objective():
    """to_ir() includes margin_objective dict with correct refs."""
    model = SimpleMarginModel()
    ir = model.to_ir()
    assert "margin_objective" in ir
    mo_ir = ir["margin_objective"]
    assert mo_ir["name"] == "mass margin"
    # plus_var ref should end with "A_allow"
    assert "A_allow" in mo_ir["plus_var"]
    assert "B_mass" in mo_ir["minus_var"]


def test_to_ir_no_margin_objective():
    """to_ir() has no margin_objective key for plain models."""

    class Plain(Model):
        """Minimal model with no margin_objective."""

        def setup(self):
            x = Variable("x")
            self.cost = x
            return [x >= 1]

    ir = Plain().to_ir()
    assert "margin_objective" not in ir


def test_margin_solution_table():
    """MarginSolution.table() returns a non-empty string."""
    sol = SimpleMarginModel().solve(verbosity=0)
    table = sol.derived.table()
    assert isinstance(table, str)
    assert "mass margin" in table
    assert "A_allow" in table or "c_frac" in table
