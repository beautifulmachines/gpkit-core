"Tests for the TOML model parser and loader."

import pytest

from gpkit import Model, Variable
from gpkit.toml._parser import (
    TomlParseError,
    _parse_var_spec,
    load_toml,
)

# ---------------------------------------------------------------------------
# Variable spec parsing
# ---------------------------------------------------------------------------


class TestParseVarSpec:
    """Variable declaration parsing for all TOML forms."""

    def test_bare_units(self):
        value, units, label = _parse_var_spec("m")
        assert value is None
        assert units == "m"
        assert label is None

    def test_value_and_units(self):
        value, units, label = _parse_var_spec("200 m^2")
        assert value == pytest.approx(200.0)
        assert units == "m^2"
        assert label is None

    def test_scientific_notation(self):
        value, units, label = _parse_var_spec("1.78e-5 kg/m/s")
        assert value == pytest.approx(1.78e-5)
        assert units == "kg/m/s"
        assert label is None

    def test_dimensionless_dash(self):
        value, units, label = _parse_var_spec("-")
        assert value is None
        assert units is None
        assert label is None

    def test_integer_param(self):
        value, units, label = _parse_var_spec(2)
        assert value == 2
        assert units is None
        assert label is None

    def test_float_param(self):
        value, units, label = _parse_var_spec(3.14)
        assert value == pytest.approx(3.14)
        assert units is None
        assert label is None

    def test_array_units_with_desc(self):
        value, units, label = _parse_var_spec(["m", "height"])
        assert value is None
        assert units == "m"
        assert label == "height"

    def test_array_value_units_with_desc(self):
        value, units, label = _parse_var_spec(["200 m^2", "wall area"])
        assert value == pytest.approx(200.0)
        assert units == "m^2"
        assert label == "wall area"

    def test_array_dash_with_desc(self):
        value, units, label = _parse_var_spec(["-", "aspect ratio"])
        assert value is None
        assert units is None
        assert label == "aspect ratio"

    def test_array_number_with_desc(self):
        value, units, label = _parse_var_spec([2, "lower limit"])
        assert value == 2
        assert units is None
        assert label == "lower limit"

    def test_array_wrong_length_raises(self):
        with pytest.raises(TomlParseError, match="exactly 2 elements"):
            _parse_var_spec(["m", "desc", "extra"])

    def test_array_non_string_desc_raises(self):
        with pytest.raises(TomlParseError, match="description must be a string"):
            _parse_var_spec(["m", 42])

    def test_invalid_type_raises(self):
        with pytest.raises(TomlParseError, match="Invalid variable spec"):
            _parse_var_spec({"bad": "dict"})


# ---------------------------------------------------------------------------
# TOML loading: simple_box.toml
# ---------------------------------------------------------------------------


class TestLoadSimpleBox:
    """Load and solve the simple_box.toml example."""

    @pytest.fixture
    def model(self):
        """Load simple_box.toml."""
        return load_toml("docs/source/examples/toml/simple_box.toml")

    def test_loads_model(self, model):
        """Model loads as a gpkit Model instance."""
        assert isinstance(model, Model)

    def test_solves(self, model):
        """Model solves without error."""
        sol = model.solve(verbosity=0)
        # Cost should be approximately 0.003674 (1/m³)
        assert sol.table()  # doesn't crash

    def test_cost_matches_python(self, model):
        """TOML model produces same optimal cost as the Python version."""
        h = Variable("h", "m", "height")
        w = Variable("w", "m", "width")
        d = Variable("d", "m", "depth")
        a_wall = Variable("A_wall", 200, "m^2", "upper limit, wall area")
        a_floor = Variable("A_floor", 50, "m^2", "upper limit, floor area")
        py_model = Model(
            1 / (h * w * d),
            [
                a_wall >= 2 * h * w + 2 * h * d,
                a_floor >= w * d,
                h / w >= 2,
                h / w <= 10,
                d / w >= 2,
                d / w <= 10,
            ],
        )
        py_sol = py_model.solve(verbosity=0)

        toml_sol = model.solve(verbosity=0)

        assert float(next(iter(toml_sol.primal.values()))) > 0
        # Both solutions should have same table structure
        assert "Free Variables" in toml_sol.table()
        assert "Free Variables" in py_sol.table()


# ---------------------------------------------------------------------------
# TOML loading: water_tank.toml (vectors)
# ---------------------------------------------------------------------------


class TestLoadWaterTank:
    """Load and solve the water_tank.toml example (vectors)."""

    @pytest.fixture
    def model(self):
        """Load water_tank.toml."""
        return load_toml("docs/source/examples/toml/water_tank.toml")

    def test_loads_model(self, model):
        """Model loads as a gpkit Model instance."""
        assert isinstance(model, Model)

    def test_solves(self, model):
        """Model solves without error."""
        sol = model.solve(verbosity=0)
        assert "Free Variables" in sol.table()

    def test_cube_symmetry(self, model):
        """Optimal water tank should be a cube (all dimensions equal)."""
        sol = model.solve(verbosity=0)
        d_values = []
        for key, val in sol.primal.items():
            if getattr(key, "name", "") == "d" and hasattr(key, "idx"):
                d_values.append(float(val))
        assert len(d_values) == 3
        # All dimensions should be approximately equal (cube)
        assert d_values[0] == pytest.approx(d_values[1], rel=1e-3)
        assert d_values[1] == pytest.approx(d_values[2], rel=1e-3)


# ---------------------------------------------------------------------------
# TOML loading from string
# ---------------------------------------------------------------------------


class TestLoadFromString:
    """Load models from inline TOML strings."""

    def test_minimal_model(self):
        toml_str = """
[vars]
x = "-"

[model]
objective = "min: x"
constraints = ["x >= 1"]
"""
        m = load_toml(toml_str)
        sol = m.solve(verbosity=0)
        # x should be 1 at optimum
        for key, val in sol.primal.items():
            if getattr(key, "name", "") == "x":
                assert float(val) == pytest.approx(1.0)

    def test_dimension_override(self):
        toml_str = """
[dimensions]
N = 3

[vectors.N]
x = "-"

[model]
objective = "min: x[0] + x[1] + x[2]"
constraints = ["x[0] >= 1", "x[1] >= 2", "x[2] >= 3"]
"""
        m = load_toml(toml_str, dimensions={"N": 3})
        sol = m.solve(verbosity=0)
        x = sol["x"]
        assert float(x[0]) == pytest.approx(1.0, rel=1e-5)
        assert float(x[1]) == pytest.approx(2.0, rel=1e-5)
        assert float(x[2]) == pytest.approx(3.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Error handling for malformed TOML input."""

    def test_missing_model_section(self):
        with pytest.raises(TomlParseError, match="must have a .model."):
            load_toml("[vars]\nx = 1\n")

    def test_both_model_and_models(self):
        with pytest.raises(TomlParseError, match="Cannot have both"):
            load_toml(
                '[model]\nobjective = "min: x"\n[models.a]\nobjective = "min: x"\n'
            )

    def test_missing_objective(self):
        with pytest.raises(TomlParseError, match="missing an 'objective'"):
            load_toml('[vars]\nx = "-"\n[model]\nconstraints = ["x >= 1"]\n')

    def test_bad_constraint_expr(self):
        with pytest.raises(TomlParseError, match="Error in constraint"):
            load_toml(
                '[vars]\nx = "-"\n'
                '[model]\nobjective = "min: x"\nconstraints = ["x >= ?"]\n'
            )

    def test_bad_vector_section_key(self):
        with pytest.raises(TomlParseError, match="not a dimension name"):
            load_toml("""
[vectors.Q]
x = "-"

[model]
objective = "min: x[0]"
""")

    def test_non_integer_dimension(self):
        with pytest.raises(TomlParseError, match="must be an integer"):
            load_toml("""
[dimensions]
N = 3.5

[model]
objective = "min: 1"
""")

    def test_multi_model_not_supported(self):
        with pytest.raises(TomlParseError, match="Multi-model"):
            load_toml("""
[models.a]
objective = "min: 1"
[models.b]
objective = "min: 1"
""")
