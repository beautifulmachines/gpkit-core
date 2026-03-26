"Tests for gpkit.toml save_subs / load_subs / apply_subs."

import io

import pytest

from gpkit import Model, Variable
from gpkit.toml import apply_subs, load_subs, save_subs
from gpkit.toml._parser import TomlParseError
from gpkit.units import qty, units

# ---------------------------------------------------------------------------
# Shared model fixtures
# ---------------------------------------------------------------------------


class Engine(Model):
    """Simple engine submodel with thrust and efficiency variables."""

    def setup(self):
        """Set up engine with thrust and efficiency constants."""
        thrust = Variable("T", 2000, "N", "thrust")
        eta = Variable("eta", 0.85, "-", "efficiency")
        self.cost = thrust
        return [thrust >= thrust, eta >= eta]  # pylint: disable=comparison-with-itself


class Aircraft(Model):
    """Aircraft model with weight and an Engine submodel."""

    def setup(self):
        """Set up aircraft with weight and engine."""
        weight = Variable("W", 5000, "N", "weight")
        self.engine = Engine()  # pylint: disable=attribute-defined-outside-init
        self.cost = weight
        return [weight >= weight, self.engine]  # pylint: disable=comparison-with-itself


class FreeVarsModel(Model):
    """Model with no substituted values — all variables are free."""

    def setup(self):
        """Set up two free variables."""
        x = Variable("x", "m")
        y = Variable("y", "m")
        self.cost = x
        return [x >= y]


# ---------------------------------------------------------------------------
# save_subs
# ---------------------------------------------------------------------------


class TestSaveSubs:
    """Tests for save_subs()."""

    def test_save_subs_marker(self):
        """_gpkit_subs = true marker is present in output."""
        model = Aircraft()
        toml_str = save_subs(model)
        assert "_gpkit_subs = true" in toml_str

    def test_save_subs_sections(self):
        """Correct [Aircraft] and [Aircraft.Engine] section headers are emitted."""
        model = Aircraft()
        toml_str = save_subs(model)
        assert "[Aircraft]" in toml_str
        assert "[Aircraft.Engine]" in toml_str

    def test_save_subs_values(self):
        """Variable values are formatted correctly."""
        model = Aircraft()
        toml_str = save_subs(model)
        assert 'W = "5000 N"' in toml_str
        assert 'T = "2000 N"' in toml_str
        assert "eta = 0.85" in toml_str

    def test_save_subs_monomial_value(self):
        "Monomial-valued substitution (created via units()) is serialized correctly."

        class Orbit(Model):
            """Orbital mechanics model."""

            def setup(self):
                """Set up with gravitational parameter and radius."""
                mu = Variable("mu", "km^2/s^2")
                r = Variable("r", "km")
                self.cost = r
                return [r >= r, mu >= mu], {
                    mu: 9.0 * units("km^2/s^2")
                }  # pylint: disable=comparison-with-itself

        model = Orbit()
        toml_str = save_subs(model)
        assert 'mu = "9 km^2/s^2"' in toml_str

    def test_save_subs_skips_callables(self):
        """Callable substitution values (e.g. sweep functions) are not emitted."""

        class SweepModel(Model):
            """Model with a callable (sweep) substitution."""

            def setup(self):
                """Set up with a callable substitution."""
                x = Variable("x", "m")
                self.cost = x
                return [x >= x], {
                    x: lambda _: 1.0
                }  # pylint: disable=comparison-with-itself

        model = SweepModel()
        toml_str = save_subs(model)
        # callable sub must not appear as a variable entry
        assert "x =" not in toml_str

    def test_save_subs_writes_file(self, tmp_path):
        """path= kwarg writes the file; content matches return value."""
        model = Aircraft()
        path = tmp_path / "subs.toml"
        returned = save_subs(model, path=path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == returned

    def test_save_subs_precision(self):
        """Values with more than 4 sig figs are preserved exactly."""

        class PhysicsModel(Model):
            """Model with a high-precision physical constant."""

            def setup(self):
                """Set up with gravitational acceleration."""
                g = Variable("g", 9.80665, "m/s^2", "gravitational acceleration")
                self.cost = g
                return [g >= g]  # pylint: disable=comparison-with-itself

        model = PhysicsModel()
        toml_str = save_subs(model)
        assert "9.80665" in toml_str

    def test_save_subs_no_subs(self):
        """Model with no substituted variables produces only the header."""
        model = FreeVarsModel()
        toml_str = save_subs(model)
        assert "_gpkit_subs = true" in toml_str
        assert "[" not in toml_str.replace("_gpkit_subs = true", "")


# ---------------------------------------------------------------------------
# load_subs
# ---------------------------------------------------------------------------


SIMPLE_SUBS_TOML = b"""\
_gpkit_subs = true

[Aircraft]
W = "5000 N"

[Aircraft.Engine]
T = "2000 N"
eta = 0.85
"""


class TestLoadSubs:
    """Tests for load_subs()."""

    def test_load_subs_structure(self):
        """Returns correct nested structure keyed by lineage path."""
        subs = load_subs(io.BytesIO(SIMPLE_SUBS_TOML))
        assert "Aircraft" in subs
        assert "Aircraft.Engine" in subs
        assert "W" in subs["Aircraft"]
        assert "T" in subs["Aircraft.Engine"]
        assert "eta" in subs["Aircraft.Engine"]

    def test_load_subs_dimensioned(self):
        """Dimensioned string values are parsed to pint Quantities."""
        subs = load_subs(io.BytesIO(SIMPLE_SUBS_TOML))
        weight_val = subs["Aircraft"]["W"]
        # Should be a pint Quantity whose magnitude in N is approximately 5000
        assert hasattr(weight_val, "to")  # pint Quantity
        assert weight_val.to("N").magnitude == pytest.approx(5000.0)

    def test_load_subs_dimensionless(self):
        """Dimensionless float values are returned as plain floats."""
        subs = load_subs(io.BytesIO(SIMPLE_SUBS_TOML))
        eta_val = subs["Aircraft.Engine"]["eta"]
        assert eta_val == pytest.approx(0.85)

    def test_load_subs_bad_marker(self):
        """Missing _gpkit_subs marker raises TomlParseError."""
        bad_toml = b"[Aircraft]\nW = 5000\n"
        with pytest.raises(TomlParseError, match="_gpkit_subs"):
            load_subs(io.BytesIO(bad_toml))

    def test_load_subs_accepts_filelike(self):
        """load_subs works with a binary file-like object, not just a path."""
        subs = load_subs(io.BytesIO(SIMPLE_SUBS_TOML))
        assert "Aircraft" in subs

    def test_load_subs_from_path(self, tmp_path):
        """load_subs works with a file path."""
        path = tmp_path / "subs.toml"
        path.write_bytes(SIMPLE_SUBS_TOML)
        subs = load_subs(path)
        assert "Aircraft" in subs


# ---------------------------------------------------------------------------
# apply_subs
# ---------------------------------------------------------------------------


class TestApplySubs:
    """Tests for apply_subs()."""

    def _aircraft_weight_vk(self, model):
        """Return the VarKey for W in the Aircraft section."""
        return next(vk for vk in model.unique_varkeys if vk.name == "W")

    def test_apply_subs_roundtrip(self):
        """save → load → apply leaves the model substitutions unchanged."""
        model = Aircraft()
        subs = load_subs(io.BytesIO(save_subs(model).encode()))
        apply_subs(model, subs)
        weight_vk = self._aircraft_weight_vk(model)
        assert float(model.substitutions[weight_vk]) == pytest.approx(5000.0)

    def test_apply_subs_override(self):
        """Modifying a loaded subs value and applying reflects in the model."""
        model = Aircraft()
        subs = load_subs(io.BytesIO(save_subs(model).encode()))
        subs["Aircraft"]["W"] = qty("N") * 6000.0
        apply_subs(model, subs)
        weight_vk = self._aircraft_weight_vk(model)
        assert float(model.substitutions[weight_vk]) == pytest.approx(6000.0)

    def test_apply_subs_from_path(self, tmp_path):
        """apply_subs accepts a file path directly."""
        model = Aircraft()
        path = tmp_path / "subs.toml"
        save_subs(model, path=path)
        # Overwrite W in the file
        content = path.read_text(encoding="utf-8").replace(
            'W = "5000 N"', 'W = "7000 N"'
        )
        path.write_text(content, encoding="utf-8")
        apply_subs(model, path)
        weight_vk = self._aircraft_weight_vk(model)
        assert float(model.substitutions[weight_vk]) == pytest.approx(7000.0)

    def test_apply_subs_unknown_lineage(self):
        """A lineage path not in the model tree warns and is skipped."""
        model = Aircraft()
        subs = {"OldModel": {"x": 1.0}}
        with pytest.warns(UserWarning, match="OldModel"):
            apply_subs(model, subs)

    def test_apply_subs_unknown_var(self):
        """An unknown variable name in a known section warns and is skipped."""
        model = Aircraft()
        subs = {"Aircraft": {"old_var": 1.0, "W": qty("N") * 5000.0}}
        with pytest.warns(UserWarning, match="old_var"):
            apply_subs(model, subs)
        # W should still be applied
        weight_vk = self._aircraft_weight_vk(model)
        assert float(model.substitutions[weight_vk]) == pytest.approx(5000.0)

    def test_apply_subs_model_evolution(self):
        """Subs file from a model with more variables works on a trimmed model.

        Simulates removing a submodel: the old subs file has entries for
        Aircraft.Engine which no longer exists.  The Aircraft section still
        applies; the Engine section warns and is skipped.
        """
        # Build old subs dict directly (simulates a subs file from a previous
        # version of Aircraft that had an Engine submodel)
        old_subs = {
            "Aircraft": {"W": qty("N") * 5000.0},
            "Aircraft.Engine": {"T": qty("N") * 2000.0, "eta": 0.85},
        }

        # New model: Aircraft without Engine
        class Aircraft(Model):  # pylint: disable=redefined-outer-name
            """Trimmed aircraft model without Engine submodel."""

            def setup(self):
                """Set up with weight only."""
                weight = Variable("W", 5000, "N", "weight")
                self.cost = weight
                return [weight >= weight]  # pylint: disable=comparison-with-itself

        model_new = Aircraft()
        with pytest.warns(UserWarning, match="Aircraft.Engine"):
            apply_subs(model_new, old_subs)

        # W (which still exists) must be applied
        weight_vk = next(vk for vk in model_new.unique_varkeys if vk.name == "W")
        assert float(model_new.substitutions[weight_vk]) == pytest.approx(5000.0)
