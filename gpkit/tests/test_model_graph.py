"""Tests for Model graph: _children, submodels, walk(), and get_var()."""

# pylint: disable=invalid-name,attribute-defined-outside-init

import pytest

from gpkit import Model, Variable
from gpkit.exceptions import AmbiguousVariable, VariableNotFound


class TestModelGraph:
    """Tests for Model._children explicit graph (GRAPH-01)."""

    def test_flat_model_has_empty_children(self):
        x = Variable("x")
        m = Model(x, [x >= 1])
        assert not m.submodels

    def test_single_child_in_children(self):
        class _MGWing(Model):
            def setup(self):
                S = Variable("S")
                self.cost = S
                return [S >= 10]

        class _MGAircraft(Model):
            wing: "_MGWing"

            def setup(self):
                W = Variable("W")
                self.wing = _MGWing()
                self.cost = W
                return [W >= self.wing.cost * 1.2, self.wing]

        a = _MGAircraft()
        assert len(a.submodels) == 1
        assert a.submodels[0] is a.wing
        # Confirm get_var() resolves through the child attr mapping
        assert a.get_var("wing.S").key.name == "S"

    def test_two_children_order_preserved(self):
        class _MGSubA(Model):
            def setup(self):
                x = Variable("x")
                self.cost = x
                return [x >= 1]

        class _MGSubB(Model):
            def setup(self):
                y = Variable("y")
                self.cost = y
                return [y >= 2]

        class _MGTop(Model):
            a: "_MGSubA"
            b: "_MGSubB"

            def setup(self):
                W = Variable("W")
                self.a = _MGSubA()
                self.b = _MGSubB()
                self.cost = W
                return [W >= self.a.cost + self.b.cost, self.a, self.b]

        t = _MGTop()
        assert len(t.submodels) == 2
        assert t.submodels[0] is t.a
        assert t.submodels[1] is t.b

    def test_dict_return_children_detected(self):
        """Children inside a dict constraint return are found by _scan_for_children."""

        class _MGInner(Model):
            def setup(self):
                x = Variable("x")
                self.cost = x
                return [x >= 1]

        class _MGOuter(Model):
            inner: "_MGInner"

            def setup(self):
                W = Variable("W")
                self.inner = _MGInner()
                self.cost = W
                return {"constraints": [self.inner, W >= self.inner.cost * 1.1]}

        m = _MGOuter()
        assert m.submodels == [m.inner]

    def test_children_of_same_class_are_distinguishable(self):
        """Two children of the same class are distinguished by attr name."""

        class _MGSub(Model):
            def setup(self):
                x = Variable("x")
                self.cost = x
                return [x >= 1]

        class _MGMultiTop(Model):
            sub1: "_MGSub"
            sub2: "_MGSub"

            def setup(self):
                W = Variable("W")
                self.sub1 = _MGSub()
                self.sub2 = _MGSub()
                self.cost = W
                return [W >= self.sub1.cost + self.sub2.cost, self.sub1, self.sub2]

        t = _MGMultiTop()
        x1 = t.get_var("sub1.x")
        x2 = t.get_var("sub2.x")
        assert x1.key != x2.key  # different VarKeys despite same class


class TestSubmodels:
    """Tests for model.submodels and model.walk() (GRAPH-03)."""

    def test_submodels_returns_list(self):
        class _SMWing(Model):
            def setup(self):
                S = Variable("S")
                self.cost = S
                return [S >= 10]

        class _SMAircraft(Model):
            wing: "_SMWing"

            def setup(self):
                W = Variable("W")
                self.wing = _SMWing()
                self.cost = W
                return [W >= self.wing.cost * 1.2, self.wing]

        a = _SMAircraft()
        assert a.submodels == [a.wing]

    def test_submodels_returns_direct_children_only(self):
        """submodels does not include grandchildren."""

        class _SMSpar(Model):
            def setup(self):
                t = Variable("t")
                self.cost = t
                return [t >= 0.01]

        class _SMWingWithSpar(Model):
            spar: "_SMSpar"

            def setup(self):
                S = Variable("S")
                self.spar = _SMSpar()
                self.cost = S
                return [S >= 10, self.spar]

        class _SMAircraftNested(Model):
            wing: "_SMWingWithSpar"

            def setup(self):
                W = Variable("W")
                self.wing = _SMWingWithSpar()
                self.cost = W
                return [W >= 1.2 * self.wing.cost, self.wing]

        a = _SMAircraftNested()
        assert len(a.submodels) == 1  # only wing, not spar
        assert a.submodels[0] is a.wing

    def test_walk_yields_all_descendants_depth_first(self):
        class _WalkSpar(Model):
            def setup(self):
                t = Variable("t")
                self.cost = t
                return [t >= 0.01]

        class _WalkWing(Model):
            spar: "_WalkSpar"

            def setup(self):
                S = Variable("S")
                self.spar = _WalkSpar()
                self.cost = S
                return [S >= 10, self.spar]

        class _WalkAircraft(Model):
            wing: "_WalkWing"

            def setup(self):
                W = Variable("W")
                self.wing = _WalkWing()
                self.cost = W
                return [W >= 1.2 * self.wing.cost, self.wing]

        a = _WalkAircraft()
        walked = list(a.walk())
        assert walked[0] is a.wing
        assert walked[1] is a.wing.spar
        assert len(walked) == 2

    def test_budget_rollup_pattern(self):
        """sum(child.W for child in model.submodels) works without helper."""

        class _BudgetComp(Model):
            def setup(self, w_min):
                self.W = Variable("W")
                self.cost = self.W
                return [self.W >= w_min]

        class _BudgetSystem(Model):
            comp1: "_BudgetComp"
            comp2: "_BudgetComp"

            def setup(self):
                W_total = Variable("W_total")
                self.comp1 = _BudgetComp(1.0)
                self.comp2 = _BudgetComp(2.0)
                self.cost = W_total
                return [
                    W_total >= self.comp1.W + self.comp2.W,
                    self.comp1,
                    self.comp2,
                ]

        s = _BudgetSystem()
        # Verify submodels gives direct children
        assert len(s.submodels) == 2
        # Budget rollup via submodels outside setup() - core GRAPH-03 pattern
        W_rollup = sum(child.W for child in s.submodels)
        assert W_rollup is not None
        sol = s.solve(verbosity=0)
        assert sol.cost == pytest.approx(3.0, rel=1e-3)


class TestGetVar:
    """Tests for Model.get_var() dotted-path resolver (GRAPH-02)."""

    @pytest.fixture
    def aircraft(self):
        """Simple two-level aircraft model for get_var() tests."""

        class _GVWing(Model):
            def setup(self):
                S = Variable("S")
                self.cost = S
                return [S >= 10]

        class _GVAircraft(Model):
            wing: "_GVWing"

            def setup(self):
                W = Variable("W")
                self.wing = _GVWing()
                self.cost = W
                return [W >= self.wing.cost * 1.2, self.wing]

        return _GVAircraft()

    def test_get_var_one_level(self, aircraft):
        """model.get_var('wing.S') returns the Wing's S Variable."""
        S = aircraft.get_var("wing.S")
        assert S.key.name == "S"

    def test_get_var_own_variable(self, aircraft):
        """model.get_var('W') returns Aircraft's own W."""
        W = aircraft.get_var("W")
        assert W.key.name == "W"

    def test_get_var_not_found_leaf(self, aircraft):
        """Raises VariableNotFound for missing variable name."""
        with pytest.raises(VariableNotFound, match="nonexistent"):
            aircraft.get_var("nonexistent")

    def test_get_var_not_found_child(self, aircraft):
        """Raises VariableNotFound for missing child attribute."""
        with pytest.raises(VariableNotFound, match="fuselage"):
            aircraft.get_var("fuselage.W")

    def test_get_var_ambiguous(self):
        """AmbiguousVariable is importable and is a LookupError subclass."""
        assert issubclass(AmbiguousVariable, LookupError)

    def test_get_var_two_levels(self):
        """model.get_var('wing.spar.t') traverses two levels."""

        class _GVSpar(Model):
            def setup(self):
                t = Variable("t")
                self.cost = t
                return [t >= 0.01]

        class _GVWingWithSpar(Model):
            spar: "_GVSpar"

            def setup(self):
                S = Variable("S")
                self.spar = _GVSpar()
                self.cost = S
                return [S >= 10, self.spar]

        class _GVAircraftDeep(Model):
            wing: "_GVWingWithSpar"

            def setup(self):
                W = Variable("W")
                self.wing = _GVWingWithSpar()
                self.cost = W
                return [W >= 1.2 * self.wing.cost, self.wing]

        a = _GVAircraftDeep()
        t = a.get_var("wing.spar.t")
        assert t.key.name == "t"

    def test_get_var_works_before_solve(self, aircraft):
        """get_var() works on an unsolved model."""
        S = aircraft.get_var("wing.S")
        assert S is not None  # no solve needed

    def test_get_var_two_children_same_class(self):
        """get_var distinguishes sub1 vs sub2 of same class."""

        class _GVSub(Model):
            def setup(self):
                x = Variable("x")
                self.cost = x
                return [x >= 1]

        class _GVTop(Model):
            sub1: "_GVSub"
            sub2: "_GVSub"

            def setup(self):
                W = Variable("W")
                self.sub1 = _GVSub()
                self.sub2 = _GVSub()
                self.cost = W
                return [W >= self.sub1.cost + self.sub2.cost, self.sub1, self.sub2]

        t = _GVTop()
        x1 = t.get_var("sub1.x")
        x2 = t.get_var("sub2.x")
        assert x1.key != x2.key  # different VarKeys from different lineage contexts
