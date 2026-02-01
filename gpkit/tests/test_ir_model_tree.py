"""Tests for model_tree structural metadata in the IR (Phase 5)."""

# pylint: disable=invalid-name

import json

from gpkit import Model, Variable


class TestModelTree:
    """Tests for model_tree structural metadata in the IR."""

    def test_flat_model(self):
        """Flat model with no sub-models: model_tree has no children."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        ir = m.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Model"
        assert tree["children"] == []
        assert len(tree["constraint_indices"]) == 2
        assert tree["constraint_indices"] == [0, 1]
        # All variables should appear in the root node
        assert sorted(tree["variables"]) == sorted(ir["variables"].keys())

    def test_one_level_nesting(self):
        """Aircraft > Wing: tree has one child."""

        class Wing(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                S = Variable("S", 100, label="wing area")
                W = Variable("W", label="wing weight")
                self.cost = W
                return [W >= S * 0.1]

        class Aircraft(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                W = Variable("W", label="total weight")
                wing = Wing()
                self.cost = W
                return [W >= wing.cost * 1.2, wing]

        ac = Aircraft()
        ir = ac.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Aircraft"
        assert tree["instance_id"].startswith("Aircraft")
        assert len(tree["children"]) == 1

        wing = tree["children"][0]
        assert wing["class"] == "Wing"
        assert "Wing" in wing["instance_id"]
        assert wing["children"] == []

        # Aircraft owns its W variable
        ac_w_vars = [v for v in tree["variables"] if v.endswith(".W")]
        assert len(ac_w_vars) == 1
        # Wing owns S and W
        wing_vars = wing["variables"]
        assert any(v.endswith(".S") for v in wing_vars)
        assert any(v.endswith(".W") for v in wing_vars)

    def test_two_level_nesting(self):
        """Aircraft > Wing > Spar: tree has nested children."""

        class Spar(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                t = Variable("t", label="spar thickness")
                self.cost = t
                return [t >= 0.01]

        class Wing(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                S = Variable("S", label="wing area")
                spar = Spar()
                self.cost = S + spar.cost
                return [S >= 1, spar]

        class Aircraft(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                W = Variable("W", label="total weight")
                wing = Wing()
                self.cost = W
                return [W >= wing.cost * 1.2, wing]

        ac = Aircraft()
        ir = ac.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Aircraft"
        assert len(tree["children"]) == 1

        wing = tree["children"][0]
        assert wing["class"] == "Wing"
        assert len(wing["children"]) == 1

        spar = wing["children"][0]
        assert spar["class"] == "Spar"
        assert spar["children"] == []
        assert any(v.endswith(".t") for v in spar["variables"])

    def test_reused_submodel(self):
        """Widget with two Sub instances: same class, different instance_id."""

        class Sub(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                m = Variable("m")
                self.cost = m
                return [m >= 1]

        class Widget(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                s1 = Sub()
                s2 = Sub()
                self.cost = s1.cost + s2.cost
                return [s1, s2]

        w = Widget()
        ir = w.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Widget"
        assert len(tree["children"]) == 2

        sub0, sub1 = tree["children"]
        assert sub0["class"] == "Sub"
        assert sub1["class"] == "Sub"
        assert sub0["instance_id"] != sub1["instance_id"]
        # Each Sub owns its own m variable
        assert len(sub0["variables"]) == 1
        assert len(sub1["variables"]) == 1
        assert sub0["variables"][0] != sub1["variables"][0]
        assert sub0["variables"][0].endswith(".m")
        assert sub1["variables"][0].endswith(".m")

    def test_constraint_ownership(self):
        """Constraint indices correctly map to the flat constraint list."""

        class Sub(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                x = Variable("x")
                self.cost = x
                return [x >= 1]

        class Top(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                y = Variable("y")
                sub = Sub()
                self.cost = y + sub.cost
                return [y >= 2, sub]

        m = Top()
        ir = m.to_ir()
        tree = ir["model_tree"]

        # Top owns constraint 0 (y >= 2), Sub owns constraint 1 (x >= 1)
        assert tree["constraint_indices"] == [0]
        sub_tree = tree["children"][0]
        assert sub_tree["constraint_indices"] == [0 + 1]  # == [1]

        # All constraint indices should cover the full flat list
        all_indices = set(tree["constraint_indices"])
        for child in tree["children"]:
            all_indices.update(child["constraint_indices"])
        assert all_indices == set(range(len(ir["constraints"])))

    def test_variable_ownership(self):
        """Each variable in the IR appears in exactly one tree node."""

        class Wing(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                S = Variable("S", 100, label="wing area")
                W = Variable("W", label="wing weight")
                self.cost = W
                return [W >= S * 0.1]

        class Aircraft(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                W = Variable("W", label="total weight")
                wing = Wing()
                self.cost = W
                return [W >= wing.cost * 1.2, wing]

        ac = Aircraft()
        ir = ac.to_ir()
        tree = ir["model_tree"]

        # Collect all variables from all tree nodes
        def collect_vars(node):
            result = list(node["variables"])
            for child in node["children"]:
                result.extend(collect_vars(child))
            return result

        tree_vars = collect_vars(tree)
        ir_vars = set(ir["variables"].keys())

        # Every IR variable appears in some tree node
        assert set(tree_vars) == ir_vars
        # No variable appears in more than one node
        assert len(tree_vars) == len(set(tree_vars))

    def test_model_tree_json_serializable(self):
        """model_tree survives JSON round-trip."""

        class Wing(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                S = Variable("S", 100, label="wing area")
                self.cost = S
                return [S >= 1]

        class Aircraft(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                W = Variable("W", label="total weight")
                wing = Wing()
                self.cost = W
                return [W >= wing.cost, wing]

        ac = Aircraft()
        ir = ac.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        assert "model_tree" in ir2
        assert ir2["model_tree"]["class"] == "Aircraft"
        assert len(ir2["model_tree"]["children"]) == 1

    def test_phase4_tests_still_pass_with_model_tree(self):
        """model_tree is additive: Phase 4 round-trip still works."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        ir = m.to_ir()
        assert "model_tree" in ir

        # from_ir ignores model_tree and still produces a solvable model
        m2 = Model.from_ir(ir)
        sol = m.solve(verbosity=0)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4
