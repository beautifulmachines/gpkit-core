"""Tests for gpkit/report.py: ReportSection IR, build_report_ir(), model.report()"""

import json

import pytest

from gpkit import Model, Var, Variable, Vectorize


# ── Dataclass structure ───────────────────────────────────────────────────────

class TestReportDataclasses:
    """Tests for VarEntry, CGroup, and ReportSection dataclasses."""

    def test_var_entry_in_report(self):
        """VarEntry holds name, latex, value, sensitivity, units, label fields."""
        from gpkit.report import VarEntry
        ve = VarEntry(
            name="x",
            latex="x",
            value=3.14,
            sensitivity=0.5,
            units="m",
            label="chord length",
        )
        assert ve.name == "x"
        assert ve.latex == "x"
        assert ve.value == pytest.approx(3.14)
        assert ve.sensitivity == pytest.approx(0.5)
        assert ve.units == "m"
        assert ve.label == "chord length"

    def test_cgroup_in_report(self):
        """CGroup holds label and constraints fields."""
        from gpkit.report import CGroup
        cg = CGroup(label="Drag", constraints=[("x", ">=", "1")])
        assert cg.label == "Drag"
        assert len(cg.constraints) == 1

    def test_report_section_dataclass(self):
        """ReportSection has title, description, assumptions, variables, constraint_groups, children."""
        from gpkit.report import CGroup, ReportSection, VarEntry
        ve = VarEntry(name="x", latex="x", value=1.0, sensitivity=None, units="-", label="")
        cg = CGroup(label="", constraints=[])
        rs = ReportSection(
            title="Wing",
            description="Wing structural model",
            assumptions=["thin airfoil"],
            variables=[ve],
            constraint_groups=[cg],
            children=[],
        )
        assert rs.title == "Wing"
        assert rs.description == "Wing structural model"
        assert rs.assumptions == ["thin airfoil"]
        assert len(rs.variables) == 1
        assert len(rs.constraint_groups) == 1
        assert rs.children == []

    def test_report_section_to_dict(self):
        """ReportSection.to_dict() returns a JSON-serializable dict."""
        from gpkit.report import CGroup, ReportSection, VarEntry
        ve = VarEntry(name="x", latex="x", value=2.0, sensitivity=0.1, units="m", label="span")
        cg = CGroup(label="Aero", constraints=[("x", ">=", "1")])
        rs = ReportSection(
            title="Fuselage",
            description="fuselage drag model",
            assumptions=[],
            variables=[ve],
            constraint_groups=[cg],
            children=[],
        )
        d = rs.to_dict()
        assert isinstance(d, dict)
        assert d["title"] == "Fuselage"
        # JSON round-trip
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["variables"][0]["name"] == "x"
        assert restored["constraint_groups"][0]["label"] == "Aero"


# ── build_report_ir() ────────────────────────────────────────────────────────

class TestBuildReportIR:
    """Tests for build_report_ir() tree traversal."""

    def test_build_report_ir_simple(self):
        """build_report_ir(simple_model) returns ReportSection with title=class name."""
        from gpkit.report import ReportSection, build_report_ir

        class _RSSimple(Model):
            """A simple test model."""
            def setup(self):
                x = Variable("x_rs_simple")
                return [x >= 1]

        m = _RSSimple()
        ir = build_report_ir(m)
        assert isinstance(ir, ReportSection)
        assert ir.title == "_RSSimple"
        # variables populated from unique_varkeys
        assert len(ir.variables) >= 1
        assert ir.children == []

    def test_build_report_ir_nested(self):
        """build_report_ir(nested_model) returns ReportSection with children."""
        from gpkit.report import ReportSection, build_report_ir

        class _RSChild(Model):
            def setup(self):
                x = Variable("x_rs_child")
                return [x >= 1]

        class _RSParent(Model):
            def setup(self):
                child = _RSChild()
                y = Variable("y_rs_parent")
                return [y >= 1, child]

        m = _RSParent()
        ir = build_report_ir(m)
        assert isinstance(ir, ReportSection)
        assert ir.title == "_RSParent"
        assert len(ir.children) == 1
        assert ir.children[0].title == "_RSChild"

    def test_build_report_ir_dict_cgroups(self):
        """build_report_ir produces named CGroups when setup() returns a dict."""
        from gpkit.report import build_report_ir

        class _RSDict(Model):
            def setup(self):
                x = Variable("x_rsd1")
                y = Variable("y_rsd1")
                return {"Drag": [x >= 1], "Lift": [y >= 1]}

        m = _RSDict()
        ir = build_report_ir(m)
        group_labels = [cg.label for cg in ir.constraint_groups]
        assert "Drag" in group_labels
        assert "Lift" in group_labels


# ── model.report() entry point ───────────────────────────────────────────────

class TestModelReport:
    """Tests for model.report() entry point."""

    def test_report_dict_json_serializable(self):
        """model.report(format='dict') returns a JSON-serializable dict."""

        class _RSReport(Model):
            """Test model for report()."""
            def setup(self):
                x = Variable("x_rsrep")
                return [x >= 1]

        m = _RSReport()
        result = m.report(format="dict")
        assert isinstance(result, dict)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        # Structure checks
        assert "title" in result
        assert "variables" in result
        assert "children" in result

    def test_report_substitutions(self):
        """model.report(format='dict', substitutions={x: val}) uses override value without mutation."""

        class _RSSubst(Model):
            def setup(self):
                x = Variable("x_rssub")
                return [x >= 1]

        m = _RSSubst()
        original_subs = dict(m.substitutions)
        # Find the x varkey
        x_vk = next(iter(m.unique_varkeys))
        result = m.report(format="dict", substitutions={x_vk: 42.0})
        assert isinstance(result, dict)
        # Substitutions must not be mutated
        assert m.substitutions == original_subs

    def test_report_unknown_format_raises(self):
        """model.report(format='unknown') raises ValueError."""

        class _RSFmt(Model):
            def setup(self):
                x = Variable("x_rsfmt")
                return [x >= 1]

        m = _RSFmt()
        with pytest.raises(ValueError, match="not yet implemented"):
            m.report(format="unknown_format_xyz")
