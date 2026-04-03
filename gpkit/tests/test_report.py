"""Tests for gpkit/report.py: ReportSection IR, build_report_ir(), model.report()"""

import json

import pytest

from gpkit import Model, Variable
from gpkit.report import (
    CGroup,
    ReportSection,
    VarEntry,
    build_report_ir,
    render_markdown,
    render_text,
)

# ── Dataclass structure ───────────────────────────────────────────────────────


class TestReportDataclasses:
    """Tests for VarEntry, CGroup, and ReportSection dataclasses."""

    def test_var_entry_in_report(self):
        """VarEntry holds name, latex, value, sensitivity, units, label fields."""
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
        """CGroup holds raw constraint objects, not pre-rendered strings."""
        x = Variable("x_cg_test_raw")
        c = x >= 1
        cg = CGroup(label="Drag", constraints=[c])
        assert cg.label == "Drag"
        assert len(cg.constraints) == 1
        assert not isinstance(cg.constraints[0], str)

    def test_report_section_dataclass(self):
        """ReportSection has all required fields."""
        ve = VarEntry(
            name="x", latex="x", value=1.0, sensitivity=None, units="-", label=""
        )
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
        assert not rs.children

    def test_report_section_to_dict(self):
        """ReportSection.to_dict() returns a JSON-serializable dict."""
        ve = VarEntry(
            name="x", latex="x", value=2.0, sensitivity=0.1, units="m", label="span"
        )
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

        class _RSSimple(Model):
            """A simple test model."""

            def setup(self):
                x = Variable("x_rs_simple")
                return [x >= 1]

        m = _RSSimple()
        ir = build_report_ir(m)
        assert isinstance(ir, ReportSection)
        assert ir.title == "_RSSimple"
        assert len(ir.variables) >= 1
        assert not ir.children

    def test_build_report_ir_nested(self):
        """build_report_ir(nested_model) returns ReportSection with children."""

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
        """model.report(fmt='dict') returns a JSON-serializable dict."""

        class _RSReport(Model):
            """Test model for report()."""

            def setup(self):
                x = Variable("x_rsrep")
                return [x >= 1]

        m = _RSReport()
        result = m.report(fmt="dict")
        assert isinstance(result, dict)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert "title" in result
        assert "variables" in result
        assert "children" in result

    def test_report_substitutions(self):
        """model.report(fmt='dict', substitutions=...) uses override without mutation"""

        class _RSSubst(Model):
            def setup(self):
                x = Variable("x_rssub")
                return [x >= 1]

        m = _RSSubst()
        original_subs = dict(m.substitutions)
        x_vk = next(iter(m.unique_varkeys))
        result = m.report(fmt="dict", substitutions={x_vk: 42.0})
        assert isinstance(result, dict)
        assert m.substitutions == original_subs

    def test_report_unknown_format_raises(self):
        """model.report(fmt='unknown') raises ValueError."""

        class _RSFmt(Model):
            def setup(self):
                x = Variable("x_rsfmt")
                return [x >= 1]

        m = _RSFmt()
        with pytest.raises(ValueError, match="not yet implemented"):
            m.report(fmt="unknown_format_xyz")


# ── Text format renderer ─────────────────────────────────────────────────────


class TestRenderText:
    """Tests for render_text() plain-text renderer."""

    def _simple_model_class(self):
        class _TxtSimple(Model):
            """A simple model for text rendering tests."""

            def setup(self):
                x = Variable("x_txt")
                y = Variable("y_txt")
                return [x >= 1, y >= x]

        return _TxtSimple

    def test_report_text_returns_string(self):
        """model.report(fmt='text') returns a non-empty string."""
        m = self._simple_model_class()()
        result = m.report(fmt="text")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_report_text_contains_class_name(self):
        """model.report(fmt='text') contains model class name as section header."""
        m = self._simple_model_class()()
        result = m.report(fmt="text")
        assert "_TxtSimple" in result

    def test_report_text_nested_indentation(self):
        """Nested submodel sections have 2-space indentation."""

        class _TxtChild(Model):
            def setup(self):
                c = Variable("c_txt_nested")
                return [c >= 1]

        class _TxtParent(Model):
            def setup(self):
                child = _TxtChild()
                p = Variable("p_txt_nested")
                return [p >= 1, child]

        m = _TxtParent()
        result = m.report(fmt="text")
        # Child section header should appear indented
        assert "_TxtChild" in result
        # Child section should be indented relative to parent
        split_lines = result.splitlines()
        parent_line = next((ln for ln in split_lines if "_TxtParent" in ln), None)
        child_line = next((ln for ln in split_lines if "_TxtChild" in ln), None)
        assert parent_line is not None
        assert child_line is not None
        assert len(child_line) - len(child_line.lstrip()) >= 2

    def test_report_text_constraint_operator_alignment(self):
        """Constraint block appears in text output."""

        class _TxtConstraints(Model):
            def setup(self):
                x = Variable("x_ca")
                y = Variable("y_ca")
                return [x >= 1, y >= x]

        m = _TxtConstraints()
        result = m.report(fmt="text")
        # Should contain constraint section header
        assert "Constraints" in result
        # Should contain variable names from constraints
        assert "x_ca" in result

    def test_report_text_with_solution(self):
        """model.report(solution=sol, fmt='text') includes variable values."""

        class _TxtSolM(Model):
            def setup(self):
                x = Variable("x_tsol_t1")
                x_max = Variable("x_tsol_max", 10)
                return [x >= 2, x <= x_max]

        m = _TxtSolM()
        sol = m.solve(verbosity=0)
        result = m.report(solution=sol, fmt="text")
        assert isinstance(result, str)
        assert "x_tsol" in result

    def test_report_text_substitutions(self):
        """model.report(fmt='text', substitutions=...) does not mutate model."""

        class _TxtSubst(Model):
            def setup(self):
                x = Variable("x_rssub2")
                return [x >= 1]

        m = _TxtSubst()
        original_subs = dict(m.substitutions)
        x_vk = next(iter(m.unique_varkeys))
        result = m.report(fmt="text", substitutions={x_vk: 5.0})
        assert isinstance(result, str)
        # Model should not be mutated
        assert m.substitutions == original_subs

    def test_render_text_direct(self):
        """render_text(ir) returns hierarchical text from a ReportSection IR."""
        ve = VarEntry(
            name="x", latex="x", value=1.0, sensitivity=None, units="m", label="span"
        )
        cg = CGroup(label="Load", constraints=[])
        ir = ReportSection(
            title="Wing",
            description="Wing model",
            assumptions=["thin airfoil"],
            variables=[ve],
            constraint_groups=[cg],
            children=[],
        )
        out = render_text(ir)
        assert isinstance(out, str)
        assert "Wing" in out
        assert "thin airfoil" in out
        assert "x" in out

    def test_render_text_assumptions_present(self):
        """render_text includes assumptions section when present."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=["steady level flight", "rigid structure"],
            variables=[],
            constraint_groups=[],
            children=[],
        )
        out = render_text(ir)
        assert "steady level flight" in out
        assert "rigid structure" in out


# ── Markdown format renderer ──────────────────────────────────────────────────


class TestRenderMarkdown:
    """Tests for render_markdown() markdown renderer."""

    def _simple_model_class(self):
        class _MdSimple(Model):
            """A simple model for markdown rendering tests."""

            def setup(self):
                x = Variable("x_md")
                y = Variable("y_md")
                return [x >= 1, y >= x]

        return _MdSimple

    def test_report_md_returns_string(self):
        """model.report(fmt='md') returns a non-empty string."""
        m = self._simple_model_class()()
        result = m.report(fmt="md")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_report_md_contains_h1_header(self):
        """model.report(fmt='md') contains '# ' header for top-level model."""
        m = self._simple_model_class()()
        result = m.report(fmt="md")
        assert "# _MdSimple" in result

    def test_report_md_variable_latex(self):
        """Markdown report contains $ delimiters around variable LaTeX."""
        m = self._simple_model_class()()
        result = m.report(fmt="md")
        # Should contain at least one $...$ for a variable
        assert "$" in result

    def test_report_md_pipe_table_header(self):
        """Markdown variable section contains pipe-table headers."""
        m = self._simple_model_class()()
        result = m.report(fmt="md")
        assert "| Variable" in result
        assert "Value" in result
        assert "Units" in result

    def test_report_md_nested_h2_header(self):
        """Nested submodels use ## header in markdown output."""

        class _MdChild(Model):
            def setup(self):
                c = Variable("c_md_nested")
                return [c >= 1]

        class _MdParent(Model):
            def setup(self):
                child = _MdChild()
                p = Variable("p_md_nested")
                return [p >= 1, child]

        m = _MdParent()
        result = m.report(fmt="md")
        assert "# _MdParent" in result
        assert "## _MdChild" in result

    def test_report_md_with_solution(self):
        """model.report(solution=sol, fmt='md') puts variable values in pipe table."""

        class _MdSolM(Model):
            def setup(self):
                x = Variable("x_mdsol_t1")
                x_max = Variable("x_mdsol_max", 10)
                return [x >= 3, x <= x_max]

        m = _MdSolM()
        sol = m.solve(verbosity=0)
        result = m.report(solution=sol, fmt="md")
        assert isinstance(result, str)
        assert "|" in result  # pipe table from variable table

    def test_render_markdown_direct(self):
        """render_markdown(ir) returns markdown from a ReportSection IR."""
        ve = VarEntry(
            name="x", latex="x", value=2.0, sensitivity=0.5, units="m", label="span"
        )
        cg = CGroup(label="Aero", constraints=[])
        ir = ReportSection(
            title="Fuselage",
            description="Fuselage drag model",
            assumptions=[],
            variables=[ve],
            constraint_groups=[cg],
            children=[],
        )
        out = render_markdown(ir)
        assert isinstance(out, str)
        assert "# Fuselage" in out
        assert "|" in out  # pipe table

    def test_render_markdown_description(self):
        """render_markdown includes description text when present."""
        ir = ReportSection(
            title="Propulsion",
            description="Turbofan engine sizing",
            assumptions=[],
            variables=[],
            constraint_groups=[],
            children=[],
        )
        out = render_markdown(ir)
        assert "Turbofan engine sizing" in out

    def test_render_markdown_assumptions(self):
        """render_markdown formats assumptions as bold assumption block."""
        ir = ReportSection(
            title="Structural",
            description="",
            assumptions=["beam theory", "isotropic material"],
            variables=[],
            constraint_groups=[],
            children=[],
        )
        out = render_markdown(ir)
        assert "beam theory" in out
        assert "isotropic material" in out
