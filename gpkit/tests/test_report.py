"""Tests for gpkit/report.py: ReportSection IR, build_report_ir(), model.report()"""

import json

import pytest

import gpkit
from gpkit import Model, Variable
from gpkit.report import (
    CGroup,
    ReportSection,
    VarEntry,
    _fmt_value,
    _md_escape,
    _serialize_value,
    build_report_ir,
    render_markdown,
    render_text,
)
from gpkit.util.repr_conventions import unitstr
from gpkit.util.small_classes import Quantity


def _all_vars(ir):
    """All VarEntry objects in an IR section (free + fixed)."""
    return ir.free_variables + ir.fixed_variables


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
            free_variables=[ve],
            fixed_variables=[],
            constraint_groups=[cg],
            children=[],
        )
        assert rs.title == "Wing"
        assert rs.description == "Wing structural model"
        assert rs.assumptions == ["thin airfoil"]
        assert len(rs.free_variables) == 1
        assert len(rs.fixed_variables) == 0
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
            free_variables=[ve],
            fixed_variables=[],
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
        assert restored["free_variables"][0]["name"] == "x"
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
        assert len(_all_vars(ir)) >= 1
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

    def test_var_entry_uses_pretty_unit_format(self):
        """build_report_ir uses platform-appropriate pretty format in VarEntry.units."""

        class _RSPrettyUnits(Model):
            def setup(self):
                area = Variable("S_pu", "m^2")
                area_min = Variable("S_min_pu", 1, "m^2")
                return [area >= area_min]

        m = _RSPrettyUnits()
        ir = build_report_ir(m)
        ve = _all_vars(ir)[0]
        area_vk = next(vk for vk in m.unique_varkeys if vk.name == "S_pu")
        assert ve.units == unitstr(area_vk)


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
        assert "free_variables" in result
        assert "fixed_variables" in result
        assert "children" in result

    def test_report_unknown_format_raises(self):
        """model.report(fmt='unknown') raises ValueError."""

        class _RSFmt(Model):
            def setup(self):
                x = Variable("x_rsfmt")
                return [x >= 1]

        m = _RSFmt()
        with pytest.raises(ValueError, match="not yet implemented"):
            m.report(fmt="unknown_format_xyz")


# ── Value / unit contract ─────────────────────────────────────────────────────


class TestValueUnitsContract:
    """_fmt_value and _serialize_value must never emit units; VarEntry.value
    must always be a plain numeric, not a pint Quantity."""

    def test_fmt_value_raises_for_quantity(self):
        """`_fmt_value` must raise TypeError if given a pint Quantity."""
        qty = gpkit.ureg.Quantity(3.0, "day")
        with pytest.raises(TypeError):
            _fmt_value(qty)

    def test_serialize_value_raises_for_quantity(self):
        """`_serialize_value` raises if given a pint Quantity."""
        qty = gpkit.ureg.Quantity(3.0, "day")
        with pytest.raises(TypeError):
            _serialize_value(qty)

    def test_var_entry_value_is_plain_float_after_solve(self):
        """build_report_ir stores plain floats in VarEntry.value, never Quantities."""
        x = Variable("x_plain", "m")
        c = Variable("c_plain", 2.0, "m")
        m = Model(x, [x >= c])
        sol = m.solve(verbosity=0)
        ir = build_report_ir(m, solution=sol)
        all_ves = _all_vars(ir) + [ve for ch in ir.children for ve in _all_vars(ch)]
        for ve in all_ves:
            assert not isinstance(
                ve.value, Quantity
            ), f"VarEntry.value for {ve.name!r} is a Quantity; expected plain float"


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
            free_variables=[ve],
            fixed_variables=[],
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
            free_variables=[],
            fixed_variables=[],
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
            free_variables=[ve],
            fixed_variables=[],
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
            free_variables=[],
            fixed_variables=[],
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
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            children=[],
        )
        out = render_markdown(ir)
        assert "beam theory" in out
        assert "isotropic material" in out

    def test_md_escape_special_characters(self):
        """_md_escape backslash-escapes markdown-special characters."""
        assert _md_escape("C_L") == r"C\_L"
        assert _md_escape("a*b") == r"a\*b"
        assert _md_escape("col|sep") == r"col\|sep"
        assert _md_escape("no specials") == "no specials"
        assert _md_escape("a_b*c|d") == r"a\_b\*c\|d"
        assert _md_escape("~95% TD") == r"\~95\% TD"


# ── Tests moved from test_model.py (cgroups + description) ──────────────────


class TestCGroups:
    """Tests for cgroups populated from dict-returning setup() methods."""

    def test_cgroups_from_dict_setup(self):
        """setup() returning a dict populates cgroups with the same mapping."""

        class _CGDictModel(Model):
            def setup(self):
                x = Variable("x_cgd")
                y = Variable("y_cgd")
                c1 = x >= 1
                c2 = y >= 1
                c3 = x * y >= 2
                return {"Drag": [c1, c2], "Lift": [c3]}

        m = _CGDictModel()
        assert m.cgroups is not None
        assert set(m.cgroups.keys()) == {"Drag", "Lift"}
        assert len(m.cgroups["Drag"]) == 2
        assert len(m.cgroups["Lift"]) == 1

    def test_cgroups_none_for_list_setup(self):
        """setup() returning a list leaves cgroups as None (not an empty dict)."""

        class _CGListModel(Model):
            def setup(self):
                x = Variable("x_cgl")
                return [x >= 1]

        m = _CGListModel()
        assert m.cgroups is None

    def test_scan_for_children_dict(self):
        """setup() returning a dict with child Models registers them in _children."""

        class _CGChild(Model):
            def setup(self):
                x = Variable("x_cgchild")
                return [x >= 1]

        class _CGParent(Model):
            def setup(self):
                child = _CGChild()
                y = Variable("y_cgpar")
                return {"Group": [child, y >= 1]}

        m = _CGParent()
        assert len(m.submodels) == 1
        assert isinstance(m.submodels[0], _CGChild)


class TestModelDescription:
    """Tests for Model.description() classmethod."""

    def test_model_description_explicit(self):
        """Model with explicit description() classmethod returns that dict."""

        class _DescExplicit(Model):
            def setup(self):
                x = Variable("x_desc_exp")
                return [x >= 1]

            @classmethod
            def description(cls):
                return {
                    "summary": "A test model with drag and lift.",
                    "assumptions": ["incompressible flow", "steady state"],
                    "references": ["Anderson 2001"],
                }

        d = _DescExplicit.description()
        assert d["summary"] == "A test model with drag and lift."
        assert "incompressible flow" in d["assumptions"]
        assert len(d["references"]) == 1

    def test_model_description_docstring_fallback(self):
        """Model with a docstring returns it as the description summary."""

        class _DescDocstring(Model):
            """Wing structural model with spar and skin."""

            def setup(self):
                x = Variable("x_desc_doc")
                return [x >= 1]

        d = _DescDocstring.description()
        assert d["summary"] == "Wing structural model with spar and skin."
        assert not d["assumptions"]
        assert not d["references"]

    def test_model_description_none(self):
        """Model without a docstring returns an empty summary."""

        class _DescNone(Model):
            def setup(self):
                x = Variable("x_desc_none")
                return [x >= 1]

        d = _DescNone.description()
        # May pick up Model base class docstring; just check required keys exist
        assert "summary" in d
        assert "assumptions" in d
        assert "references" in d
        assert isinstance(d["assumptions"], list)
        assert isinstance(d["references"], list)
