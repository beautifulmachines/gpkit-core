"""Tests for model description metadata, IR wiring, and renderer output.

Split from test_report.py to keep that file within the line-count limit.
Covers: Model.description(), class-attr assumptions/references, ReportSection
fields (references, front_matter, toc, objective_*), build_report_ir wiring,
and renderer output for all new fields.
"""

import pytest

from gpkit import Model, Variable
from gpkit.report import (
    ReportSection,
    build_report_ir,
    render_markdown,
    render_text,
)

# ── ReportSection references / front_matter / toc fields ─────────────────────


class TestReportSectionReferences:
    """Tests for references, front_matter, and toc fields on ReportSection."""

    def test_references_field_exists(self):
        """ReportSection has a references field defaulting to []."""
        ir = ReportSection(
            title="Wing",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        assert not ir.references

    def test_references_in_to_dict(self):
        """ReportSection.to_dict() includes references."""
        ir = ReportSection(
            title="Wing",
            description="",
            assumptions=[],
            references=["Anderson 2001"],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        d = ir.to_dict()
        assert "references" in d
        assert d["references"] == ["Anderson 2001"]

    def test_front_matter_field_exists(self):
        """ReportSection has a front_matter field defaulting to ''."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        assert ir.front_matter == ""

    def test_toc_field_exists(self):
        """ReportSection has a toc field defaulting to False."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        assert ir.toc is False

    def test_front_matter_toc_in_to_dict(self):
        """front_matter and toc appear in to_dict()."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            front_matter="# Study",
            toc=True,
        )
        d = ir.to_dict()
        assert d["front_matter"] == "# Study"
        assert d["toc"] is True


# ── build_report_ir description wiring ───────────────────────────────────────


class TestBuildReportIRDescription:
    """build_report_ir() pulls description from model.description() classmethod."""

    def test_build_ir_description_from_docstring(self):
        """build_report_ir uses docstring as section description."""

        class _IRAero(Model):
            """Aerodynamics submodel."""

            def setup(self):
                x = Variable("x_ir_aero")
                return [x >= 1]

        ir = build_report_ir(_IRAero())
        assert ir.description == "Aerodynamics submodel."

    def test_build_ir_assumptions_from_description(self):
        """build_report_ir populates assumptions from model.description()."""

        class _IRAss(Model):
            def setup(self):
                x = Variable("x_ir_ass")
                return [x >= 1]

            @classmethod
            def description(cls):
                return {
                    "summary": "Test",
                    "assumptions": ["beam theory"],
                    "references": [],
                }

        ir = build_report_ir(_IRAss())
        assert "beam theory" in ir.assumptions

    def test_build_ir_references_from_description(self):
        """build_report_ir populates references from model.description()."""

        class _IRRef(Model):
            def setup(self):
                x = Variable("x_ir_ref")
                return [x >= 1]

            @classmethod
            def description(cls):
                return {
                    "summary": "",
                    "assumptions": [],
                    "references": ["Drela 2013"],
                }

        ir = build_report_ir(_IRRef())
        assert "Drela 2013" in ir.references

    def test_build_ir_class_attrs_wired(self):
        """build_report_ir picks up class-attr assumptions/references."""

        class _IRClassAttr(Model):
            """Structural model."""

            assumptions = ["no buckling"]
            references = ["Raymer 2018"]

            def setup(self):
                x = Variable("x_ir_cls")
                return [x >= 1]

        ir = build_report_ir(_IRClassAttr())
        assert ir.description == "Structural model."
        assert "no buckling" in ir.assumptions
        assert "Raymer 2018" in ir.references

    def test_build_ir_anonymous_no_description(self):
        """Anonymous model sections have empty description, assumptions, references."""
        x = Variable("x_ir_anon")
        m = Model(1 / x, [x >= 1])
        ir = build_report_ir(m)
        assert ir.description == ""
        assert not ir.assumptions
        assert not ir.references

    def test_build_ir_front_matter_kwarg(self):
        """build_report_ir passes front_matter to root ReportSection only."""

        class _IRFM(Model):
            def setup(self):
                x = Variable("x_ir_fm")
                return [x >= 1]

        ir = build_report_ir(_IRFM(), front_matter="# Study\nIntro text.")
        assert ir.front_matter == "# Study\nIntro text."

    def test_build_ir_toc_kwarg(self):
        """build_report_ir passes toc=True to root ReportSection."""

        class _IRToc(Model):
            def setup(self):
                x = Variable("x_ir_toc")
                return [x >= 1]

        ir = build_report_ir(_IRToc(), toc=True)
        assert ir.toc is True


# ── Renderer: references ──────────────────────────────────────────────────────


class TestRenderReferences:
    """Tests that renderers include references output."""

    def _ir_with_refs(self):
        return ReportSection(
            title="Wing",
            description="A wing model.",
            assumptions=["steady state"],
            references=["Anderson 2001", "Drela 2013"],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )

    def test_render_text_references(self):
        """render_text includes references block when present."""
        out = render_text(self._ir_with_refs())
        assert "Anderson 2001" in out
        assert "Drela 2013" in out

    def test_render_markdown_references(self):
        """render_markdown includes **References:** line when present."""
        out = render_markdown(self._ir_with_refs())
        assert "**References:**" in out
        assert "Anderson 2001" in out
        assert "Drela 2013" in out

    def test_render_text_no_references_block_when_empty(self):
        """render_text omits references block when list is empty."""
        ir = ReportSection(
            title="Wing",
            description="",
            assumptions=[],
            references=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        out = render_text(ir)
        assert "References" not in out

    def test_render_markdown_no_references_when_empty(self):
        """render_markdown omits **References:** when list is empty."""
        ir = ReportSection(
            title="Wing",
            description="",
            assumptions=[],
            references=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        out = render_markdown(ir)
        assert "**References:**" not in out


# ── Renderer: front_matter and toc ───────────────────────────────────────────


class TestRenderFrontMatterToc:
    """Tests that renderers handle front_matter and toc fields."""

    def test_render_markdown_front_matter(self):
        """render_markdown prepends front_matter text before the root heading."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            front_matter="# Design Study\nIntroductory text.",
        )
        out = render_markdown(ir)
        assert out.startswith("# Design Study")
        assert "Introductory text." in out

    def test_render_markdown_toc_marker(self):
        """render_markdown inserts [TOC] marker when toc=True."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            toc=True,
        )
        out = render_markdown(ir)
        assert "[TOC]" in out

    def test_render_text_front_matter(self):
        """render_text prepends front_matter before the section header."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            front_matter="Intro text.",
        )
        out = render_text(ir)
        assert out.startswith("Intro text.")

    def test_render_text_toc_ignored(self):
        """render_text silently ignores toc=True (no TOC facility in plain text)."""
        ir = ReportSection(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            toc=True,
        )
        # Should not raise; output is plain text with no [TOC] marker
        out = render_text(ir)
        assert "[TOC]" not in out


# ── Objective section ─────────────────────────────────────────────────────────


class TestObjective:
    """Tests for objective expression and value in ReportSection and renderers."""

    def _solved_box(self):
        # pylint: disable=import-outside-toplevel
        from gpkit.examples.simple_box import Box

        m = Box()
        sol = m.solve(verbosity=0)
        return m, sol

    def test_objective_fields_on_report_section(self):
        """ReportSection has all four objective_* fields defaulting to empty/None."""
        ir = ReportSection(
            title="Test",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        assert ir.objective_str == ""
        assert ir.objective_latex == ""
        assert ir.objective_value is None
        assert ir.objective_units == ""

    def test_objective_in_to_dict(self):
        """ReportSection.to_dict() includes all four objective fields."""
        ir = ReportSection(
            title="Test",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            objective_str="1/(h·w·d)",
            objective_latex=r"\frac{1}{h w d}",
            objective_value=3.67e-3,
            objective_units="1/m³",
        )
        d = ir.to_dict()
        assert d["objective_str"] == "1/(h·w·d)"
        assert d["objective_latex"] == r"\frac{1}{h w d}"
        assert pytest.approx(d["objective_value"]) == 3.67e-3
        assert d["objective_units"] == "1/m³"

    def test_build_ir_populates_objective_unsolved(self):
        """build_report_ir sets objective_str and latex even without a solution."""

        class _ObjModel(Model):
            def setup(self):
                x = Variable("x_obj", "m")
                y = Variable("y_obj", "m")
                self.cost = x * y
                return [x >= y, y >= x]

        m = _ObjModel()
        ir = build_report_ir(m)
        assert ir.objective_str != ""
        assert ir.objective_latex != ""
        assert ir.objective_value is None  # no solution provided
        assert ir.objective_units != ""

    def test_build_ir_populates_objective_solved(self):
        """build_report_ir populates objective_value from solution."""
        m, sol = self._solved_box()
        ir = build_report_ir(m, solution=sol)
        assert ir.objective_value is not None
        assert ir.objective_value == pytest.approx(sol.cost)

    def test_build_ir_objective_empty_for_constant_cost(self):
        """build_report_ir leaves objective fields empty when cost has no variables."""

        class _ConstCost(Model):
            def setup(self):
                x = Variable("x_cc")
                return [x >= 1]

        m = _ConstCost()
        ir = build_report_ir(m)
        assert ir.objective_str == ""
        assert ir.objective_latex == ""

    def test_build_ir_objective_empty_for_submodel(self):
        """Child models in a hierarchy have empty objective fields."""

        class _Child(Model):
            def setup(self):
                x = Variable("x_child_obj")
                return [x >= 1]

        class _Parent(Model):
            child = None  # set in setup()

            def setup(self):
                self.child = _Child()
                x = Variable("x_child_obj2")
                y = Variable("y_parent_obj")
                self.cost = y
                return [self.child, y >= x, x >= 1]

        m = _Parent()
        ir = build_report_ir(m)
        assert ir.objective_str != ""  # parent has a real cost
        assert ir.children[0].objective_str == ""  # child does not

    def test_render_text_objective(self):
        """render_text includes objective expression and value when present."""
        m, sol = self._solved_box()
        out = m.report(solution=sol, fmt="text")
        assert "Objective" in out
        assert "minimize" in out.lower()

    def test_render_markdown_objective(self):
        """render_markdown includes objective heading and value when present."""
        m, sol = self._solved_box()
        out = m.report(solution=sol, fmt="md")
        assert "Objective" in out
        assert "minimize" in out.lower()

    def test_render_text_no_objective_when_empty(self):
        """render_text omits objective section when cost has no variables."""

        class _NoObj(Model):
            def setup(self):
                x = Variable("x_noobj")
                return [x >= 1]

        m = _NoObj()
        out = m.report(fmt="text")
        assert "Objective" not in out

    def test_render_markdown_no_objective_when_empty(self):
        """render_markdown omits objective section when cost has no variables."""

        class _NoObjMd(Model):
            def setup(self):
                x = Variable("x_noobjmd")
                return [x >= 1]

        m = _NoObjMd()
        out = m.report(fmt="md")
        assert "Objective" not in out
