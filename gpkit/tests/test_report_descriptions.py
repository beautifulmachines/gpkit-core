"""Tests for model description metadata, IR wiring, and renderer output.

Split from test_report.py to keep that file within the line-count limit.
Covers: Model.description(), class-attr assumptions/references, ModelView
fields (references, front_matter, toc, objective_*), model_view wiring,
and renderer output for all new fields.
"""

import dataclasses
import inspect
from typing import ClassVar

import pytest

from gpkit import Model, Variable
from gpkit.display.report import render_markdown, render_report, render_text
from gpkit.model_view import ModelView, model_view

# ── ModelView references / front_matter / toc fields ─────────────────────


class TestModelViewReferences:
    """Tests for references, front_matter, and toc fields on ModelView."""

    def test_references_field_exists(self):
        """ModelView has a references field defaulting to []."""
        ir = ModelView(
            title="Wing",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        assert not ir.references

    def test_references_in_to_dict(self):
        """ModelView.to_dict() includes references."""
        ir = ModelView(
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

    def test_preamble_field_exists(self):
        """A section carries its model's own preamble, defaulting to ''."""
        ir = ModelView(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
        )
        assert ir.preamble == ""

    def test_preamble_in_to_dict(self):
        ir = ModelView(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            preamble="Derived in Anderson ch. 5.",
        )
        assert ir.to_dict()["preamble"] == "Derived in Anderson ch. 5."

    def test_a_section_carries_no_document_parameters(self):
        """front_matter and toc describe a document, not a model (#319).

        They are renderer arguments; a section that carried them would make a
        model's data depend on how it is being printed.
        """
        fields = {f.name for f in dataclasses.fields(ModelView)}
        assert not fields & {"front_matter", "toc"}


# ── model_view description wiring ───────────────────────────────────────


class TestBuildReportIRDescription:
    """model_view() pulls description from model.description() classmethod."""

    def test_build_ir_description_from_docstring(self):
        """model_view uses docstring as section description."""

        class _IRAero(Model):
            """Aerodynamics submodel."""

            def setup(self):
                x = Variable("x_ir_aero")
                return [x >= 1]

        ir = model_view(_IRAero())
        assert ir.description == "Aerodynamics submodel."

    def test_build_ir_assumptions_from_description(self):
        """model_view populates assumptions from model.description()."""

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

        ir = model_view(_IRAss())
        assert "beam theory" in ir.assumptions

    def test_build_ir_references_from_description(self):
        """model_view populates references from model.description()."""

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

        ir = model_view(_IRRef())
        assert "Drela 2013" in ir.references

    def test_build_ir_class_attrs_wired(self):
        """model_view picks up class-attr assumptions/references."""

        class _IRClassAttr(Model):
            """Structural model."""

            assumptions: ClassVar = ["no buckling"]
            references: ClassVar = ["Raymer 2018"]

            def setup(self):
                x = Variable("x_ir_cls")
                return [x >= 1]

        ir = model_view(_IRClassAttr())
        assert ir.description == "Structural model."
        assert "no buckling" in ir.assumptions
        assert "Raymer 2018" in ir.references

    def test_build_ir_anonymous_no_description(self):
        """Anonymous model sections have empty description, assumptions, references."""
        x = Variable("x_ir_anon")
        m = Model(1 / x, [x >= 1])
        ir = model_view(m)
        assert ir.description == ""
        assert not ir.assumptions
        assert not ir.references

    def test_build_ir_takes_no_document_parameters(self):
        "It describes a model; how that model is printed is not its input."

        class _IRFM(Model):
            def setup(self):
                x = Variable("x_ir_fm")
                return [x >= 1]

        params = inspect.signature(model_view).parameters
        assert not params.keys() & {"front_matter", "toc"}
        model_view(_IRFM())  # and it still builds

    def test_build_ir_carries_each_models_own_preamble(self):
        "Per section, unlike front_matter, which applies once at the root."

        class _Child(Model):
            @classmethod
            def report_preamble(cls):
                return "Child notes."

            def setup(self):
                self.y = Variable("y_ir_pre")
                return [self.y >= 1]

        class _Parent(Model):
            @classmethod
            def report_preamble(cls):
                return "Parent notes."

            def setup(self):
                self.child = _Child()
                return [self.child, self.child.y >= 1]

        ir = model_view(_Parent())
        assert ir.preamble == "Parent notes."
        assert [c.preamble for c in ir.children] == ["Child notes."]


# ── Renderer: references ──────────────────────────────────────────────────────


class TestRenderReferences:
    """Tests that renderers include references output."""

    def _ir_with_refs(self):
        return ModelView(
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
        ir = ModelView(
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
        ir = ModelView(
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
    """front_matter and toc are render_report arguments, not section fields."""

    @staticmethod
    def _section(**kwargs):
        return ModelView(
            title="Aircraft",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            **kwargs,
        )

    def test_markdown_front_matter(self):
        out = render_report(
            self._section(), fmt="md", front_matter="# Design Study\nIntroductory text."
        )
        assert out.startswith("# Design Study")
        assert "Introductory text." in out

    def test_markdown_toc_marker(self):
        assert "[TOC]" in render_report(self._section(), fmt="md", toc=True)

    def test_text_front_matter(self):
        out = render_report(self._section(), fmt="text", front_matter="Intro text.")
        assert out.startswith("Intro text.")

    def test_text_toc_ignored(self):
        "Plain text has no TOC facility, so the argument is accepted and dropped."
        out = render_report(self._section(), fmt="text", toc=True)
        assert "[TOC]" not in out

    def test_a_preamble_renders_per_section(self):
        "Unlike front_matter, which is applied once by the renderer."
        child = self._section(preamble="Child notes.")
        root = self._section(preamble="Root notes.", children=[child])
        out = render_report(root, fmt="text", front_matter="Intro.")
        assert out.startswith("Intro.")
        assert out.count("Root notes.") == 1
        assert out.count("Child notes.") == 1

    def test_the_dict_form_still_carries_both(self):
        "They are document metadata, so the dict renderer supplies them."
        d = render_report(self._section(), fmt="dict", front_matter="# S", toc=True)
        assert (d["front_matter"], d["toc"]) == ("# S", True)

    def test_a_dict_section_carries_neither(self):
        ir = self._section()
        assert not ir.to_dict().keys() & {"front_matter", "toc"}


# ── Objective section ─────────────────────────────────────────────────────────


class TestObjective:
    """Tests for objective expression and value in ModelView and renderers."""

    def _solved_box(self):
        from gpkit.examples.simple_box import Box  # noqa: PLC0415

        m = Box()
        sol = m.solve(verbosity=0)
        return m, sol

    def test_objective_fields_on_report_section(self):
        """ModelView has all four objective_* fields defaulting to empty/None."""
        ir = ModelView(
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
        """ModelView.to_dict() includes all four objective fields."""
        ir = ModelView(
            title="Test",
            description="",
            assumptions=[],
            free_variables=[],
            fixed_variables=[],
            constraint_groups=[],
            objective_str="1/(h⋅w⋅d)",
            objective_latex=r"\frac{1}{h w d}",
            objective_value=3.67e-3,
            objective_units="1/m³",
        )
        d = ir.to_dict()
        assert d["objective_str"] == "1/(h⋅w⋅d)"
        assert d["objective_latex"] == r"\frac{1}{h w d}"
        assert pytest.approx(d["objective_value"]) == 3.67e-3
        assert d["objective_units"] == "1/m³"

    def test_build_ir_populates_objective_unsolved(self):
        """model_view sets objective_str and latex even without a solution."""

        class _ObjModel(Model):
            def setup(self):
                x = Variable("x_obj", "m")
                y = Variable("y_obj", "m")
                self.cost = x * y
                return [x >= y, y >= x]

        m = _ObjModel()
        ir = model_view(m)
        assert ir.objective_str != ""
        assert ir.objective_latex != ""
        assert ir.objective_value is None  # no solution provided
        assert ir.objective_units != ""

    def test_build_ir_populates_objective_solved(self):
        """model_view populates objective_value from solution.

        Box uses cost=1/(h*w*d), so the direction flips to "maximize" and
        objective_value is the maximized volume (1/sol.cost), not sol.cost.
        """
        m, sol = self._solved_box()
        ir = model_view(m, solution=sol)
        assert ir.objective_value is not None
        assert ir.objective_direction == "maximize"
        assert ir.objective_value == pytest.approx(1.0 / sol.cost)

    def test_build_ir_objective_empty_for_constant_cost(self):
        """model_view leaves objective fields empty when cost has no variables."""

        class _ConstCost(Model):
            def setup(self):
                x = Variable("x_cc")
                return [x >= 1]

        m = _ConstCost()
        ir = model_view(m)
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
        ir = model_view(m)
        assert ir.objective_str != ""  # parent has a real cost
        assert ir.children[0].objective_str == ""  # child does not

    def test_render_text_objective(self):
        """Objective is not auto-rendered in text output."""
        m, sol = self._solved_box()
        out = m.report(solution=sol, fmt="text")
        assert "Objective" not in out

    def test_render_markdown_objective(self):
        """Objective is not auto-rendered in markdown output."""
        m, sol = self._solved_box()
        out = m.report(solution=sol, fmt="md")
        assert "Objective" not in out

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
