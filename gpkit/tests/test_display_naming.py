"""Tests for DisplayScope, the lineage-display naming rule."""

import pytest

from gpkit.display import DisplayScope
from gpkit.util.repr_conventions import also_excluding
from gpkit.varkey import VarKey

AIRCRAFT = (("Aircraft", 0),)
WING = (("Aircraft", 0), ("Wing", 0))
SPAR = (("Aircraft", 0), ("Wing", 0), ("Spar", 0))
FUSELAGE = (("Aircraft", 0), ("Fuselage", 0))
TANK = (("Aircraft", 0), ("Fuselage", 0), ("Tank", 0))


class TestDescendants:
    """Variables at or below the anchor render as suffixes of their path."""

    def test_owned_variable_is_bare(self):
        """A variable living at the anchor needs no path at all."""
        S = VarKey("S", lineage=WING)
        scope = DisplayScope(anchor=WING, shown=[S])
        assert scope.name(S) == "S"
        assert scope.owns(S)

    def test_unambiguous_descendant_is_bare(self):
        """A descendant with no name collision shortens all the way down."""
        t = VarKey("t", lineage=SPAR)
        S = VarKey("S", lineage=WING)
        scope = DisplayScope(anchor=WING, shown=[t, S])
        assert scope.name(t) == "t"

    def test_colliding_descendants_widen_together(self):
        """Colliding names widen to the shortest depth that separates them."""
        spar_t = VarKey("t", lineage=SPAR)
        skin_t = VarKey("t", lineage=WING + (("Skin", 0),))
        scope = DisplayScope(anchor=WING, shown=[spar_t, skin_t])
        assert scope.name(spar_t) == "Spar.t"
        assert scope.name(skin_t) == "Skin.t"

    def test_collision_with_anchor_owned_name(self):
        """An owned variable cannot widen, so the descendant is what moves."""
        wing_t = VarKey("t", lineage=WING)
        spar_t = VarKey("t", lineage=SPAR)
        scope = DisplayScope(anchor=WING, shown=[wing_t, spar_t])
        assert scope.name(wing_t) == "t"
        assert scope.name(spar_t) == "Spar.t"

    def test_widens_only_as_far_as_needed(self):
        """Deeper lineage is not shown once the names are distinguishable."""
        a = VarKey("t", lineage=WING + (("Spar", 0), ("Cap", 0)))
        b = VarKey("t", lineage=WING + (("Skin", 0), ("Ply", 0)))
        scope = DisplayScope(anchor=WING, shown=[a, b])
        assert scope.name(a) == "Cap.t"
        assert scope.name(b) == "Ply.t"


class TestAscendingPaths:
    """Variables outside the anchor render as relative paths, never bare."""

    def test_sibling_branch_gets_up_marker(self):
        """The `..` case: up to the common ancestor, then down the sibling."""
        m = VarKey("m", lineage=TANK)
        scope = DisplayScope(anchor=WING, shown=[m])
        assert scope.name(m) == "..Fuselage.Tank.m"
        assert not scope.owns(m)

    def test_ancestor_variable(self):
        """A variable owned by a parent model ascends with no descent."""
        W = VarKey("W", lineage=AIRCRAFT)
        scope = DisplayScope(anchor=WING, shown=[W])
        assert scope.name(W) == "..W"

    def test_ascent_depth_shows_in_marker(self):
        """N levels up is a run of N+1 dots."""
        W = VarKey("W", lineage=AIRCRAFT)
        scope = DisplayScope(anchor=SPAR, shown=[W])
        assert scope.name(W) == "...W"

    def test_globally_unique_foreign_name_still_qualified(self):
        """Uniqueness does not license rendering a foreign variable bare.

        `eta` collides with nothing anywhere, but it does not live under the
        anchor, so a bare `eta` would tell the reader it is the anchor's own.
        """
        eta = VarKey("eta", lineage=(("Motor", 0),))
        S = VarKey("S", lineage=WING)
        scope = DisplayScope(anchor=WING, shown=[eta, S])
        assert scope.name(eta) == "...Motor.eta"

    def test_ascending_path_is_not_shortened(self):
        """Ascending paths stay full-length even with nothing to collide with."""
        m = VarKey("m", lineage=TANK)
        scope = DisplayScope(anchor=WING, shown=[m])
        assert scope.name(m) == "..Fuselage.Tank.m"

    def test_descendant_widens_around_a_foreign_name(self):
        """A fixed ascending name is still an obstacle descendants widen past."""
        foreign = VarKey("t", lineage=(("Motor", 0),))
        spar_t = VarKey("t", lineage=SPAR)
        scope = DisplayScope(anchor=WING, shown=[foreign, spar_t])
        assert scope.name(foreign) == "...Motor.t"
        assert scope.name(spar_t) == "t"


class TestDegenerateCases:
    """The root anchor reduces the rule to shortest-unambiguous-suffix."""

    def test_root_anchor_no_lineage(self):
        x = VarKey("x")
        scope = DisplayScope(shown=[x])
        assert scope.name(x) == "x"

    def test_root_anchor_shortens_suffixes(self):
        a = VarKey("eta", lineage=(("Motor", 0),))
        b = VarKey("eta", lineage=(("Actuator", 0),))
        c = VarKey("F_tip", lineage=WING)
        scope = DisplayScope(shown=[a, b, c])
        assert scope.name(a) == "Motor.eta"
        assert scope.name(b) == "Actuator.eta"
        assert scope.name(c) == "F_tip"

    def test_empty_scope(self):
        """An empty scope still names keys, at full relative path."""
        x = VarKey("x", lineage=WING)
        scope = DisplayScope(anchor=AIRCRAFT)
        assert scope.name(x) == "Wing.x"


class TestUnshownKeys:
    """Keys the scope was not told about get their full relative path."""

    def test_unshown_descendant_is_not_shortened(self):
        t = VarKey("t", lineage=SPAR)
        scope = DisplayScope(anchor=WING, shown=[VarKey("S", lineage=WING)])
        assert scope.name(t) == "Spar.t"

    def test_unshown_foreign_key_still_ascends(self):
        m = VarKey("m", lineage=TANK)
        scope = DisplayScope(anchor=WING, shown=[VarKey("S", lineage=WING)])
        assert scope.name(m) == "..Fuselage.Tank.m"


class TestVectors:
    """Vector elements are named through their parent veckey."""

    def test_elements_share_the_parent_name(self):
        veckey = VarKey("c", lineage=WING, shape=(3,))
        elements = [
            VarKey("c", lineage=WING, shape=(3,), idx=(i,), veckey=veckey)
            for i in range(3)
        ]
        scope = DisplayScope(anchor=WING, shown=elements)
        assert {scope.name(e) for e in elements} == {"c"}
        assert scope.name(veckey) == "c"

    def test_siblings_do_not_count_as_a_collision(self):
        """Three elements of one vector must not widen each other's path."""
        veckey = VarKey("c", lineage=SPAR, shape=(2,))
        elements = [
            VarKey("c", lineage=SPAR, shape=(2,), idx=(i,), veckey=veckey)
            for i in range(2)
        ]
        scope = DisplayScope(anchor=WING, shown=elements)
        assert {scope.name(e) for e in elements} == {"c"}


class TestLatex:
    """latex() renders the same resolved path as a subscript."""

    def test_bare_owned_name(self):
        S = VarKey("S", lineage=WING)
        scope = DisplayScope(anchor=WING, shown=[S])
        assert scope.latex(S) == "S"

    def test_descendant_path_becomes_subscript(self):
        spar_t = VarKey("t", lineage=SPAR)
        skin_t = VarKey("t", lineage=WING + (("Skin", 0),))
        scope = DisplayScope(anchor=WING, shown=[spar_t, skin_t])
        assert scope.latex(spar_t) == r"{t}_{\text{spar}}"

    def test_ascending_path_marks_the_ascent(self):
        m = VarKey("m", lineage=TANK)
        scope = DisplayScope(anchor=WING, shown=[m])
        assert scope.latex(m) == r"{m}_{\text{..},\text{fuselage},\text{tank}}"

    def test_merges_into_an_existing_subscript(self):
        """A name that latexifies to a subscript keeps it, path appended."""
        vk = VarKey("m_wet", lineage=SPAR)
        scope = DisplayScope(anchor=WING, shown=[vk, VarKey("m_wet", lineage=WING)])
        assert scope.latex(vk) == r"m_{\text{wet},\text{spar}}"


class TestScopeIdentity:
    """Scopes are cheap values: hashable and comparable."""

    def test_equal_scopes_compare_and_hash_equal(self):
        S = VarKey("S", lineage=WING)
        one = DisplayScope(anchor=WING, shown=[S])
        two = DisplayScope(anchor=WING, shown=[S])
        assert one == two
        assert hash(one) == hash(two)
        assert len({one, two}) == 1

    def test_different_anchors_differ(self):
        S = VarKey("S", lineage=WING)
        assert DisplayScope(anchor=WING, shown=[S]) != DisplayScope(
            anchor=AIRCRAFT, shown=[S]
        )

    def test_shown_accepts_any_iterable(self):
        S = VarKey("S", lineage=WING)
        assert DisplayScope(anchor=WING, shown=[S]) == DisplayScope(
            anchor=WING, shown={S}
        )


@pytest.mark.parametrize(
    "anchor,lineage,expected",
    [
        ((), (), "x"),
        ((), WING, "Aircraft.Wing.x"),
        (AIRCRAFT, WING, "Wing.x"),
        (WING, WING, "x"),
        (WING, AIRCRAFT, "..x"),
        (WING, FUSELAGE, "..Fuselage.x"),
        (SPAR, FUSELAGE, "...Fuselage.x"),
    ],
)
def test_relative_path_rendering(anchor, lineage, expected):
    """The path rule alone, with nothing to shorten against."""
    x = VarKey("x", lineage=lineage)
    assert DisplayScope(anchor=anchor).name(x) == expected


class TestFormatFlags:
    """A scope carries format flags and goes where a bare flag set went."""

    def test_membership_and_iteration(self):
        scope = DisplayScope(excluded={"units", "idx"})
        assert "units" in scope
        assert "vec" not in scope
        assert set(scope) == {"units", "idx"}

    def test_also_excluding_preserves_the_scope(self):
        S = VarKey("S", lineage=WING)
        scope = DisplayScope(anchor=WING, shown=[S], excluded={"units"})
        narrowed = scope.also_excluding("ast_units")
        assert narrowed.excluded == {"units", "ast_units"}
        assert narrowed.anchor == scope.anchor
        assert narrowed.shown == scope.shown

    def test_also_excluding_a_bare_set(self):
        """Callers that never build a scope keep getting a plain flag set."""
        assert also_excluding({"units"}, "root") == frozenset({"units", "root"})

    def test_lineage_flag_suppresses_the_path(self):
        m = VarKey("m", lineage=TANK)
        scope = DisplayScope(anchor=WING, shown=[m], excluded={"lineage"})
        assert scope.name(m) == "m"
        assert scope.latex(m) == "m"

    def test_modelnums_flag_drops_instance_numbers(self):
        vk = VarKey("x", lineage=(("Aircraft", 0), ("Wing", 2)))
        shown = [vk, VarKey("x", lineage=FUSELAGE)]  # collide, so a segment shows
        assert DisplayScope(anchor=AIRCRAFT, shown=shown).name(vk) == "Wing2.x"
        assert (
            DisplayScope(anchor=AIRCRAFT, shown=shown, excluded={"modelnums"}).name(vk)
            == "Wing.x"
        )

    def test_scope_is_hashable_for_render_caches(self):
        """parse_ast caches rendered strings keyed on the threaded context."""
        S = VarKey("S", lineage=WING)
        scope = DisplayScope(anchor=WING, shown=[S], excluded={"units"})
        assert {scope: 1}[scope] == 1


class TestVarKeyComposition:
    """VarKey.str_without/latex accept a scope and compose path + name + idx."""

    def test_str_without_uses_the_scope(self):
        t = VarKey("t", lineage=SPAR)
        scope = DisplayScope(anchor=WING, shown=[t])
        assert t.str_without(scope) == "t"
        assert t.str_without({"units"}) == "Aircraft.Wing.Spar.t"

    def test_vector_parent_gets_a_slice_suffix(self):
        c = VarKey("c", lineage=SPAR, shape=(3,))
        scope = DisplayScope(anchor=WING, shown=[c])
        assert c.str_without(scope) == "c[:]"

    def test_vector_element_gets_its_index(self):
        veckey = VarKey("c", lineage=SPAR, shape=(3,))
        el = VarKey("c", lineage=SPAR, shape=(3,), idx=(1,), veckey=veckey)
        scope = DisplayScope(anchor=WING, shown=[el])
        assert el.str_without(scope) == "c[1]"

    def test_idx_flag_suppresses_the_suffix(self):
        veckey = VarKey("c", lineage=SPAR, shape=(3,))
        el = VarKey("c", lineage=SPAR, shape=(3,), idx=(1,), veckey=veckey)
        scope = DisplayScope(anchor=WING, shown=[el], excluded={"idx"})
        assert el.str_without(scope) == "c"

    def test_latex_puts_the_path_in_the_subscript(self):
        spar_t = VarKey("t", lineage=SPAR)
        skin_t = VarKey("t", lineage=WING + (("Skin", 0),))
        scope = DisplayScope(anchor=WING, shown=[spar_t, skin_t])
        assert spar_t.latex(scope) == r"{t}_{\text{spar}}"

    def test_latex_vector_parent_wears_the_arrow(self):
        c = VarKey("c", lineage=SPAR, shape=(3,))
        scope = DisplayScope(anchor=SPAR, shown=[c])
        assert c.latex(scope) == r"\vec{c}"

    def test_latex_element_uses_brackets_not_a_subscript(self):
        """Index follows the name in latex just as it does in text."""
        veckey = VarKey("c", lineage=SPAR, shape=(3,))
        el = VarKey("c", lineage=SPAR, shape=(3,), idx=(1,), veckey=veckey)
        other = VarKey("c", lineage=WING + (("Skin", 0),))  # collide, so Spar shows
        scope = DisplayScope(anchor=WING, shown=[el, other])
        assert el.latex(scope) == r"{c}_{\text{spar}}[1]"

    def test_latex_and_text_agree_on_ordering(self):
        """Both formats read path, then name, then index."""
        veckey = VarKey("c", lineage=TANK, shape=(2,))
        el = VarKey("c", lineage=TANK, shape=(2,), idx=(0,), veckey=veckey)
        scope = DisplayScope(anchor=WING, shown=[el])
        assert el.str_without(scope) == "..Fuselage.Tank.c[0]"
        assert el.latex(scope) == r"{c}_{\text{..},\text{fuselage},\text{tank}}[0]"
