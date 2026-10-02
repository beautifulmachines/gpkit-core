"""Tests for ConstraintKey: the constraint counterpart of VarKey."""

import tomllib
from importlib import import_module
from pathlib import Path

import pytest

from gpkit import Model, Var, Variable
from gpkit.constraintkey import ConstraintKey
from gpkit.constraints.set import keyed_constraints, walk_owned


def _catalog():
    "Every catalog model, as (id, class) pairs."
    path = Path(__file__).resolve().parent.parent.parent / "catalog.toml"
    with open(path, "rb") as f:
        entries = tomllib.load(f).get("models", [])
    return [
        (e.get("id", e["class"]), getattr(import_module(e["module"]), e["class"]))
        for e in entries
    ]


_CATALOG = _catalog()
_IDS = [cid for cid, _ in _CATALOG]


class _KeyChild(Model):
    y = Var("-")

    def setup(self):
        return [self.y >= 2, self.y <= 9]


class _GroupedTop(Model):
    t = Var("-")

    def setup(self, extra=False):
        timing = [self.t >= 1, self.t <= 9]
        if extra:
            timing.insert(0, self.t >= 0.5)
        return {"Timing": timing, "Sizing": [self.t >= 2]}


class _KeyTop(Model):
    def setup(self):
        x = Variable("x_key")
        self.a = _KeyChild()
        self.b = _KeyChild()
        return [x >= self.a.y, x <= 10, self.a, self.b]


class TestConstraintKey:
    """The key is a lineage plus a way to tell siblings apart, like VarKey."""

    def test_ref_is_path_and_discriminator(self):
        k = ConstraintKey(path=("Aircraft", "Wing"), index=2)
        assert k.ref == "Aircraft.Wing[2]"
        assert str(k) == k.ref

    def test_a_group_is_another_path_segment(self):
        "Named groups extend the path; the key keeps one shape either way."
        k = ConstraintKey(path=("Aircraft", "Wing", "Geometry"), index=1)
        assert k.ref == "Aircraft.Wing.Geometry[1]"

    def test_model_numbers_appear_like_varkey(self):
        "A second instance of a class is Wing1, as it is in a VarKey's ref."
        assert (
            ConstraintKey(path=("Aircraft", "Wing1"), index=0).ref
            == "Aircraft.Wing1[0]"
        )

    def test_an_empty_path_still_keys(self):
        "A model with no lineage names nothing, so the index alone identifies."
        assert ConstraintKey(index=3).ref == "[3]"

    def test_equality_and_hash_are_by_ref(self):
        a = ConstraintKey(path=("Wing",), index=1)
        b = ConstraintKey(path=("Wing",), index=1)
        c = ConstraintKey(path=("Wing",), index=2)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert len({a, b, c}) == 2

    def test_compares_to_its_ref_string(self):
        "VarKey does this, so a consumer holding a ref can look up with it."
        assert ConstraintKey(path=("Wing",), index=1) == "Wing[1]"

    def test_usable_as_a_dict_key(self):
        a = ConstraintKey(path=("Wing",), index=0)
        same = ConstraintKey(path=("Wing",), index=0)
        assert {a: 0.5}[same] == pytest.approx(0.5)

    def test_round_trips_through_the_ir(self):
        k = ConstraintKey(path=("Aircraft", "Wing", "Geometry"), index=1)
        assert ConstraintKey.from_ir(k.to_ir()) == k


class TestKeyedConstraints:
    """keyed_constraints pairs the one walk with one key per constraint."""

    def test_follows_the_walk(self):
        m = _KeyTop()
        keyed = list(keyed_constraints(m))
        walked = [c for _, c in walk_owned(m)]
        assert [c for _, c in keyed] == walked or all(
            a is b for (_, a), b in zip(keyed, walked)
        )
        assert len(keyed) == len(walked)

    def test_index_restarts_per_model(self):
        "Each model numbers its own constraints, so a sibling's edits can't shift it."
        refs = [k.ref for k, _ in keyed_constraints(_KeyTop())]
        assert refs == [
            "_KeyTop[0]",
            "_KeyTop[1]",
            "_KeyTop._KeyChild[0]",
            "_KeyTop._KeyChild[1]",
            "_KeyTop._KeyChild1[0]",
            "_KeyTop._KeyChild1[1]",
        ]

    def test_index_restarts_per_group(self):
        "A group names the constraints inside it, so the index counts within it."
        refs = [k.ref for k, _ in keyed_constraints(_GroupedTop())]
        assert refs == [
            "_GroupedTop.Timing[0]",
            "_GroupedTop.Timing[1]",
            "_GroupedTop.Sizing[0]",
        ]

    def test_a_group_rename_does_not_shift_other_groups(self):
        "Inserting into one group leaves the others' keys alone."
        before = {k.ref for k, _ in keyed_constraints(_GroupedTop())}
        after = {k.ref for k, _ in keyed_constraints(_GroupedTop(extra=True))}
        assert "_GroupedTop.Sizing[0]" in before & after

    @pytest.mark.parametrize("build", [c for _, c in _CATALOG], ids=_IDS)
    def test_keys_are_unique(self, build):
        refs = [k.ref for k, _ in keyed_constraints(build.default())]
        assert len(refs) == len(set(refs))

    @pytest.mark.parametrize("build", [c for _, c in _CATALOG], ids=_IDS)
    def test_keys_are_stable_across_builds(self, build):
        "Two builds of one model give keys that compare equal -- what VarKey does."
        first = [k for k, _ in keyed_constraints(build.default())]
        second = [k for k, _ in keyed_constraints(build.default())]
        assert first == second
