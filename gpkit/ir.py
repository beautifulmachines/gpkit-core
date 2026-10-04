"The IR documents produced by Model.to_ir() and Solution.to_ir(), and diffs over them."

from .units import ureg
from .util.repr_conventions import unitstr

# Nothing validates this yet, and it has not tracked the schema changes made so
# far -- see the issue on versioning policy before relying on it.
IR_VERSION = "1.0"


def ir_units(obj) -> str:
    """The unit string an IR document carries, for anything bearing units.

    Full unit names ("meter**2", "USD"), because that is the only spelling that is
    both canonical and readable back by the unit registry:

    - VarKey.unitrepr keeps the user's own spelling, so m^2, m ** 2 and m*m all
      survive distinctly, by design, for TOML round-trips
    - the symbol form drops to glyphs a registry cannot parse -- USD becomes "$"
    - the display form substitutes a middle dot for products ("m⋅N"), and is
      platform-dependent besides

    Display is unaffected: unitstr() formats from the live units, not from this.
    """
    return unitstr(obj, "%s", ":C")


def _by_quantity(refs) -> dict:
    """Index refs by what identifies the quantity, dropping the units.

    A variable is a physical quantity; the unit it was declared in is a
    presentation choice, so feet and metres name the same variable.  Units are
    the ref's last "|"-delimited field (see VarKey._compute_ref), which is all
    this needs -- so it reads refs alone and works for a section of values or of
    bare sensitivities alike.

    A name declared twice in one document with different units would collide, so
    those keep their units and stay distinct.
    """
    stripped = {}
    for ref in refs:
        head, sep, _units = ref.rpartition("|")
        stripped.setdefault(head if sep else ref, []).append(ref)
    index = {}
    for key, matching in stripped.items():
        for ref in matching:
            index[key if len(matching) == 1 else ref] = ref
    return index


def _compare_values(baseline: dict, scenario: dict) -> tuple[float | None, bool]:
    """Ratio between two values, converting units, or None where undefined.

    The second return says the dimension changed, which is a different finding
    from a number moving.
    """
    b = ureg.Quantity(baseline["value"], baseline.get("units") or "dimensionless")
    s = ureg.Quantity(scenario["value"], scenario.get("units") or "dimensionless")
    if b.dimensionality != s.dimensionality:
        return None, True
    converted = s.to(b.units).magnitude
    if not b.magnitude:
        # Growing away from zero is an unbounded ratio, and JSON cannot carry
        # infinity -- the caller reports it as from_zero instead. Zero to zero
        # is no change. Shrinking *to* zero is just -1, so it needs none of this.
        return (0.0 if not converted else None), False
    return converted / b.magnitude - 1, False


def _diff_section(baseline: dict, scenario: dict, relative: bool) -> dict:
    """Partition two ref-keyed maps, and compare what they share.

    Values compare multiplicatively -- GP quantities are positive and the problem
    is convex in log space.  Sensitivities are already log derivatives, so they
    compare additively, and need no unit conversion at all.
    """
    b_index, s_index = _by_quantity(baseline), _by_quantity(scenario)  # keys only
    matched, dimension_changed, redeclared, from_zero = {}, {}, {}, []
    for key, b_ref in b_index.items():
        if key not in s_index:
            continue
        s_ref = s_index[key]
        if relative:
            rel, dim_changed = _compare_values(baseline[b_ref], scenario[s_ref])
            if dim_changed:
                dimension_changed[b_ref] = [
                    baseline[b_ref].get("units", ""),
                    scenario[s_ref].get("units", ""),
                ]
            elif rel is None:  # grew away from zero: the ratio is unbounded
                from_zero.append(b_ref)
            else:
                matched[b_ref] = float(rel)
        else:
            matched[b_ref] = float(scenario[s_ref] - baseline[b_ref])
        if s_ref != b_ref and b_ref not in dimension_changed:
            redeclared[b_ref] = s_ref
    section = {
        "matched": matched,
        "only_in_baseline": [r for k, r in b_index.items() if k not in s_index],
        "only_in_scenario": [r for k, r in s_index.items() if k not in b_index],
        "dimension_changed": dimension_changed,
        "from_zero": from_zero,
        "units_redeclared": redeclared,
    }
    return {k: v for k, v in section.items() if v}


def _section_changed(section: dict, tol: float) -> bool:
    "Any key gained or lost, units or dimension changed, or a number beyond tol."
    for bucket in (
        "only_in_baseline",
        "only_in_scenario",
        "dimension_changed",
        "from_zero",
    ):
        if section.get(bucket):
            return True
    return any(abs(v) > tol for v in section.get("matched", {}).values())


def diff_solutions(baseline: dict, scenario: dict, tol: float = 1e-6) -> dict:
    """Compare two Solution.to_ir() documents.

    Reports which quantities each document has and how the shared ones moved. It
    carries neither values nor units, because both inputs already do, keyed by
    the same refs -- so it joins them the way a solution IR joins a model IR.
    Keys are the baseline's refs.
    """
    diff = {"cost": {}}
    rel, dim_changed = _compare_values(baseline["cost"], scenario["cost"])
    if dim_changed:  # the objective was rewritten, e.g. from a mass to a cost
        diff["cost"]["dimension_changed"] = [
            baseline["cost"].get("units", ""),
            scenario["cost"].get("units", ""),
        ]
    elif rel is None:
        diff["cost"]["from_zero"] = True
    else:
        diff["cost"]["rel"] = float(rel)

    sections = {}
    for name in ("primal", "constants"):
        section = _diff_section(baseline.get(name, {}), scenario.get(name, {}), True)
        if section:
            sections[name] = diff[name] = section
    sens = _diff_section(
        baseline["sensitivities"]["variables"],
        scenario["sensitivities"]["variables"],
        False,
    )
    if sens:
        sections["sensitivities"] = sens
        diff["sensitivities"] = {"variables": sens}

    diff["changed"] = (
        "dimension_changed" in diff["cost"]
        or "from_zero" in diff["cost"]
        or abs(diff["cost"].get("rel", 0)) > tol
        or any(_section_changed(s, tol) for s in sections.values())
    )
    return diff
