# Model Inventory and Conformance Audit

This document inventories the major GP model variants in the Beautiful Machines ecosystem and
audits their conformance to the nine canonical patterns in
[Modeling Conventions](modeling_conventions.md).

Audit date: 2026-02-28. This is a read-only reference — no model files were modified as part of
this audit. Non-conformant entries in existing models are expected; these models predate the
conventions spec.

---

## 1. Model Variant Inventory (STRUCT-02)

The table below describes the unique structural features of each model variant. "Unique features"
means what this model demonstrates or requires that distinguishes it from the others — not an
exhaustive feature list.

| Model | Unique Features | Conventions Used | Notes |
|---|---|---|---|
| **gassolar/gas** | Multi-segment mission (climb + loiter), Breguet endurance range equation, SP wing option via `WingSP`, variable-altitude climb | `PhysicalComponent`/`PerformanceModel` base classes, `summing_vars` for weight rollup, `Vectorize` for flight segments, string subscripts (legacy) | Primary gassolar test case; SP option demonstrates signomial programming extension |
| **gassolar/solar** | Battery + solar cell energy balance, seasonal and latitude-dependent wind speed fits, solar irradiance model, SP structural flexibility option | `PhysicalComponent`/`PerformanceModel` base classes, `summing_vars` for weight rollup, solar/battery energy constraints as a separate analysis layer | Solar architecture baseline; energy balance replaces Breguet equation |
| **gassolar/solar_detailed** | Blade Element Momentum Theory (BEMT) propeller model (SP), BoxSpar structural spar (GP + SP), multi-pod vectorization, `relaxed_constants` SP convergence pattern | Highest-fidelity solar configuration; heavily SP; `relaxed_constants` pattern for SP convergence is unique here | Most complex model; SP-heavy throughout propulsion and structure |
| **jho** | DF70 engine model with real measured power curve, cylindrical (not elliptical) fuselage, pylon structural component, `AircraftLoading` separate from mission model, as-built parameter values (flight-tested to spec) | `PhysicalComponent`/`PerformanceModel` base classes, `summing_vars`, some scalar setup args (non-conformant), string subscripts (legacy) | Only model here with a real aircraft built and flown to its optimization spec; as-built provenance preserved in module docstring |
| **lunar/apollo** | `PhysicalComponent`/`PerformanceModel` base class pattern used explicitly, rocket equation via `te_exp_minus1` helper, multi-burn architecture (trans-lunar injection, lunar orbit insertion, descent), pure mass budget (no aerodynamics or lift) | `PhysicalComponent`/`PerformanceModel` inheritance as primary pattern; demonstrates the base class pattern in a non-aircraft context | Demonstrates PhysicalComponent/PerformanceModel inheritance pattern; useful as a reference for non-aircraft domain models |
| **liberty** | Class-level `Var` descriptors as primary pattern (first model explicitly using this), `m_dry`/`m_prop` as class-level `Var`s, nuclear-electric spacecraft propulsion, Falcon Heavy launch vehicle interface, Mars transit and capture burn architecture | `Var` descriptor pattern throughout (conformant); attribute access for substitutions (conformant); minimal `Model` subclass without `PhysicalComponent`; `sum(comp.m_dry ...)` rollup pattern | First model explicitly using `Var` descriptor as primary convention; minimal subclassing; establishes new canonical pattern |

---

## 2. JHO / Gassolar Integration Decision (STRUCT-01)

> **Decision:** JHO is integrated into gassolar as the `gassolar/jho` variant. The jho repo is
> archived.
>
> **Rationale:** gassolar already models the same airframe class (gas long-endurance UAV). Having
> two separate repos with near-identical dependency stacks creates maintenance overhead without
> benefit. JHO's as-built status — the only model here with a real aircraft built and
> flight-tested to its specs — is preserved in the model's module docstring and in this
> inventory. The as-built provenance is a documentation concern, not a code-structure concern.
>
> **Long-term:** The as-built configuration may become a tagged parameterized instance of the
> shared gas model in a future phase, allowing both the optimization model and the as-built
> verification to share a single constraint structure with different substitution sets.

---

## 3. Pattern Conformance Audit

Rows are model variants; columns are the nine canonical patterns from
[Modeling Conventions](modeling_conventions.md). Cells: **conformant**, **non-conformant**, or
**N/A** (pattern does not apply to this model).

**Legend:**
- `conformant` — model follows the canonical pattern
- `non-conformant` — model uses a legacy or incorrect form (expected for pre-convention models)
- `N/A` — pattern is not applicable to this model's domain or structure
- `(check)` — audit not yet confirmed for this variant; pending code review

| Pattern | gassolar/gas | gassolar/solar | gassolar/solar_detailed | jho | lunar/apollo | liberty |
|---|---|---|---|---|---|---|
| **9.1 Cost setting at call site** | non-conformant (uses string subscript `model["MTOW"]`) | non-conformant (check) | non-conformant (check) | non-conformant (uses `model.loiter["t"]` string subscript) | non-conformant | conformant |
| **9.2 Weight rollup via attr access** | non-conformant (uses `summing_vars`) | non-conformant (`summing_vars`) | non-conformant (`summing_vars`) | non-conformant (uses `summing_vars`) | N/A (mass budget only; no summing pattern) | conformant (`sum(comp.m_dry ...)`) |
| **9.3 setup() struct args vs. substitutions** | conformant (`sp=False` selects Wing subclass) | conformant | conformant | non-conformant (passes scalar Var value as setup arg) | conformant | conformant |
| **9.4 Substitution keys as Variable objects** | non-conformant (some string keys at call site) | non-conformant (check) | non-conformant (check) | non-conformant | non-conformant | conformant |
| **5 / 9.x setup() return contract** | conformant (returns tuple) | conformant (check) | conformant (check) | non-conformant (stores AND returns in some models) | conformant | conformant |
| **6 / 9.x Submodel attrs stored in setup()** | conformant | conformant (check) | conformant (check) | conformant | conformant | conformant |
| **9.5 Vectorize (orchestrator vs. leaf)** | conformant | conformant | conformant | conformant | N/A (no multi-condition structure) | N/A (no multi-condition structure) |
| **8 / Peer-to-peer equality, attr access** | non-conformant (uses string subscripts for coupling vars) | non-conformant (check) | non-conformant (check) | non-conformant | N/A | N/A |
| **CONV-06 No string subscripts in new code** | non-conformant (legacy; predates CONV-06) | non-conformant (legacy) | non-conformant (legacy) | non-conformant (legacy) | non-conformant (legacy) | conformant |

**Notes on the audit:**

1. **Non-conformant entries are expected.** All models except `liberty` predate the conventions
   spec. The audit is a diagnostic baseline, not a remediation action list for Phase 2.

2. **CONV-06 (no string subscripts in new code)** applies only to code written during v1. Existing
   models are not required to be retrofitted in Phase 2.

3. **`(check)` entries** indicate that solar and solar_detailed variants share significant code
   with the gas variant and are likely non-conformant in the same patterns, but a line-by-line
   audit of those specific files was not completed. Treat as non-conformant pending review.

4. **liberty** is fully conformant because it was written during v1 with the conventions spec in
   mind. It serves as the reference implementation for new models.

5. **lunar/apollo** is non-conformant on CONV-06 (string subscripts exist in the legacy code)
   but the `PhysicalComponent`/`PerformanceModel` pattern it demonstrates is otherwise correct.

---

## 4. Cross-Reference

See also: [Modeling Conventions](modeling_conventions.md)

*Audit performed 2026-02-28. Non-conformant entries in existing models are expected — these models
predate the conventions spec. CONV-06 applies to new code only. Future phases may address
retrofitting.*
