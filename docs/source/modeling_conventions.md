# Modeling Conventions

This document describes the canonical patterns for building composable gpkit models. The patterns
generalize across engineering domains — aircraft, beams, thermal systems, spacecraft — and are
illustrated with concrete code from the included examples.

---

## 1. Core Abstraction: Component and Performance

Every gpkit model fits one of two roles, distinguished by its `setup()` signature:

| Role | `setup()` signature | Contains |
|---|---|---|
| **Component** | `setup(self)` or `setup(self, N)` | Static geometry, materials, structural properties |
| **Perf** | `setup(self, static, state)` | Constraints for one operating point |

These are naming conventions, not framework enforcement. A model *is* a Perf model if it follows
the `(static, state)` pattern.

**State** is just the second argument to a Perf model — a bundle of variables describing one
operating condition (airspeed, altitude, temperature, orbital position, applied load, …). It may
be a few variables or a full Model subclass; what matters is that it carries condition-specific
data that the Perf model needs.

The same pattern appears across domains:

| Aircraft | Structural beam | Spacecraft power |
|---|---|---|
| `Aircraft` → Component | `Beam` → Component | `SolarPanel` → Component |
| `FlightState` → State | `LoadCase` → State | `OrbitPoint` → State |
| `AircraftP` → Perf | `BeamSection` → Perf | `PanelPerf` → Perf |
| `Mission` → multi-condition | `BeamAnalysis` → multi-condition | `Orbit` → multi-condition |

Domain-specific names are fine. "Component" and "Perf" are the conceptual roles; class names can
reflect the engineering domain.

---

## 2. Component

A Component models the static properties of a physical object — geometry, materials, and
structural constraints that do not depend on operating conditions.

```python
class Wing(Model):
    """Aircraft wing model"""

    W = Var("lbf", "weight")
    S = Var("ft^2", "surface area")
    rho = Var("lbf/ft^2", "areal density", value=1)
    A = Var("-", "aspect ratio", value=27)
    c = Var("ft", "mean chord")

    def setup(self):
        return [self.c == (self.S / self.A) ** 0.5, self.W >= self.S * self.rho]

    def perf(self, state):
        return WingAero(self, state)
```

**Variable declarations** use the `Var` class descriptor. Variables are accessible as `self.W`,
`self.S`, etc. inside `setup()` and as `wing.W`, `wing.S` at the call site. The `Var` descriptor
initializes the variable correctly inside the `NamedVariables` context during `Model.__init__`,
so lineage (`Wing.W`) is stamped automatically.

`Var(units, label, *, value=None)` — `default` sets a substitution on the variable. Variables
with defaults act as constants in the optimization unless explicitly freed by the caller.

**Sub-model selection via class attributes** allows behavior to vary by subclassing without
mutating shared classes:

```python
class Wing(Model):
    sparModel = CapSpar     # override in subclasses to swap spar type
    fillModel = WingCore
    skinModel = WingSkin

    def setup(self, N=5):
        if self.sparModel:
            self.spar = self.sparModel(N, self.planform)
        if self.fillModel:
            self.foam = self.fillModel(self.planform)
        ...

class SolarWing(Wing):      # solar aircraft needs different structural layup
    sparModel = BoxSpar
    fillModel = None
    skinModel = WingSecondStruct
```

This is Pattern A (class attribute override). See Section 8 for the anti-pattern it replaces.

---

## 3. State

A State model is a bundle of variables describing one operating condition. It is not a special
framework concept — it is simply what we call the second argument to a Perf model.

```python
class FlightState(Model):
    """Context for evaluating flight physics"""

    V = Var("knots", "true airspeed", value=40)
    mu = Var("N*s/m^2", "dynamic viscosity", value=1.628e-5)
    rho = Var("kg/m^3", "air density", value=0.74)

    def setup(self):
        pass
```

States may contain constraints (e.g., atmosphere model correlations, wind speed fits) or may be
pure variable bundles. If a State `setup()` returns nothing, that is fine — the variables are
still initialized and usable.

A State is also a Component from gpkit's perspective: it has no external model dependencies in
its `setup()`. This means a Mission can pass the same State to multiple Perf models, and all of
them share the same Variable objects for that operating condition.

---

## 4. Performance Model (Perf)

A Perf model writes constraints linking a Component's properties to a State's conditions. Its
`setup()` always takes `(self, static, state)`.

```python
class WingAero(Model):
    """Wing aerodynamics"""

    CD = Var("-", "drag coefficient")
    CL = Var("-", "lift coefficient")
    e = Var("-", "Oswald efficiency", value=0.9)
    Re = Var("-", "Reynold's number")
    D = Var("lbf", "drag force")

    def setup(self, wing, state):
        self.wing = wing
        self.state = state

        c = wing.c           # Variable identity is the link — no explicit linking needed
        A = wing.A
        S = wing.S
        rho = state.rho
        V = state.V
        mu = state.mu

        return [
            self.D >= 0.5 * rho * V**2 * self.CD * S,
            self.Re == rho * V * c / mu,
            self.CD >= 0.074 / self.Re**0.2 + self.CL**2 / np.pi / A / self.e,
        ]
```

**One operating point only.** A Perf model writes constraints for a single element — never
coupling constraints between conditions. Coupling belongs at the multi-condition level (Section 5).

**`setup()` Return Value Contract.** The return value of `setup()` determines what constraints
and submodels the parent model collects. Use the form appropriate for your case:

| Return value | When to use |
|---|---|
| `None` (or nothing) | `setup()` only sets instance attributes; all constraints live in submodels |
| flat list `[c1, c2, submodel, ...]` | Simple model; constraints and submodels mix freely |
| tuple `(submodelA, [c1, c2])` | Multiple groups; all elements are flattened by gpkit |
| `(constraints, {var: fn})` | Post-solve computed variables only (callback dict as second element) |

**Anti-pattern: storing AND returning.** Do not assign `self.constraints = [...]` AND include
`self` in the returned list. If `self` is returned as a submodel, the parent model will traverse
its constraints through the submodel relationship; returning the same list separately causes
double-counting.

**`.perf()` method.** A Component may define a `perf()` convenience method that constructs the
associated Perf model:

```python
class Wing(Model):
    ...
    def perf(self, state):
        return WingAero(self, state)
```

`wing.perf(state)` and `WingAero(wing, state)` are equivalent. The `.perf()` method decouples
the caller from the concrete Perf class, which is useful when the Perf class may be subclassed
or swapped. For simple cases with one fixed Perf class, calling `WingAero(wing, state)` directly
is also fine.

---

## 5. Multi-Condition Analysis

Multi-condition analysis combines multiple Perf instances (one per operating condition) with
coupling constraints between them. Use `Vectorize(N)` to create N simultaneous instances.

```python
class Mission(Model):
    """A sequence of flight segments"""

    def setup(self, aircraft):
        self.aircraft = aircraft

        with Vectorize(4):         # four flight segments — all share the same aircraft
            self.fs = FlightSegment(aircraft)

        Wburn = self.fs.aircraftp.Wburn
        Wfuel = self.fs.aircraftp.Wfuel
        self.takeoff_fuel = Wfuel[0]

        return {
            "fuel constraints": [
                Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1],  # fuel burns backward in time
                Wfuel[-1] >= Wburn[-1],
            ],
            "flight segment": self.fs,
        }
```

**Coupling constraints live here, not in the Perf model.** The Mission model sees the
full sequence and owns the inter-element relationships. For example:

```python
# Time-coupled (Mission): fuel chain
Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1]

# Space-coupled (Beam): deflection accumulates
th[1:] >= th[:-1] + 0.5 * dx * (M[1:] + M[:-1]) / EI
```

The Perf model writes constraints for one element only; the multi-condition model stitches them
together.

**Composability.** A multi-condition model is itself a Component — it has no external model
dependencies. It can be used as a Perf in a higher-level analysis. There is no ceiling on
composition depth.

**Internal discretization.** When `N` comes from a `setup()` argument (not an external Vectorize
context), declare vector variables inside `setup()`:

```python
class Beam(Model):
    EI = Var("N*m^2", "bending stiffness")
    L = Var("m", "overall beam length", value=5)

    def setup(self, N):
        with Vectorize(N):
            q = Variable("q", 100, "N/m", "distributed load")
            V = Variable("V", "N", "internal shear")
            M = Variable("M", "N*m", "internal moment")
            th = Variable("th", "-", "slope")
            w = Variable("w", "m", "displacement")

        self.w_tip = w[-1]
        return {
            "shear integration": V[:-1] >= V[1:] + 0.5 * self.dx * (q[:-1] + q[1:]),
            "moment integration": M[:-1] >= M[1:] + 0.5 * self.dx * (V[:-1] + V[1:]),
            "theta integration": th[1:] >= th[:-1] + 0.5 * self.dx * (M[1:] + M[:-1]) / self.EI,
            "displacement integration": w[1:] >= w[:-1] + 0.5 * self.dx * (th[1:] + th[:-1]),
        }
```

Scalar `Var` declarations (like `EI`) automatically become vector variables when the model is
instantiated inside an external `Vectorize(N)` context.

**Orchestrator vs. leaf: who owns the N dimension.** There are two distinct roles in multi-condition modeling:

- **Orchestrator** (e.g., `FlightSegment`, `Mission`): uses `with Vectorize(N)` *inside* `setup()` to create N simultaneous operating-point instances. The orchestrator model owns the multi-condition expansion.
- **Leaf Component** (e.g., `Wing`, `Motor`, `SolarCell`): never uses `with Vectorize(N)` internally. Its scalar `Var` descriptors automatically become vector variables when the leaf is instantiated *inside* an external `Vectorize` context owned by a parent model.

This means: `Wing` does not need to know about N. If `FlightSegment.setup()` wraps `Wing()` construction inside `with Vectorize(N)`, each `Wing.S`, `Wing.W`, etc. becomes a length-N vector automatically. Adding internal `Vectorize` to a leaf Component would create a double-vectorization that is almost certainly wrong.

---

## 6. Pressure Signature and Drop-in Compatibility

Every model has a **pressure signature**: which variables it bounds, and in which direction.

- **Upward pressure** on `x`: the model contains `x >= f(...)`, preventing `x → 0`.
- **Downward pressure** on `x`: the model contains `x <= f(...)`, preventing `x → ∞`.

The pressure signature is always derivable from a model's constraints. It is a useful mental
model for reasoning about composition: when you remove a sub-model or swap it for another, ask
which pressures the original was providing. Any variable that loses all upward or downward
pressure in the combined problem will cause the solver to fail with an unbounded-variable error.
gpkit reports these at solve time via `check_bounds`.

**Drop-in compatibility.** Model B drops in for Model A if B provides all the pressures the
system was relying on A for. Two common patterns:

1. *Fidelity swap*: same variables bounded in same directions, different constraints (e.g.,
   replace a drag polar fit with a higher-fidelity CFD-calibrated curve).
2. *Constant → model*: a substituted constant is replaced by a model that computes it, relaxing
   one or both sides.

**Named constraint groups** (dict return from `setup()`) support the pressure-signature workflow:
the group name appears in solution printouts, making it easy to identify which constraint is
binding and why.

---

## 7. Variable Linkage Priority Order

gpkit links variables by identity — when two models share the same `Variable` object, any
solution value applies to both simultaneously. Priority order:

1. **Pass the Component, access `.attr`** (primary). `WingAero(wing, state)` uses `wing.S`,
   which is the same `Variable` object everywhere `wing.S` appears. Variable identity is the
   link — no explicit linking needed.

2. **Equality constraints** — for peer-to-peer connections between parallel sub-systems:
   ```python
   self.prop.Q == self.motor.Q
   self.prop.omega == self.motor.omega
   ```

3. **String subscript `model["varname"]`** — legacy pattern, avoid in new code. Fragile: breaks
   if the variable name changes.

The correct access pattern inside a Perf model is to use the Component's attribute directly:

```python
def setup(self, wing, state):
    c = wing.c           # correct — uses wing's Variable object
    S = wing.S           # correct
    V = state.V          # correct
```

Not:
```python
def setup(self, wing, state):
    c = wing["c"]        # fragile — string key lookup
```

**Submodel attribute storage is the mechanism that makes attribute access work.** Every submodel
the caller may need to reference after construction must be stored as `self.attr` inside
`setup()`. Storing submodels only in a list is insufficient:

```python
# CORRECT — each submodel stored as self.attr
def setup(self, aircraft):
    self.fs = FlightState()
    self.perf = aircraft.perf(self.fs)
    self.wing = aircraft.wing
    return [self.fs, self.perf]

# WRONG — wing only in a list; prevents mission.aircraft.wing.S access later
def setup(self, aircraft):
    components = [FlightState(), aircraft.perf(aircraft.fs)]
    return components
```

Storing a submodel only in a list (e.g., `self.components = [wing, fuselage]`) prevents tree
navigation like `mission.aircraft.wing.planform.S`. The obligation lives in `setup()`: if the
caller will need to reference it, assign it to `self`.

---

## 8. Anti-patterns

### A. Global class mutation to change sub-model structure

```python
# BAD — modifies HorizontalTail for every future use in this process
HorizontalTail.sparModel = BoxSparGP
HorizontalTail.fillModel = None
TailBoom.__bases__ = (BoxSparGP,)   # replaces base class globally!
```

**Fix**: create explicit subclasses that override class attributes. Do not mutate shared classes.

```python
class SolarHTail(HorizontalTail):
    sparModel = BoxSparGP
    fillModel = None
    skinModel = WingSecondStruct

class SolarTailBoom(TailBoom):
    spar_model = BoxSparGP
    secondaryWeight = True

# Pass the subclasses where needed:
emp = Empennage(N=5, htail_cls=SolarHTail, tailboom_cls=SolarTailBoom)
```

For `__bases__`-style mutations, the Component class needs to expose a class attribute for the
choice. See `TailBoom.spar_model` and `Wing.sparModel` for examples of this pattern.

### B. Class mutation inside `setup()`

```python
# BAD — mutates WingSP globally even though this is called per-instance
def setup(self, sp=False):
    if sp:
        WingSP.fillModel = None   # global side effect
        self.wing = WingSP()
```

**Fix**: define the subclass at module level and use it directly:

```python
class _WingSP(WingSP):
    fillModel = None

def setup(self, sp=False):
    if sp:
        self.wing = _WingSP()
```

### C. Perf model class mutation via instance attribute

```python
# BAD — changes Propeller's perf class globally by setting a class attribute
class Propulsor(Model):
    prop_flight_model = ActuatorProp   # class-level default

    def setup(self):
        self.prop = Propeller()
        Propeller.flight_model = self.prop_flight_model  # global class mutation!
```

**Fix**: set the attribute on the instance, and accept the perf class as a constructor argument:

```python
class Propulsor(Model):
    prop_flight_model = ActuatorProp   # class-level default

    def setup(self, prop_flight_model=None):
        self.prop = Propeller()
        self.prop.flight_model = prop_flight_model or type(self).prop_flight_model  # instance only
```

Callers that need a different perf model pass it explicitly:
```python
p = Propulsor(prop_flight_model=BladeElementProp)
```

### D. Perf model string key access

```python
# BAD — fragile string key; breaks on rename
Poper >= state["(P/S)_{min}"] * static.solarcells["S"] * static.solarcells["\\eta"]
```

**Fix**: access variables through the component's attribute:

```python
Poper >= state.PSmin * static.solarcells.S * static.solarcells.etasolar
```

### E. String keys in substitution dictionaries

```python
# BAD — string key in substitutions dict; same fragility as model["varname"]
model.substitutions["Vv"] = 0.04
model.substitutions["latitude"] = 45
```

`model.substitutions["varname"]` relies on gpkit's name-matching lookup: it returns the first
variable whose name matches the string, fails silently on rename, and cannot be type-checked.

**Fix**: use Variable object keys via attribute access:

```python
# CORRECT — Variable object as key; exact reference, rename-safe
self.emp.substitutions[self.emp.vtail.Vv] = 0.04
model.substitutions[model.fs.latitude] = 45
```

This is the substitution-side complement of Anti-pattern D. See Section 9.4 for the full
Substitution-Based Configuration decision.

---

## 9. Additional Patterns

These patterns address common implementation choices not covered by sections 1–8. Each entry
states the canonical decision, the GP or engineering rationale behind it, and the code form.

---

### 9.1 Cost Function Setting

**Decision:** Cost is set at the call site on the top-level model, after construction. Never
inside `setup()`.

**Rationale:** The cost function is the optimization objective — a modeling *choice*, not a model
*property*. In GP, the objective is a monomial or posynomial expression chosen by the analyst.
The same constraint structure may be optimized for minimum takeoff weight, maximum endurance, or
minimum power draw depending on the study. Baking cost into `setup()` removes this flexibility
without any GP-structural benefit; the cost is not a constraint and does not affect the feasible
region.

```python
# CORRECT — set at call site, after construction, using attribute access
model = Mission()
model.cost = model.aircraft.MTOW       # minimize takeoff weight

# Or for a different study objective using the same model:
model.cost = model.fs.t_flight         # maximize endurance (invert: minimize 1/t)

# WRONG — cost baked into setup(); objective cannot be changed without modifying the class
class Mission(Model):
    def setup(self):
        ...
        self.cost = self["MTOW"]       # anti-pattern: string subscript + hard-coded objective
```

**Anti-pattern note:** The API does not prevent setting cost inside `setup()`. It is a natural
mistake because cost looks like any other model property. The error only surfaces when you try to
re-use the model for a different optimization objective.

---

### 9.2 Weight (Mass) Rollup

**Decision:** For new code, use `sum(comp.W for comp in self.components)` where `W` is a `Var`
descriptor. `summing_vars` is legacy; it still works but is not preferred for new code.

**Rationale:** A weight rollup is a posynomial inequality in GP:
`W_total >= W_1 + W_2 + ... + W_n`. Using `Var` descriptors and attribute access makes each
summand an explicit Variable object — exact references, type-safe, rename-safe. `summing_vars`
requires a string name and returns a list through a dictionary lookup, which has the same
fragility as string key access everywhere else.

```python
# CORRECT — attribute access, no string; summands are explicit Variable objects
self.m_dry >= sum(comp.m_dry for comp in self.components)

# LEGACY — works but not preferred for new code
Wzfw >= sum(summing_vars(components, "W")) + Wpay + Wavn
```

**Anti-pattern note:** `summing_vars` appears throughout existing models and still functions
correctly. The audit table in `model_inventory.md` flags non-conformant uses for reference; do
not retrofit during Phase 2.

---

### 9.3 Scalar Configuration Arguments to `setup()`

**Decision:** `setup()` arguments are for configuration that changes model *structure* — which
submodels are included, the dimension N for Vectorize. Constants and parameters that can vary at
solve time belong as substitutions set at the call site.

**Rationale:** `setup()` is called once during construction, before any solving. A `setup()`
argument changes the set of constraints and variables gpkit builds. A substitution changes the
value of a variable but leaves the constraint structure unchanged. This distinction is fundamental
to how gpkit constructs and solves models: structural choices (which wing subclass, how many
segments) must be made at construction time; parameter choices (latitude, payload, speed) can be
deferred to substitutions and changed between solves without rebuilding the model.

```python
# CORRECT — sp=False selects which Wing subclass to instantiate (structural choice)
def setup(self, sp=False):
    self.wing = WingSP() if sp else WingGP()
    self.fuselage = Fuselage()
    ...

# WRONG — latitude is a parameter, not a structural choice; set as substitution instead
class Mission(Model):
    def setup(self, latitude=45):                       # anti-pattern
        self.lat = Variable("latitude", latitude, "deg", "latitude")

# CORRECT replacement: latitude as a substitution at the call site
model = Mission()
model.substitutions[model.fs.latitude] = 45            # set at call site; can change without rebuild
```

---

### 9.4 Substitution-Based Configuration

**Decision:** Substitution keys must be Variable objects, not strings. Use
`model.substitutions[model.submodel.var] = value`. Never `model.substitutions["varname"] = value`
in new code.

**Rationale:** String keys rely on gpkit's name-matching lookup, which returns the first variable
whose name matches — fragile, fails silently on rename, and cannot be verified by static
analysis. Variable object keys are exact references: renaming the variable at the `Var` declaration
automatically updates any code that uses attribute access to reach it. This is the same
correctness argument as for constraint-level attribute access (CONV-06) applied to the
substitutions dict.

```python
# CORRECT — Variable object as key; attribute access chain is exact and rename-safe
self.emp.substitutions[self.emp.vtail.Vv] = 0.04
model.substitutions[model.fs.altitude] = 10000        # ft

# WRONG — string key; first-match lookup, breaks silently on rename (CONV-06 violation)
model.substitutions["Vv"] = 0.04
model.substitutions["altitude"] = 10000
```

See also: Anti-pattern E in Section 8.

---

### 9.5 Conditional Vectorize

**Decision:** Use `with Vectorize(N)` inside `setup()` when the model *itself* orchestrates N
operating points (e.g., FlightSegment creating N FlightState + Perf pairs). Do NOT use internal
`Vectorize` for leaf Components — their scalar `Var` descriptors automatically become vectors
when they are instantiated inside an external `Vectorize` context from a parent model.

**Rationale:** `Vectorize` is a context manager that marks new Variables as members of an
N-element vector. The question is: who owns the N dimension? If `FlightSegment` creates N
`FlightState` instances inside `with Vectorize(N)`, `FlightSegment` owns the multi-condition
structure and N is its setup argument. If `Wing` is a leaf component, it should not know about
N — its scalar `Var` descriptors become vectors automatically when `Wing` is constructed inside
an external `Vectorize` block. Adding `with Vectorize(N)` inside a leaf's `setup()` would create
a double-vectorization (an N×M tensor when only an N-vector was intended).

```python
# CORRECT — orchestrator model uses Vectorize internally (owns the N dimension)
class FlightSegment(Model):
    def setup(self, aircraft, N=5):
        with Vectorize(N):
            self.fs = FlightState()                    # N FlightState instances
            self.perf = aircraft.perf(self.fs)         # N Perf instances
        return [self.fs, self.perf, ...]

# CORRECT — leaf Component: NO Vectorize inside setup()
class Wing(Model):
    W = Var("lbf", "weight")    # scalar Var; becomes vector automatically when Wing is
    S = Var("ft^2", "area")     # instantiated inside an external with Vectorize(N) block

    def setup(self):
        return [self.W >= self.S * self.rho]   # no Vectorize here; parent owns N

# WRONG — leaf tries to own its own vectorization (double-vectorization risk)
class Wing(Model):
    def setup(self, N=5):
        with Vectorize(N):                     # anti-pattern for a leaf component
            self.W = Variable("W", "lbf", "weight")
```

---

## Quick Reference

| Concept | Signature | Typical contents |
|---|---|---|
| Component | `setup(self)` | `Var` declarations, structural constraints |
| Component with discretization | `setup(self, N)` | `Var` scalars + `Vectorize(N)` for arrays |
| Perf model | `setup(self, static, state)` | Constraints for one operating point |
| Multi-condition / Mission | `setup(self, component, N=4)` | `Vectorize(N)`, coupling constraints |

| Task | Pattern |
|---|---|
| Declare a variable | `W = Var("lbf", "weight")` (class level) |
| Declare a constant | `A = Var("-", "aspect ratio", value=27)` |
| Swap sub-model type | Class attribute override: `class SolarWing(Wing): sparModel = BoxSpar` |
| Connect two components | Equality constraint: `prop.Q == motor.Q` |
| Make Perf accessible from Component | `def perf(self, state): return WingAero(self, state)` |
| Set optimization objective | `model.cost = model.aircraft.MTOW` (call site, after construction) |
| Roll up component masses | `self.m_dry >= sum(comp.m_dry for comp in self.components)` |
| Pass structural config to setup() | `setup(self, sp=False)` — selects subclass, not a parameter value |
| Set a parameter at call site | `model.substitutions[model.fs.latitude] = 45` (Variable object key) |
| Vectorize an orchestrator | `with Vectorize(N): self.fs = FlightState()` inside orchestrator `setup()` |

---

See also: [Model Inventory and Conformance Audit](model_inventory.md)
