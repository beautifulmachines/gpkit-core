# Modeling Conventions

This document describes the canonical patterns for building composable gpkit models. The patterns
generalize across engineering domains — from aircraft to energy systems to spacecraft — and are
illustrated with concrete code from the included examples.

---

## 1. Core Abstraction: Component, State, and Performance

Every gpkit model fits one of four roles, distinguished by its `setup()` signature:

| Role | `setup()` signature | Contains |
|---|---|---|
| **Component** | `setup(self)` or `setup(self, N)` | Static geometry, materials, structural properties |
| **State** | `setup(self)` | One operating condition — what changes across analysis points |
| **Perf** | `setup(self, static, state)` | Constraints linking one Component to one State |
| **Multi-condition** | `setup(self, component, N=...)` | N States, N Perfs, coupling constraints |

These are naming conventions, not framework enforcement. A model *is* a Perf model if it follows
the `(static, state)` pattern.

**State is the thing that gets vectorized.** When a multi-condition model creates N analysis
points with `with Vectorize(N)`, it is creating N State instances — N different operating
conditions. The Component (the aircraft, the beam, the solar panel) is shared across all N
points. The Perf model evaluates one (Component, State) pair; by vectorizing State, the
multi-condition model creates N Perf instances in parallel.

This decomposition is the key insight: Component captures what doesn't change; State captures
what does. A State may carry a few variables or a full set of atmospheric/environmental
correlations, but its role is always the same — bundle the condition-specific data that Perf
needs to evaluate one operating point.

The same pattern appears across domains:

| Aircraft | Structural beam | Spacecraft power |
|---|---|---|
| `Aircraft` → Component | `Beam` → Component | `SolarPanel` → Component |
| `FlightState` → State | `LoadCase` → State | `OrbitPoint` → State |
| `AircraftP` → Perf | `BeamSection` → Perf | `PanelPerf` → Perf |
| `Mission` → multi-condition | `BeamAnalysis` → multi-condition | `Orbit` → multi-condition |

Domain-specific names are fine. "Component," "State," "Perf," and "Multi-condition" are the
conceptual roles; class names can reflect the engineering domain.

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

`Var(units, label, *, value=None)` — `value` sets a default substitution on the variable. Variables
with values act as constants in the optimization unless explicitly freed by the caller.

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

**Sub-component decomposition.** Components can contain other Components. There is no limit on
depth; the solver flattens the entire constraint tree regardless of nesting level. The convention:

1. Instantiate each sub-component in `setup()` and store it as `self.attr`.
2. Pass sub-components to each other as constructor arguments when one needs another's variables.
3. Return all sub-components from `setup()` so gpkit collects their constraints.

```python
class Wing(Model):
    """Wing: planform geometry + structural sub-components"""

    W    = Var("lbf", "wing weight")
    mfac = Var("-",   "wing weight margin factor", value=1.2)

    spar_model = CapSpar     # override in subclasses to swap spar type
    fill_model = WingCore
    skin_model = WingSkin

    def setup(self, N=5):
        self.planform = Planform(N)          # stored as self.attr — navigable from outside
        self.components = []

        if self.skin_model:
            self.skin = self.skin_model(self.planform)       # planform passed in
            self.components.append(self.skin)
        if self.spar_model:
            self.spar = self.spar_model(N, self.planform)    # planform passed in
            self.components.append(self.spar)
        if self.fill_model:
            self.foam = self.fill_model(self.planform)
            self.components.append(self.foam)

        return [
            self.W / self.mfac >= sum(c.W for c in self.components),
            self.planform,
            self.components,
        ]
```

`Planform(N)` is passed to `CapSpar(N, self.planform)` so the spar can reference
`planform.b`, `planform.cave`, etc. by variable identity — no explicit "connect" step.
All sub-components are stored as `self.attr` so callers can navigate after construction:
`wing.planform.AR`, `wing.spar.W`, `wing.foam.W`. Storing sub-components only in the
returned list (without `self.attr` assignment) prevents this navigation.

A Perf model receiving a composite Component navigates as deep as needed:

```python
class WingAero(Model):
    def setup(self, static, state):
        AR   = static.planform.AR     # navigate into sub-component
        cmac = static.planform.cmac
        tau  = static.planform.tau
        ...
```

`static` is the Wing; `static.planform` is the Planform sub-component;
`static.planform.AR` is the exact Variable object that appears in Planform's own
constraints. Identity is the link — not string lookup, not copying.

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
`setup()` typically takes `(self, static, state)`.

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

**One operating point only.** A Perf model writes constraints for a single point — never
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

**The dependency principle: why `setup()` receives model instances.**

The `(self, static, state)` signature is not imposed by the framework — it follows from how
gpkit links variables. Variables link by *object identity*: two constraints referencing the
same `Variable` object are linked automatically. There is no separate "connect" step.

The implication for `setup()` argument design: **pass the models whose variables appear in
your constraints.** If `WingAero` constrains `wing.planform.AR`, `wing.planform.cmac`, and
`state.rho`, then `setup()` must receive both `wing` (to navigate into its planform sub-component)
and `state`. Receiving copies of those values would create new, disconnected Variable objects
that the structural constraints never see.

The `(static, state)` naming maps to the two fundamental inputs of any physics equation:

| Argument | What it represents | Example |
|---|---|---|
| `static` | What the component *is* — geometry, materials, topology | Wing chord, aspect ratio, spar thickness |
| `state` | What the conditions *are* — environment, operating point | Airspeed, density, viscosity |

**Fewer or more arguments are correct when physics requires it.** The `(static, state)` form
is the common case for a single-component Perf. Some simpler Perfs need only one input:

```python
class BreguetEndurance(Model):
    def setup(self, perf):        # only needs performance outputs, not the static component
        BSFC    = perf.engine.BSFC
        W_end   = perf.W_end
        W_start = perf.W_start
        ...
```

Higher-level *coupling* models — those that write constraints linking two subsystems rather than
evaluating one subsystem at one condition — naturally have more arguments:

```python
class MarsTransitInjection(Model):
    """Couples rocket capability to payload mass — neither is 'static' vs 'state'"""
    def setup(self, rocket, payload):
        self.burn = Burn(rocket, self.v_p)
        return [
            self.burn.m_prop <= rocket.m_prop_max,
            self.burn.m_cutoff >= payload.m + rocket.m_dry,
            ...
        ]
```

Here `rocket` and `payload` are both "static" in the sense of being design components, but
neither plays the role of operating conditions. The `(static, state)` convention applies to
single-component Perf models; coupling models receive whatever they constrain.

The rule is always the same: design arguments to match your constraint dependencies.

**The `state` argument vectorizes transparently.** When an orchestrator creates N instances
inside `with Vectorize(N)`, the `state` passed to each Perf becomes an N-element vector.
The Perf model's code does not change — `state.V` and `state.rho` are written identically —
but those variables are now length-N arrays, and the returned constraints are N constraints in
parallel. The multi-condition structure is invisible to the Perf model.

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

**Internal discretization.** A spatial discretization follows the same pattern as multi-condition
analysis: define a leaf element model, then vectorize it in the parent. `BeamElement` plays the
same role as `FlightSegment` — a model for one element, vectorized to produce N simultaneous
instances. The integration constraints live in the parent `Beam`, just as coupling constraints
live in `Mission`.

```python
class BeamElement(Model):
    """Loads and displacements at one cross-section — the spatial analogue of FlightState."""

    q  = Var("N/m", "distributed load")
    V  = Var("N",   "internal shear")
    M  = Var("N*m", "bending moment")
    th = Var("-",   "slope")
    w  = Var("m",   "deflection")

    def setup(self):
        pass   # no intra-element constraints; integration lives in the parent


class Beam(Model):
    """Vectorizes BeamElement — the spatial analogue of Mission."""

    EI = Var("N*m^2", "bending stiffness")
    L  = Var("m",     "beam length", value=5)

    def setup(self, N=5):
        self.dx = self.L / N

        with Vectorize(N):
            self.elem = BeamElement()    # N cross-sections; all vars become shape-(N,) arrays

        e = self.elem    # shorthand

        return [
            # Trapezoidal integration: shear and moment from tip to root
            e.V[:-1] >= e.V[1:]  + 0.5 * self.dx * (e.q[:-1]  + e.q[1:]),
            e.M[:-1] >= e.M[1:]  + 0.5 * self.dx * (e.V[:-1]  + e.V[1:]),
            # Slope and deflection from root to tip
            e.th[1:] >= e.th[:-1] + 0.5 * self.dx * (e.M[1:]  + e.M[:-1]) / self.EI,
            e.w[1:]  >= e.w[:-1]  + 0.5 * self.dx * (e.th[1:] + e.th[:-1]),
        ]
```

Scalar `Var` declarations (like `EI`) automatically become vector variables when the model is
instantiated inside an external `Vectorize(N)` context.

**Discrete integration: the unifying pattern.**

`BeamElement` and `FlightSegment` play the same role: a model for a single element, vectorized
by its parent to create N simultaneous instances. The integration constraints in `Beam` and
`Mission` are structurally identical — they differ only in governing equation.

Two coupling patterns appear whenever N elements are integrated:

| Pattern | Purpose | Form |
|---|---|---|
| **Accumulator** | Sum a consumed or produced quantity across all N elements | `total >= per_element.sum()` |
| **Continuity chain** | Enforce state consistency between adjacent elements | `x_end[:-1] >= x_start[1:]` |

**Accumulator** — for quantities that add up (fuel burned, energy consumed, load accumulated):

```python
# Fuel: sum W_fuel across all N segments into a mission total
self.W_fuel_total >= self.be.W_fuel.sum()

# Beam: sum element loads into a total applied force (if needed)
self.F_total >= self.elem.q.sum() * self.dx
```

`.sum()` on a `NomialArray` produces a single posynomial, GP-tractable as an inequality.

**Continuity chain** — for state variables that must be consistent across adjacent elements:

```python
# Mission: aircraft weight at end of segment i >= weight at start of segment i+1
self.perf.W_end[:-1] >= self.perf.W_start[1:]

# Beam: same structure; slope and deflection chain root-to-tip
e.th[1:] >= e.th[:-1] + curvature_term
```

`[:-1]` selects elements 0 through N−2; `[1:]` selects 1 through N−1. The result is
N−1 element-wise constraints. Boundary conditions on the terminal element(s) are set separately.

The two patterns are complementary, not alternatives. A fuel burn model uses both: the
accumulator sums per-segment burn into a total; the continuity chain enforces weight bookkeeping
across segment boundaries.

**Why `>=` instead of `==`?** A constraint where both sides are sums of terms
(posynomial = posynomial) is not GP-tractable in general. The `>=` form is a posynomial
inequality, which is GP-tractable. It is also physically conservative: the optimizer cannot
underestimate accumulated loads or fuel consumption. For monotone quantities (fuel strictly
decreases; shear strictly accumulates toward the root under positive loading), the inequality
binds at the optimum, so no fidelity is lost.

**Integration direction** is a modeling choice: choose the direction that makes the inequality
conservative in the physically meaningful sense. For fuel, the chain runs backward in time
(fuel remaining decreases). For beam shear, the chain runs tip-to-root (shear accumulates
inward). Slope and deflection chain root-to-tip (they accumulate outward).

**Not limited to trapezoidal.** The trapezoidal rule (`0.5*dx*(f[i] + f[i+1])`) is one
approximation. Forward Euler (`dx*f[i]`), backward Euler (`dx*f[i+1]`), and higher-order
schemes work equally well — each produces a different posynomial expression on the right-hand
side, all GP-tractable as inequalities. The trapezoidal rule is a reasonable default: it is
second-order accurate and symmetric.

**Recipe for any new discretized physics model:**
1. Define a leaf element Model with per-element state variables.
2. Vectorize it in the parent: `with Vectorize(N): self.elem = MyElement()`.
3. Write the governing ODE as a trapezoidal (or other) inequality chain.
4. Set boundary conditions on the terminal element(s) separately.
5. Accumulate totals with `.sum()` where needed.

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
| Accumulate a consumable total | `self.W_fuel >= self.be.W_fuel.sum()` |
| Chain state continuity across N elements | `self.perf.W_end[:-1] >= self.perf.W_start[1:]` |

---

See also: Model Inventory and Conformance Audit — `.planning/model_inventory.md` in the beautifulmachines meta-repo (covers cross-repo model variants: gassolar, jho, lunar/apollo, liberty)
