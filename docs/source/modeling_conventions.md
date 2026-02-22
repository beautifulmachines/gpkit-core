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
    rho = Var("lbf/ft^2", "areal density", default=1)
    A = Var("-", "aspect ratio", default=27)
    c = Var("ft", "mean chord")

    upper_unbounded = ("W",)
    lower_unbounded = ("c", "S")

    def setup(self):
        return [self.c == (self.S / self.A) ** 0.5, self.W >= self.S * self.rho]

    def perf(self, state):
        return WingAero(self, state)
```

**Variable declarations** use the `Var` class descriptor. Variables are accessible as `self.W`,
`self.S`, etc. inside `setup()` and as `wing.W`, `wing.S` at the call site. The `Var` descriptor
initializes the variable correctly inside the `NamedVariables` context during `Model.__init__`,
so lineage (`Wing.W`) is stamped automatically.

`Var(units, label, *, default=None)` — `default` sets a substitution on the variable. Variables
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

    V = Var("knots", "true airspeed", default=40)
    mu = Var("N*s/m^2", "dynamic viscosity", default=1.628e-5)
    rho = Var("kg/m^3", "air density", default=0.74)

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
    e = Var("-", "Oswald efficiency", default=0.9)
    Re = Var("-", "Reynold's number")
    D = Var("lbf", "drag force")

    upper_unbounded = ("D", "Re", "wing.A", "state.mu")
    lower_unbounded = ("CL", "wing.S", "state.mu", "state.rho", "state.V")

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

    upper_unbounded = ("aircraft.wing.c", "aircraft.wing.A")
    lower_unbounded = ("aircraft.W",)

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
    L = Var("m", "overall beam length", default=5)

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

---

## 6. Pressure Signature and Drop-in Compatibility

Every model has a **pressure signature**: which variables it bounds, and in which direction.

- **Upward pressure** on `x`: the model contains `x >= f(...)`, preventing `x → 0`.
- **Downward pressure** on `x`: the model contains `x <= f(...)`, preventing `x → ∞`.

`upper_unbounded` / `lower_unbounded` class-level tuples declare where a model provides *no*
downward/upward pressure — meaning external models must provide those bounds. gpkit validates
these declarations at construction time.

```python
class Wing(Model):
    upper_unbounded = ("W",)    # nothing in Wing prevents W → ∞; caller must bound it
    lower_unbounded = ("c", "S")  # nothing in Wing prevents c,S → 0; caller must bound them
```

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

### D. Self-equality no-ops

```python
# BAD — mathematically meaningless; creates spurious constraints
constraints = [rho == rho, mu == mu, h == h]
```

Variables with substitutions (constants) do not need fake equality constraints. Remove them.

### E. Perf model string key access

```python
# BAD — fragile string key; breaks on rename
Poper >= state["(P/S)_{min}"] * static.solarcells["S"] * static.solarcells["\\eta"]
```

**Fix**: access variables through the component's attribute:

```python
Poper >= state.PSmin * static.solarcells.S * static.solarcells.etasolar
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
| Declare a constant | `A = Var("-", "aspect ratio", default=27)` |
| Swap sub-model type | Class attribute override: `class SolarWing(Wing): sparModel = BoxSpar` |
| Connect two components | Equality constraint: `prop.Q == motor.Q` |
| Expose pressure signature | Class-level `upper_unbounded`/`lower_unbounded` tuples |
| Make Perf accessible from Component | `def perf(self, state): return WingAero(self, state)` |
