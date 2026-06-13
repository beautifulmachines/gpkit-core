# gpkit-core — Developer Reference

Curated reference for working in this repo or consuming it from another project (e.g., tradespace).
Not a substitute for the full docs, but enough to orient quickly and find the right APIs.

---

## Package Structure

```
gpkit/
├── __init__.py          exports: Model, Variable, Var, VarKey, ArrayVariable, VectorVariable,
│                                 Monomial, Posynomial, Signomial, NomialArray,
│                                 GeometricProgram, SequentialGeometricProgram,
│                                 ConstraintSet, SignomialEquality, Objective,
│                                 units, ureg
├── model.py             Model class — the central abstraction
├── varkey.py            VarKey — symbolic variable identity
├── solutions.py         Solution, Sensitivities, RawSolution dataclasses
├── report.py            Hierarchical model reporting (Phase 3.5)
├── nomials/             Monomial, Posynomial, Signomial, NomialArray
├── constraints/
│   ├── set.py           ConstraintSet, build_model_tree()
│   └── ...
├── programs/
│   ├── gp.py            GeometricProgram — solve() lives here
│   └── sgp.py           SequentialGeometricProgram (signomial)
├── toml/                TOML serialization tools (see section below)
├── interactive/         Jupyter-only widgets — NOT for web UI (see note below)
└── tests/               pytest suite
```

---

## Core Classes

### `Model` (`gpkit/model.py`)

The base class for all GP models. Subclass it and implement `setup()`.

```python
class Wing(Model):
    def setup(self):
        S = Variable("S", "m^2", "Wing area")
        W = Variable("W", "N", "Wing weight")
        return [W >= 9.81 * S]    # list → unnamed constraints
        # OR: {"Structural": [...], "Aero": [...]}  → named constraint groups

model = Wing()
model.solve(solver="cvxopt")   # returns Solution
```

**Key attributes** (set after `__init__`):
- `model.cost` — the objective expression (Monomial/Posynomial)
- `model.substitutions` — `{VarKey: value}` dict of fixed parameters
- `model.cgroups` — `dict | None`; named constraint groups if `setup()` returned a dict
- `model._children` — list of direct child Model instances

**Key methods**:
- `model.solve(solver=..., verbosity=0)` → `Solution`
- `model.to_ir()` → `dict` — full serializable IR (see IR section)
- `model.report(solution=None, fmt="text"|"dict"|"md"|"latex", toc=False)` → see Report section
- `model.walk()` — generator, depth-first over all descendant Models
- `model.submodels()` — direct children in setup() definition order
- `model.get_var("wing.S")` — resolve dotted path to Variable

**Connecting submodels**: Pass the child instance, access its attributes:
```python
class Aircraft(Model):
    def setup(self):
        self.wing = Wing()
        S = self.wing.S          # shared identity — same variable
        return [self.wing]
```

---

### `VarKey` / `Variable` / `Var` (`gpkit/varkey.py`)

`Variable(name, units, label)` creates a symbolic variable. `Var` is the descriptor-based class-level variant.

**Key attributes**:
- `vk.name` — plain name string, e.g. `"S"`
- `vk.units` — unit string, e.g. `"m^2"` or `None`
- `vk.label` — description string
- `vk.lineage` — tuple of `(model_name, instance_num)` pairs tracing ancestry
- `vk.models` — just the model name strings from lineage
- `vk.ref` — canonical string key used in IR maps, e.g. `"Aircraft0.Wing0.S|m²"`. Format: `"{instance_id}.{name}|{units}"` or just `"{name}|{units}"` for root-level vars.
- `vk.idx` — tuple of indices for vector elements (or `None`)
- `vk.shape` — shape tuple for vector variables (or `None`)
- `vk.veckey` — parent VarKey for vector elements

**IR serialization**:
```python
vk.to_ir()   # → dict with: name, lineage, units, label, idx, shape
VarKey.from_ir(d)  # → VarKey
```

---

### `Solution` / `Sensitivities` (`gpkit/solutions.py`)

Returned by `model.solve()`.

```python
@dataclass(frozen=True, slots=True)
class Solution:
    cost: float
    primal: VarMap          # free variables: VarKey → value (with units)
    constants: VarMap       # fixed variables: VarKey → value
    sens: Sensitivities
    meta: dict              # model ref, warnings, solver info

@dataclass(frozen=True, slots=True)
class Sensitivities:
    constraints: dict       # VarKey/constraint → sensitivity float
    models: dict            # lineage string → sensitivity float
    variables: VarMap       # VarKey → sensitivity float
    variablerisk: VarMap
```

**Accessing values**:
```python
sol[varkey]              # → Quantity (value with units)
sol.variables            # combined primal + constants
sol.subinto(expression)  # substitute solution into an expression
sol.cost_breakdown()     # cost allocation across constraints
sol.table()              # formatted text output
```

**JSON export**:
```python
sol.savejson("out.json")
# Format: {varkey_ref: {"v": value_or_array, "u": unit_string}}
```

---

## IR Format (`Model.to_ir()`)

Returns a JSON-serializable dict. This is the primary interface between gpkit-core and external tools like tradespace.

```python
{
    "gpkit_ir_version": "1.0",

    "variables": {
        "<ref>": {             # ref = VarKey.ref, e.g. "Aircraft0.Wing0.S|m²"
            "name": str,
            "lineage": [...],  # list of [model_name, instance_num] pairs; omitted if empty
            "units": str,      # omitted if dimensionless
            "label": str,      # omitted if empty
            "shape": [...],    # omitted for scalars; present for vector parents and elements
            "idx": [...],      # omitted for scalars and vector parents; present for vector elements
            "veckey_ref": str, # omitted for scalars and vector parents; present for vector elements
        },
        ...
    },
    # The variables dict includes ALL three kinds of VarKey:
    #   scalar:        no "shape", no "idx"  — an ordinary single variable
    #   vector parent: has "shape", no "idx" — the parent key for a VectorVariable;
    #                                          its ref appears in model_tree.variables
    #   vector element: has "shape", "idx", and "veckey_ref" — one element of a vector;
    #                                          veckey_ref points to the parent's ref

    "cost": { ... },           # nomial IR dict

    "constraints": [           # flat list, same order as flatiter()
        { ... },               # constraint IR dicts
        ...
    ],

    "model_tree": {            # see model_tree section below
        "class": str,
        "instance_id": str,
        "variables": [str, ...],
        "constraint_indices": [int, ...],
        "children": [...]
    },

    "substitutions": {         # omitted if empty
        "<ref>": float,
        ...
    }
}
```

---

## Model Tree (`build_model_tree`)

Lives in `gpkit/constraints/set.py`. Called by `to_ir()`. Returns the nested component hierarchy.

**Each node**:
```python
{
    "class": str,                    # Model subclass name, e.g. "Wing"
    "instance_id": str,              # dotted path, e.g. "Aircraft0.Wing0"; "" for root
    "variables": [str, ...],         # sorted list of VarKey.ref strings owned by this node
    "constraint_indices": [int, ...],# indices into the flat constraints list from to_ir()
    "children": [                    # direct child models, same structure
        { ... },
        ...
    ]
}
```

`constraint_indices` references the `constraints` array in the same `to_ir()` dict. Index `i` in `constraint_indices` → `ir["constraints"][i]`.

---

## Model Report (`model.report()`)

Implemented in `gpkit/report.py` (Phase 3.5). Returns a human-readable or structured summary.

```python
model.report(
    solution=None,        # Solution — if provided, includes solved values + sensitivities
    fmt="text",           # "text" | "md" | "latex" | "dict"
    front_matter="",      # prepend raw text/markdown
    toc=False             # insert TOC marker (markdown only)
)
```

**`fmt="dict"` return structure** (recursively nested):
```python
{
    "title": str,                    # model class name
    "lineage_path": str,             # e.g. "Aircraft.Wing"
    "magic_prefix": str,             # for variable name context stripping
    "objective_direction": str,      # "minimize" or "maximize"
    "objective_label": str,          # human label if cost is a single var; else ""
    "is_anonymous": bool,            # True for bare Model(...) instances
    "description": str,              # from Model.description()["summary"]
    "assumptions": [str, ...],
    "references": [str, ...],
    "front_matter": str,
    "toc": bool,
    "objective_str": str,
    "objective_latex": str,
    "objective_value": float | None, # None if unsolved
    "objective_units": str,

    "free_variables": [              # optimized variables
        {
            "name": str,
            "latex": str,
            "value": float | None,
            "sensitivity": float | None,
            "units": str,
            "label": str,
            "source": str
        },
        ...
    ],

    "fixed_variables": [ ... ],      # same structure as free_variables

    "constraint_groups": [
        {
            "label": str,            # "" for unnamed groups
            "constraints": [str, ...]# constraint text representations
        },
        ...
    ],

    "children": [ ... ]              # same structure, recursively for submodels
}
```

**Named constraint groups** come from `setup()` returning a dict:
```python
def setup(self):
    return {"Drag": [c1, c2], "Structural": [c3]}
```
These appear in `model.cgroups` (in memory) and in `report(fmt="dict")["constraint_groups"]`. They are **not** present in `to_ir()` directly — only in the report.

---

## Solve Flow

```
model.solve(solver="cvxopt", verbosity=0)
  → GeometricProgram(model)          # or SequentialGeometricProgram for signomial
      → solver call (cvxopt / mosek)
      → RawSolution(x, nu, la, cost, status)
  → Solution(cost, primal, constants, sens, meta)
```

**Supported solvers**: `"cvxopt"` (default, open source), `"mosek_cli"`, `"mosek_conif"`.

**`RawSolution`**:
```python
@dataclass
class RawSolution:
    x: Sequence      # primal variables in log scale; recover via exp(x)
    nu: Sequence     # inequality dual variables (sensitivities)
    la: Sequence     # equality dual variables
    cost: float
    status: str      # "optimal" | "infeasible" | "unbounded" | ...
    meta: dict
```

---

## TOML Tools (`gpkit/toml/`)

For specifying models and parameters in TOML files. Vision: user-editable model specs without Python.

```python
from gpkit.toml import load_toml, to_toml, save_subs, load_subs, apply_subs

model = load_toml("path/to/model.toml")
toml_str = to_toml(model)
save_subs({"S": 10.0}, "subs.toml")
subs = load_subs("subs.toml")
apply_subs(model, subs)
```

**TOML variable spec syntax**:
```toml
S = "m^2"              # free variable with units
W = ["200 N", "Wing weight"]   # fixed parameter with value, units, label
n = 2                  # dimensionless parameter
x = "-"                # dimensionless free variable
```

---

## Model Composition Patterns

**Preferred pattern** (class with descriptor variables):
```python
class Wing(Model):
    S = Var("S", "m^2", "Wing area")    # class-level descriptor
    W = Var("W", "N", "Wing weight")
    def setup(self):
        return [self.W >= 9.81 * self.S]
```

**Connecting submodels** — pass the instance, not its variables:
```python
class Aircraft(Model):
    def setup(self):
        self.wing = Wing()
        # self.wing.S is the same VarKey as Wing's S — shared identity
        W_total = Variable("W", "N")
        return [self.wing, W_total >= self.wing.W]
```

**Naming convention**: `Component` for physical parts, `Perf`/`Performance` for analysis points, `State` for operating conditions.

---

## Model Lineage and the Model Graph

Each `VarKey` carries a `lineage` tuple tracing it back through nested model instantiation. This is how tradespace (and the report) know which component owns which variable.

```python
vk.lineage    # → (("Aircraft", 0), ("Wing", 0))
vk.models     # → ("Aircraft", "Wing")
vk.instance_id  # → "Aircraft0.Wing0"
vk.ref        # → "Aircraft0.Wing0.S|m²"
```

The model graph itself is explicit in `model_tree` from `to_ir()`. No need to re-derive it from lineage strings.

---

## Interactive Module (Jupyter Only)

`gpkit/interactive/` contains Jupyter widget integrations using `ipywidgets` and `ipysankeywidget`. These are **not** usable in a plain browser context:

- `interactive/modelinteract.py` — slider-based interactive solve in Jupyter
- `interactive/sankey.py` — Sankey widget (Jupyter)
- `interactive/plotting.py`, `plot_sweep.py` — Jupyter plotting

The tradespace web UI replaces this functionality with proper browser-native implementations.

---

## Testing

```bash
make test                          # recommended
uv run pytest gpkit/tests -v       # direct
uv run pytest gpkit/tests/test_ir.py -v   # single file
make coverage                      # with coverage report
```

Test files in `gpkit/tests/`: `test_model.py`, `test_ir.py` (most comprehensive — covers `to_ir()`, `build_model_tree()`), `test_report.py`, `test_constraints.py`, `test_nomials.py`.

---

## Notes for Tradespace Integration

- **Primary interface**: `model.to_ir()` for structure, `model.solve()` → `Solution` for results
- **Concurrent-safe solve**: Build a fresh model per request (call the model's constructor), apply substitutions on the fresh instance, solve, discard. Do NOT share model instances across requests.
- **Opportunity**: `model.copy()` would simplify concurrent-safe patterns — not yet implemented.
- **Named constraint groups**: Available via `model.report(fmt="dict")["constraint_groups"]` but not in `to_ir()`. If tradespace needs them in the IR, add them to `to_ir()` in gpkit-core.
- **Solution → JSON**: `sol.savejson()` for full primal dump; for sensitivity maps, iterate `sol.sens.variables` and `sol.sens.models`.
