TOML models
***********

A gpkit model can be written as a TOML document instead of Python. The document declares
variables and states constraints as expression strings; ``load_toml`` builds a ``Model`` from it
and ``to_toml`` writes one back out.

.. code-block:: python

    from gpkit.toml import load_toml, to_toml

    model = load_toml("water_tank.toml")   # a path, or a TOML string
    sol = model.solve()
    print(to_toml(model))

``load_toml`` also takes ``dimensions=`` to override vector sizes and ``substitutions=`` to apply
further fixed values after the model is built.

A complete model
================

.. literalinclude:: examples/toml/water_tank.toml
   :language: toml

Variables
=========

Each key under ``[vars]`` declares one variable. The value is a *spec*, optionally paired with a
description:

.. code-block:: toml

    S = "m^2"                      # free variable with units
    x = "-"                        # free, dimensionless
    L = "6 m"                      # fixed parameter
    n = 2                          # fixed, dimensionless
    W = ["200 N", "wing weight"]   # any of the above, plus a description

A spec with a leading number fixes the variable at that value; a spec that is only units leaves
it free for the solver to choose.

``to_toml`` escapes what it writes, so a description may hold anything. When writing one by hand,
TOML's literal strings take no escapes at all, which is easier for LaTeX:
``deta = ["-", '\Delta (2y/b)']``.

Vectors
=======

A ``[vectors.<size>]`` section declares vector variables, all of the given length. The specs are
the same as for scalars, with one addition: a fixed vector either shares one value across every
element, or lists them all in index order.

.. literalinclude:: examples/toml/vector_values.toml
   :language: toml

A vector with *some* elements fixed and others free has no spelling, and ``to_toml`` refuses to
write one rather than emit a document that would read back as a different model.

Named dimensions
================

``[dimensions]`` gives integer names that can size a vector section and appear in expressions, so
a discretization can be changed in one place — or from Python, with
``load_toml(path, dimensions={"N": 10})``.

.. literalinclude:: examples/toml/beam.toml
   :language: toml

Note the constraints use numpy-style slicing (``V[:-1] >= V[1:] + ...``) to state one relation
across the whole vector.

The model section
=================

``[model]`` holds the objective and the constraints:

.. code-block:: toml

    [model]
    objective = "min: A"        # or "max: ..."
    constraints = [
      "A >= 2*(d[0]*d[1] + d[0]*d[2] + d[1]*d[2])",
      "V == d[0]*d[1]*d[2]",
    ]

Expressions use ``+``, ``*``, ``/`` and ``**``, with the comparisons ``>=``, ``<=`` and ``==``.
Three function calls are allowed and no others: ``sum()`` and ``prod()`` over a vector, and
``units('W')`` for a unit literal where a bare conversion factor is needed.

``-`` works between *numbers* only — a named dimension or a literal, as in ``(N-1)*dx`` below.
Subtracting one variable from another is not a GP-compatible constraint and gpkit's variables
have no subtraction operator, so it is rejected rather than quietly turned into a signomial. To
write a constraint of that form, build the model in Python under ``SignomialsEnabled`` (see
:doc:`signomialprogramming`).

Submodels
=========

Several models in one document each get a ``[models.<id>]`` section, which carries that model's
own variables, constraints, and a ``submodels`` list.

.. literalinclude:: examples/toml/wing_aircraft.toml
   :language: toml

The root model is the one no other model lists as a submodel; its section carries the
``objective``. A variable declared in another model can be named either plainly, as ``W_w``
above, or qualified as ``wing.W_w``; ``to_toml`` always writes the qualified form, since a bare
name would be ambiguous against a local variable of the same name.

A model's own keys must precede its ``[models.<id>.vectors.<size>]`` sub-tables, since a TOML key
belongs to the most recently opened table.

Substitutions
=============

A model's fixed values can also be written to their own document, separate from the model
structure, and applied back to a model later:

.. code-block:: python

    from gpkit.toml import save_subs, apply_subs

    save_subs(model, "subs.toml")     # also returns the TOML string
    apply_subs(model, "subs.toml")    # in place; a dict from load_subs works too

``save_subs`` emits one section per model that has substituted variables, headed by its lineage
path (``[Mission.Aircraft]``). ``apply_subs`` matches on that path and the variable name, warning
rather than raising when a name no longer exists, so a stored set of values stays usable as the
model changes.

Current limits
==============

A TOML round-trip is not yet lossless. Known gaps, each tracked as an issue:

* a reloaded model's variable and constraint refs do not match the original's, because the model
  graph is flattened on load (#298)
* named constraint groups are not written (#299)
* vector constraints are scalarized into one line per element (#141)
* a variable whose name is not a valid identifier cannot be written or read (#310)
* a document's ``name`` and ``description`` are neither read nor written (#314)
