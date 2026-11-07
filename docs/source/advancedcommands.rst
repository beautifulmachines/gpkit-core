Advanced Commands
*****************

.. _migp:

Choice Variables
================
If MOSEK 9 is installed, GPkit supports discretized free variables with the ``mosek_conif`` solver.
Choice variables are free in the sense of having multiple possible choices, but discretized in the
sense of having a limited set of possible solutions.

.. literalinclude:: examples/migp.py

If solved with the mosek_conif solver, the script above will print::

    Optimal Cost
    ------------
     [ 1.5       2.15      2.8       3.22     ... ]

    ~~~~~~~~
    WARNINGS
    ~~~~~~~~
    No Dual Solution
    ----------------
    This model has the discretized choice variables [x] and hence no dual
    solution. You can fix those variables to their optimal value and get
    sensitivities to the resulting continuous problem by updating your model's
    substitions with `sol["choicevariables"]`.
    ~~~~~~~~

    Swept Variables
    ---------------
    numerator : [ 0.5
                  1.15
                  1.8
                  2.45
                  3.1
                  3.75
                  4.4
                  5.05
                  5.7
                  6.35
                  7         ]

    Free Variables
    --------------
    x : [ 1
          1
          1
          2
          2
          2
          2
          2
          2
          3
          3         ]

Note that the optimal values for ``x`` are discretized, clicking from 1
to 2 to 3 during the sweep, and that the solution has no dual variables.


Derived Variables
=================

Evaluated Fixed Variables
-------------------------

Some fixed variables may be derived from the values of other fixed variables.
For example, air density, viscosity, and temperature are functions of altitude.
These can be represented by a substitution or value that is a one-argument function
accepting ``model.substitutions`` (for details, see `Substitutions`_ below).

.. literalinclude:: examples/evaluated_fixed_variables.py

These functions are automatically differentiated with the `ad <https://pypi.org/project/ad/>`_ package to provide more accurate sensitivities. In some cases this requires functions from ``ad.admath`` instead of their python or numpy equivalents; the `ad documentation <https://pypi.org/project/ad/>`_ details how to do this.


Computed Variables
------------------

Some variables may be computed from the values of solved variables
after the optimization is performed. For example, if the efficiency :math:`\nu` of a motor is not a GP-compatible
variable, but :math:`(1-\nu)` is a valid GP variable, then :math:`\nu` can be calculated after solving.
These can be set up using the ``model.computed`` dictionary.

.. literalinclude:: examples/evaluated_free_variables.py

Note that computed variables should not be used in constructing your model constraints!
For variables computed during a solution (linked substitutions), see :ref:`sgp`.


.. _Sweeps:

Sweeps
======

Sweeps are useful for analyzing tradeoff surfaces. ``model.sweep({x: [1, 2, 3]})`` solves the model once per value and returns a ``Sweep``, a list of solutions, without changing the model's substitutions. Given several variables, as in ``model.sweep({x: xs, y: ys})``, it steps through them together, so the value lists must have the same length; to sweep a grid, pass every combination, e.g. built with ``itertools.product``. The method ``model.autosweep({sweepvar: (start, end)}, tol=0.01)`` sweeps one variable given only its bounds, solving enough points in between to keep the error in the log of the cost under ``tol``. For details see `1D Autosweeps`_ below.

1D Autosweeps
-------------
If you're only sweeping over a single variable, autosweeping lets you specify a
tolerance for cost error instead of a number of exact positions to solve at.
GPkit will then search the sweep segment for a locally optimal number of sweeps
that can guarantee a max absolute error on the log of the cost.

Accessing variable and cost values from an autosweep is slightly different, as
can be seen in this example:

.. literalinclude:: examples/autosweep.py

If you need access to the raw solutions arrays, the smallest simplex tree containing
any given point can be gotten with ``min_bst = bst.min_bst(val)``, the extents of that tree with ``bst.bounds`` and solutions of that tree with ``bst.sols``. More information is in ``help(bst)``.


Tight ConstraintSets
====================

Tight ConstraintSets will warn if any inequalities they contain are not
tight (that is, the right side does not equal the left side) after solving. This
is useful when you know that a constraint *should* be tight for a given model,
but representing it as an equality would be non-convex.

.. literalinclude:: examples/tight_constraintsets.py

Loose ConstraintSets
====================

Loose ConstraintSets will warn if any GP-compatible constraints they contain are
not loose (that is, their sensitivity is above some threshold after solving). This
is useful when you want a constraint to be inactive for a given model because
it represents an important model assumption (such as a fit only valid over a particular interval).

.. literalinclude:: examples/loose_constraintsets.py

Substitutions
=============

Substitutions are a general-purpose way to change every instance of one variable into either a number or another variable.

Substituting into Posynomials, NomialArrays, and GPs
-----------------------------------------------------

The examples below all use Posynomials and NomialArrays, but the syntax is identical for GPs (except when it comes to sweep variables).

.. literalinclude:: examples/substitutions.py

Here the variable ``x`` is being replaced with ``3`` in three ways: first by substituting for ``x`` directly, then by substituting for the ``VarKey("x")``, then by substituting the string "x". In all cases the substitution is understood as being with the VarKey: when a variable is passed in the VarKey is pulled out of it, and when a string is passed in it is used as an argument to the Posynomial's ``varkeys`` dictionary.

Substituting multiple values
----------------------------

.. literalinclude:: examples/sub_multi_values.py

To substitute in multiple variables, pass them in as a dictionary where the keys are what will be replaced and values are what it will be replaced with. Note that you can also substitute for VectorVariables by their name or by their NomialArray.

Substituting with nonnumeric values
-----------------------------------

You can also substitute in sweep variables (see Sweeps_), strings, and monomials:

.. literalinclude:: examples/substituting_nonnumeric_values.py

Note that units are preserved, and that the value can be either a string (in which case it just renames the variable), a varkey (in which case it changes its description, including the name) or a Monomial (in which case it substitutes for the variable with a new monomial).

Updating ConstraintSet substitutions
------------------------------------
ConstraintSets have a ``.substitutions`` KeyDict attribute which will be substituted before solving.
This KeyDict accepts variable names, VarKeys, and Variable objects as keys, and can be updated (or deleted from)
like a regular Python dictionary to change the substitutions that will be used at solve-time. If a ConstraintSet itself
contains ConstraintSets, it and all its elements share pointers to the same substitutions dictionary object,
so that updating any one of them will update all of them.


Fixed Variables
---------------

When a Model is created, any fixed Variables are used to form a dictionary: ``{var: var.value for var in self.varlocs if var.value is not None}``. This dictionary is then substituted into the Model's cost and constraints before the ``substitutions`` argument is (and hence values are supplanted by any later substitutions).

``solution[p]`` substitutes the solution's values into the scalar expression ``p`` and returns the result as a quantity in ``p``'s units; if ``p`` has a variable the solution doesn't hold, it raises ``KeyError``. For a single variable, scalar or vector, this is just its value, whether it was free or fixed. For a vector expression, compute with the values instead, e.g. ``2*solution[v]``.

Freeing Fixed Variables
-----------------------

After creating a Model, it may be useful to "free" a fixed variable and resolve.  This can be done using the command ``del m.substitutions["x"]``, where ``m`` is a Model.  An example of how to do this is shown below.

.. literalinclude:: examples/freeing_fixed_variables.py
