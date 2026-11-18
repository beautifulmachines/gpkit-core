.. _signomialprogramming:

Signomial Programming
*********************

Signomial programming finds a local solution to a problem of the form:


.. math:: \begin{array}{lll}\text{}
    \text{minimize} & g_0(x) & \\
    \text{subject to} & f_i(x) = 1, & i = 1,....,m \\
                      & g_i(x) - h_i(x) \leq 1, & i = 1,....,n
                      \end{array}

where each :math:`f` is monomial while each :math:`g` and :math:`h` is a posynomial.

This requires multiple solutions of geometric programs, and so will take longer to solve than an equivalent geometric programming formulation.

In general, when given the choice of which variables to include in the positive-posynomial / :math:`g` side of the constraint, the modeler should:

    #. maximize the number of variables in :math:`g`,
    #. prioritize variables that are in the objective,
    #. then prioritize variables that are present in other constraints.

The ``.localsolve`` syntax was chosen to emphasize that signomial programming returns a local optimum. For the same reason, calling ``.solve`` on an SP will raise an error.

Each GP in the sequence replaces :math:`h` with a monomial lower bound, so it is a local approximation of the signomial program. Unless each :math:`h` comes with a guess (below), the first GP is built at a default point where every unset variable is 1 in its units. That GP need not be conservative, so ``localsolve`` adds a slack variable that lets each approximated constraint loosen, and penalizes it in the cost.

Guessing term shares
====================

By the weighted arithmetic-geometric mean inequality, for any shares :math:`w_k \geq 0` that sum to 1,

.. math:: h(x) = \sum_k u_k(x) \geq \prod_k \left(\frac{u_k(x)}{w_k}\right)^{w_k},

with equality wherever term :math:`u_k` is the fraction :math:`w_k` of :math:`h`. A term with zero share drops out of the product. So the monomial bound is fixed by how :math:`h`'s value splits among its terms, and a modeler often knows that split better than any starting point: which term dominates, or which can be left out at first. ``guess`` writes it down as the greater-than side of the constraint:

.. literalinclude:: examples/pressure_vessel.py

Shares are relative weights, normalized to sum to 1. A zero share for the cap term drops it from the first bound, so the first GP counts the caps at the cylinder's full thickness. With a guess on every signomial constraint:

* **The first GP is conservative.** Its feasible set lies inside the signomial program's, so its solution, when it has one, is a feasible design. ``m.local_gp()`` builds that GP; ``m.gp()`` and ``m.solve()`` still raise, because the model itself is not a GP.
* **No slack is needed.** ``localsolve`` condenses each later GP at the previous solution, where the bound is exact, so the previous design stays feasible and the cost never rises.
* **An infeasible first GP raises** ``Infeasible``: the guess left out a term the design needs, and should give it some share.

A term written on both sides of a guessed constraint raises: cancel it, and guess the shares of what remains. An ``x0`` sets the first shares of any constraint whose greater-than side it collapses to a monomial: values that make the ratios between its terms numbers. Other constraints start from their guesses.

.. literalinclude:: examples/pressure_vessel_output.txt
    :language: breakdowns

The same model can be written in TOML (see :doc:`tomlmodels`).

Example Usage
=============

.. literalinclude:: examples/simple_sp.py

When using the ``localsolve`` method, the ``reltol`` argument specifies the relative tolerance of the solver: that is, by what percent does the solution have to improve between iterations? If any iteration improves less than that amount, the solver stops and returns its value.

If you wish to start the local optimization at a particular point :math:`x_0`, however, you may do so by putting that position (a dictionary formatted as you would a substitution) as the ``x0`` argument.

.. _sgp:

Sequential Geometric Programs
=============================

The method of solving local GP approximations of a non-GP compatible model can be generalized, at the cost of the general smoothness and lack of a need for trust regions that SPs guarantee.

For some applications, it is useful to call external codes which may not be GP compatible.  Imagine we wished to solve the following optimization problem:

.. math:: \begin{array}{lll}\text{}
    \text{minimize} & y & \\
    \text{subject to} & y \geq \sin(x) \\
                      & \frac{\pi}{4} \leq x \leq \frac{\pi}{2}
                      \end{array}

This problem is not GP compatible due to the :math:`sin(x)` constraint.  One approach might be to take the first term of the Taylor expansion of :math:`sin(x)` and attempt to solve:

.. literalinclude:: examples/sin_approx_example.py

.. literalinclude:: examples/sin_approx_example_output.txt
    :language: breakdowns

Assume we have some external code which is capable of evaluating our incompatible function:

.. literalinclude:: examples/external_function.py

Now, we can create a ConstraintSet that allows GPkit to treat the incompatible constraint as though it were a signomial programming constraint:

.. literalinclude:: examples/external_constraint.py

and replace the incompatible constraint in our GP:

.. literalinclude:: examples/external_sp.py

.. literalinclude:: examples/external_sp_output.txt
    :language: breakdowns

which is the expected result.  This method has been generalized to larger problems, such as calling XFOIL and AVL.

If you wish to start the local optimization at a particular point :math:`x_0`, however, you may do so by putting that position (a dictionary formatted as you would a substitution) as the ``x0`` argument
