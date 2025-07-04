# pylint: disable=import-outside-toplevel
"""Script for running all gpkit unit tests"""

from gpkit.tests.helpers import run_tests


def import_tests():
    """Get a list of all GPkit unit test TestCases"""
    tests = []

    from gpkit.tests import t_tools

    tests += t_tools.TESTS

    from gpkit.tests import t_sub

    tests += t_sub.TESTS

    from gpkit.tests import t_vars

    tests += t_vars.TESTS

    from gpkit.tests import t_nomials

    tests += t_nomials.TESTS

    from gpkit.tests import t_constraints

    tests += t_constraints.TESTS

    from gpkit.tests import t_nomial_array

    tests += t_nomial_array.TESTS

    from gpkit.tests import t_model

    tests += t_model.TESTS

    from gpkit.tests import t_solution_array

    tests += t_solution_array.TESTS

    from gpkit.tests import t_examples

    tests += t_examples.TESTS

    from gpkit.tests import t_varmap

    tests += t_varmap.TESTS

    from gpkit.tests import t_util

    tests += t_util.TESTS

    return tests


def run(xmloutput=False, tests=None, verbosity=1):
    """Run all gpkit unit tests.

    Arguments
    ---------
    xmloutput: bool
        If true, generate xml output files for continuous integration
    """
    if tests is None:
        tests = import_tests()
    if xmloutput:
        run_tests(tests, xmloutput="test_reports")
    else:  # pragma: no cover
        run_tests(tests, verbosity=verbosity)


if __name__ == "__main__":  # pragma: no cover
    run()
