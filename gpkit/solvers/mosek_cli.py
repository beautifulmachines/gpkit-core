"""Module for using the MOSEK EXPOPT command line interface

Example
-------
``result = _mosek.cli_expopt.imize(cs, A, p_idxs, "gpkit_mosek")``

"""

import errno
import os
import shutil
import stat
import sys
import tempfile
from subprocess import CalledProcessError, check_output

from ..exceptions import (
    DualInfeasible,
    InvalidLicense,
    PrimalInfeasible,
    UnknownInfeasible,
)
from ..globals import settings
from ..solutions import RawSolution


def _handle_remove_readonly(func, path, exc_or_exc_info):  # pragma: no cover
    """Handle rmtree errors by changing permissions and retrying.

    Works with both onerror (Python < 3.12) and onexc (Python >= 3.12).
    """
    # onerror passes (exc_type, exc_value, exc_tb) tuple
    # onexc passes the exception directly
    exc = exc_or_exc_info[1] if isinstance(exc_or_exc_info, tuple) else exc_or_exc_info
    if func in (os.rmdir, os.remove) and getattr(exc, "errno", None) == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        func(path)  # try again


def _safe_rmtree(path):  # pragma: no cover
    """Remove directory tree, handling read-only files."""
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_handle_remove_readonly)
    else:
        # pylint: disable-next=deprecated-argument
        shutil.rmtree(path, ignore_errors=False, onerror=_handle_remove_readonly)


def optimize_generator(path=None, **_):
    """Constructor for the MOSEK CLI solver function.

    Arguments
    ---------
    path : str (optional)
        The directory in which to put the MOSEK CLI input/output files.
        By default uses a system-appropriate temp directory.
    """
    tmpdir = path is None
    if tmpdir:
        path = tempfile.mkdtemp()
    filename = path + os.sep + "gpkit_mosek"
    if "mosek_bin_dir" in settings:
        if settings["mosek_bin_dir"] not in os.environ["PATH"]:
            os.environ["PATH"] += ":" + settings["mosek_bin_dir"]

    def optimize(prob, **_):
        """Interface to the MOSEK "mskexpopt" command line solver

        Definitions
        -----------
        n is the number of monomials in the gp
        m is the number of variables in the gp
        p is the number of posynomials in the gp

        Arguments
        ---------
        c : floats array of shape n
            Coefficients of each monomial
        A: floats array of shape (m,n)
            Exponents of the various free variables for each monomial.
        p_idxs: ints array of shape n
            Posynomial index of each monomial
        filename: str
            Filename prefix for temporary files

        Returns
        -------
        dict
            Contains the following keys
                "success": bool
                "objective_sol" float
                    Optimal value of the objective
                "primal_sol": floats array of size m
                    Optimal value of the free variables. Note: not in logspace.
                "dual_sol": floats array of size p
                    Optimal value of the dual variables, in logspace.

        Raises
        ------
        RuntimeWarning
            If the format of mskexpopt's output file is unexpected.

        """
        write_output_file(filename, prob.c, prob.A, prob.p_idxs)

        # run mskexpopt and print stdout
        solution_filename = filename + ".sol"
        try:
            for logline in check_output(
                ["mskexpopt", filename, "-sol", solution_filename]
            ).split(b"\n"):
                print(logline)
        except CalledProcessError as e:
            # invalid license return codes:
            #   expired: 233 (linux)
            #   missing: 240 (linux)
            if e.returncode in [233, 240]:  # pragma: no cover
                raise InvalidLicense() from e
            raise UnknownInfeasible() from e
        with open(solution_filename, encoding="utf-8") as f:
            _, probsta = f.readline()[:-1].split("PROBLEM STATUS      : ")
            if probsta == "PRIMAL_INFEASIBLE":
                raise PrimalInfeasible()
            if probsta == "DUAL_INFEASIBLE":
                raise DualInfeasible()
            if probsta != "PRIMAL_AND_DUAL_FEASIBLE":
                raise UnknownInfeasible("PROBLEM STATUS: " + probsta)

            _, solsta = f.readline().split("SOLUTION STATUS     : ")
            # line looks like "OBJECTIVE           : 2.763550e+002"
            objective_val = float(f.readline().split()[2])
            assert_equal(f.readline(), "")
            assert_equal(f.readline(), "PRIMAL VARIABLES")
            assert_equal(f.readline(), "INDEX   ACTIVITY")
            primal_vals = read_vals(f)
            # read_vals reads the dividing blank line as well
            assert_equal(f.readline(), "DUAL VARIABLES")
            assert_equal(f.readline(), "INDEX   ACTIVITY")
            dual_vals = read_vals(f)

        if tmpdir:
            _safe_rmtree(path)

        return RawSolution(
            status=solsta[:-1],
            cost=objective_val,
            x=primal_vals,
            nu=dual_vals,
            la=prob.compute_la(dual_vals),
            meta={"solver": "mosek_cli"},
        )

    return optimize


def write_output_file(filename, c, A, p_idxs):
    "Writes a mosekexpopt compatible GP description to `filename`."
    with open(filename, "w", encoding="utf-8") as f:
        numcon = p_idxs[-1]
        numter, numvar = map(int, A.shape)
        for n in [numcon, numvar, numter]:
            f.write(f"{n}\n")

        f.write("\n*c\n")
        f.writelines([f"{x:.20e}\n" for x in c])

        f.write("\n*p_idxs\n")
        f.writelines([f"{x}\n" for x in p_idxs])

        f.write("\n*t j A_tj\n")
        f.writelines([f"{i} {j} {v:.20e}\n" for i, j, v in zip(A.row, A.col, A.data)])


def assert_equal(received, expected):
    "Asserts that a file's next line is as expected."
    if expected.rstrip() != received.rstrip():  # pragma: no cover
        errstr = repr(expected) + " is not the same as " + repr(received)
        raise RuntimeWarning("could not read mskexpopt output file: " + errstr)


def read_vals(fil):
    "Read numeric values until a blank line occurs."
    vals = []
    line = fil.readline()
    while line not in ["", "\n"]:
        # lines look like "1       2.390776e+000   \n"
        vals.append(float(line.split()[1]))
        line = fil.readline()
    return vals
