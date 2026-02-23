"Docstring-parsing methods"

import ast
import inspect


class parse_variables:  # pylint:disable=invalid-name,too-few-public-methods
    """decorator for adding local Variables from a string.

    Generally called as `@parse_variables(__doc__, globals())`.
    """

    def __init__(self, string, scopevars=None):
        self.string = string
        self.scopevars = scopevars
        if scopevars is None:
            raise DeprecationWarning("""
parse_variables is no longer used directly with exec, but as a decorator:

    @parse_variables(__doc__, globals())
    def setup(...):

""")

    def __call__(self, function):  # pylint:disable=too-many-locals
        orig_lines, lineno = inspect.getsourcelines(function)
        indent_length = 0
        while orig_lines[1][indent_length] in [" ", "\t"]:
            indent_length += 1
        first_indent_length = indent_length
        setup_lines = 1
        while "):" not in orig_lines[setup_lines]:
            setup_lines += 1
        next_indented_idx = setup_lines + 1
        # get the next indented line
        while len(orig_lines[next_indented_idx]) <= indent_length + 1:
            next_indented_idx += 1
        while orig_lines[next_indented_idx][indent_length] in [" ", "\t"]:
            indent_length += 1
        second_indent = orig_lines[next_indented_idx][:indent_length]
        parse_lines = [
            second_indent + line + "\n"
            for line in parse_varstring(self.string).split("\n")
        ]
        parse_lines += [second_indent + "# (@parse_variables spacer line)\n"]
        parse_lines += [second_indent + "# (setup spacer line)\n"] * setup_lines
        # make ast of these new lines, insert it into the original ast
        new_lines = (
            orig_lines[1 : setup_lines + 1]
            + parse_lines
            + orig_lines[setup_lines + 1 :]
        )
        new_src = "\n".join(
            [
                line[first_indent_length:-1]
                for line in new_lines
                if "#" not in line[:first_indent_length]
            ]
        )
        new_ast = ast.parse(new_src, "<parse_variables>")
        ast.increment_lineno(new_ast, n=lineno - len(parse_lines))
        code = compile(
            new_ast, inspect.getsourcefile(function), "exec", dont_inherit=True
        )  # don't inherit __future__ from here
        out = {}
        exec(code, self.scopevars, out)  # pylint: disable=exec-used
        return out[function.__name__]


# pylint: disable=no-member
def parse_varstring(string):
    "Parses a string to determine what variables to create from it"
    consts = check_and_parse_flag(string, "Constants\n")
    variables = check_and_parse_flag(string, "Variables\n")
    vecvars = check_and_parse_flag(string, "Variables of length", vv_declare)
    out = ["# " + line.strip() for line in string.split("\n")]
    # imports, to be updated if more things are parsed above
    out[0] = "from gpkit import Variable, VectorVariable" + "  " + out[0]
    for lines, indexs in (consts, variables, vecvars):
        for line, index in zip(lines.split("\n"), indexs):
            out[index] = f"{line}  # from '{out[index][1:].strip()}'"
    return "\n".join(out)


def vv_declare(string, flag, idx2, countstr):
    "Turns Variable declarations into VectorVariable ones"
    length = string[len(flag) : idx2].strip()
    return countstr.replace("Variable(", f"VectorVariable({length}, ")


# pylint: disable=too-many-locals
def check_and_parse_flag(string, flag, declaration_func=None):
    "Checks for instances of flag in string and parses them."
    overallstr = ""
    originalstr = string
    lineidxs = []
    for _ in range(string.count(flag)):
        countstr = ""
        idx = string.index(flag)
        string = string[idx:]
        if idx == -1:
            idx = 0
            skiplines = 0
        else:
            skiplines = 2
        idx2 = 0 if "\n" in flag else string.index("\n")
        for line in string[idx2:].split("\n")[skiplines:]:
            if not line.strip():  # whitespace only
                break
            try:
                units = line[line.index("[") + 1 : line.index("]")]
            except ValueError as err:
                raise ValueError(
                    "A unit declaration bracketed by [] was"
                    " not found on the line reading:\n"
                    f"    {line}"
                ) from err
            nameval = line[: line.index("[")].split()
            labelstart = line.index("]") + 1
            if labelstart >= len(line):
                label = ""
            else:
                while line[labelstart] == " ":
                    labelstart += 1
                label = line[labelstart:].replace("\\", "\\\\").replace("'", "\\'")
            countstr += variable_declaration(nameval, units, label, line)
            # note that this is a 0-based line indexing
            lineidxs.append(originalstr[: originalstr.index(line)].count("\n"))
        if declaration_func is None:
            overallstr += countstr
        else:
            overallstr += declaration_func(string, flag, idx2, countstr)
        string = string[idx2 + len(flag) :]
    return overallstr, lineidxs


PARSETIP = (
    "Is this line following the format `Name (optional Value) [Units]"
    " (Optional Description)` without any whitespace in the Name or"
    " Value fields?"
)


def variable_declaration(nameval, units, label, line):
    "Turns parsed output into a Variable declaration"
    if len(nameval) > 2:
        raise ValueError(
            f"while parsing the line '{line}', additional fields"
            f" (separated by whitespace) were found between Value"
            f" '{nameval[1]}' and the Units `{units}`. {PARSETIP}"
        )
    if len(nameval) == 2:
        out = (
            f"{nameval[0]} = self.{nameval[0]} = "
            f"Variable('{nameval[0]}', {nameval[1]}, '{units}', '{label}')"
        )
    elif len(nameval) == 1:
        out = (
            f"{nameval[0]} = self.{nameval[0]} = "
            f"Variable('{nameval[0]}', '{units}', '{label}')"
        )
    return out + "\n"  # pylint: disable= possibly-used-before-assignment
