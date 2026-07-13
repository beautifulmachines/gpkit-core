"context-local construction state and lazily-loaded settings"

import os
import sys
import tomllib
from collections import defaultdict
from contextvars import ContextVar

from .build import build


def load_settings(path=None, trybuild=True):
    "Load the TOML settings file at path; return settings dict"
    if path is None:
        path = os.sep.join(
            [os.path.dirname(os.path.dirname(__file__)), "env", "settings.toml"]
        )
    try:
        with open(path, "rb") as settingsfile:
            settings_ = tomllib.load(settingsfile)
    except (OSError, tomllib.TOMLDecodeError):  # pragma: no cover
        settings_ = {"installed_solvers": []}
    if not settings_["installed_solvers"] and trybuild:  # pragma: no cover
        # Bootstrap diagnostics go to stderr: the load can be triggered from
        # inside stdout captures (example regeneration, StdoutCaptured) whose
        # output must contain only the captured program's own stdout.
        print("Found no installed solvers, beginning a build.", file=sys.stderr)
        build()
        settings_ = load_settings(path, trybuild=False)
        if settings_["installed_solvers"]:
            print(
                f"""
GPkit is now installed with solver(s) {", ".join(settings_["installed_solvers"])}
To incorporate new solvers at a later date, run `gpkit.build()`.

If you encounter any bugs or issues using GPkit, please open a new issue at
https://github.com/beautifulmachines/gpkit-core/issues/new.

We hope you find the engineering-design models at
https://github.com/beautifulmachines/gpkit-models/ useful for your own applications.

Enjoy!
""",
                file=sys.stderr,
            )
        else:
            print(
                """
=============
Build failed!  :(
=============
You may need to install a solver and then `import gpkit` again.
Please post the output above to
https://github.com/beautifulmachines/gpkit-core/issues/new
so we can prevent others from having to see this message.

        Thanks!  :)
""",
                file=sys.stderr,
            )
    settings_.setdefault(
        "default_solver",
        settings_["installed_solvers"][0] if settings_["installed_solvers"] else "",
    )
    return settings_


class _Settings:
    """Dict-like view of the settings file, loaded on first access.

    Deferring the load keeps `import gpkit` free of filesystem reads and of
    the solver build that a missing settings file triggers.
    """

    def __init__(self):
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = load_settings()
        return self._data

    def __getitem__(self, key):
        return self._load()[key]

    def __setitem__(self, key, value):
        self._load()[key] = value

    def __contains__(self, key):
        return key in self._load()

    def __repr__(self):
        return repr(self._load())


settings = _Settings()


# Construction state lives in ContextVars: each thread (and each asyncio
# task) sees its own value, so concurrent model builds don't interfere.
_signomials_enabled = ContextVar("signomials_enabled", default=False)
_vectorization = ContextVar("vectorization", default=())
_lineage = ContextVar("lineage", default=())
_modelnums = ContextVar("modelnums")
_namedvars = ContextVar("namedvars")


def _context_dict(var, factory):
    "Return var's value in the current context, initializing it if unset."
    try:
        return var.get()
    except LookupError:
        value = factory()
        var.set(value)
        return value


class SignomialsEnabledMeta(type):
    "Metaclass to implement falsiness for SignomialsEnabled"

    def __bool__(cls):
        return _signomials_enabled.get()


class SignomialsEnabled(metaclass=SignomialsEnabledMeta):
    """Class to put up and tear down signomial support in an instance of GPkit.

    Example
    -------
        >>> import gpkit
        >>> x = gpkit.Variable("x")
        >>> y = gpkit.Variable("y", 0.1)
        >>> with SignomialsEnabled():
        >>>     constraints = [x >= 1-y]
        >>> gpkit.Model(x, constraints).localsolve()
    """

    def __enter__(self):
        self._token = _signomials_enabled.set(True)

    def __exit__(self, type_, val, traceback):
        _signomials_enabled.reset(self._token)


class VectorizeMeta(type):
    "Exposes the current vectorization shape as a class attribute."

    @property
    def vectorization(cls):
        "the current vectorization shape"
        return _vectorization.get()


class Vectorize(metaclass=VectorizeMeta):
    """Creates an environment in which all variables are
    extended in an additional dimension.
    """

    def __init__(self, dimension_length):
        self.dimension_length = dimension_length

    def __enter__(self):
        "Enters a vectorized environment."
        self._token = _vectorization.set(
            (self.dimension_length,) + _vectorization.get()
        )

    def __exit__(self, type_, val, traceback):
        "Leaves a vectorized environment."
        _vectorization.reset(self._token)


class NamedVariablesMeta(type):
    "Exposes the current naming context as class attributes."

    @property
    def lineage(cls):
        "the current model nesting"
        return _lineage.get()

    @property
    def modelnums(cls):
        "the number of models of each lineage"
        return _context_dict(_modelnums, lambda: defaultdict(int))

    @property
    def namedvars(cls):
        "variables created in the current nesting"
        return _context_dict(_namedvars, lambda: defaultdict(list))


class NamedVariables(metaclass=NamedVariablesMeta):
    """Creates an environment in which all variables have
    a model name and num appended to their varkeys.
    """

    @classmethod
    def reset_modelnumbers(cls):
        "Clear all model number counters"
        cls.modelnums.clear()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        "Enters a named environment."
        lineage = NamedVariables.lineage
        modelnums = NamedVariables.modelnums
        num = modelnums[(lineage, self.name)]
        modelnums[(lineage, self.name)] += 1
        lineage += ((self.name, num),)
        self._token = _lineage.set(lineage)
        return lineage, NamedVariables.namedvars[lineage]

    def __exit__(self, type_, val, traceback):
        "Leaves a named environment."
        del NamedVariables.namedvars[NamedVariables.lineage]
        _lineage.reset(self._token)
