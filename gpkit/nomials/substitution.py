"Scripts to parse and collate substitutions"

import warnings as pywarnings

import numpy as np

from ..util.small_scripts import splitsweep
from ..varmap import VarSet


def new_parse_subs(varkeys, substitutions):
    out = dict()
    if not isinstance(varkeys, VarSet):
        raise NotImplementedError
    for var, val in substitutions.items():
        if hasattr(val, "__len__") and not hasattr(val, "shape"):
            # cast for shape and lookup by idx
            val = np.array(val) if not hasattr(val, "units") else val
            # else case is pint bug: Quantity's have __len__ but len() raises
        for key in varkeys.keys(var):
            if key.shape and getattr(val, "shape", None):
                if key.shape == val.shape:
                    out[key] = val[key.idx]
                    continue
                raise ValueError(
                    f"cannot substitute array of shape {val.shape} for"
                    f" variable {key.veckey} of shape {key.shape}."
                )
            out[key] = val
    return out


def parse_subs(varkeys, substitutions, clean=False):
    "Seperates subs into the constants, sweeps, linkedsweeps actually present."
    constants, sweep, linkedsweep = {}, {}, {}
    if clean:
        for var in varkeys:
            if dict.__contains__(substitutions, var):
                sub = dict.__getitem__(substitutions, var)
                append_sub(sub, [var], constants, sweep, linkedsweep)
    else:
        if not isinstance(varkeys, VarSet):
            varkeys = VarSet(varkeys)
        for var in substitutions:
            key = getattr(var, "key", var)
            if key in varkeys:
                sub, keys = substitutions[var], varkeys.keys(key)
                append_sub(sub, keys, constants, sweep, linkedsweep)
    return constants, sweep, linkedsweep


def append_sub(sub, keys, constants, sweep, linkedsweep):
    # pylint: disable=too-many-branches
    "Appends sub to constants, sweep, or linkedsweep."
    sweepsub, sweepval = splitsweep(sub)
    if sweepsub:  # if the whole key is swept
        sub = sweepval
    for key in keys:
        if not key.shape or not getattr(sub, "shape", hasattr(sub, "__len__")):
            value = sub
        else:
            with pywarnings.catch_warnings():
                pywarnings.filterwarnings("error")
                try:
                    sub = np.array(sub) if not hasattr(sub, "shape") else sub
                # pylint: disable=fixme
                except ValueError:  # pragma: no cover  # TODO: coverage this
                    # ragged nested sequences, eg [[2]], [3, 4]], in py3.7+
                    sub = np.array(sub, dtype=object)
            if key.shape == sub.shape:
                value = sub[key.idx]
                if sweepsub and np.prod(key.shape) != len(keys):
                    value = sub  # handle coincidental length match case
                sweepel, sweepval = splitsweep(value)
                if sweepel:  # if only an element is swept
                    value = sweepval
                    sweepsub = True
            elif sweepsub:
                try:
                    np.broadcast(sub, np.empty(key.shape))
                except ValueError as exc:
                    raise ValueError(
                        f"cannot sweep variable {key.veckey} of shape {key.shape}"
                        f" with array of shape {sub.shape}; array shape"
                        f" must either be {key.shape} or {('N',) + key.shape}"
                    ) from exc
                idx = (slice(None),) + key.descr["idx"]
                value = sub[idx]
            else:
                raise ValueError(
                    f"cannot substitute array of shape {sub.shape} for"
                    f" variable {key.veckey} of shape {key.shape}."
                )
        if hasattr(value, "__call__") and not hasattr(value, "key"):
            linkedsweep[key] = value
        elif sweepsub:
            sweep[key] = value
        else:
            try:
                assert np.isnan(value)
            except (AssertionError, TypeError, ValueError):
                constants[key] = value
