"Scripts to parse and collate substitutions"

import warnings as pywarnings

import numpy as np

from ..varmap import VarSet


def parse_subs(varkeys, substitutions):
    out = dict()
    if not isinstance(varkeys, VarSet):
        raise NotImplementedError
    for var, val in substitutions.items():
        if hasattr(val, "__len__") and not hasattr(val, "shape"):
            # cast for shape and lookup by idx
            val = np.array(val) if not hasattr(val, "units") else val
            # else case is pint bug: Quantity's have __len__ but len() raises
        if hasattr(val, "__call__") and not hasattr(val, "key"):
            # linked case -- eventually make explicit instead of using subs
            continue
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


def parse_sweep(varkeys, sweeps):
    out = dict()
    if not sweeps:
        return {}
    for var, val in sweeps.items():
        keys = varkeys.keys(var)
        for key in keys:
            if not key.shape:
                out[key] = val
                continue
            with pywarnings.catch_warnings():
                pywarnings.filterwarnings("error")
                try:
                    val = np.array(val) if not hasattr(val, "shape") else val
                # pylint: disable=fixme
                except ValueError:  # pragma: no cover  # TODO: coverage this
                    # ragged nested sequences, eg [[2]], [3, 4]], in py3.7+
                    val = np.array(val, dtype=object)
            if key.shape == val.shape:
                # this silly case goes away once standard sweep lens enforced
                value = val if np.prod(key.shape) != len(keys) else val[key.idx]
                # handles coincidental length match case
                out[key] = value
                continue
            try:
                np.broadcast(val, np.empty(key.shape))
            except ValueError as exc:
                raise ValueError(
                    f"cannot sweep variable {key.veckey} of shape {key.shape}"
                    f" with array of shape {val.shape}; array shape"
                    f" must either be {key.shape} or {('N',) + key.shape}"
                ) from exc
            idx = (slice(None),) + key.descr["idx"]
            out[key] = val[idx]
    return out


def parse_linked(varkeys, substitutions):
    out = dict()
    for var, val in substitutions.items():
        for key in varkeys.keys(var):
            if hasattr(val, "__call__") and not hasattr(val, "key"):
                out[key] = val
    return out
