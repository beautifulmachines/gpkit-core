"Scripts to parse and collate substitutions"

import numpy as np

from ..varmap import VarSet


def is_linked(val):
    "Return True if val is a linked (callable-computed) substitution value."
    return hasattr(val, "__call__") and not hasattr(val, "key")


def parse_subs(varkeys, substitutions):
    "Return constant mappings {VarKey: val} w/ broadcasting and shape checks."
    out = {}
    if not isinstance(varkeys, VarSet):
        raise NotImplementedError
    for var, val in substitutions.items():
        if hasattr(val, "__len__") and not hasattr(val, "shape"):
            # cast for shape and lookup by idx
            val = np.array(val) if not hasattr(val, "units") else val
            # else case is pint bug: Quantity's have __len__ but len() raises
        if is_linked(val):
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


def parse_linked(varkeys, substitutions):
    "Extract linked (callable-computed) substitutions as {VarKey: function}."
    out = {}
    for var, val in substitutions.items():
        for key in varkeys.keys(var):
            if is_linked(val):
                out[key] = val
    return out
