"Implements the VarMap class"

from collections.abc import MutableMapping

import numpy as np


def is_veckey(key):
    if getattr(key, "shape", None) and not getattr(key, "idx", None):
        # it has a shape but no index
        return True
    return False


class VarMap(MutableMapping):
    """A simple mapping from VarKey to value, with lookup by canonical
    name string or veckey

    Maintains:
      - _data: dict mapping VarKey to value
      - _by_name: dict mapping str (VarKey.name) to list of VarKeys
      - _by_vec: dict mapping veckey to values stored in np array
    """

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._by_name = {}
        self._by_vec = {}
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            vks = self._by_name.get(key, set())
            if not vks:
                raise KeyError(key)
            if len(vks) == 1:
                return self._data[next(iter(vks))]
            raise KeyError(f"Multiple VarKeys for name '{key}': {vks}")
        key = key.key  # works for both Variables and VarKeys
        if is_veckey(key):
            return self._by_vec[key]
        return self._data[key]

    def __setitem__(self, key, value):
        if not hasattr(key, "name"):
            raise TypeError("VarMap keys must be VarKey instances")
        self._data[key] = value
        name = key.name
        if name not in self._by_name:
            self._by_name[name] = set()
        self._by_name[name].add(key)
        # handle vector element case
        idx = getattr(key, "idx", None)
        if idx:
            veckey = key.veckey
            if veckey not in self._by_vec:
                self._by_vec[veckey] = np.full(veckey.shape, np.nan)
            self._by_vec[veckey][idx] = value

    def __delitem__(self, key):
        name = key.name
        del self._data[key]
        self._by_name[name].discard(key)
        # if _by_name[name] refers to no other vks, remove it
        if not self._by_name[name]:
            del self._by_name[name]
        # handle vector element case
        idx = getattr(key, "idx", None)
        if idx:
            veckey = key.veckey
            self._by_vec[veckey][idx] = np.nan
        # handle full veckey case
        if is_veckey(key):
            del self._by_vec[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._by_name and bool(self._by_name[key])
        return key in self._data

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def by_name(self, name):
        """Return all VarKeys for a given name string."""
        return set(self._by_name.get(name, set()))
