"Implements the VarMap class"

from collections.abc import MutableMapping


class VarMap(MutableMapping):
    """A simple mapping from VarKey to value, with lookup by canonical name string.

    Maintains:
      - _data: dict mapping VarKey to value
      - _by_name: dict mapping str (VarKey.name) to list of VarKeys
    """

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._by_name = {}
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            vks = self._by_name.get(key, set())
            if not vks:
                raise KeyError(key)
            if len(vks) == 1:
                return self._data[next(iter(vks))]
            raise KeyError(f"Multiple VarKeys for name '{key}': {vks}")
        return self._data[key]

    def __setitem__(self, key, value):
        if not hasattr(key, "name"):
            raise TypeError("VarMap keys must be VarKey instances")
        self._data[key] = value
        name = key.name
        if name not in self._by_name:
            self._by_name[name] = set()
        self._by_name[name].add(key)

    def __delitem__(self, key):
        name = key.name
        del self._data[key]
        self._by_name[name].discard(key)
        # if _by_name[name] refers to no other vks, remove it
        if not self._by_name[name]:
            del self._by_name[name]

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
