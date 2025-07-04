"Implements the VarMap class"

from collections.abc import MutableMapping

import numpy as np

from .units import Quantity
from .util.small_scripts import is_sweepvar, veclinkedfn


def _nested_lookup(nested_keys, val_dict):
    if nested_keys is None:
        return float("nan")
    if isinstance(nested_keys, np.ndarray):
        return [_nested_lookup(row, val_dict) for row in nested_keys]
    return val_dict[nested_keys]


def is_veckey(key):
    "return True iff this key corresponds to a VectorVariable"
    if getattr(key, "shape", None) and not getattr(key, "idx", None):
        # it has a shape but no index
        return True
    return False


class VarSet(set):
    """
    A lightweight *set* of :class:`gpkit.VarKey` objects with constant-time
    look-ups by **canonical name** or **vector parent (veckey)**.

    Unlike :class:`~gpkit.VAR.VarMap`, *VarSet stores no values — it is
    purely a membership container plus two derived indices:

    Attributes
    ----------
    _by_name : dict[str, set[VarKey]]
        Maps each canonical name string (``VarKey.name``) to the set of
        keys sharing that name. Maps to veckey, not elements, for vectors.
    _by_vec : dict[VarKey, numpy.ndarray]
        Maps a vector parent key (veckey) to an ``ndarray(dtype=object)``
        whose element at index *i* is the scalar key representing
        ``veckey[i]``.  Positions for which no element key has been
        registered contain ``None``.

    Invariants
    ----------
    * Every key in the VarSet appears exactly **once** in the underlying
      Python set and once in either index.
    * The indices are **derivative**—mutating the container via
      :py:meth:`add`, :py:meth:`update`, or :py:meth:`discard` keeps
      ``_by_name`` and ``_by_vec`` in sync automatically.
    * ``len(varset)`` equals the number of scalar VarKey instances
      currently registered (any parent veckeys are not included)

    See Also
    --------
    VarMap
        Mapping from VarKey → value that contains a VarSet internally.
    """

    pass


class VarMap(MutableMapping):
    """A simple mapping from VarKey to value, with lookup by canonical
    name string or veckey

    Maintains:
      - _data: dict mapping VarKey to value
      - _by_name: dict mapping str (VarKey.name) to set of VarKeys
      - _by_vec: dict mapping veckey to keys stored in nested list structure
    """

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._by_name = {}
        self._by_vec = {}
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        _, val = self.item(key)
        return val

    def item(self, key):
        "get the (varkey, value) pair associated with a (str or key)"
        key = getattr(key, "key", None) or key  # handles Variable case
        try:
            return (key, self._data[key])  # single varkey case
        except KeyError as kerr:
            if is_veckey(key):  # vector case
                return (key, _nested_lookup(self._by_vec[key], self._data))
            if isinstance(key, str):  # by name lookup
                vk = self._key_from_name(key)
                return self.item(vk)
            raise kerr

    def _key_from_name(self, name):
        vks = self.keys_by_name(name)
        if not vks:
            raise KeyError(f"unrecognized key {name}")
        if len(vks) == 1:
            (vk,) = vks
            return vk
        raise KeyError(f"Multiple VarKeys for name '{name}': {vks}")

    def __setitem__(self, key, value):
        key = getattr(key, "key", None) or key  # handles Variable case
        if isinstance(key, str):
            key = self._key_from_name(key)
        if isinstance(value, Quantity):
            value = value.to(key.units).magnitude
        if is_veckey(key):
            if key not in self._by_vec:
                raise NotImplementedError
            if is_sweepvar(value):
                self._data[key] = value
                return
            if hasattr(value, "__call__"):  # a linked vector-function
                # this case temporarily borrowed from keymap, should refactor
                key.vecfn = value
                value = np.empty(key.shape, dtype="object")
                it = np.nditer(value, flags=["multi_index", "refs_ok"])
                while not it.finished:
                    i = it.multi_index
                    it.iternext()
                    value[i] = veclinkedfn(key.vecfn, i)
            # to setitem via a veckey, the keys must already be registered.
            vks = set(self._by_vec[key].flat)
            if np.prod(key.shape) != len(vks):
                raise ValueError(
                    f"{key} shape is {key.shape}, but _by_vec[{key}] only has "
                    f" {len(vks)} registered keys."
                )  # now we know all vks are present
            if not np.shape(value):
                self.update({vk: value for vk in vks})
                return
            if np.shape(value) == key.shape:
                for vk in vks:
                    self[vk] = np.array(value)[vk.idx]
                return
            raise NotImplementedError
        self._register_key(key)
        self._data[key] = value

    def _register_key(self, key):
        "adds the key to _by_name and, if applicable, _by_vec"
        if not hasattr(key, "name"):
            raise TypeError("VarMap keys must be VarKey instances")
        idx = getattr(key, "idx", None)
        name = key.name if not idx else key.veckey.name
        if name not in self._by_name:
            self._by_name[name] = set()
        self._by_name[name].add(key if not idx else key.veckey)
        if idx:
            if key.veckey not in self._by_vec:
                self._by_vec[key.veckey] = np.empty(key.shape, dtype=object)
            self._by_vec[key.veckey][idx] = key

    def register_keys(self, keys):
        "register a set of keys to this mapping, without values yet"
        for key in keys:
            self._register_key(key)

    def __delitem__(self, key):
        key = getattr(key, "key", None) or key  # handles Variable case
        if is_veckey(key):
            raise NotImplementedError
        if isinstance(key, str):
            key = self._key_from_name(key)
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
            self._by_vec[veckey][idx] = None
            if not self._by_vec[veckey].any():
                del self._by_vec[veckey]

    def _primary_keys(self):
        "keys; uses veckeys instead of individual element keys where applicable"
        ks = set(self._data)
        for vk, vks in self._by_vec.items():
            ks -= set(vks.flat)
            ks.add(vk)
        return ks

    def primary_items(self):
        "like items, but using veckeys and ignoring element keys/items"
        return ((k, self[k]) for k in self._primary_keys())

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._by_name and bool(self._by_name[key])
        key = getattr(key, "key", key)  # handle Variable case
        if key in self._by_vec:
            return True
        return key in self._data

    def update(self, *args, **kwargs):  # pylint: disable=arguments-differ
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def keys_by_name(self, name):
        """Return all VarKeys for a given name string."""
        return set(self._by_name.get(name, set()))

    def quantity(self, key):
        "Return a quantity corresponding to self[key]"
        clean_key, val = self.item(key)
        return Quantity(val, clean_key.units or "dimensionless")
