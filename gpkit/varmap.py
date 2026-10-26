"Implements the VarMap class"

from collections.abc import MutableMapping

import numpy as np

from .display.naming import DisplayScope
from .units import Quantity, qty
from .util.small_scripts import veclinkedfn


def _nested_lookup(nested_keys, val_dict):
    "Values in a vector's shape, nan where an element has none."
    if isinstance(nested_keys, np.ndarray):
        return [_nested_lookup(row, val_dict) for row in nested_keys]
    return val_dict.get(nested_keys, float("nan"))


def is_veckey(key):
    "return True iff this key corresponds to a VectorVariable"
    return bool(getattr(key, "shape", None)) and not getattr(key, "idx", None)


class VarSet(set):
    """
    A lightweight *set* of :class:`gpkit.VarKey` objects with constant-time
    look-ups by **canonical name** or **vector parent (veckey)**.

    Unlike :class:`~gpkit.varmap.VarMap`, *VarSet stores no values — it is
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

    def __init__(self, keys=()):
        super().__init__()
        self._by_name = {}
        self._by_vec = {}
        self._by_ref = {}
        self.update(keys)

    def add(self, key):
        self._register_key(key)
        super().add(key)

    def discard(self, key):
        if key not in self:
            return
        name = key.name
        self._by_name[name].discard(key)
        if not self._by_name[name]:
            del self._by_name[name]
        self._by_ref.pop(key.ref, None)
        # handle vector element case
        idx = getattr(key, "idx", None)
        if idx is not None:
            veckey = key.veckey
            self._by_vec[veckey][idx] = None
            if not self._by_vec[veckey].any():
                del self._by_vec[veckey]
        super().discard(key)

    def by_name(self, name):
        """Return all VarKeys for a given name string."""
        if not isinstance(name, str):
            return set()
        return self._by_name.get(name, set()).copy()

    def by_vec(self, veckey):
        "Return np.array of keys for a given veckey"
        return np.array(self._by_vec.get(veckey, ()))

    def keys(self, key):
        "set of all keys in self that the input key refers to"
        key = getattr(key, "key", None) or key  # Variable case
        # Only add key to out if it's a VarKey directly in the set, not a str lookup
        out = {key} if not isinstance(key, str) and super().__contains__(key) else set()
        # For full ref strings (e.g. "Motor.x"), resolve via _by_ref
        if isinstance(key, str) and key in self._by_ref:
            out.add(self._by_ref[key])
        for k in self.by_name(key).union((key,) if is_veckey(key) else ()):
            if is_veckey(k):
                out.update(self.by_vec(k).flat)
                out.discard(None)  # present if vec elements missing
            else:
                out.add(k)
        return out

    def resolve(self, key):
        "Return *one* canonical VarKey for (Variable, str, or veckey) key"
        if hasattr(key, "key"):
            return key.key
        if isinstance(key, str) and key in self._by_ref:
            return self._by_ref[key]
        vks = self.by_name(key)
        if not vks:
            raise KeyError(f"unrecognized key {key}")
        if len(vks) == 1:
            (vk,) = vks
            return vk
        raise KeyError(f"{key} refers to multiple keys {vks}")

    def clean(self, mapping):
        "return a *new dict* with all keys in mapping resolved."
        return {self.resolve(k): v for k, v in mapping.items()}

    def update(self, keys):
        for k in keys:
            self.add(k)

    def vector_parents(self):
        """The veckey of every vector with an element in this set.

        Veckeys are not members, so iteration and `len` never yield one: this is
        the only way to enumerate the vector level.
        """
        return set(self._by_vec)

    def vector_parent_keys(self):
        "set(self) but with veckeys and no individual vector element keys"
        ks = set(self)
        for vk, vks in self._by_vec.items():
            ks -= set(vks.flat)
            ks.add(vk)
        return ks

    def to_ir(self):
        "{ref: VarKey.to_ir()} for every member and vector parent, sorted by ref."
        keys = set(self) | self.vector_parents()
        return {vk.ref: vk.to_ir() for vk in sorted(keys, key=lambda v: v.ref)}

    def _register_key(self, key):
        "adds the key to _by_name, _by_ref, and, if applicable, _by_vec"
        name = getattr(key, "name", None)
        if name is None:
            raise TypeError("VarSet keys must be VarKey instances")
        if name not in self._by_name:
            self._by_name[name] = set()
        idx = getattr(key, "idx", None)
        self._by_name[name].add(key if not idx else key.veckey)
        self._by_ref[key.ref] = key
        if idx is not None:
            if key.veckey not in self._by_vec:
                self._by_vec[key.veckey] = np.empty(key.shape, dtype=object)
            self._by_vec[key.veckey][idx] = key

    def __contains__(self, key):
        key = getattr(key, "key", None) or key  # Variable case
        if super().__contains__(key):
            return True
        if isinstance(key, str) and key in self._by_ref:
            return True
        return key in self._by_name or key in self._by_vec


class FrozenVarSet(VarSet):
    """A VarSet that cannot be changed once built.

    For a key set that is declared in one go rather than accumulated, so it is
    safe to share -- including as a class-level default.
    """

    _MSG = "a FrozenVarSet cannot be changed after construction"
    _sealed = False

    def __init__(self, keys=()):
        super().__init__(keys)
        self._sealed = True

    def add(self, key):
        if self._sealed:
            raise TypeError(self._MSG)
        super().add(key)

    def discard(self, key):
        if self._sealed:
            raise TypeError(self._MSG)
        super().discard(key)

    def _frozen(self, *_args, **_kwargs):
        raise TypeError(self._MSG)

    # set's own mutators bypass add/discard, so they would desync the indices
    clear = pop = remove = _frozen
    difference_update = intersection_update = symmetric_difference_update = _frozen
    __ior__ = __iand__ = __isub__ = __ixor__ = _frozen


EMPTY_VARSET = FrozenVarSet()


class VarMap(MutableMapping):
    """A mapping from VarKey to value, with lookup by name string or veckey.

    `register_keys` adds a key without a value, so the varset this resolves
    against may name more keys than the mapping holds. `in`, `len` and iteration
    follow the values.
    """

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._varset = VarSet()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        _, val = self.item(key)
        return val

    def item(self, key):
        """get the (varkey, value) pair associated with a (str or key)

        A veckey gives the whole vector, nan where an element has no value.
        Raises KeyError if no element has one, so a vector is here on the same
        terms as a scalar: when the mapping holds a value for it.
        """
        key = self._varset.resolve(key)
        try:
            return (key, self._data[key])  # single varkey case
        except KeyError:
            elements = self._varset.by_vec(key)
            if any(vk in self._data for vk in elements.flat if vk is not None):
                return (key, _nested_lookup(elements, self._data))
            raise

    @property
    def varset(self):
        "The VarSet this map's keys resolve through."
        return self._varset

    def __setitem__(self, key, value):
        key = self._varset.resolve(key)
        if isinstance(value, Quantity):
            # stored in the key's units, which is what quantity() reads back
            value = value.to(key.units or "dimensionless").magnitude
        if is_veckey(key):
            # Setting a whole vector writes its elements, so their keys have to
            # be registered first -- they carry idx and units this cannot invent.
            vks = {vk for vk in self._varset.by_vec(key).flat if vk is not None}
            size = np.prod(key.shape)
            if len(vks) != size:
                raise ValueError(
                    f"cannot set {key} by veckey: {len(vks)} of {size} element "
                    f"keys are registered here"
                )
            if not np.shape(value):
                self.update({vk: value for vk in vks})
                return
            if np.shape(value) == key.shape:
                for vk in vks:
                    self[vk] = np.array(value)[vk.idx]
                return
            raise ValueError(
                f"cannot set {key} of shape {key.shape} from a value of shape "
                f"{np.shape(value)}"
            )
        self._varset.add(key)
        self._data[key] = value

    def register_keys(self, keys):
        "register a set of keys to this mapping, without values yet"
        self._varset.update(keys)

    def __delitem__(self, key):
        key = self._varset.resolve(key)
        if is_veckey(key):
            raise NotImplementedError
        del self._data[key]
        self._varset.discard(key)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        key = getattr(key, "key", key)  # handle Variable case
        if not isinstance(key, str) and not is_veckey(key):
            return key in self._data  # fast path: scalar VarKey
        # string or veckey: resolve via VarSet, then check _data
        try:
            self.item(key)
            return True
        except KeyError:
            return False

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def quantity(self, key):
        "Return a quantity corresponding to self[key]"
        clean_key, val = self.item(key)
        return Quantity(val, clean_key.units or "dimensionless")


def _elementwise_linked(veckey, fn):
    "One linked function per element of veckey, each reading its own index."
    out = np.empty(veckey.shape, dtype="object")
    for idx in np.ndindex(out.shape):
        out[idx] = veclinkedfn(fn, idx)
    return out


class Substitutions(VarMap):
    """The values prescribed for a problem's parameters.

    Handles two value types a plain VarMap does not: a constant Monomial, such
    as `3*units("day")`, stored as its coefficient; and a linked function, over
    a vector stored as one function per element.
    """

    def __setitem__(self, key, value):
        key = self._varset.resolve(key)
        if hasattr(value, "hmap") and not any(value.hmap.keys()):
            # A Monomial carrying variables passes through, so solve-time
            # validation is what reports it.
            (value,) = value.hmap.to(key.units or qty("dimensionless")).values()
        elif callable(value) and is_veckey(key):
            value = _elementwise_linked(key, value)
        super().__setitem__(key, value)


def display_names(varkeys, excluded=()):
    """{VarKey: its shortest unambiguous name among varkeys}.

    excluded is the format flags ("units", "idx", ...) the names are spelled
    with.
    """
    varkeys = list(varkeys)
    scope = DisplayScope(shown=varkeys, excluded=excluded)
    return {vk: vk.str_without(scope) for vk in varkeys}
