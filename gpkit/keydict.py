"Implements KeyMap and KeySet classes"

from collections import defaultdict
from collections.abc import Hashable

import numpy as np

from .util.small_classes import FixedScalar, Quantity
from .util.small_scripts import is_sweepvar, isnan

DIMLESS_QUANTITY = Quantity(1, "dimensionless")
INT_DTYPE = np.dtype(int)


def clean_value(key, value):
    """Gets the value of variable-less monomials, so that
    `x.sub({x: gpkit.units.m})` and `x.sub({x: gpkit.ureg.m})` are equivalent.

    Also converts any quantities to the key's units, because quantities
    can't/shouldn't be stored as elements of numpy arrays.
    """
    if isinstance(value, FixedScalar):
        value = value.value
    if isinstance(value, Quantity):
        value = value.to(key.units or "dimensionless").magnitude
    return value


class KeyMap:
    """Helper class to provide KeyMapping to interfaces.

    Mapping keys
    ------------
    A KeyMap keeps an internal list of VarKeys as
    canonical keys, and their values can be accessed with any object whose
    `key` attribute matches one of those VarKeys, or with strings matching
    any of the multiple possible string interpretations of each key:

    For example, after creating the KeyDict kd and setting kd[x] = v (where x
    is a Variable or VarKey), v can be accessed with by the following keys:
     - x
     - x.key
     - x.name (a string)
     - "x_modelname" (x's name including modelname)

    Note that if a item is set using a key that does not have a `.key`
    attribute, that key can be set and accessed normally.
    """

    collapse_arrays = False
    keymap = []
    log_gets = False
    vks = varkeys = None

    def __init__(self, *args, **kwargs):
        "Passes through to super().__init__ via the `update()` method"
        self.keymap = defaultdict(set)
        self._unmapped_keys = set()
        self.owned = set()
        self.update(*args, **kwargs)  # pylint: disable=no-member

    def parse_and_index(self, key):
        "Returns key if key had one, and veckey/idx for indexed veckeys."
        try:
            key = key.key
            if self.collapse_arrays and key.idx:
                return key.veckey, key.idx
            return key, None
        except AttributeError:
            if self.vks is None and self.varkeys is None:
                return key, self.update_keymap()
        # looks like we're in a substitutions dictionary
        if self.varkeys is None:
            self.varkeys = KeySet(self.vks)
        if key not in self.varkeys:
            raise KeyError(key)
        newkey, *otherkeys = self.varkeys[key]
        if otherkeys:
            if all(k.veckey == newkey.veckey for k in otherkeys):
                return newkey.veckey, None
            raise ValueError(
                f"{key} refers to multiple keys in this "
                "substitutions KeyDict. Use "
                f"`.variables_byname({key})` to see all of them."
            )
        if self.collapse_arrays and newkey.idx:
            return newkey.veckey, newkey.idx
        return newkey, None

    def __contains__(self, key):
        "In a winding way, figures out if a key is in the KeyDict"
        # pylint: disable=no-member
        try:
            key, idx = self.parse_and_index(key)
        except KeyError:
            return False
        except ValueError:  # multiple keys correspond
            return True
        if not isinstance(key, Hashable):
            return False
        if super().__contains__(key):
            if idx:
                try:
                    val = super().__getitem__(key)[idx]
                    return True if is_sweepvar(val) else not isnan(val).any()
                except TypeError as err:
                    val = super().__getitem__(key)
                    raise TypeError(
                        f"{key} has an idx, but its value in this"
                        " KeyDict is the scalar {val}."
                    ) from err
                except IndexError as err:
                    val = super().__getitem__(key)
                    raise IndexError(
                        f"key {key} with idx {idx} is out of " " bounds for value {val}"
                    ) from err
        return key in self.keymap

    def update_keymap(self):
        "Updates the keymap with the keys in _unmapped_keys"
        copied = set()  # have to copy bc update leaves duplicate sets
        for key in self._unmapped_keys:
            for mapkey in key.keys:
                if mapkey not in copied and mapkey in self.keymap:
                    self.keymap[mapkey] = set(self.keymap[mapkey])
                    copied.add(mapkey)
                self.keymap[mapkey].add(key)
        self._unmapped_keys = set()


class KeySet(KeyMap, set):
    "KeyMaps that don't collapse arrays or store values."

    collapse_arrays = False

    def update(self, keys):
        "Iterates through the dictionary created by args and kwargs"
        for key in keys:
            self.keymap[key].add(key)
        self._unmapped_keys.update(keys)
        super().update(keys)

    def __getitem__(self, key):
        "Gets the keys corresponding to a particular key."
        key, _ = self.parse_and_index(key)
        return self.keymap[key]
