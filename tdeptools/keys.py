"""Dictionary keys naming"""
from collections import namedtuple

# keys
_keys = [
    "positions",
    "forces",
    "energy_total",
    "energy_kinetic",
    "energy_potential",
    "temperature",
    "stress",
    "pressure",
    "dielectric_tensor",
    "born_charges",
]
_dct = {key: key for key in _keys}
keys = namedtuple("keys", _dct.keys())(**_dct)
