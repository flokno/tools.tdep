"""Namedtuple with all valid key names"""
from collections import namedtuple

# keys
_keys = [
    "cell",
    "volume",
    "natoms",
    "positions",
    "positions_cartesian",
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
