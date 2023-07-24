import numpy as np
from ase import Atoms


def get_special_points_cart(atoms: Atoms):
    """Return the special points in Cartesian coords."""
    bandpath = atoms.cell.bandpath()
    special_points = bandpath.special_points
    rc = atoms.cell.reciprocal().T
    special_points_cart = {key: rc @ val for key, val in special_points.items()}

    return special_points_cart
