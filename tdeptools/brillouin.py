import numpy as np
from ase import Atoms


def get_q_points_cart(
    q_point: np.ndarray, cell_reciprocal: np.ndarray, decimals: int = 12
) -> np.ndarray:
    return (q_point @ cell_reciprocal).round(decimals=decimals)


def get_special_points_cart(atoms: Atoms):
    """Return the special points in Cartesian coords."""
    bandpath = atoms.cell.bandpath()
    special_points = bandpath.special_points
    rc = atoms.cell.reciprocal()
    special_points_cart = {
        key: get_q_points_cart(val, rc) for key, val in special_points.items()
    }

    return special_points_cart
