import numpy as np


def to_voigt(array: np.ndarray) -> np.ndarray:
    """Convert [..., 3, 3] array to [..., 6] in Voigt form"""
    new_array = np.zeros((*np.shape(array)[:-2], 6))

    new_array[:, 0] = array[:, 0, 0]
    new_array[:, 1] = array[:, 1, 1]
    new_array[:, 2] = array[:, 2, 2]
    new_array[:, 3] = array[:, 2, 1]
    new_array[:, 4] = array[:, 2, 0]
    new_array[:, 5] = array[:, 1, 0]

    return new_array


def from_voigt(array: np.ndarray) -> np.ndarray:
    """Convert [..., 6] array in Voigt form to [..., 3, 3]"""
    new_array = np.zeros((*np.shape(array)[:-1], 3, 3))

    new_array[:, 0, 0] = array[:, 0]
    new_array[:, 1, 1] = array[:, 1]
    new_array[:, 2, 2] = array[:, 2]
    new_array[:, 2, 1] = array[:, 3]
    new_array[:, 1, 2] = array[:, 3]
    new_array[:, 2, 0] = array[:, 4]
    new_array[:, 0, 2] = array[:, 4]
    new_array[:, 1, 0] = array[:, 5]
    new_array[:, 0, 1] = array[:, 5]

    return new_array


class Fix:
    """Remove small numbers"""
    def __init__(self, decimals: int = 3):
        self.decimals = decimals

    def __call__(self, array: np.ndarray, decimals: int = None) -> np.ndarray:
        if decimals is None:
            decimals = self.decimals

        tmp = np.around(array, decimals=decimals)
        tmp += 1e-2 ** decimals
        return tmp.round(decimals=decimals)
