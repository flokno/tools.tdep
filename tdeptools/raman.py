import numpy as np


def _flatten(matrix):
    """Flatten a 3x3 matrix to Voigt form"""
    M = matrix
    d11, d22, d33 = np.diag(M)
    d32, d31, d21 = M[2, 1], M[2, 0], M[1, 0]

    return d11, d22, d33, d32, d31, d21


def intensity(tensor: np.ndarray, vec_in: np.ndarray, vec_out: np.ndarray) -> float:
    """Raman intensity for given incoming and outgoing field

    Args:
        tensor: a 3x3 matrix representing the Raman tensor
        vec_in: 3d vector representing incoming field
        vec_out: 3d vector representing outgoing field

    Returns:
        intensity: (vec_out @ tensor @ vec_in) ** 2
    """
    # normalize vectors
    vec_in = np.asarray(vec_in)
    vec_out = np.asarray(vec_out)

    vec_in = vec_in / np.linalg.norm(vec_in)
    vec_out = vec_out / np.linalg.norm(vec_out)

    intensity = (vec_out @ tensor @ vec_in) ** 2

    return intensity


def intensity_parallel(tensor: np.ndarray, vec: np.ndarray):
    """Raman intensity for E_in = E_out = vec"""
    return intensity(tensor=tensor, vec_in=vec, vec_out=vec)


def intensity_isotropic(dielectric_matrix: np.ndarray):
    """Raman intensity, isotropically averaged


    [1] D. Porezag and M. R. Pederson, Phys Rev B 54, 7830 (1996).
    [2] J. M. Skelton et al., Phys Chem Chem Phys 19, 12452 (2017).
    """
    d11, d22, d33, d32, d31, d21 = _flatten(dielectric_matrix)

    intensity = 45 / 9 * np.sum([d11, d22, d33]) ** 2
    intensity += 7 / 2 * ((d11 - d22) ** 2 + (d11 - d33) ** 2 + (d22 - d33) ** 2)
    intensity += 7 / 2 * 6 * (d32**2 + d31**2 + d21**2)

    return intensity


def po_average(I_abq, direction1, direction2, nangles: int = 361):
    """PO average the Raman tensor for each mode/frequency"""

    angles = np.linspace(0, 360, nangles)

    I_qp_para = np.zeros([len(I_abq), nangles])
    I_qp_perp = np.zeros([len(I_abq), nangles])

    for iq, Iab in enumerate(I_abq):
        for jj, angle in enumerate(angles / 180 * np.pi):
            vec1 = np.cos(angle) * direction1 + np.sin(angle) * direction2
            angle += np.pi / 2
            vec2 = np.cos(angle) * direction1 + np.sin(angle) * direction2
            # parallel: vec1=vec2
            # perp: angle(vec2) = angle(vec1) + pi/2
            I_qp_para[iq, jj] = intensity(Iab, vec_in=vec1, vec_out=vec1)
            I_qp_perp[iq, jj] = intensity(Iab, vec_in=vec1, vec_out=vec2)

    return I_qp_para, I_qp_perp, angles
