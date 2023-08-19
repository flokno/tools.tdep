import numpy as np
from .kramers_kronig import get_oscillator_complex



def get_mode_resolved_BEC(
    born_charges: np.ndarray, eigenvectors: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Get mode-resolved Born effective charges

    Z_s = \sum_i Z_i / m_i^1/2 |is>
    """
    n_atoms = len(masses)
    n_bands = n_atoms * 3
    Z_mode = np.zeros([n_bands, 3])
    # oscillator_strength = np.zeros([n_bands, 3, 3])

    for ss, ev in enumerate(eigenvectors):
        for ii in range(n_atoms):
            ev_i = ev[3 * ii : 3 * ii + 3]
            m_i = masses[ii]
            u_i = ev_i / (m_i ** 0.5)  # displacement for mode s, atom i
            # echo(f"{ss}, {ii}, u_is = {u_i}")
            Z_i = born_charges[ii]

            Z_s = Z_i @ u_i

            Z_mode[ss] += Z_s
        # oscillator_strength[ss] = np.outer(Z_mode[ss], Z_mode[ss])

    return Z_mode
