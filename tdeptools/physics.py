"""Function related to physics, e.g., Bose-Einstein distribution"""

import numpy as np

from tdeptools.konstanter import (
    lo_hbar_Joule,
    lo_amu_to_kg,
    lo_frequency_Hartree_to_THz,
    lo_frequency_THz_to_Hartree,
    lo_kb_Hartree,
)

amplitude_to_angstrom = 1e10 / np.sqrt(lo_amu_to_kg)
THz_to_K = lo_frequency_THz_to_Hartree / lo_kb_Hartree


def n_BE(
    omega: float,
    temperature: float = 0,
    quantum: bool = True,
    negative: bool = True,
    zero_frequency_as_inf: bool = False,
    eps: float = 2 * np.finfo(float).eps,
) -> np.ndarray:
    """Calculates the Bose-Einstein distribution.

    For negative frequencies: n(-w) = -n(w) - 1

    Args:
        omega: A float value representing the frequency of the system. (in THz)
        temperature: A float value representing the temperature of the system (in K)
        quantum: A boolean value indicating whether to use quantum statistics
        negative: Calculate for negative frequencies
        zero_frequency_as_inf: Return infinity for zero frequency
        eps: A small value to avoid division by zero.

    Returns:
        n : A float value representing the Bose-Einstein distribution.
    """
    # initialize n
    n = 0.0 * np.asarray(omega)

    if negative:
        n[omega < eps] = -1.0

    # deal with temperature == 0
    if temperature < eps:
        return n
    else:
        x = omega * THz_to_K / temperature

    mask = (abs(x) > eps) & (abs(x) < 100)  # n will be _very_ small or large elsewhere

    if quantum:
        n[mask] = 1 / (np.exp(x) - 1)[mask]
    else:
        n[mask] = 1 / x[mask]

    if not negative:
        n[omega < eps] = 0.0

    # deal with omega == 0
    mask = abs(omega) < eps
    if zero_frequency_as_inf:
        n[mask] = np.nan
    else:
        n[mask] = 0.0

    return n


def freq2amplitude(
    omega: float, temperature: float, quantum: bool = True
) -> np.ndarray:
    """Convert frequency to amplitude.

    Args:
        omega (float): Frequency of the oscillation (in THz).
        temperature (float): Temperature of the system (in Kelvin).
        quantum (bool, optional): Whether or not to use quantum statistics.

    Returns:
        A (float): Amplitude of the oscillation in AMU^1/2 * AA.
    """

    n = n_BE(omega=omega, temperature=temperature, quantum=quantum)
    A = 0.0 * np.asarray(omega)

    if quantum:
        n += 0.5

    mask = np.asarray(omega) > 1e-12

    # old:
    # A[mask] = np.sqrt(n / omega * lo_frequency_Hartree_to_THz)[mask]

    # correct:
    # A = \sqrt( \hbar (n + 1/2) / \omega )
    # [hbar] = J s = kg m^2 s^-1
    # [omega] = THz = 1e12 s^-1
    # [hbar / omega] = kg m^2 / 1e12
    # kg^1/2 m =  (AMU * amu_to_kg)^1/2 * 1e10 AA
    A[mask] = amplitude_to_angstrom * np.sqrt(lo_hbar_Joule * n / omega * 1e-12)[mask]

    return A
