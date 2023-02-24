"""Function related to physics, e.g., Bose-Einstein distribution"""
import numpy as np

from tdeptools.konstanter import (
    lo_frequency_Hartree_to_THz,
    lo_frequency_THz_to_Hartree,
    lo_kb_Hartree,
)

THz_to_K = lo_frequency_THz_to_Hartree / lo_kb_Hartree


def n_BE(omega: float, temperature: float = 0, quantum: bool = True):
    """Calculates the Bose-Einstein distribution.

    Args:
        omega: A float value representing the frequency of the system. (in THz)
        temperature: A float value representing the temperature of the system (in K)
        quantum: A boolean value indicating whether to use quantum statistics

    Returns:
        n : A float value representing the Bose-Einstein distribution.
    """
    n = 0.0 * omega

    if temperature < 1e-12:
        return n

    x = omega * THz_to_K / temperature

    if np.all(x < 1e-12):
        return n

    mask = (x > 1e-5) & (x < 100)  # n will be _very_ small or large elsewhere

    if quantum:
        n[mask] = 1 / (np.exp(x) - 1)[mask]
    else:
        n[mask] = 1 / x[mask]

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
        A (float): Amplitude of the oscillation.
    """

    n = n_BE(omega=omega, temperature=temperature, quantum=quantum)
    A = 0.0 * np.asarray(omega)

    if quantum:
        n += 0.5

    mask = np.asarray(omega) > 1e-12

    A[mask] = np.sqrt(n / omega * lo_frequency_Hartree_to_THz)[mask]

    return A
