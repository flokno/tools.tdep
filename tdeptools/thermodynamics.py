import numpy as np

from tdeptools.helpers import to_voigt

from . import konstanter as k


def get_free_energy(energies, temperatures):
    """Compute harmonic free energy for temperatures

    Args:
        energies[Nq, Ns]: mode energies in eV
        temperatures[NT]: temperatures in K
    Returns:
        free energy in eV

    Formula:
        NT: number of temperatures
        Nq: number of q-points
        Ns: number of branches (3N)
        x_q = e_q / kT
        F_T = sum_q ( 0.5 * e_q + kT * ln(1 - exp(-x_q)))
    """
    Nq = energies.shape[0]
    energies = np.array(energies).flatten()
    temperatures = np.array(temperatures)
    temperatures[temperatures < k.lo_tiny] = k.lo_tiny  # handle zeros
    kT = k.lo_kb_eV * temperatures
    b_T = 1 / kT
    b_T[b_T > k.lo_huge] = 0  # k.lo_huge  # handle zeros
    x_Tq = energies[None, :] * b_T[:, None]
    # ln_arg = 1 - exp(...)
    ln_arg = 1 - np.exp(-x_Tq)
    ln_arg[ln_arg < k.lo_tiny] = 1
    ln = np.log(ln_arg)

    F_T = (0.5 * energies[None, :] + kT[:, None] * ln).sum(axis=1)
    F_T /= Nq

    return F_T


def get_heat_capacity(energies, temperatures):
    """Compute heat capacity for temperatures

    Args:
        energies[Nq]: mode energies in eV
        temperatures[NT]: temperatures in K
    Returns:
        heat capacity in eV/K

    Formula:
        NT: number of temperatures
        Nq: number of q-points
        Ns: number of branches (3N)

        x_q = e_q / k_B / T
        n_q = 1 / (exp(x_q) - 1)
        c_V = k_B * 1/Nq * sum_q x_q^2 . exp(x_q) * n_q**2
    """
    Nq = energies.shape[0]

    c_qsT = get_mode_heat_capacity(energies, temperatures)

    c_T = c_qsT.sum(axis=(0, 1)) / Nq

    return c_T


def get_mode_heat_capacity(energies, temperatures):
    """Compute mode heat capacity for temperatures

    Args:
        energies[Nq, Ns]: mode energies in eV
        temperatures[NT]: temperatures in K
    Returns:
        heat capacity[Nq, Ns, NT] in eV/K

    Formula:
        NT: number of temperatures
        Nq: number of q-points
        Ns: number of branches (3N)

        x_q = e_q / kT
        n_q = 1 / (exp(x_q) - 1)
        cs_V = k_B * x_q^2 . exp(x_q) * n_q**2
    """
    energies = np.array(energies)
    temperatures = np.array(temperatures)
    temperatures[temperatures < k.lo_tiny] = k.lo_huge  # handle zeros
    x_Tq = energies[:, :, None] / temperatures[None, None, :] / k.lo_kb_eV
    n_Tq = np.exp(x_Tq) - 1  # compute inverse first
    n_Tq[n_Tq < k.lo_tiny] = k.lo_huge  # handle zeros
    n_Tq = 1 / n_Tq
    # mean = 1/3N . sum_q
    c_T = (x_Tq ** 2 * np.exp(x_Tq) * n_Tq ** 2) * k.lo_kb_eV

    return c_T


def get_pressure_volume(energies, grueneisens, temperatures) -> np.ndarray:
    """Compute quasi-harmonic pressure times volume: p = - dF/dV

    Args:
        energies[Nq, Ns]: mode energies in eV
        grueneisens[Nq, Ns]: mode Grueneiesen parameters
        temperatures[NT]: temperatures in K
    Returns:
        pressure in GPa * AA**3

    Formula:
        NT: number of temperatures
        Nq: number of q-points
        Ns: number of branches (3N)
        gamma_q: Grueneisen parameter
        n_q = 1 / (exp(e_q / kB / T) - 1)
        pV = 1/Nq * sum_q gamma_q * e_q (1/2 + n_q)
    """
    Nq = energies.shape[0]
    energies = np.array(energies)
    grueneisens = np.array(grueneisens)
    temperatures = np.array(temperatures)

    temperatures[temperatures < k.lo_tiny] = k.lo_huge  # handle zeros
    x_Tq = energies[None, :] / temperatures[:, None, None] / k.lo_kb_eV
    n_Tq = np.exp(x_Tq) - 1  # compute inverse first
    n_Tq[n_Tq < k.lo_tiny] = k.lo_huge  # handle zeros
    n_Tq = 1 / n_Tq

    pV_T = (grueneisens[None, :] * energies[None, :] * (0.5 + n_Tq)).sum(axis=(1, 2))
    pV_T /= Nq
    pV_T *= k.lo_pressure_eVA_to_GPa

    return pV_T


def get_stress_volume(
    energies, grueneisens, temperatures, voigt: bool = True
) -> np.ndarray:
    """Compute quasi-harmonic stress times volume: s = dF/deps

    Args:
        energies[Nq, Ns]: mode energies in eV
        grueneisens[Nq, Ns, 3, 3]: mode Grueneiesen parameters
        temperatures[NT]: temperatures in K
        voigt: return in Voigt form
    Returns:
        pressure in GPa * AA**3

    Formula:
        NT: number of temperatures
        Nq: number of q-points
        Ns: number of branches (3N)
        gamma_q_ab: Grueneisen tensor
        n_q = 1 / (exp(e_q / kB / T) - 1)
        sV_ab = 1/Nq * sum_q gamma_q_ab * e_q (1/2 + n_q)
    """
    Nq = energies.shape[0]
    energies = np.array(energies)
    grueneisens = np.array(grueneisens)
    temperatures = np.array(temperatures)

    _ = None

    temperatures[temperatures < k.lo_tiny] = k.lo_huge  # handle zeros
    # cast to [T, q, s, a, b]
    x_Tq = energies[_, :, :, _, _] / temperatures[:, _, _, _, _] / k.lo_kb_eV
    n_Tq = np.exp(x_Tq) - 1  # compute inverse first
    n_Tq[n_Tq < k.lo_tiny] = k.lo_huge  # handle zeros
    n_Tq = 1 / n_Tq

    sV_T = -(grueneisens[_, :] * energies[_, :, :, _, _] * (0.5 + n_Tq)).sum(axis=(1, 2))
    sV_T /= Nq
    sV_T *= k.lo_pressure_eVA_to_GPa

    if voigt:
        return to_voigt(sV_T)

    return sV_T
