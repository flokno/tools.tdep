"""Dictionary with physical dimensions"""
from tdeptools.keys import keys

time, atom, cart = "time", "atom", "cart"
dimensions = {
    keys.cell: (cart, cart),
    keys.positions: (atom, cart),
    keys.positions_cartesian: (atom, cart),
    keys.forces: (atom, cart),
    keys.stress: (cart, cart),
    keys.dielectric_tensor: (cart, cart),
    keys.born_charges: (atom, cart, cart),
}


site = "site"
cart2 = "cart2"
mode = "mode"
mode2 = "mode2"
atom_unique = "atom_unique"
frequency = "frequency"
dimensions_phonon_self_energy = {
    "incident_wavevector": [cart],
    "q_point": [cart],
    "dynamical_matrix_im": (mode, mode2),
    "dynamical_matrix_re": (mode, mode2),
    "eigenvectors_im": (mode, mode2),
    "eigenvectors_re": (mode, mode2),
    "harmonic_frequencies": [mode],
    "frequencies": [frequency],
    "imaginary_isotope_selfenergy": (mode, frequency),
    "imaginary_threephonon_selfenergy": (mode, frequency),
    "lifetime": [mode],
    "peak_fwhm": [mode],
    "peak_mid": [mode],
    "real_fourphonon_selfenergy": (mode, frequency),
    "real_threephonon_selfenergy": (mode, frequency),
    "spectralfunction_per_mode": (mode, frequency),
    "spectralfunction_per_site": (site, frequency),
    "spectralfunction_per_unique_atom": (atom_unique, frequency),
    "grid_dimensions": [cart],
    "atomic_numbers": [atom],
    "fractional_coordinates": (atom, cart),
    "cartesian_coordinates": (atom, cart),
    "latticevectors": (cart, cart2),
    "reciprocal_latticevectors": (cart, cart2),
}
