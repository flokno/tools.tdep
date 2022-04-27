import sys

import numpy as np

# imaginary i
lo_imag = 1.0j
# pi
lo_pi = 3.141592653589793
# 2*pi
lo_twopi = 6.283185307179586

# Ok it's a little weird that I have redefined my constants lately, but they
# where not completely consistent. Now I went to NIST and have made sure it actually
# make sense, and is consistent with each other and so on. These are the
# recommended values from NIST as of November 2017

# eV to Joule
lo_eV_to_Joule = 1.6021766208e-19
lo_Joule_to_eV = 1.0 / lo_eV_to_Joule
# eV to Hartree
lo_Hartree_to_eV = 27.21138602
lo_eV_to_Hartree = 1.0 / lo_Hartree_to_eV
# Hartree to Joule
lo_Hartree_to_Joule = lo_Hartree_to_eV * lo_eV_to_Joule
lo_Joule_to_Hartree = 1.0 / lo_Hartree_to_Joule
# Bohr radius to m
lo_bohr_to_m = 0.52917721067e-10
lo_m_to_bohr = 1.0 / lo_bohr_to_m
# Bohr radius to angstrom
lo_bohr_to_A = 0.52917721067
lo_A_to_bohr = 1.0 / lo_bohr_to_A
# Planck constant
lo_hbar_eV = 6.582119514e-16
lo_hbar_Joule = lo_hbar_eV * lo_eV_to_Joule
lo_hbar_Hartree = lo_hbar_eV * lo_eV_to_Hartree
# Boltzmann constant
lo_kb_eV = 8.6173303e-5
lo_kb_Hartree = lo_kb_eV * lo_eV_to_Hartree
lo_kb_Joule = lo_kb_eV * lo_eV_to_Joule
# Speed of light in m/s
lo_c_ms = 299792458
# Atomic and electron mass unit
lo_amu_to_kg = 1.660539040e-27
lo_emu_to_kg = 9.10938356e-31
lo_amu_to_emu = lo_amu_to_kg / lo_emu_to_kg
lo_emu_to_amu = 1.0 / lo_amu_to_emu

# convert forces in eV/A to atomic units, Hartree/bohr
lo_force_eVA_to_HartreeBohr = lo_eV_to_Hartree / lo_A_to_bohr
lo_force_HartreeBohr_to_eVA = 1.0 / lo_force_eVA_to_HartreeBohr
# convert volumes
lo_volume_A_to_bohr = lo_A_to_bohr ** 3
lo_volume_bohr_to_A = 1.0 / lo_volume_A_to_bohr
# convert pressure in GPa to Hartree/bohr^3
lo_pressure_GPa_to_HartreeBohr = 1e9 * (lo_bohr_to_m ** 3) / lo_Hartree_to_Joule
lo_pressure_HartreeBohr_to_GPa = 1.0 / lo_pressure_GPa_to_HartreeBohr
# convert pressure in eV/A to atomic units, Hartree/bohr
lo_pressure_eVA_to_HartreeBohr = lo_eV_to_Hartree / (lo_A_to_bohr) ** 3
lo_pressure_HartreeBohr_to_eVA = 1.0 / lo_pressure_eVA_to_HartreeBohr
lo_pressure_eVA_to_GPa = lo_pressure_eVA_to_HartreeBohr * lo_pressure_HartreeBohr_to_GPa
# convert time to atomic units
lo_time_au_to_s = lo_hbar_Joule / lo_Hartree_to_Joule
lo_time_s_to_au = 1.0 / lo_time_au_to_s
lo_time_au_to_fs = 1e15 * lo_time_au_to_s
lo_time_fs_to_au = 1.0 / lo_time_au_to_fs
# convert velocities
lo_velocity_au_to_ms = lo_bohr_to_m / lo_time_au_to_s
lo_velocity_ms_to_au = 1.0 / lo_velocity_au_to_ms
lo_velocity_au_to_Afs = lo_velocity_au_to_ms * 1e-5
lo_velocity_Afs_to_au = 1.0 / lo_velocity_au_to_Afs
# convert forceconstants
lo_forceconstant_1st_eVA_to_HartreeBohr = lo_eV_to_Hartree / (lo_A_to_bohr)
lo_forceconstant_2nd_eVA_to_HartreeBohr = lo_eV_to_Hartree / (lo_A_to_bohr ** 2)
lo_forceconstant_3rd_eVA_to_HartreeBohr = lo_eV_to_Hartree / (lo_A_to_bohr ** 3)
lo_forceconstant_4th_eVA_to_HartreeBohr = lo_eV_to_Hartree / (lo_A_to_bohr ** 4)
lo_forceconstant_1st_HartreeBohr_to_eVA = 1.0 / lo_forceconstant_1st_eVA_to_HartreeBohr
lo_forceconstant_2nd_HartreeBohr_to_eVA = 1.0 / lo_forceconstant_2nd_eVA_to_HartreeBohr
lo_forceconstant_3rd_HartreeBohr_to_eVA = 1.0 / lo_forceconstant_3rd_eVA_to_HartreeBohr
lo_forceconstant_4th_HartreeBohr_to_eVA = 1.0 / lo_forceconstant_4th_eVA_to_HartreeBohr
# convert phonon frequencies
lo_frequency_Hartree_to_Hz = 1.0 / lo_hbar_Hartree
lo_frequency_Hartree_to_THz = 1e-12 / lo_twopi / lo_hbar_Hartree
lo_frequency_Hartree_to_meV = 1000.0 * lo_Hartree_to_eV
lo_frequency_Hartree_to_icm = 1.0 / lo_hbar_Hartree / lo_twopi / lo_c_ms / 100.0
lo_frequency_THz_to_Hartree = 1.0 / lo_frequency_Hartree_to_THz
lo_frequency_meV_to_Hartree = 1.0 / lo_frequency_Hartree_to_meV
lo_frequency_Hz_to_eV = lo_Hartree_to_eV / lo_frequency_Hartree_to_Hz
# convert group velocities
lo_groupvel_Hartreebohr_to_ms = lo_bohr_to_m / lo_time_au_to_s
lo_groupvel_ms_to_Hartreebohr = 1.0 / lo_groupvel_Hartreebohr_to_ms
# thermal conductivity, atomic units to SI(W/mK)
lo_kappa_au_to_SI = lo_Hartree_to_Joule / lo_time_au_to_s / lo_bohr_to_m
lo_kappa_SI_to_au = 1.0 / lo_kappa_au_to_SI

# Some default tolerances
# Tolerance for realspace distances to be 0
lo_tol = 1e-5
# Tolerance for realspace squared distances to be 0
lo_sqtol = lo_tol ** 2
# Tolerance for reciprocal distances to be 0
lo_rectol = 1e-6
# Tolerance for reciprocal squared distances
lo_sqrectol = lo_rectol ** 2
# Tolerance for angles in degrees to be 0
lo_degreetol = 1e-4
# Tolerance for angles in radians to be 0
lo_radiantol = lo_degreetol * 180.0 / lo_pi
# Tolerance for phonon frequencies to be 0.
lo_freqtol = lo_tol * 1e-4
# Tolerance for phonon group velocities to be zero
lo_phonongroupveltol = lo_tol * 1e-5
# Tolerance for temperatures to be 0, in K
lo_temperaturetol = 1e-3
# large number
lo_huge = sys.float_info.max
# small number
lo_tiny = sys.float_info.min
# large integer
lo_hugeint = sys.maxsize

# Variable that holds exit status, for catching exceptions
lo_status = 0
# A strange vector I use to make some things more deterministic. Don't ask.
lo_degenvector = np.array([1.0, 1.33, 1.45624623]) * 1e-8

# The default gnuplot terminal, decided by precompiler flags.
# if GPwxt
# default gnuplot terminal on linux
lo_gnuplotterminal = "wxt"
# elif GPaqua
# default gnuplot terminal on osx
lo_gnuplotterminal = "aqua"
# elif GPqt
# default gnuplot terminal on osx
lo_gnuplotterminal = "qt"
# else
# fallback gnuplot terminal
lo_gnuplotterminal = "qt"
# endif

# Some exit codes for when things go horribly wrong.

# Something has an unexpected dimension
lo_exitcode_baddim = 1
# BLAS or LAPACK fails
lo_exitcode_blaslapack = 2
# Unphysical value, i.e. negative temperature or something like that.
lo_exitcode_physical = 3
# Bad symmetry
lo_exitcode_symmetry = 4
# Something off with the arguments sent to the routine
lo_exitcode_param = 5
# IO error
lo_exitcode_io = 6
# MPI problem
lo_exitcode_mpi = 7

# Some constant strings to use in the documentation
lo_author = "Olle Hellman"
lo_version = "1.2"
lo_licence = "MIT"
