import numpy as np
from warnings import warn


def get_oscillator_complex(
    x0: float, gamma: float, xmax: float = 50, nx: int = 10000, tol: float = 1e-9
):
    """Return complex damped oscillator

    y(x) = 1 / (x^2 + 2i * gamma * x - x0^2)

    Args:
        x0: eigenfrequency (undamped)
        gamma: dampening
        xmax: max. frequency (domain will be [-xmax, xmax])
        nx: frequency points

    returns:
        (xx, yy):
            xx: the frequency axis
            yy: the (complex!) oscillator response
    """

    xx = np.linspace(-xmax, xmax, num=nx, endpoint=True)

    yy = 1 / (xx ** 2 + 2.0j * gamma * xx - x0 ** 2)

    diff = max(abs(yy[0]), abs(yy[-1]))
    if diff > tol:
        warn(f"** Oscillator has not decayed to zero on the boundaries: {diff}")

    return xx, yy


def get_kk_imag_from_real_even(xs: np.ndarray, ys: np.ndarray, eta: float = 1e-5):
    """Kramers-Kronig transform real to imaginary part from transfrom on [-inf, inf]"""
    assert len(xs) == len(ys)

    dx = xs[1] - xs[2]
    chi1 = ys.copy()
    chi2 = np.zeros_like(ys)

    for ii, x in enumerate(xs):

        dw = xs - x + eta * 1.0j

        chi2[ii] = (chi1 / dw).real.sum()

    return chi2 / np.pi * dx


def get_kk_real_from_imag_even(xs: np.ndarray, ys: np.ndarray, eta: float = 1e-8):
    """Kramers-Kronig transform imaginary to real part from transform on [-inf, inf]"""
    return -get_kk_imag_from_real_even(xs=xs, ys=ys, eta=eta)


# currently used transforms
get_kk_imag_from_real = get_kk_imag_from_real_even
get_kk_real_from_imag = get_kk_real_from_imag_even


# More uneven transforms:
def get_kk_imag_from_real_uneven(xs: np.ndarray, ys: np.ndarray, eta: float = 1e-5):
    """Kramers-Kronig transform real to imaginary part from transfrom on [0, inf]"""
    assert len(xs) == len(ys)

    mask = xs >= 0

    dx = (xs[1] - xs[0]).real
    xs_square_eta = (xs[mask] + eta * 1.0j) ** 2

    chi1 = ys.copy()
    chi2 = np.zeros_like(ys)
    chi1_mask = chi1[mask]

    for ii, x in enumerate(xs):

        dw = xs_square_eta - x ** 2

        chi2[ii] = (x * chi1_mask / dw).real.sum()

    return -2 * chi2.real / np.pi * dx


def get_kk_real_from_imag_uneven(xs: np.ndarray, ys: np.ndarray, eta: float = 1e-5):
    """Kramers-Kronig transform imaginary to real part from transform on [0, inf]"""
    assert len(xs) == len(ys)

    mask = xs >= 0

    dx = (xs[1] - xs[0]).real
    xs_mask = xs[mask]
    xs_square_eta = (xs_mask + eta * 1.0j) ** 2

    chi1 = ys.copy()
    chi2 = np.zeros_like(ys)
    chi1_mask = chi1[mask]

    for ii, x in enumerate(xs):

        dw = xs_square_eta - x ** 2

        chi2[ii] = (xs_mask * chi1_mask / dw).real.sum()

    return 2 * chi2 / np.pi * dx
