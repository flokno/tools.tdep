import numpy as np
from scipy import signal as sl
from scipy import interpolate as si

from .physics import n_BE


def get_bose_weighted_DOS(x, y, temperature, zero_pad=True):
    """Get zero-padded DOS exteneded to negative frequencies incl. Bose-weighted

    Args:
        x: frequency
        y: DOS
        temperature: temperature in K
        zero_pad: whether to zero-pad the DOS

    Returns:
        x_full: extended frequency
        y_full: extended DOS
        y_full_weighted: extended Bose-weighted DOS
    """
    _x = x.copy()
    _y = y.copy()

    if zero_pad:
        _x = np.concatenate([x, x[1:] + x.max()])
        _y = np.concatenate([y, np.zeros_like(x[1:])])

    _n = n_BE(_x, temperature=temperature)

    # extend from [0, w_max] to [-w_max, w_max]
    x_full = np.concatenate([-_x[::-1], _x[1:]])
    y_full = np.concatenate([_y[::-1], _y[1:]])

    # with n, n+1 (Antistokes/Stokes)
    n_full = np.concatenate([_n[::-1], _n[1:] + 1])
    y_full_weighted = y_full * n_full

    return x_full, y_full, y_full_weighted


def get_weighted_2w_DOS(x, y, temperature, xmin=0, zero_pad=True):
    """Get Bose-weighted 2\omega-DOS exteneded to negative frequencies"""
    _x, _y, _f = get_bose_weighted_DOS(x, y, temperature, zero_pad=zero_pad)

    # need to weight with (n+1)**2
    _n = n_BE(_x, temperature=temperature)

    # interpolate to get 2\omega
    f = si.interp1d(2 * _x, _y, kind="cubic", fill_value=0)
    _y = f(_x)

    _f = (2 * _n + 1) * (1 + n_BE(2 * _x, temperature=temperature)) * _y
    _f = (_n + 1) ** 2 * _y

    _f[abs(_x) < xmin] = 0

    return _x, _y, _f


def get_convoluted_DOS(x, y, temperature, zero_pad=True):
    """Get convolution of weighted DOS -> 2nd order Raman intensity"""
    x_full, y_full, _ = get_bose_weighted_DOS(x, y, temperature, zero_pad=zero_pad)
    NN = len(x_full)
    conv = sl.convolve(y_full, y_full)[NN // 2 : 3 * NN // 2]  # [len(f_full) - 1 :]

    return x_full, conv / conv.size


def get_convoluted_weighted_DOS(x, y, temperature, zero_pad=True):
    """Get convolution of weighted DOS -> 2nd order Raman intensity"""
    x_full, _, f_full = get_bose_weighted_DOS(x, y, temperature, zero_pad=zero_pad)
    assert np.isfinite(f_full).all()  # otherwise the convolution will be NaN
    conv = sl.convolve(f_full, f_full, mode="same")

    return x_full, conv / conv.size
