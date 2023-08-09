import numpy as np
from tdeptools.kramers_kronig import (
    get_oscillator_complex,
    get_kk_imag_from_real,
    get_kk_real_from_imag,
    get_kk_real_from_imag_uneven,
)


tol = 1e-4  # the error we tolerate
etas = 0.1 ** np.arange(3, 14)
x0 = 3
gamma = 1
xx, yy = get_oscillator_complex(x0=x0, gamma=gamma, xmax=50, nx=4000, tol=1e-3)


def test_real_from_imag(tol=tol):
    real_true = yy.real

    for eta in etas:
        real_kk = get_kk_real_from_imag(xx, yy.imag, eta=eta)

        mse = np.sum((real_true - real_kk) ** 2) / np.sum(real_true ** 2)

        assert mse < tol, (eta, mse)


def test_imag_from_real(tol=tol):
    imag_true = yy.imag

    for eta in etas:
        imag_kk = get_kk_imag_from_real(xx, yy.real, eta=eta)

        mse = np.sum((imag_true - imag_kk) ** 2) / np.sum(imag_true ** 2)

        assert mse < tol, (eta, mse)


def test_real_from_imag_uneven(tol=tol):
    _x = np.linspace(0, 50, 2000)
    _y = 1 / (_x ** 2 + 2.0j * gamma * _x - x0 ** 2)

    real_true = _y.real

    for eta in etas:
        real_kk = get_kk_real_from_imag_uneven(_x, _y.imag, eta=eta)

        mse = np.sum((real_true - real_kk) ** 2) / np.sum(real_true ** 2)

        assert mse < tol, (eta, mse)


if __name__ == "__main__":
    test_real_from_imag()
    test_imag_from_real()
    test_real_from_imag_uneven()
