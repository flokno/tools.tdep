import numpy as np
import pytest

from tdeptools.physics import n_BE


@pytest.mark.parametrize("N_w", [3, 4, 10000, 10001])
@pytest.mark.parametrize("w_max", [10, 1e2, 1e3])
@pytest.mark.parametrize("temperature", [0, 100, 1000])
def test_n_BE_negative(
    N_w: int,
    w_max: float,
    temperature: float,
    eps: float = np.finfo(float).eps,
):
    frequencies = np.linspace(-w_max, w_max, N_w)
    frequencies_negative = -frequencies

    occupations = n_BE(frequencies, temperature=temperature)
    occupations_negative = n_BE(frequencies_negative, temperature=temperature)

    # only compoare non-zero frequencies
    mask = abs(frequencies) > eps
    _x = occupations[mask]
    _y = -occupations_negative[mask] - 1

    np.testing.assert_allclose(_y, _x, atol=1e-9)

    # check the 0 frequency
    mask = abs(frequencies) < eps
    _x = occupations[mask]
    _y = occupations_negative[mask]
    np.testing.assert_almost_equal(_y, _x)


if __name__ == "__main__":
    pytest.main([__file__])
