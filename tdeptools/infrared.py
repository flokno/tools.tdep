import numpy as np


def get_oscillator_complex(x0: float, gamma: float, xmax: float = 50, nx: int = 10000):
    """Return damped oscillator

    y(x) = 1 / (x^2 + 2i*gamma*x - x0^2)

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
    if diff > 1e-9:
        print(f"** Oscillator has not decayed to zero on the boundaries: {diff}")

    return xx, yy
