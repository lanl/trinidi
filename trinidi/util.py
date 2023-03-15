"""Some util functions."""

import numpy as np

MASS_OF_NEUTRON = 939.56542052 * 1e6 / (299792458) ** 2  # mass neutron [eV s²/m²]


def fibs(N):
    r"""Generate a list of the first N Fibonacci numbers.

    The Fibonacci sequence is defined as

    .. math::
        f(n+2) = f(n+1) + f(n)\;,

    where :math:`f(1) = f(2) = 0`.
    """
    fib = lambda n: n if n < 2 else fib(n - 1) + fib(n - 2)

    return np.array([fib(i) for i in range(N)])


def time2energy(flight_path_length, time):
    r"""Convert time-of-flight to energy of the neutron

    .. math::
        E = \frac{1}{2} m \left( \frac{L}{t} \right) \; ,

    where :math:`E` is the energy, :math:`m` is the mass, :math:`t` is
    the time-of-flight of the neutron, and :math:`L` is the flight path
    lenghth.


    Args:
        flight_path_length: flight path length in :math:`\mathrm{m}`.
        time: Time-of-flight in :math:`\mathrm{μs}`

    Returns:
        Energy of the neutron in :math:`\mathrm{eV}`.
    """
    m = MASS_OF_NEUTRON  # [eV s²/m²]
    L = flight_path_length  # meters
    t = time / 1e6  # seconds
    return m / 2 * (L / t) ** 2  # eV


def energy2time(flight_path_length, energy):
    r"""Convert energy of the neutron to time-of-flight

    .. math::
        E = \frac{1}{2} m \left( \frac{L}{t} \right) \; ,

    where :math:`E` is the energy, :math:`m` is the mass, :math:`t` is
    the time-of-flight of the neutron, and :math:`L` is the flight path
    lenghth.

    Args:
        flight_path_length: flight path length in :math:`\mathrm{m}`.
        energy:  Energy of the neutron in :math:`\mathrm{eV}`.

    Returns:
        Time-of-flight in :math:`\mathrm{μs}`.

    """
    L = flight_path_length
    m = MASS_OF_NEUTRON  # [eV s²/m²]
    E = energy  # eV
    t = L * np.sqrt(m / 2 / E)  # seconds
    return t * 1e6  # micro seconds
