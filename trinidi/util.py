"""Some util functions."""

import numpy as np

MASS_OF_NEUTRON = 939.56542052 * 1e6 / (299792458) ** 2  # [eV s²/m²]

TOF_LABEL = "Time-of-flight [μs]"
ENERGY_LABEL = "Energy [eV]"


def time2energy(time, flight_path_length):
    r"""Convert time-of-flight to energy of the neutron.

    .. math::
        E = \frac{1}{2} m \left( \frac{L}{t} \right) \; ,

    where :math:`E` is the energy, :math:`m` is the mass, :math:`t` is
    the time-of-flight of the neutron, and :math:`L` is the flight path
    lenghth.

    Args:
        time: Time-of-flight in :math:`\mathrm{μs}`.
        flight_path_length: flight path length in :math:`\mathrm{m}`.

    Returns:
        Energy of the neutron in :math:`\mathrm{eV}`.
    """
    m = MASS_OF_NEUTRON  # [eV s²/m²]
    L = flight_path_length  # m
    t = time / 1e6  # s
    return m / 2 * (L / t) ** 2  # eV


def energy2time(energy, flight_path_length):
    r"""Convert energy to time-of-flight of the neutron.

    .. math::
        t = L \sqrt{ \frac{m}{2 E} }

    where :math:`E` is the energy, :math:`m` is the mass, :math:`t` is
    the time-of-flight of the neutron, and :math:`L` is the flight path
    lenghth.

    Args:
        energy:  Energy of the neutron in :math:`\mathrm{eV}`.
        flight_path_length: flight path length in :math:`\mathrm{m}`.

    Returns:
        Time-of-flight in :math:`\mathrm{μs}`.

    """
    L = flight_path_length  # m
    m = MASS_OF_NEUTRON  # eV s²/m²
    E = energy  # eV
    t = L * np.sqrt(m / 2 / E)  # s
    return t * 1e6  # μs


def no_nan_divide(x, y):
    """Return `x/y`, with 0 instead of NaN where `y` is 0.

    Args:
        x: Numerator.
        y: Denominator.

    Returns:
        `x / y` with 0 wherever `y == 0`.
    """
    return np.where(y != 0, np.divide(x, np.where(y != 0, y, 1)), 0)
