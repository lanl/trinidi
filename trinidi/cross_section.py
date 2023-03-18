"""Cross section routines"""

import os

import numpy as np
from scipy import interpolate
from si_prefix import si_format

from trinidi.util import time2energy

# Store absolute path to xsdata.npy
__xsdata_filename__ = os.path.join(os.path.dirname(__file__), "../data/xsdata.npy")
if not os.path.exists(__xsdata_filename__):
    raise Exception(
        f"__xsdata_filename__: `{__xsdata_filename__}` does not exist. Please ensure submodule `data` is up to date."
    )

global __xsdata__


def __xsdata_load_once__():
    """Function that loads the cross section data dictionary `__xsdata__` once when necessary."""

    if "__xsdata__" not in globals():
        global __xsdata__
        __xsdata__ = np.load(__xsdata_filename__, allow_pickle=True)[()]

    return __xsdata__


def avail():
    """Returns list of all available isotopes."""
    xsdata = __xsdata_load_once__()
    return xsdata["isotopes"]


def info(isotopes=None):
    r"""Print energy ranges for isotopes.

    Args:
        isotopes (optional, list): list of isotope symbols.
        If `None` all available isotopes are used.
    """

    xsdata = __xsdata_load_once__()
    if isotopes is None:
        isotopes = xsdata["isotopes"]
        print_total = True
    else:
        print_total = False

    for isotope in isotopes:
        iid = xsdata["isotopes"].index(isotope)
        E = xsdata["energies"][iid]
        E0 = si_format(E[0])
        E1 = si_format(E[-1])
        print(f"{isotope}\t{E0}eV to {E1}eV")

    if print_total:
        n = len(xsdata["isotopes"])
        print(f"Total number of isotopes: {n}")

    return


def create_xsdict(isotopes, t, flight_path_length, samples_per_bin=10):
    """Creates a cross section dictionary."""
    xsdict = np.zeros([len(isotopes), t.size])
    Δt = abs(t[1] - t[0])
    E = time2energy(t, flight_path_length)
    xsdata = __xsdata_load_once__()

    for i, isotope in enumerate(isotopes):

        if isotope not in xsdata["isotopes"]:
            raise ValueError(f"Isotope {isotope} cannot be found in data base.")

        iid = xsdata["isotopes"].index(isotope)
        xs_raw = xsdata["cross_sections"][iid]
        E_raw = xsdata["energies"][iid]
        interpolator = interpolate.interp1d(E_raw, xs_raw)

        if not (min(E_raw) < min(E) and max(E) < max(E_raw)):
            raise ValueError(
                f"Requested energy range [{si_format(min(E))}eV, {si_format(max(E))}eV] for {isotope} is too large. Must be within [{si_format(min(E_raw))}eV, {si_format(max(E_raw))}eV]."
            )

        if samples_per_bin == 1:

            xs = xs + interpolate.interp1d(E_raw, xs_raw)(E)

        else:

            xs = np.zeros(t.size)
            for r in np.linspace(-1 / 2, 1 / 2, samples_per_bin):

                E_shift = time2energy(t + r * Δt, flight_path_length)
                xs = xs + interpolator(E_shift) / samples_per_bin

        xsdict[i] = xs

    return xsdict
