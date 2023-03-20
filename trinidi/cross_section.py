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
            If `None` all available isotopes are displayed.
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


def create_xsdict(isotopes, t_F, flight_path_length, samples_per_bin=10):
    """Creates a cross section dictionary."""
    xsdict = np.zeros([len(isotopes), t_F.size])
    Δt = abs(t_F[1] - t_F[0])
    E = time2energy(t_F, flight_path_length)
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
            xs = np.zeros(t_F.size)
            for r in np.linspace(-1 / 2, 1 / 2, samples_per_bin):
                E_shift = time2energy(t_F + r * Δt, flight_path_length)
                xs = xs + interpolator(E_shift) / samples_per_bin

        xsdict[i] = xs

    return xsdict


def plot_xsdict(ax, D, isotopes, t_F=None, E=None):
    """Plot a cross section dictionary.

    Args:
        ax: Matplotlib axis.
        D: Cross section dictionary with shape `(N_m, N_F)`.
        isotopes: List of isotope symbols with length `N_m`.
        t_F (optional): Time-of-flight array with size `N_F`. Exactly one of `t_F` and `E` must be `None`.
        E (optional): Energy array with size `N_F`. Exactly one of `t_F` and `E` must be `None`.
    """

    if t_F is not None and E is None:
        xax = t_F
        xax_label = "Time-of-flight [μs]"
    elif t_F is None and E is not None:
        xax = E
        xax_label = "Energy [eV]"
    else:
        raise ValueError("Exactly one of t_F and E must be None")

    for i in range(len(isotopes)):
        ax.plot(xax, D[i], label=isotopes[i], alpha=0.6)

    ax.set_yscale("log")
    ax.set_title("Cross Section Dictionary [cm²/mol]")
    ax.legend()
    ax.set_xlabel(xax_label)


class XSDict:
    """docstring for XSDict"""

    def __repr__(self):
        return f"""{type(self)}
isotopes = {self.isotopes}
N_m = {self.N_m}

t_F = [{self.t_F[0]:.3f} μs, ..., {self.t_F[-1]:.3f} μs]
Δt = {self.Δt:.3f} μs
N_F = {self.N_F}
flight_path_length = {self.flight_path_length:.3f} m
E = [{self.E[-1]:.3f} eV, ..., {self.E[0]:.3f} eV]

samples_per_bin = {self.samples_per_bin}

shape = {self.values.shape} = (N_m, N_F)
values = {self.values.__repr__()}

        """

    def __init__(self, isotopes, t_F, flight_path_length, samples_per_bin=10):
        """Initialize a XSDict object.

        Args:
            isotopes (TYPE): Description
            t_F (TYPE): Description
            flight_path_length (TYPE): Description
            samples_per_bin (int, optional): Description

        Raises:
            ValueError: Description
        """
        self.isotopes = isotopes.copy()
        self.N_m = len(self.isotopes)

        self.t_F = t_F
        self.Δt = self.t_F[1] - self.t_F[0]
        if self.Δt < 0:
            raise ValueError("t_F must be equispaced and increasing")
        self.N_F = self.t_F.size

        self.flight_path_length = flight_path_length
        self.E = time2energy(self.t_F, self.flight_path_length)

        self.samples_per_bin = samples_per_bin
        self.values = create_xsdict(
            isotopes, t_F, flight_path_length, samples_per_bin=10
        )

    def plot(self, ax, function_of_energy=False):
        """Plot a XSDict.

        Args:
            ax (TYPE): Description
            function_of_energy (bool, optional): Description
        """
        if function_of_energy:
            plot_xsdict(ax, self.values, self.isotopes, E=self.E)
        else:
            plot_xsdict(ax, self.values, self.isotopes, t_F=self.t_F)

    def merge(self, merge_isotopes, merge_weights, new_key):
        """Merge entries of an XSDict.

        Args:
            merge_isotopes (TYPE): Description
            merge_weights (TYPE): Description
            new_key (TYPE): Description

        Raises:
            ValueError: Description
        """
        not_in = list(set(merge_isotopes) - set(self.isotopes))
        if not_in:
            raise ValueError(
                f"merge_isotopes must be subset of self.isotopes. {not_in} not part of {self.isotopes}"
            )

        target = min([self.isotopes.index(isotope) for isotope in merge_isotopes])

        D = list(self.values)
        d = np.zeros_like(D[target])

        for isotope in merge_isotopes:
            i = self.isotopes.index(isotope)

            d = d + D[i] * merge_weights[i]
            del D[i]
            del self.isotopes[i]

        self.isotopes.insert(target, new_key)

        D.insert(target, d)
        self.values = np.array(D)
