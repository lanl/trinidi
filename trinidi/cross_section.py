"""Cross section routines"""

import os

import numpy as np

from scipy import interpolate
from si_prefix import si_format

from trinidi import util

# Store absolute path to xsdata.npy
__xsdata_filename__ = os.path.join(os.path.dirname(__file__), "data/xsdata.npy")
if not os.path.exists(__xsdata_filename__):
    raise Exception(
        f"__xsdata_filename__: `{__xsdata_filename__}` does not exist. Please ensure submodule `data` is up to date."
    )

global __xsdata__


def __xsdata_load_once__():
    """Loads the cross section data dictionary `__xsdata__` once when necessary."""

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


class XSDict:
    r"""XSDict cross section dictionary class.

    The cross section of the :math:`i^{\mathrm{th}}` time-of-flight bin corresponds to the
    average cross section in the time-of-flight interval `[t_F[i] - Δt/2, t_F[i] + Δt/2]`.

    The attribute `self.values` stores the cross section values and is a `numpy.ndarray`
    of size `(N_m, N_F)`, where `N_m` is the number of isotopes (`len(isotopes)`) and `N_m`
    is the number of time-of-flight bins (`t_F.size`).

    """

    def __repr__(self):
        return f"""{type(self)}
    isotopes = {self.isotopes}
    N_m = {self.N_m}

    t_F = [{self.t_F[0]:.3f} μs, ..., {self.t_F[-1]:.3f} μs]
    Δt = {self.Δt:.3f} μs
    N_F = {self.N_F}
    flight_path_length = {self.flight_path_length:.3f} m
    E = [{self.E[0]:.3f} eV, ..., {self.E[-1]:.3f} eV]

    samples_per_bin = {self.samples_per_bin}

    values.shape = {self.values.shape} = (N_m, N_F)
        """

    def __init__(self, isotopes, t_F, flight_path_length, samples_per_bin=10):
        r"""Initialize a XSDict object.

        Args:
            isotopes (list): Isotope symbols e.g. `["U-235", "U-238"]`.
            t_F (array): time-of-flight array of the neutrons in :math:`\mathrm{μs}`.
            flight_path_length (scalar): flight path length in :math:`\mathrm{m}`.
            samples_per_bin (int, optional): Default 10. Likely this parameter need not be
                modified. Number of samples used within time-of-flight bin to approximate average
                cross section within bin. If `samples_per_bin == 1`, center of the time-of-flight
                bin is used.
        """
        self.isotopes = isotopes.copy()
        self.N_m = len(self.isotopes)

        self.t_F = t_F
        self.Δt = self.t_F[1] - self.t_F[0]
        if self.Δt < 0:
            raise ValueError("t_F must be equispaced and increasing")
        self.N_F = self.t_F.size

        self.flight_path_length = flight_path_length
        self.E = util.time2energy(self.t_F, self.flight_path_length)

        self.samples_per_bin = samples_per_bin
        self.values = self._get_cross_section_values(
            isotopes, self.t_F, self.flight_path_length, self.samples_per_bin
        )

    def plot(self, ax, function_of_energy=False):
        """Plot the cross sections of a XSDict object.

        Args:
            ax: Matplotlib axis used for plotting.
            function_of_energy (bool, optional): `True` plots the dictionary as a function
                of energy. Default `False` plots it as a function of time-of-flight.
        """
        if function_of_energy:
            xax = self.E
            xax_label = util.ENERGY_LABEL
        else:
            xax = self.t_F
            xax_label = util.TOF_LABEL

        for d, isotopes in zip(self.values, self.isotopes):
            ax.plot(xax, d, label=isotopes, alpha=0.6)

        ax.set_yscale("log")
        ax.set_title("Cross Section Dictionary [cm²/mol]")
        ax.legend()
        ax.set_xlabel(xax_label)

    def merge(self, merge_isotopes, merge_weights, new_key):
        """Merge cross section entries of an XSDict.

        This function can be used to combine cross section entries using a weighted sum.
        The list `merge_isotopes` must be a subset of `self.isotopes`. The `merge_weights` are
        usually the natural abundance fractions of the isotopes, summing to <=1. The returned
        list of `isotopes` has the unchanged isotopes where the merged isotopes are replaced by
        the `new_key` string. (I.e. they updated list `isotopes` is likely not stricly only isotopes.)

        Args:
            merge_isotopes (list): list of isotope symbols to be summed.
            merge_weights (list): list of weights to be used for weighted sum.
            new_key (str): New symbol to be used in place of the merged cross sections.

        Returns:
            (list) Updated list of isotopes.
        """
        not_in = list(set(merge_isotopes) - set(self.isotopes))
        if not_in:
            raise ValueError(
                f"merge_isotopes must be subset of self.isotopes. {not_in} not part of {self.isotopes}"
            )

        # index of the first merged isotope; where new_key will be inserted.
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
        self.N_m = len(self.isotopes)

        return self.isotopes

    def _get_cross_section_values(self, isotopes, t_F, flight_path_length, samples_per_bin):
        """Creates a cross section dictionary."""

        xsdict = np.zeros([len(isotopes), t_F.size])
        Δt = abs(t_F[1] - t_F[0])
        E = util.time2energy(t_F, flight_path_length)
        xsdata = __xsdata_load_once__()

        for i, isotope in enumerate(isotopes):
            if isotope not in xsdata["isotopes"]:
                raise ValueError(f"Isotope {isotope} cannot be found in data base.")

            iid = xsdata["isotopes"].index(isotope)
            xs_raw = xsdata["cross_sections"][iid]
            E_raw = xsdata["energies"][iid]
            xs_interp = interpolate.interp1d(E_raw, xs_raw)

            if not (min(E_raw) < min(E) and max(E) < max(E_raw)):
                raise ValueError(
                    f"Requested energy range [{si_format(min(E))}eV, {si_format(max(E))}eV]"
                    f"for {isotope} is too large. Must be within "
                    f"[{si_format(min(E_raw))}eV, {si_format(max(E_raw))}eV]."
                )

            if samples_per_bin == 1:
                xs = xs + interpolate.interp1d(E_raw, xs_raw)(E)

            else:
                xs = np.zeros(t_F.size)
                # compute average cross section using samples_per_bin samples per time-of-flight bin
                for r in np.linspace(-1 / 2, 1 / 2, samples_per_bin):
                    E_shift = util.time2energy(t_F + r * Δt, flight_path_length)
                    xs = xs + xs_interp(E_shift) / samples_per_bin

            xsdict[i] = xs

        return xsdict
