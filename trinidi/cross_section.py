"""Cross section routines"""

import os

import numpy as np

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


def new_dict():
    """Generate cross section dictionary"""

    xsdata = __xsdata_load_once__()

    n = len(xsdata["isotopes"])
    print(f"Total number of isotopes: {n}")
