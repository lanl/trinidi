# Copyright (C) 2023 by Thilo Balke
# All rights reserved. BSD 3-clause License.
# This file is part of the TRINIDI package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Time-of-Flight Resonance Imaging with Neutrons for Isotopic Density Inference (TRINIDI)
is a Python package for estimating isotopic densities using neutron transmission resonance images.
"""

__version__ = "0.0.1a1"


from .cross_section import *
from .resolution import *
from .util import *

__all__ = []

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
