#!/usr/bin/env python
# coding: utf-8

# Isotropic Total Variation (Accelerated PGM)
# ===========================================
#
# This example demonstrates the use of class
# to solve isotropic total variation (TV) regularization. It solves the
# denoising problem
#
#   $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - \mathbf{x}
#   \|^2 + \lambda R(\mathbf{x}) + \iota_C(\mathbf{x}) \;,$$
#
# where $R$ is a TV regularizer, $\iota_C(\cdot)$ is the indicator function
# of constraint set $C$, and $C = \{ \mathbf{x} \, | \, x_i \in [0, 1] \}$,
# i.e. the set of vectors with components constrained to be in the interval
# $[0, 1]$. The problem is solved seperately with $R$ taken as isotropic
# and anisotropic TV regularization
#
# The solution via PGM is based on the approach in <cite data-cite="beck-2009-tv"/>,
# which involves constructing a dual for the constrained denoising problem.
# The PGM solution minimizes the resulting dual. In this case, switching
# between the two regularizers corresponds to switching between two
# different projectors.
#
#

# Create a ground truth image.

# In[3]:


print("Hello World!")

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100)
plt.plot(x)
plt.show()
