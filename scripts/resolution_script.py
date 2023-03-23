r"""Example script for resolution submodule"""

import matplotlib.pyplot as plt
import numpy as np

from trinidi import resolution

Δt = 0.30
t_A = np.arange(72, 720, Δt)
N_A = t_A.size
flight_path_length = 10
num_kernels = 5

projection_shape = (1,)
output_shape = projection_shape + (N_A,)

g = lambda t_A: resolution.lanl_fp5_kernel(t_A, Δt, flight_path_length)
kernels = resolution.equispaced_kernels(t_A, num_kernels, g)
R = resolution.ResolutionOperator(output_shape, t_A, kernels=kernels)
t_F = R.t_F

x = np.random.rand(*R.input_shape) ** 0.5
y = R(x)


fig, ax = plt.subplots(1, 1, figsize=[8, 6], sharex=True)
ax = np.atleast_1d(ax)
ax[0].plot(t_F, x.flatten(), label="Original", alpha=0.75)
ax[0].plot(t_A, y.flatten(), label="Blurred", alpha=0.75)
ax[0].legend()
ax[0].set_xlabel("Time in μs")
fig.suptitle("Comparison of imput signal and output signal of resolution blurring")
plt.show()
