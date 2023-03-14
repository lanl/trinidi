""" Resolution Operator """

import numpy as np
from jax import device_put
from scico.linop import Convolve, LinearOperator


class ResolutionOperator:
    """Resolution Operator Class"""

    def __repr__(self):
        return f"""{type(self)}

input_shape = {self.input_shape}
output_shape = {self.output_shape}

projection_shape = {self.projection_shape}

        """

    def __init__(self, output_shape, kernels=None):
        """Initialize a ResolutionOperator object.

        Args:
            output_shape: Output shape of operator, i.e. measurement shape.
            kernels (list of nd-arrays): list of convolution kernels.
                'None' results in identity operator.
        """
        self.output_shape = output_shape

        if kernels == None:
            kernels = [1]

        if len(kernels) > 1:
            self.kernels = kernels
        else:
            raise ValueError("Number of kernels must be at least 1.")

        self.projection_shape = self.output_shape[:-1]
        self.N_A = self.output_shape[-1]

        kernel_sizes = [k.size for k in self.kernels]
        self.N_buffer_lo = int((max(kernel_sizes[:2]) - 1) / 2)
        self.N_buffer_hi = int((max(kernel_sizes[-2:]) - 1) / 2)
        self.N_F = self.N_buffer_lo + self.N_A + self.N_buffer_hi

        # Creating operators
        self.input_shape = self.projection_shape + (self.N_F,)

        self.W = self._get_weights()

        # Convolution operators with different stds
        self.Hks = []
        for k, kernel in enumerate(self.kernels):
            Hk = self._get_Hk(kernel, self.input_shape)
            self.Hks.append(Hk)

            HWk = LinearOperator(
                input_shape=self.input_shape,
                eval_fn=lambda x, Hk=Hk, Wk=self.W[k]: Hk(x) * Wk,
            )

            if k == 0:
                self.H = HWk
            else:
                self.H = self.H + HWk

        if self.N_buffer_hi > 0:
            self.G = lambda x: ((x.T)[self.N_buffer_lo : -self.N_buffer_hi]).T
        else:
            self.G = lambda x: ((x.T)[self.N_buffer_lo :]).T

        self.R = lambda x: self.G(self.H(x))

    def __call__(self, x):
        return self.R(x)

    def call_on_any_array(self, array):
        """Call Resolution operator on any array that has the same
        number of TOA bins (N_A).
        """
        if array.shape[-1] != self.input_shape[-1]:
            raise ValueError(
                f"array shape not compatible. array.shape[-1] ({array.shape[-1]}) != input_shape[-1] ({self.input_shape[-1]})"
            )
        output_shape = array.shape[:-1] + (self.output_shape[-1],)
        R_ = self.__class__(output_shape, self.kernels)
        return R_(array)

    def compute_t_F(self, t_A):
        """Finds time-of-flight array so that R(t_F) = t_A

        Args:
            t_A (array): Time-of-arrival equi-spaced increasing array

        Returns:
            t_F (array): Time-of-flight equi-spaced increasing array
        """

        x = np.arange(self.N_F)
        u = self.call_on_any_array(x)

        slope = (u[-1] - u[0]) / (u.size - 1)
        offset = u[0]

        Δt = t_A[1] - t_A[0]  # desired slope
        t0 = t_A[0]  # desired offset
        t_F = ((x - offset) / slope) * Δt + t0

        return t_F

    def _get_weights(self):
        def triangle(size, center=0, radius=1):
            x = np.arange(size)
            y = 1 - np.abs(x - center) / radius
            y = np.maximum(y, 0)
            return y

        K = len(self.kernels)
        N_F = self.N_F

        if K > 1:
            W = np.zeros([K, N_F])
            for i in range(K):
                W[i] = triangle(
                    N_F, center=(N_F - 1) / (K - 1) * i, radius=(N_F - 1) / (K - 1)
                )

        else:
            W = np.ones([K, N_F])

        return W

    def _get_Hk(self, kernel, input_shape):
        h = kernel.copy()
        h = np.require(h, dtype=np.float32)
        kernel_shape = tuple(np.ones_like(self.projection_shape)) + h.shape
        h = h.reshape(kernel_shape)
        h = device_put(h)

        return Convolve(h, input_shape=input_shape, mode="same", jit=True)


kernels = [
    np.array([1]),
    np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]),
]

t_A = np.arange(70, 700, 0.32)
N_A = t_A.size


output_shape = (1, N_A)


flight_path_length = 10.4


R = ResolutionOperator(output_shape, kernels)
t_F = R.compute_t_F(t_A)
x = np.random.rand(*R.input_shape)

y = R(x)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=[12, 8], sharex=True)
ax = np.atleast_1d(ax)
ax[0].plot(t_F, x[0], label="x", alpha=0.75)
ax[0].plot(t_A, y[0], label="y", alpha=0.75)
ax[0].legend(prop={"size": 8})
fig.suptitle("")
# plt.savefig('')
plt.show()
