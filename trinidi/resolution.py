""" Resolution Operator """

import numpy as np

from jax import device_put

from scico.linop import Convolve, LinearOperator

from trinidi.util import time2energy


class ResolutionOperator:
    """ResolutionOperator Class"""

    def __repr__(self):
        return f"""{type(self)}
    input_shape = {self.input_shape} = projection_shape + (N_F,)
    output_shape = {self.output_shape} = projection_shape + (N_A,)

    projection_shape = {self.projection_shape}
    N_F = {self.input_shape[-1]}
    N_A = {self.output_shape[-1]}
        """

    def __init__(self, output_shape, t_A, kernels=None):
        """Initialize a ResolutionOperator object.

        Args:
            output_shape: Output shape of operator, i.e. measurement shape.
            kernels (list of nd-arrays): list of convolution kernels.
                'None' results in identity operator. Each kernel must
                sum to 1.
        """
        self.output_shape = output_shape
        self.t_A = t_A

        if kernels == None:
            kernels = [np.array([1])]

        if len(kernels) >= 1:
            self.kernels = kernels
        else:
            raise ValueError("Number of kernels must be at least 1.")

        for k in self.kernels:
            if np.any(k < 0):
                raise ValueError("Kernels must me non-negative")
            if np.abs(np.sum(k) - 1) > 1e-3:
                raise ValueError("Kernels must sum to 1.")

        self.projection_shape = self.output_shape[:-1]
        self.N_A = self.output_shape[-1]

        kernel_sizes = [k.size for k in self.kernels]
        N_buffer_lo = int((max(kernel_sizes[:2]) - 1) / 2)
        N_buffer_hi = int((max(kernel_sizes[-2:]) - 1) / 2)
        self.N_F = N_buffer_lo + self.N_A + N_buffer_hi

        # Creating operators
        self.input_shape = self.projection_shape + (self.N_F,)

        self.W = self._get_weights(self.kernels, self.N_F)

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

        if N_buffer_hi > 0:
            self.G = lambda x: ((x.T)[N_buffer_lo:-N_buffer_hi]).T
        else:
            self.G = lambda x: ((x.T)[N_buffer_lo:]).T

        self.R = lambda x: self.G(self.H(x))

        if self.projection_shape == (1,):
            self.single = self

        else:
            single_output_shape = (
                1,
                self.N_A,
            )
            self.R_single = self.__class__(single_output_shape, None, kernels=self.kernels)

        if self.t_A is not None:
            self.t_F = self.compute_t_F(self.t_A)

    def __call__(self, x):
        return self.R(x)

    def single(self, x):
        r"""Call Resolution operator on an array of size `(1, N_F)`."""
        return self.R_single(x)

    def call_on_any_array(self, array):
        """Call ResolutionOperator on an array of size `(..., N_F)`.

        Note: New ResolutionOperator object is created every time this function is called.
        Thus, is is not recommended to use this function when fast performance is required.
        """
        if array.shape[-1] != self.input_shape[-1]:
            raise ValueError(
                f"array shape not compatible. array.shape[-1] ({array.shape[-1]}) != input_shape[-1] ({self.input_shape[-1]})"
            )
        output_shape = array.shape[:-1] + (self.output_shape[-1],)
        R_ = self.__class__(output_shape, None, kernels=self.kernels)
        return R_(array)

    def compute_t_F(self, t_A):
        r"""Finds time-of-flight array so that :math:`t_F^\top R \approx t_A^\top`.

        Args:
            t_A (array): Time-of-arrival equi-spaced increasing array.

        Returns:
            t_F (array): Time-of-flight equi-spaced increasing array.
        """
        x = np.arange(self.N_F)
        u = np.array(self.call_on_any_array(x))

        slope = (u[-1] - u[0]) / (u.size - 1)
        offset = u[0]

        Δt = t_A[1] - t_A[0]  # desired slope
        t0 = t_A[0]  # desired offset
        t_F = ((x - offset) / slope) * Δt + t0

        return t_F

    def _get_weights(self, kernels, N_F):
        def triangle(size, center=0, radius=1):
            x = np.arange(size)
            y = 1 - np.abs(x - center) / radius
            y = np.maximum(y, 0)
            return y

        K = len(kernels)

        if K > 1:
            W = np.zeros([K, N_F])
            for i in range(K):
                W[i] = triangle(N_F, center=(N_F - 1) / (K - 1) * i, radius=(N_F - 1) / (K - 1))

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

    def plot_kernel_weights(self, ax):
        """Plot kernel weights as a function of t_F."""
        if self.t_A is not None:
            for i, w in enumerate(self.W):
                ax.plot(
                    self.t_F[w > 0],
                    w[w > 0],
                    linestyle="--",
                    label=f"w{i}",
                    alpha=0.6,
                    linewidth=1.3,
                )
            ax.legend(prop={"size": 8})
            ax.set_title("Kernel Weights")
        else:
            raise ValueError("Can only plot weights if t_A is not None at construction.")


from scipy.special import gamma


def lansce_fp5_kernel(t_A, Δt, flight_path_length):
    r"""Resolution function kernel based on :cite:`lynn2002neutron`.

    Args:
        t_A (scalar): time-of-arrival of the neutron in :math:`\mathrm{μs}`.
        Δt (scalar): time sampling bin width in :math:`\mathrm{μs}`.
        flight_path_length (scalar): flight path length in :math:`\mathrm{m}`.

    Returns:
        Kernel array.
    """
    E = time2energy(t_A, flight_path_length)

    v1 = 6
    T1 = 0.74 / np.sqrt(E) / Δt
    t1 = 0.49 / np.sqrt(E) / Δt

    v2 = 4.3
    T2 = 5.1 / np.sqrt(E) / Δt
    t2 = 2.2 / np.sqrt(E) / Δt

    w1 = 0.65

    mode = t1 + T1 * v1 / 2 - T1  # This is only true
    thresh = t2 + v2 * T2  # for these values of v, T, t

    x = np.arange(-np.ceil(thresh), np.ceil(thresh))

    fi = lambda x, v, t, T: ((x - t) ** (v / 2 - 1) / (gamma(v / 2) * T ** (v / 2))) * np.exp(
        -(x - t) / T
    )

    f1 = np.zeros_like(x)
    f1[x + mode > t1] = fi(x[x + mode > t1] + mode, v1, t1, T1)

    f2 = np.zeros_like(x)
    f2[x + mode > t2] = fi(x[x + mode > t2] + mode, v2, t2, T2)

    f = f1 * w1 + f2 * (1 - w1)
    f = f / np.sum(f)

    return f


def equispaced_kernels(t_A, num_kernels, kernel_generator):
    r"""Generate a list of resolution function kernels that correspond to equispaced time-of-flights and a given kernel_generator.

    Args:
        t_A (array): time-of-arrival array of the neutrons in :math:`\mathrm{μs}`.
        num_kernels (int): Number of kernels to be generated. If `1`, single kernel with average
            time-of-arrival is being generated.
        kernel_generator (function): Function with single argument that generates a kernel array
            based on the time-of-arrival.

    Returns:
        list of Kernel arrays.
    """
    if num_kernels > 1:
        t_As = np.linspace(t_A[0], t_A[-1], num=num_kernels)
    else:
        t_As = [np.mean(t_A)]

    return [kernel_generator(t) for t in t_As], t_As
