"""Some simulation related functions and classes."""

import numpy as np

from trinidi import cross_section, reconstruct, resolution


class SimpleSpectrum:
    """Represent spectrum y(t_A) using only a few coefficient, coeffs.

    y(t) ≈ exp( a_0 + a_1 (log t) + a_2 (log t)^2 + a_3 (log t)^3)

    where t:     t_A
          a_i:   coeffs

    """

    def __repr__(self):
        return f"""{type(self)}

self.coeffs = {self.coeffs}
        """

    def __init__(self, t_A=None, y=None, N=None, coeffs=None):
        """Initialize either with [y and t_A, N] or directly with coeffs.

        Args:
            t_A (array, optional): t_A vector
            y (array, optional): spectrum vector
            N (scalar, optional): number of coefficients
            coeffs (array/list, optional): coefficients
        """
        if (t_A is None) and (y is None) and (N is None) and (not (coeffs is None)):
            # coeffs

            self.coeffs = np.atleast_2d(np.array(coeffs))
            if self.coeffs.shape[0] == 1:
                self.coeffs = self.coeffs.T

            self.N = self.coeffs.size

        elif (not (t_A is None)) and (not (y is None)) and (not (N is None)) and (coeffs is None):
            # data

            self.N = N

            y = np.atleast_2d(np.array(y))
            if y.shape[0] == 1:
                y = y.T

            print(y.shape)

            A = np.stack([np.log(t_A) ** i for i in range(self.N)]).T
            self.coeffs = np.linalg.pinv(A) @ np.log(y)

        else:
            print("ERROR")

    def __call__(self, t_A):
        """Get approximation at t_A(s)

        Args:
            t_A (array): t_A vector

        Returns:
            array: approximation
        """
        A = np.stack([np.log(t_A) ** i for i in range(self.N)]).T
        return np.exp(A @ self.coeffs)


def generate_spectra(t_A, acquisition_time=1):
    r"""Generate example parameters for simulation"""
    b_raw = (
        SimpleSpectrum(coeffs=[[7.63056371], [-2.07322922], [0.11770202]])(t_A) * acquisition_time
    )
    ϕ = SimpleSpectrum(coeffs=[[2.67570648], [-0.72236513], [0.05469538]])(t_A) * acquisition_time
    N_b = 3

    P = reconstruct.background_basis(N_b, t_A.size)
    θ = (np.log(b_raw).T @ np.linalg.pinv(P)).T
    b = (np.exp(θ.T @ P)).T

    α_1 = 1.2
    α_2 = 0.8

    return ϕ, b, θ, α_1, α_2


def circle_mask(N, center=None, radius=1):
    r"""Generate a circle mask with array size of size `N`x`N`."""
    if center is None:
        center = [0, 0]

    if radius is None:
        radius = 1

    v = np.linspace(-1, 1, N, endpoint=True)
    x0, x1 = np.meshgrid(v, v)
    m = ((x0 - center[0]) ** 2 + (x1 - center[1]) ** 2) ** 0.5 <= radius
    return m


def rose_phantom(N, num_circles=5, radius=2 / 3):
    r"""Generate a rose phantom with array size of size `N`x`N`."""
    distance = 1 - radius
    angles = np.linspace(0, 2 * np.pi, num_circles, endpoint=False) - np.pi / 2
    centers = np.array([[np.cos(angle), np.sin(angle)] for angle in angles]) * distance
    masks = np.stack([circle_mask(N, center, radius=radius) for center in centers], axis=2)

    return masks


def generate_sample_data(
    isotopes,
    z,
    Δt=0.90,
    t_0=72,
    t_last=400,
    flight_path_length=10,
    projection_shape=(31, 31),
    kernels=None,
    acquisition_time=10,
):
    r"""Generate example data."""

    t_A = np.arange(t_0, t_last, Δt)
    N_A = t_A.size

    if not kernels:
        kernels = [np.array([1]), np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])]

    output_shape = projection_shape + (N_A,)

    R = resolution.ResolutionOperator(output_shape, t_A, kernels=kernels)
    t_F = R.t_F

    ϕ, b, θ, α_1, α_2 = generate_spectra(t_A, acquisition_time=10)
    N_b = θ.size

    D = cross_section.XSDict(isotopes, t_F, flight_path_length)

    Z = rose_phantom(projection_shape[0], num_circles=z.size, radius=2 / 3) * z.reshape(
        [1, 1, z.size]
    )

    v = np.random.poisson(1000, size=projection_shape + (1,))
    v = v / v.mean()

    Φ = v @ ϕ.T
    B = v @ b.T

    Y_o_bar = Φ + B
    Y_s_bar = α_1 * (Φ * R(np.exp(-Z @ D.values)) + α_2 * B)

    Y_o = np.random.poisson(Y_o_bar)
    Y_s = np.random.poisson(Y_s_bar)

    ground_truth_params = {"z": z, "α_1": α_1, "α_2": α_2, "θ": θ}

    return Y_o, Y_s, ground_truth_params, Z, t_A, flight_path_length, N_b, kernels
