"""Some reconstruction functions and classes."""


class ProjectionRegion:
    r"""ProjectionRegion class.

    This class is used to handle the Ω regions indicating the regions where the sample image
    has no areal density, or uniformly dense areal density. Used for averaging over many pixels.

    """

    def __init__(self, projection_mask):
        """Initialize a ProjectionRegion object.

        If the counting data, `Y`, has shape `projection_shape + (N_A,)`, the `projection_mask`
        array must have size `projection_shape + (1,)`.

        Args:
            projection_mask (boolean or binary array): `projection_mask` array used for averaging.
        """

        self.projection_shape = projection_mask.shape[:-1]
        self.mask = (projection_mask > 0) * 1.0
        self.ω = self.mask / self.mask.sum()

    def average(self, Y):
        r"""Compute ::math::`ω^\top Y`,

            where ::math::`\omega` has shape `projection_shape + (1,)` and ::math::`Y` has shape
            `projection_shape + (N_A,)`, thus the result has shape `(1, N_A)`.

        Args:
            Y: ::math::`Y` array.
            ω (optional): ::math::`\omega` array. When `None`, the average over all projections is
                being computed, i.e. it is equivalent to ::math::`(\forall i) \omega_i = 1/N`, where
                `N` is the number of projections, `np.prod(projection_shape)`.

        Returns:
            The ::math::`ω^\top Y` array.
        """
        projection_dims = tuple([i for i, _ in enumerate(self.projection_shape)])
        return np.sum(self.ω * Y, axis=projection_dims)[np.newaxis, :]

    def imshow(self, ax, title=None):
        r"""Show the projection_mask as an image"""
        if len(self.projection_shape) != 2:
            raise ValueError("imshow only possible when projection_shape is 2D.")

        im = ax.imshow(self.mask[:, :, 0], vmin=0)
        if title:
            ax.set_title(title)

        return im

    def plot_contours(self, ax, color="red", alpha=0.5):
        r"""Show the projection_mask as contour"""
        if len(self.projection_shape) != 2:
            raise ValueError("plot_contours only possible when projection_shape is 2D.")

        cs = ax.contour(self.mask[:, :, 0], colors=color, alpha=alpha, levels=[0.5])

        return cs


def plot_densities(fig, ax, Z, isotopes):
    r"""Show areal densities. `ax` must be list."""

    for i, isotope in enumerate(isotopes):
        z = Z[:, :, i]
        vmax = np.percentile(z, 99.9)
        vmin = np.percentile(z, 0.1)
        im = ax[i].imshow(z, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[i], format="%.1e")
        ax[i].set_title(f"{isotope}")
        ax[i].axis("off")

    fig.suptitle("Areal Densities [mol/cm²]")

    return fig, ax


import matplotlib.pyplot as plt
import numpy as np

from trinidi import cross_section, resolution, util

Δt = 0.90
t_A = np.arange(72, 720, Δt)
N_A = t_A.size
flight_path_length = 10

kernels = [np.array([1]), np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])]

isotopes = ["U-238", "Pu-239", "Ta-181"]
projection_shape = (31, 31)


output_shape = projection_shape + (N_A,)

R = resolution.ResolutionOperator(output_shape, kernels)
t_F = R.compute_t_F(t_A)
D = cross_section.XSDict(isotopes, t_F, flight_path_length)


ϕ, b, θ, α_1, α_2 = util.generate_spectra(t_A, acquisition_time=3)


fig, ax = plt.subplots(1, 1, figsize=[12, 8], sharex=True)
ax = np.atleast_1d(ax)
ax[0].plot(t_A, b.flatten(), label="b", alpha=0.75)
ax[0].plot(t_A, (ϕ + b).flatten(), label="ϕ+b", alpha=0.75)
ax[0].set_xlabel(util.TOF_LABEL)
ax[0].legend(prop={"size": 8})


z = np.array([[0.005, 0.003, 0.004]]).T
Z = util.rose_phantom(
    projection_shape[0], num_circles=z.size, radius=2 / 3
) * z.reshape([1, 1, z.size])


fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
ax = np.atleast_1d(ax)
plot_densities(fig, ax, Z, isotopes)
plt.show()


v = np.random.poisson(1000, size=projection_shape + (1,))
v = v / v.mean()

Φ = v @ ϕ.T
B = v @ b.T

Y_o_bar = Φ + B
Y_s_bar = α_1 * (Φ * R(np.exp(-Z @ D.values)) + α_2 * B)

Y_o = np.random.poisson(Y_o_bar)
Y_s = np.random.poisson(Y_s_bar)


Ω_z = ProjectionRegion(np.prod(Z, axis=2, keepdims=True) > 0)
Ω_0 = ProjectionRegion(np.sum(Z, axis=2, keepdims=True) == 0)


fig, ax = plt.subplots(1, 3, figsize=[14, 4])
ax = np.atleast_1d(ax)

ax[0].imshow(np.sum(Y_s, axis=-1) / np.sum(Y_o, axis=-1), vmin=0)
ax[0].set_title("1Y_s / 1Y_o")

Ω_z.plot_contours(ax[0], color="red")
Ω_0.plot_contours(ax[0], color="blue")

Ω_z.imshow(ax[1], title="Ω_z")
Ω_0.imshow(ax[2], title="Ω_0")

fig.suptitle("")
plt.show()


class Parameters:
    r"""Parameter class for nuisance parameters.


    :code:`projection_shape` is the shape of the detector so usually this will
    be :code:`(N_pixels_x, N_pixels_y)` but it may be any shape including
    singleton shape.
    :code:`N_p` number of projections, :code:`np.prod(projection_shape) = N_p`.

    :code:`Y_o`, :code:`Y_s` measurements have shape :code:`projection_shape +  (N_A,)`

    :code:`N_A` is the number of measured TOF bins (TOA's)

    :code:`D` has shape :code:`(N_F, N_A)`

    :code:`N_F` is the number of theoretical TOF bins. :math:`N_F \geq N_A`

    :code:`ω_sz`, :code:`ω_s0` have shape :code:`projection_shape + (1,)`.
        :math:`ω_sz^\top` has shape :code:`(1,) + projection_shape`.

    :code:`R` has shape :code:`(N_F, N_A)`.



    """

    def __init__(
        self, Y_o, Y_s, R, D, Ω_z=None, Ω_0=None, N_b=5, β=1.0, optimization_params=None
    ):
        r"""
        Args:
            Y_o: Open beam measurement.
            Y_s: Sample measurement.
            D: Cross section dictionary.
            ω_sz: Uniformly dense region averaging vector.
            ω_s0: Open beam region averaging vector. When `None`,
                parameters will be computed equivalent to `β=0`.
            R: Resolution operator of class `ResolutionOperator`.
                When `None`, `R` is chosen internally as identity
                operator.
            β: Balancing weight between solving equation for `ω_sz`
                (`β=0`), and solving equation for `ω_s0` (`β` infinite).
                Equal weight when `β=1.0` (default).
        """

        # self.t_A = t_A
        # self.R = resolution.ResolutionOperator(Y_o.shape, kernels)
        # self.t_F = R.compute_t_F(self.t_A)

        # self.D = cross_section.XSDict(isotopes, t_F, flight_path_length)

        y_o = ""
        y_s0 = ""
        y_sz = ""

        self.v = ""
        P = ""

        apgm = ""

    def estmate(self):
        r"""Estimate Parameters"""

        apgm.solve()
        self.z = ""
        self.α_1 = ""
        self.α_2 = ""
        self.θ = ""

        self.ϕ = ""
        self.b = ""

    def save(self, file_name):
        r"""Save Parameters"""

    def load(self, file_name):
        r"""Load Parameters"""


y_sz = Ω_z.average(Y_s).T
y_s0 = Ω_0.average(Y_s).T

fig, ax = plt.subplots(1, 1, figsize=[12, 8], sharex=False)
ax = np.atleast_1d(ax)
ax[0].plot(t_A, y_sz.flatten(), label="y_sz", alpha=0.75)
ax[0].plot(t_A, y_s0.flatten(), label="y_s", alpha=0.75)
ax[0].legend(prop={"size": 8})
ax[0].set_xlabel(util.TOF_LABEL)

plt.show()


par = Parameters(Y_o, Y_s, R, D)
