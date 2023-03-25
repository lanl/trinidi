"""Some reconstruction functions and classes."""

import jax
import jax.config
import matplotlib.pyplot as plt
import numpy as np
import scico.numpy as snp
from scico.functional import (
    Functional,
    NonNegativeIndicator,
    SeparableFunctional,
    ZeroFunctional,
)
from scico.loss import SquaredL2Loss
from scico.numpy import BlockArray
from scico.operator import Operator
from scico.optimize.pgm import AcceleratedPGM, RobustLineSearchStepSize

from trinidi import cross_section, resolution, util

jax.config.update("jax_enable_x64", True)


class ProjectionRegion:
    r"""ProjectionRegion class.

    This class is used to handle the Ω regions indicating the regions where the sample image
    has no areal density, or uniformly dense areal density. Used for averaging over many pixels.

    """

    def __init__(self, projection_mask):
        r"""Initialize a ProjectionRegion object.

        If the counting data, `Y`, has shape `projection_shape + (N_A,)`, the `projection_mask`
        array must have size `projection_shape + (1,)`.

        Args:
            projection_mask (boolean or binary array): `projection_mask` array used for averaging.
        """

        self.projection_shape = projection_mask.shape[:-1]
        self.mask = (projection_mask > 0) * 1.0
        self.ω = self.mask / self.mask.sum()

    def average(self, Y):
        r"""Compute :math:`\omega^\top Y`,

            where :math:`\omega` has shape `projection_shape + (1,)` and :math:`Y` has shape
            `projection_shape + (N_A,)`, thus the result has shape `(1, N_A)`.

        Args:
            Y: :math:`Y` array.

        Returns:
            The :math:`\omega^\top Y` array.
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


class FunctionalSum(Functional):
    r"""A sum of two functionals."""

    def __repr__(self):
        return (
            "Sum of functionals of types "
            + str(type(self.functional1))
            + " and "
            + str(type(self.functional2))
        )

    def __init__(self, functional1: Functional, functional2: Functional):
        self.functional1 = functional1
        self.functional2 = functional2
        self.has_eval = functional1.has_eval and functional2.has_eval
        self.has_prox = False
        super().__init__()

    def __call__(self, x) -> float:
        return self.functional1(x) + self.functional2(x)


class Forward_zα1α2θ(Operator):
    r"""
    Pseudo forward operator
    f(z,ε,μ,θ) = ε[ {c_o - b(θ)} * q(z) +  μ b(θ)]

    c_o = (ω' C_o) / (ω'v)
    q(z) = exp(-z D) R
    b(θ) = exp(θ P)

    This gets fitted to
    c_s = (ω' C_s) / (ω'v)
    """

    def __init__(self, input_shape, y_o, D, R, P, jit: bool = True):
        self.y_o = y_o
        self.D = jax.device_put(D.values)
        self.R = R.single
        self.P = jax.device_put(P)

        super().__init__(
            input_shape=input_shape,
            jit=jit,
        )

    def _eval(self, zα1α2θ):
        z = zα1α2θ[0]
        α_1 = zα1α2θ[1]
        α_2 = zα1α2θ[2]
        θ = zα1α2θ[3]
        b = (snp.exp(θ.T @ self.P)).T
        q = self.R(snp.exp(-z.T @ self.D)).T
        y_o = self.y_o
        return α_1 * ((y_o - b) * q + α_2 * b)


class Forward_0α1α2θ(Operator):
    r"""
    Pseudo forward operator
    f(0,ε,μ,θ) = ε[ {c_o - b(θ)} * q(z) +  μ b(θ)]
               = ε[ c_o +  (μ-1) b(θ)]

    c_o = (ω' C_o) / (ω'v)
    q(0) = exp(-z D) R = 1
    b(θ) = exp(θ P)

    This gets fitted to
    c_s = (ω' C_s) / (ω'v)

    """

    def __init__(self, input_shape, y_o, P, jit: bool = True):
        self.y_o = y_o
        self.P = jax.device_put(P)

        super().__init__(
            input_shape=input_shape,
            jit=jit,
        )

    def _eval(self, zα1α2θ):
        # z = zα1α2θ[0] # not used
        α_1 = zα1α2θ[1]
        α_2 = zα1α2θ[2]
        θ = zα1α2θ[3]
        b = (snp.exp(θ.T @ self.P)).T
        y_o = self.y_o
        return α_1 * (y_o + (α_2 - 1) * b)


import copy


class Preconditioner:
    """(Z) (D)  = (ZC) (C_inverse D)
    original   conditioned
    where (C_inverse D) has unit norm rows

    C = np.diag(np.diag(np.matmul(D, D.T)))**0.5
    """

    def __init__(self, D):
        self._C = np.diag(np.diag(np.matmul(D.values, D.values.T))) ** 0.5
        self._C_inverse = np.linalg.inv(self._C)
        self.D_conditioned = copy.deepcopy(D)
        self.D_conditioned.values = self._C_inverse @ D.values

    def condition_Z(self, Z):
        r"""Condition Z"""
        return Z @ self._C

    def uncondition_Z(self, Z_conditioned):
        r"""Uncondition Z"""
        return Z_conditioned @ self._C_inverse

    def condition_zα1α2θ(self, zα1α2θ):
        r"""Condition zα1α2θ"""
        zα1α2θ_conditioned = zα1α2θ.copy()
        zα1α2θ_conditioned[0] = self.condition_Z(zα1α2θ[0].T).T
        return zα1α2θ_conditioned

    def uncondition_zα1α2θ(self, zα1α2θ_conditioned):
        r"""Uncondition zα1α2θ"""
        zα1α2θ = zα1α2θ_conditioned.copy()
        zα1α2θ[0] = self.uncondition_Z(zα1α2θ_conditioned[0].T).T
        return zα1α2θ


Δt = 0.90
t_A = np.arange(72, 400, Δt)
N_A = t_A.size
flight_path_length = 10

kernels = [np.array([1]), np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])]

isotopes = ["U-238", "Pu-239", "Ta-181"]
projection_shape = (31, 31)


output_shape = projection_shape + (N_A,)

R = resolution.ResolutionOperator(output_shape, t_A, kernels=kernels)
t_F = R.t_F
D = cross_section.XSDict(isotopes, t_F, flight_path_length)


ϕ, b, θ, α_1, α_2 = util.generate_spectra(t_A, acquisition_time=10)


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
# plt.show()


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
# plt.show()


class Parameters:
    r"""Parameter class for nuisance parameters.
    :code:`projection_shape` is the shape of the detector so usually this will
    be :code:`(N_pixels_x, N_pixels_y)` but it may be any shape including    singleton shape.
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
        self, Y_o, Y_s, R, D, Ω_z, Ω_0=None, N_b=5, β=1.0, non_negative_z=False
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

        self.R = R
        self.t_A = self.R.t_A
        self.t_F = self.R.t_F
        self.D = D
        self._pc = Preconditioner(D)
        self._D_conditioned = self._pc.D_conditioned

        self.P = util.background_basis(N_b, self.t_A.size)

        self.Y_o = Y_o
        self.Y_s = Y_s
        projection_shape = Y_s.shape[:-1]

        # Ω_o: average all projections (1')
        Ω_o = ProjectionRegion(np.ones(projection_shape + (1,)))
        self.Ω_z = Ω_z
        self.Ω_0 = Ω_0  # This may be None, equivalent to β=0

        # v = (Y_o 1/N_p) / (1'/N_p Y_o 1/N_A) where 1 is a vector of ones.
        self.v = np.mean(self.Y_o, axis=-1, keepdims=True) / np.mean(self.Y_o)

        # --- Averages y_o, y_sz
        # y_o' = (1' Y_o) / (1' v)
        self.y_o = (Ω_o.average(self.Y_o) / Ω_o.average(self.v)).T
        self.y_sz = (self.Ω_z.average(self.Y_s) / self.Ω_z.average(self.v)).T

        # --- Initialization
        self.zα1α2θ_init = self._initialize(
            self.Ω_0, self.Y_o, self.Y_s, self.y_sz, self.y_o, self.D, self.R, self.P
        )
        self._zα1α2θ_init_conditioned = self._pc.condition_zα1α2θ(self.zα1α2θ_init)

        # --- When not Ω_0:
        self._forward_z_conditioned = Forward_zα1α2θ(
            self.zα1α2θ_init.shape, self.y_o, self._D_conditioned, self.R, self.P
        )
        self.forward_z = Forward_zα1α2θ(
            self.zα1α2θ_init.shape, self.y_o, self.D, self.R, self.P
        )

        fz = SquaredL2Loss(y=jax.device_put(self.y_sz), A=self._forward_z_conditioned)
        fz.is_smooth = True

        if self.Ω_0:
            # f = || y_sz - f(z, α_1, α_2, θ) ||^2  +  || y_s0 - f(0, α_1, α_2, θ) ||^2
            self.forward_0 = Forward_0α1α2θ(self.zα1α2θ_init.shape, self.y_o, self.P)
            self.y_s0 = (self.Ω_0.average(Y_s) / self.Ω_0.average(self.v)).T
            f0 = SquaredL2Loss(y=jax.device_put(self.y_s0), A=self.forward_0)
            f0.is_smooth = True
            f = FunctionalSum(fz, f0)

        else:
            # f = || y_sz - f(z, α_1, α_2, θ) ||^2
            self.forward_0 = None
            self.y_s0 = None
            f = fz

        # --- g constraints
        if non_negative_z:
            gz = NonNegativeIndicator()
        else:
            gz = ZeroFunctional()

        gα1 = NonNegativeIndicator()
        gα2 = NonNegativeIndicator()
        gθ = ZeroFunctional()

        g = SeparableFunctional([gz, gα1, gα2, gθ])

        step_size = RobustLineSearchStepSize()
        L0 = 1e-5
        self.apgm = AcceleratedPGM(
            f=f,
            g=g,
            L0=L0,
            x0=self._zα1α2θ_init_conditioned,
            step_size=step_size,
            itstat_options={"display": True, "period": 10},
        )

    def solve(self, iterations=100):
        r"""Find parameters."""
        self.apgm.maxiter = iterations
        zα1α2θ_conditioned = self.apgm.solve()
        self.zα1α2θ = self._pc.uncondition_zα1α2θ(zα1α2θ_conditioned)
        return self.zα1α2θ

    def _initialize(self, Ω_0, Y_o, Y_s, y_sz, y_o, D, R, P):
        r"""Initialize parameters."""
        α_2 = 1

        projection_shape = Y_s.shape[:-1]
        Ω_o = ProjectionRegion(
            np.ones(projection_shape + (1,))
        )  # average all projections (1')

        if Ω_0:
            # (ω_0' Y_s 1) / (ω_0' Y_o 1)
            α_1 = np.sum(Ω_0.average(Y_s)) / np.sum(Ω_0.average(Y_o))

        else:
            # (1' Y_s 1) / (1' Y_o 1)
            α_1 = np.sum(Ω_o.average(Y_s)) / np.sum(Ω_o.average(Y_o))

        temp = np.min(y_sz / y_o) * y_o / (α_1 * α_2)
        θ = (np.log(temp.T) @ np.linalg.pinv(P)).T
        b = (np.exp(θ.T @ P)).T

        q = np.abs(util.no_nan_divide(y_sz / α_1 - α_2 * b, y_o - b))
        DR = R.call_on_any_array(D.values)
        z = (-np.log(q.T) @ np.linalg.pinv(DR)).T

        cast = lambda x: np.require(x, dtype=np.float64)
        return BlockArray([cast(z), cast(α_1), cast(α_2), cast(θ)])

    def plot_regions(self):
        r"""Plot Ω regions and corresponding spectra"""
        meas_ratio = np.sum(self.Y_s, axis=-1) / np.sum(self.Y_o, axis=-1)

        N = 4 if self.Ω_0 else 3
        fig, ax = plt.subplots(1, N, figsize=[15, 4], sharex=True)
        ax = np.atleast_1d(ax)

        im = ax[0].imshow(self.v[:, :, 0], vmin=0)
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("v")

        im = ax[1].imshow(meas_ratio, vmin=0)
        fig.colorbar(im, ax=ax[1])
        self.Ω_z.plot_contours(ax[1], color="red")
        if self.Ω_0:
            self.Ω_0.plot_contours(ax[1], color="blue")
        ax[1].set_title("1Y_s / 1Y_o")

        im = self.Ω_z.imshow(ax[2], title="Ω_z")
        fig.colorbar(im, ax=ax[2])

        if self.Ω_0:
            im = self.Ω_0.imshow(ax[3], title="Ω_0")
            fig.colorbar(im, ax=ax[3])

        fig.suptitle("Selected Regions Ω_z, Ω_0")

        fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
        ax = np.atleast_1d(ax)
        ax[0].plot(self.t_A, self.y_o.flatten(), label="y_o", alpha=0.75, color="green")
        ax[0].plot(self.t_A, self.y_sz.flatten(), label="y_sz", alpha=0.75, color="red")
        if self.Ω_0:
            ax[0].plot(
                self.t_A, self.y_s0.flatten(), label="y_s0", alpha=0.75, color="blue"
            )
        ax[0].legend(prop={"size": 8})
        ax[0].set_xlabel(util.TOF_LABEL)
        ax[0].set_title("Averaged Measurements")

        self.D.plot(ax[1])

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


par = Parameters(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0)

zα1α2θ = par.solve(iterations=50000)


par.plot_regions()
plt.show()
