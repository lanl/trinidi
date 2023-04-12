"""Some reconstruction functions and classes."""

import copy

import numpy as np

import jax
import jax.config

import matplotlib.pyplot as plt
import scico.numpy as snp
from scico.functional import (
    Functional,
    NonNegativeIndicator,
    SeparableFunctional,
    ZeroFunctional,
)
from scico.loss import PoissonLoss, SquaredL2Loss
from scico.numpy import BlockArray
from scico.operator import Operator
from scico.optimize.pgm import AcceleratedPGM, RobustLineSearchStepSize
from scipy.optimize import minimize

from trinidi import resolution, util

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
    r"""Pseudo forward operator

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


class Forward_zα1α2θ_arr:
    r"""Pseudo forward operator

    f(z,ε,μ,θ) = ε[ {c_o - b(θ)} * q(z) +  μ b(θ)]

    c_o = (ω' C_o) / (ω'v)
    q(z) = exp(-z D) R
    b(θ) = exp(θ P)

    This gets fitted to
    c_s = (ω' C_s) / (ω'v)
    """

    def __init__(self, N_m, y_o, D, R, P, jit: bool = True):
        self.y_o = y_o
        self.D = jax.device_put(D.values)
        self.R = R.single
        self.P = jax.device_put(P)
        self.N_m = N_m

    def __call__(self, zα1α2θ):
        N_m = self.N_m
        z = zα1α2θ[0:N_m][:, np.newaxis]
        α_1 = zα1α2θ[N_m : N_m + 1]
        α_2 = zα1α2θ[N_m + 1 : N_m + 2]
        θ = zα1α2θ[N_m + 2 :][:, np.newaxis]
        b = (snp.exp(θ.T @ self.P)).T
        q = self.R(snp.exp(-z.T @ self.D)).T
        y_o = self.y_o
        return α_1 * ((y_o - b) * q + α_2 * b)


class Forward_0α1α2θ(Operator):
    r"""Pseudo forward operator

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


class Forward_0α1α2θ_arr:
    r"""Pseudo forward operator

    f(0,ε,μ,θ) = ε[ {c_o - b(θ)} * q(z) +  μ b(θ)]
               = ε[ c_o +  (μ-1) b(θ)]

    c_o = (ω' C_o) / (ω'v)
    q(0) = exp(-z D) R = 1
    b(θ) = exp(θ P)

    This gets fitted to
    c_s = (ω' C_s) / (ω'v)
    """

    def __init__(self, N_m, y_o, P, jit: bool = True):
        self.y_o = y_o
        self.P = jax.device_put(P)
        self.N_m = N_m

    def __call__(self, zα1α2θ):
        N_m = self.N_m
        # z = zα1α2θ[0:N_m][:,np.newaxis]
        α_1 = zα1α2θ[N_m : N_m + 1]
        α_2 = zα1α2θ[N_m + 1 : N_m + 2]
        θ = zα1α2θ[N_m + 2 :][:, np.newaxis]
        b = (snp.exp(θ.T @ self.P)).T
        y_o = self.y_o
        return α_1 * (y_o + (α_2 - 1) * b)


def background_basis(N_b, N_t):
    r"""Creates the (N_b, N_t) background basis matrix"""
    P = np.ones([N_b, N_t])
    x = np.linspace(np.exp(-1), np.exp(1), N_t)
    for i in range(N_b):
        P[i] = np.log(x) ** i
    for i in range(N_b):
        P[i] = P[i] / np.linalg.norm(P[i])

    return P


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


def _check_compatible_RDY(R, D, Y_list=None):
    r"""Check shapes compatibility between R and D; and list of Y (optional)"""
    if R.N_F != D.N_F:
        raise ValueError(
            f"ResolutionOperator and XSDict shapes are not compatible. {R.N_F=} while {D.N_F=}."
        )

    if Y_list is not None:
        Y = Y_list[0]

        if not all(Yi.shape == Y.shape for Yi in Y_list):
            raise ValueError(
                f"Not all shapes in Y_list are the same. Have: {[Yi.shape for Yi in Y_list]}"
            )

        if R.projection_shape != Y.shape[:-1]:
            raise ValueError(
                f"ResolutionOperator and counting data shapes are not compatible. {R.projection_shape=} while {Y.shape[:-1]=}."
            )

        if R.N_A != Y.shape[-1]:
            raise ValueError(
                f"ResolutionOperator and counting data shapes are not compatible. {R.N_A=} while {Y.shape[-1]=}."
            )


def _plot_convergence_common(iteration_history, plot_residual=True, value_gt=None, figsize=[6, 4]):
    r"""Plot convergence behaviour."""
    if iteration_history is None:
        raise ValueError(
            "Iteration history is None. Can only plot convergence after .solve() has been run."
        )

    Iter = np.array(iteration_history.Iter)
    Time = np.array(iteration_history.Time)
    Objective = np.array(iteration_history.Objective)
    L = np.array(iteration_history.L)
    Residual = np.array(iteration_history.Residual)

    if plot_residual:
        fig, ax = plt.subplots(2, 1, sharex="all", figsize=figsize)
        ax = np.atleast_1d(ax)
    else:
        fig, ax = plt.subplots(1, 1, sharex="all", figsize=figsize)
        ax = np.atleast_1d(ax)

    ax[0].semilogy(Objective, label="Objective", color="blue", linewidth=1)
    if value_gt is not None:
        array_gt = np.ones(len(Objective)) * value_gt
        ax[0].semilogy(array_gt, label="Objective(Ground Truth)", color="orange", linewidth=1)
        ax[0].set_title(f"Final Objective: {Objective[-1]:.4e} (Ground Truth: {value_gt:.4e})")
    else:
        ax[0].set_title(f"Final Objective: {Objective[-1]:.4e}")
    ax[0].legend()
    ax[0].set_xlabel("Iteration")

    if plot_residual:
        ax[1].semilogy(Residual, label="Residual", color="orange")
        ax[1].set_xlabel("Iteration")
        ax[1].legend()

    fig.suptitle(f"Convergence Plots")
    return fig, ax


class ParameterEstimator:
    r"""ParameterEstimator class for nuisance parameters.

    :code:`projection_shape` is the shape of the detector so usually this will
    be :code:`(N_pixels_x, N_pixels_y)` but it may be any shape including    singleton shape.
    :code:`N_p` number of projections, :code:`np.prod(projection_shape) = N_p`.
    :code:`Y_o`, :code:`Y_s` measurements have shape :code:`projection_shape +  (N_A,)`
    :code:`N_A` is the number of measured TOF bins (TOA's)
    :code:`D` has shape :code:`(N_F, N_A)`
    :code:`N_F` is the number of theoretical TOF bins. :math:`N_F \geq N_A`
    :code:`ω_sz`, :code:`ω_s0` have shape :code:`projection_shape + (1,)`. :math:`ω_sz^\top` has shape :code:`(1,) + projection_shape`.
    :code:`R` has shape :code:`(N_F, N_A)`.
    """

    def __repr__(self):
        parameter_dict = self._to_parameter_dict(self.zα1α2θ)
        z = parameter_dict["z"]
        α_1 = parameter_dict["α_1"]
        α_2 = parameter_dict["α_2"]
        θ = parameter_dict["θ"]
        return f"""{type(self)}
    z.T = {z.T}
    α_1 = {α_1}
    α_2 = {α_2}
    θ.T = {θ.T}
        """

    def __init__(
        self,
        Y_o,
        Y_s,
        R,
        D,
        Ω_z,
        Ω_0=None,
        N_b=5,
        non_negative_z=False,
        verbose=False,
        dispperiod=10,
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

        _check_compatible_RDY(R, D, Y_list=[Y_o, Y_s])

        self.R = R
        self.D = D
        self._pc = Preconditioner(self.D)
        self.N_m = self.D.N_m

        self.Y_o = Y_o
        self.Y_s = Y_s
        projection_shape = Y_s.shape[:-1]

        self.t_A = self.R.t_A
        self.N_b = N_b
        self.P = background_basis(N_b, self.t_A.size)

        # --- Averaging regions
        # Ω_o: average all projections (1')
        Ω_o = ProjectionRegion(np.ones(projection_shape + (1,)))
        if Ω_z.projection_shape != projection_shape:
            raise ValueError(
                f"Ω_z shape does not match. {Ω_z.projection_shape=} != {projection_shape=}"
            )
        self.Ω_z = Ω_z

        if Ω_0:
            if Ω_0.projection_shape != projection_shape:
                raise ValueError(
                    f"Ω_0 shape does not match. {Ω_0.projection_shape=} != {projection_shape=}"
                )
            self.Ω_0 = Ω_0
        else:
            self.Ω_0 = None  # equivalent to β=0

        # v = (Y_o 1/N_p) / (1'/N_p Y_o 1/N_A) where 1 is a vector of ones.
        self.v = np.mean(self.Y_o, axis=-1, keepdims=True) / np.mean(self.Y_o)

        # --- Averages y_o, y_sz, y_s0
        # y_o' = (1' Y_o) / (1' v)
        self.y_o = (Ω_o.average(self.Y_o) / Ω_o.average(self.v)).T
        # y_sz' = (ω_z' Y_s) / (ω_z' v)
        self.y_sz = (self.Ω_z.average(self.Y_s) / self.Ω_z.average(self.v)).T
        if self.Ω_0:
            # y_s0' = (ω_0' Y_s) / (ω_0' v)
            self.y_s0 = (self.Ω_0.average(Y_s) / self.Ω_0.average(self.v)).T
        else:
            self.y_s0 = None

        # --- Initialization
        self.zα1α2θ_init = self._initialize(
            self.Ω_0, self.Y_o, self.Y_s, self.y_sz, self.y_o, self.D, self.R, self.P
        )
        self.zα1α2θ = self.zα1α2θ_init.copy()

        # --- Forward operators
        zα1α2θ_init_conditioned = self._pc.condition_zα1α2θ(self.zα1α2θ_init)

        forward_z_conditioned = Forward_zα1α2θ(
            self.zα1α2θ_init.shape, self.y_o, self._pc.D_conditioned, self.R, self.P
        )
        self.forward_z = Forward_zα1α2θ(self.zα1α2θ_init.shape, self.y_o, self.D, self.R, self.P)
        if self.Ω_0:
            self.forward_0 = Forward_0α1α2θ(self.zα1α2θ_init.shape, self.y_o, self.P)
        else:
            self.forward_0 = None

        # --- Losses
        fz_conditioned = SquaredL2Loss(
            y=jax.device_put(self.y_sz), A=forward_z_conditioned, scale=1
        )
        fz = SquaredL2Loss(y=jax.device_put(self.y_sz), A=self.forward_z, scale=1)

        if self.Ω_0:
            # f = || y_sz - f(z, α_1, α_2, θ) ||^2  +  || y_s0 - f(0, α_1, α_2, θ) ||^2
            f0 = SquaredL2Loss(y=jax.device_put(self.y_s0), A=self.forward_0, scale=1)
            f_conditioned = FunctionalSum(fz_conditioned, f0)
            self.f = FunctionalSum(fz, f0)

        else:
            # f = || y_sz - f(z, α_1, α_2, θ) ||^2
            f_conditioned = fz_conditioned
            self.f = fz

        # --- g constraints
        if non_negative_z:
            gz = NonNegativeIndicator()
        else:
            gz = ZeroFunctional()

        gα1 = NonNegativeIndicator()
        gα2 = NonNegativeIndicator()
        gθ = ZeroFunctional()

        g = SeparableFunctional([gz, gα1, gα2, gθ])

        # --- Optimizer APGM
        self.apgm = AcceleratedPGM(
            f=f_conditioned,
            g=g,
            L0=1e-5,
            x0=zα1α2θ_init_conditioned,
            step_size=RobustLineSearchStepSize(),
            itstat_options={"display": True, "period": dispperiod},
        )

        # ----- BFGS

        aa = zα1α2θ_init_conditioned
        x0_pc = np.concatenate([aa[0][:, 0], aa[1][np.newaxis], aa[2][np.newaxis], aa[3][:, 0]])
        A_pc1 = Forward_zα1α2θ_arr(self.N_m, self.y_o, self._pc.D_conditioned, self.R, self.P)
        A_pc2 = Forward_0α1α2θ_arr(self.N_m, self.y_o, self.P)

        f1_arr = lambda x: np.sum((self.y_sz - A_pc1(x)) ** 2)
        if self.Ω_0:
            f2_arr = lambda x: np.sum((self.y_s0 - A_pc2(x)) ** 2)
            f_arr = lambda x: f1_arr(x) + f2_arr(x)
        else:
            f_arr = f1_arr

        x_pc = minimize(f_arr, x0_pc, method="BFGS", options={"disp": verbose})

        z_pc = x_pc.x[0 : self.N_m][:, np.newaxis]
        α_1 = float(x_pc.x[self.N_m : self.N_m + 1])
        α_2 = float(x_pc.x[self.N_m + 1 : self.N_m + 2])
        θ = x_pc.x[self.N_m + 2 :][:, np.newaxis]

        z = (z_pc.T @ self._pc._C_inverse).T

        self.zα1α2θ_bfgs = self._to_zα1α2θ(z=z, α_1=α_1, α_2=α_2, θ=θ)
        self.zα1α2θ = self._to_zα1α2θ(z=z, α_1=α_1, α_2=α_2, θ=θ)

        self.iteration_history = None

    def _initialize(self, Ω_0, Y_o, Y_s, y_sz, y_o, D, R, P):
        r"""Initialize parameters."""
        α_2 = 1

        projection_shape = Y_s.shape[:-1]
        Ω_o = ProjectionRegion(np.ones(projection_shape + (1,)))  # average all projections (1')

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

        zα1α2θ_init = self._to_zα1α2θ(z=z, α_1=α_1, α_2=α_2, θ=θ)

        return zα1α2θ_init

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

    def plot_results(self):
        r"""Add docstring here."""

        y_sz = self.y_sz
        y_o = self.y_o
        y_s0 = self.y_s0

        z = self.get()["z"]
        α_1 = self.get()["α_1"]
        α_2 = self.get()["α_2"]
        θ = self.get()["θ"]
        b = (snp.exp(θ.T @ self.P)).T
        ϕ = self.y_o - b

        t_A = self.t_A

        fitz = self.forward_z(self.zα1α2θ)
        fit0 = self.forward_0(self.zα1α2θ)

        efbg = α_1 * α_2 * b

        fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
        ax = np.atleast_1d(ax)

        if y_s0 is not None:
            ax[0].plot(
                t_A,
                y_s0.flatten(),
                label="Ω_0 Measurement, y_s0",
                alpha=0.3,
                color="tab:orange",
                linewidth=3,
            )
            lab1 = "Effective Open Beam (Fit of y_s0)"
        else:
            lab1 = "Effective Open Beam"

        lab2 = "Effective Background"
        ax[0].plot(t_A, fit0.flatten(), "--", label=lab1, color="orange", linewidth=1)
        ax[0].plot(t_A, y_sz.flatten(), label="y_sz", alpha=0.3, color="tab:blue", linewidth=3)
        ax[0].plot(t_A, fitz.flatten(), "--", label="Fit of y_sz", color="tab:blue", linewidth=1)
        ax[0].plot(t_A, efbg.flatten(), "--", label=lab2, color="tab:red", linewidth=1)
        ax[0].legend(prop={"size": 8})
        ax[0].set_xlabel("TOF [μs]")

        self.D.plot(ax[1])

    def apgm_plot_convergence(self, plot_residual=True, ground_truth=None, figsize=[6, 4]):
        r"""Plot convergence behaviour."""

        if ground_truth:
            zα1α2θ = self._to_zα1α2θ(**ground_truth)
            value_gt = float(self.f(zα1α2θ))  # scalar
        else:
            value_gt = None

        fig, ax = _plot_convergence_common(
            self.iteration_history, plot_residual=plot_residual, value_gt=value_gt, figsize=figsize
        )

        return fig, ax

    def apgm_solve(self, iterations=100):
        r"""Find parameters."""
        self.apgm.maxiter = iterations

        zα1α2θ_conditioned = self.apgm.solve()
        self.iteration_history = self.apgm.itstat_object.history(transpose=True)

        self.zα1α2θ = self._pc.uncondition_zα1α2θ(zα1α2θ_conditioned)

    def get(self):
        r"""Get Parameters."""
        parameter_dict = self._to_parameter_dict(self.zα1α2θ)
        return parameter_dict

    def set(self, z=None, α_1=None, α_2=None, θ=None):
        r"""Set Parameters."""
        zα1α2θ = self._to_zα1α2θ(z=z, α_1=α_1, α_2=α_2, θ=θ)
        self._check_zα1α2θ_shape(zα1α2θ)
        self.zα1α2θ = zα1α2θ
        self.apgm.x = self._pc.condition_zα1α2θ(self.zα1α2θ)

    def save(self, file_name):
        r"""Save Parameters to file."""
        parameter_dict = self._to_parameter_dict(self.zα1α2θ)
        np.save(file_name, parameter_dict, allow_pickle=True)

    def load(self, file_name):
        r"""Load Parameters from file."""
        parameter_dict = np.load(file_name, allow_pickle=True)[()]
        self.set(**parameter_dict)

    def _to_zα1α2θ(self, z=None, α_1=None, α_2=None, θ=None):
        """Convert parameter_dict to zα1α2θ."""
        cast = lambda x: jax.device_put(np.require(x, dtype=np.float64))
        to_z = cast(z).reshape([-1, 1])
        to_α_1 = cast(α_1)
        to_α_2 = cast(α_2)
        to_θ = cast(θ).reshape([-1, 1])
        zα1α2θ = BlockArray([to_z, to_α_1, to_α_2, to_θ])
        return zα1α2θ

    def _to_parameter_dict(self, zα1α2θ):
        """Convert zα1α2θ to parameter_dict."""
        cast = lambda x: np.require(x, dtype=np.float64)
        z = cast(zα1α2θ[0])
        α_1 = float(zα1α2θ[1])
        α_2 = float(zα1α2θ[2])
        θ = cast(zα1α2θ[3])
        parameter_dict = {"z": z, "α_1": α_1, "α_2": α_2, "θ": θ}
        return parameter_dict

    def _check_zα1α2θ_shape(self, zα1α2θ):
        """Check whether zα1α2θ has compatible shape to stored selzα1α2θ."""
        if not self.zα1α2θ.shape == zα1α2θ.shape:
            raise ValueError(
                f"Incompatible shapes of zα1α2θ. Internal shape is {self.zα1α2θ.shape} but new shape is {zα1α2θ.shape}."
            )


class Forward_Z(Operator):
    r"""Pseudo forward operator
    F(Z) = ε (Φ * Q(Z) + B_s)

    Q(Z) = exp(-Z D) R
    B_s = μ B
    """

    def __init__(self, input_shape, α_1, α_2, Φ, B, D, R, jit: bool = True):
        self.α_1 = jax.device_put(α_1)
        self.α_2 = jax.device_put(α_2)
        self.Φ = jax.device_put(Φ)
        self.D = jax.device_put(D.values)
        self.B = jax.device_put(B)
        self.R = R

        super().__init__(
            input_shape=input_shape,
            jit=jit,
        )

    def _eval(self, Z):
        Q = self.R(snp.exp(-Z @ self.D))
        return self.α_1 * (self.Φ * Q + self.α_2 * self.B)


class DensityEstimator:
    r"""DensityEstimator class"""

    def __init__(
        self,
        Y_s,
        par,
        D=None,
        R=None,
        non_negative_Z=False,
        projection_transform=None,
        dispperiod=10,
    ):
        """Initialize an DensityEstimator object."""

        if D is None:
            self.D = par.D

        if R is None:
            self.R = par.R

        if projection_transform:
            v = projection_transform(par.v)
            Y_s = projection_transform(Y_s)
            self.R = resolution.ResolutionOperator(Y_s.shape, self.R.t_A, self.R.kernels)
        else:
            v = par.v

        self._pc = Preconditioner(self.D)

        _check_compatible_RDY(self.R, self.D, Y_list=[Y_s])
        if v.shape[:-1] != Y_s.shape[:-1]:
            raise ValueError(f"{v.shape[:-1]=} and {Y_s.shape[:-1]=} shapes are not compatible.")
        projection_shape = self.R.projection_shape

        d = par.get()
        z = d["z"]
        α_1 = d["α_1"]
        α_2 = d["α_2"]
        θ = d["θ"]

        b = (snp.exp(θ.T @ par.P)).T
        ϕ = par.y_o - b

        B = v @ b.T
        Φ = v @ ϕ.T

        self.Z_init = self._initialize(Φ, Y_s, α_1, α_2, self.D, self.R, B)
        Z_init_conditioned = self._initialize(Φ, Y_s, α_1, α_2, self.D, self.R, B)
        self.Z = self.Z_init.copy()

        self.forward_Z = Forward_Z(self.Z_init.shape, α_1, α_2, Φ, B, self.D, self.R)
        _forward_Z_conditioned = Forward_Z(
            self.Z_init.shape, α_1, α_2, Φ, B, self._pc.D_conditioned, self.R
        )

        f_conditioned = PoissonLoss(y=jax.device_put(Y_s), A=_forward_Z_conditioned)
        self.f = PoissonLoss(y=jax.device_put(Y_s), A=self.forward_Z)

        if non_negative_Z:
            g = NonNegativeIndicator()
        else:
            g = ZeroFunctional()

        self.apgm = AcceleratedPGM(
            f=f_conditioned,
            g=g,
            L0=1e-5,
            x0=Z_init_conditioned,
            step_size=RobustLineSearchStepSize(),
            itstat_options={"display": True, "period": dispperiod},
        )

        self.iteration_history = None

    def _initialize(self, Φ, Y_s, α_1, α_2, D, R, B):
        r"""Initialize Z."""
        Q = np.abs(util.no_nan_divide(Y_s / α_1 - α_2 * B, Φ))
        Q[Q < 1e-5] = 1e-5  # this is to prevent log(0)
        DR = R.call_on_any_array(D.values)
        Z = -np.log(Q) @ np.linalg.pinv(DR)
        return jax.device_put(Z)

    def plot_convergence(self, plot_residual=True, ground_truth=None, figsize=[6, 4]):
        r"""Plot convergence behaviour."""

        if ground_truth is not None:
            value_gt = float(self.f(ground_truth))  # scalar

        fig, ax = _plot_convergence_common(
            self.iteration_history, plot_residual=plot_residual, value_gt=value_gt, figsize=figsize
        )

        return fig, ax

    def solve(self, iterations=100):
        r"""Find parameters."""
        self.apgm.maxiter = iterations

        Z_conditioned = self.apgm.solve()
        self.iteration_history = self.apgm.itstat_object.history(transpose=True)

        self.Z = self._pc.uncondition_Z(Z_conditioned)
        return self.Z
