#!/usr/bin/env python
# coding: utf-8

"""
Demo: trinidi.reconstruct Module
==================================

This script illustrates the functionality of the `trinidi.reconstruct` submodule.
"""

import numpy as np

import matplotlib.pyplot as plt

from trinidi import cross_section, reconstruct, resolution, util

"""
Generation of Simulated data
----------------------------
We use the function below to generate the data. The details of this function are not necessarily
important for the user to understand.
"""


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

    ϕ, b, θ, α_1, α_2 = util.generate_spectra(t_A, acquisition_time=10)
    N_b = θ.size

    D = cross_section.XSDict(isotopes, t_F, flight_path_length)

    Z = util.rose_phantom(projection_shape[0], num_circles=z.size, radius=2 / 3) * z.reshape(
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


"""
Below we define what isotopes to use for the phantom and what their corresponding densities should
become.
"""

isotopes = ["U-238", "Pu-239", "Ta-181"]
z = np.array([[0.005, 0.003, 0.004]]).T


# projection_shape = (20, 20) # use this for fast execution
projection_shape = (60, 60)  # use this for the docs

Y_o, Y_s, ground_truth_params, Z, t_A, flight_path_length, N_b, kernels = generate_sample_data(
    isotopes, z, acquisition_time=10, projection_shape=projection_shape
)

"""
The arrays `Y_o` and `Y_s` contain the the open beam and sample neutron count measurements.
The first (two) axes correspond to `projection_shape`, which in most cases is equal to the
detector shape. The last axis correponds to the time-of-arrival (TOA) dimension which has
`N_A` bins.
"""
print(f"{Y_o.shape = }     (Open beam measurement)")
print(f"{Y_s.shape = }     (Sample measurement)")
print()
projection_shape = Y_o.shape[:-1]
print(f"{projection_shape = }     (Shape of the detector)")
print()
N_A = Y_o.shape[-1]
print(f"{N_A = }     (Number of TOA bins)")


"""
The corresponding array of TOAs (`t_A`) is in units of [μs] and has the number of elements, `N_A`.
"""
print(f"t_A = [{t_A[0]:.2f}, {t_A[1]:.2f}, ..., {t_A[-2]:.2f}, {t_A[-1]:.2f}] [μs]")
print(f"{t_A.shape = }")


"""
The `flight_path_length` has units of [m] a relates the neutron times with the neutron energies.
"""
print(f"{flight_path_length = } [m]")
E = util.time2energy(t_A, flight_path_length)
print(f"E = [{E[0]:.2f}, {E[1]:.2f}, ..., {E[-2]:.2f}, {E[-1]:.2f}] [eV]")


"""
The ground truth areal densities `Z` are used to compare to our estimates. `Z` has the same
`projection_shape` as the measurements. The last axis of `Z` is the number of isotopes.
"""
print(f"{Z.shape = }     (Ground truth areal densities)")

N_m = Z.shape[-1]
print(f"{N_m = }     (Number of isotopes)")
print(f"{len(isotopes) = }")


"""
We can show the areal densities `Z` using the plotting function below. The phantom consists of
several discs and disc `i` has the density `z[i]` and corresponds to `isotope[i]`.
"""


def plot_densities(fig, ax, Z, isotopes, vmaxs=None):
    r"""Show areal densities. `ax` must be list."""

    for i, isotope in enumerate(isotopes):
        z = Z[:, :, i]
        if vmaxs is None:
            vmax = np.percentile(z, 99.9)
        else:
            vmax = vmaxs[i]
        vmin = 0
        im = ax[i].imshow(z, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[i], format="%.1e")
        ax[i].set_title(f"{isotope}")
        ax[i].axis("off")

    fig.suptitle("Areal Densities [mol/cm²]")

    return fig, ax


fig, ax = plt.subplots(1, N_m, figsize=[12, 3.3])
ax = np.atleast_1d(ax)
plot_densities(fig, ax, Z, isotopes, vmaxs=z * 1.5)
plt.show()


print(f"{z.T = } [mol/cm²]")
print(f"{isotopes = }")


"""
Preparation for the Reconstruction
----------------------------------

For the reconstruction we need to define the regions `Ω_z` and `Ω_0` that correspond to the uniformly
dense region and the open beam region, respectively.

For this we use the `ProjectionRegion` class. We initialize it with a mask (boolean array) that is
of shape `projection_shape + (1,)`, indicating which pixels belong to these regions.

In this example we use the ground truth `Z` to find these regions as
 - the overlap of all discs (Ω_z)
 - the complement of the union of all discs (Ω_0).

(In the case when the ground truth is not known, the user will have to find a way to define these
regions.)
"""

Ω_z = reconstruct.ProjectionRegion(np.prod(Z, axis=2, keepdims=True) > 0)
Ω_0 = reconstruct.ProjectionRegion(np.sum(Z, axis=2, keepdims=True) == 0)


"""
We can illustrate these regions using the builtin `ProjectionRegion.plot_contour`
function and the `ProjectionRegion.imshow` function below.
"""

fig, ax = plt.subplots(1, 3, figsize=[14, 4])
ax = np.atleast_1d(ax)

ax[0].imshow(np.sum(Y_s, axis=-1) / np.sum(Y_o, axis=-1), vmin=0)
ax[0].set_title("1Y_s / 1Y_o (TOF integrated measurement ratio)")
Ω_z.plot_contours(ax[0], color="red")
Ω_0.plot_contours(ax[0], color="blue")

Ω_z.imshow(ax[1], title="Ω_z")
Ω_0.imshow(ax[2], title="Ω_0")

plt.show()

"""
Next we need to define the resolution operator, `R`. We use the same `kernels` used for the
generation of the data to initialize the `ResolutioOperator` object.
"""

R = resolution.ResolutionOperator(Y_s.shape, t_A, kernels=kernels)
print(R)

"""
Since the resolution operator inputs wider spectra than it outputs, there is an additional vector
of time-of-flights (TOF), `t_F`, with size `N_F`. The `t_F` array is calibrated such that
approximately R(t_A) = t_F.
"""

t_F = R.t_F
N_F = t_F.size

print(f"{N_A = }")
print(f"t_A = [{t_A[0]:.2f}, {t_A[1]:.2f}, ..., {t_A[-2]:.2f}, {t_A[-1]:.2f}] [μs]")
print(f"{N_F = }")
print(f"t_F = [{t_F[0]:.2f}, {t_F[1]:.2f}, ..., {t_F[-2]:.2f}, {t_F[-1]:.2f}] [μs]")


"""
The cross section dictionary uses `t_F` as calibration for the neutron energies. Below we define
the `XSDict` object.
"""

D = cross_section.XSDict(isotopes, t_F, flight_path_length)
print(D)

"""
Below we plot the average measurements in the `Ω_z` and `Ω_0` regions compared to the cross section
dictionary. We use the `ProjectionRegion.averge` function to compute these average spectra.

(Note that in `trinidi` we define vecors with `N` elements to have shape `(N,1)` and thus we
occasionally have to explicitly flatten the arrays for correct plotting etc.)
"""

fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
ax = np.atleast_1d(ax)
ax[0].plot(t_A, Ω_z.average(Y_s).flatten(), label="Ω_z.average(Y_s)", alpha=0.75)
ax[0].plot(t_A, Ω_0.average(Y_s).flatten(), label="Ω_0.average(Y_s)", alpha=0.75)
D.plot(ax[1])
ax[0].legend(prop={"size": 8})
fig.suptitle("Average spectra in Ω_z and Ω_0")
plt.show()


"""
Reconstruction of the Nuisance Parameters
-----------------------------------------
"""


"""
We define a `ParameterEstimator` object. For the background we use the same number `N_b` of basis
functions. This estimates the nuisance parameters (z, α_1, α_1, θ). Novice users do not need to
worry about the values and handling of these parameters.
"""
par = reconstruct.ParameterEstimator(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=N_b)


"""
`Advanced users`: The parameters are available through the `ParameterEstimator.get` function.
"""
d = par.get()
print(d)


"""
`Advanced users`: The parameters can manually be modified using the `ParameterEstimator.set` function.
We also provide similar `save` and `load` functions to write/read them to file.
"""
par.set(**d)
par.set(z=d["z"], α_1=d["α_1"], α_2=d["α_2"], θ=d["θ"])  # same as line above

# par.save("par.npy")
# d = par.load("par.npy")

"""
Similar to above we can easily plot the `Ω_z` and `Ω_0` regions.
"""
par.plot_regions()
plt.show()

"""
The `ParameterEstimator.plot_results` function allows to display the resulting spectra from the
estimation.
"""
par.plot_results()
plt.show()


"""
We also provide the APGM optimization routine, however we do not recommend it since tends to be
signicicantly slower.
"""

# if False:
#     par.apgm_solve(iterations=100)
#     fig, ax = par.apgm_plot_convergence(plot_residual=True, ground_truth=ground_truth_params)
#     plt.show()

# par.set(**d)


"""
Reconstruction of the Areal Densities
-------------------------------------
"""

"""
We use this simple plotting function below to keep things clean.
"""


def plot_compare(Z, str_Z, Z_hat, str_Z_hat):
    r"""Generate two plots comparing ground truth with reconstruction."""
    fig, ax = plt.subplots(1, N_m, figsize=[12, 3.3])
    ax = np.atleast_1d(ax)
    plot_densities(fig, ax, Z, isotopes, vmaxs=z * 1.5)
    fig.suptitle(f"{str_Z} [mol/cm²]")

    fig, ax = plt.subplots(1, N_m, figsize=[12, 3.3])
    ax = np.atleast_1d(ax)
    plot_densities(fig, ax, Z_hat, isotopes, vmaxs=z * 1.5)
    fig.suptitle(f"{str_Z_hat} [mol/cm²]")


"""
We define a `DensityEstimator` object. Note that if the optional arguments `D` and `R` are left
empty, the `D` and `R` operators from the `par` `Parameters` object will be used for the
reconstruction.

(If different `D` and `R` operators are desired, they need to be passed to the `DensityEstimator`
constructor.)
"""
den = reconstruct.DensityEstimator(Y_s, par, non_negative_Z=False, dispperiod=50)

"""
We reonstruct the areal densities using the `DensityEstimator.solve` function.
"""
Z_hat = den.solve(iterations=200)


"""
We can plot the convergence of the objective using the `DensityEstimator.plot_convergence`
function. The ground truth argument is optional and will additionaly display the objective to the
ground truth.
"""

den.plot_convergence(ground_truth=Z)
plt.show()

"""
We plot the resulting areal density estimates below and compare them to the ground truth.
"""

plot_compare(Z, "Ground Truth", Z_hat, "Reconstruction")
plt.show()


"""
Cropped or Binned Measurements
------------------------------

In practice, it is not always desired to reconstruct the full field of view or at full resolution.
For this reason we provide the `projection_transform` argument in the `DensityEstimator`
constructor. Perhaps we require the full field of view to estimate the nuisance parameters, but
we only want to reconstruct a small cropped region.

The `projection_transform` is the desired operation that is applied to the measurements and used
internally to handle the modification of the background and flux estimates.
"""


"""
First we show an example of binned data using the below defined `binning_2x2` function.
"""


def binning_2x2(Y):
    r"""Bin an array of shape (N, M, ...) with the result being (N//2, M//2, ...)"""
    N0 = Y.shape[0] // 2
    N1 = Y.shape[1] // 2
    Y = Y[0::2][:N0] + Y[1::2][:N0]
    Y = Y[:, 0::2][:, :N1] + Y[:, 1::2][:, :N1]
    return Y


"""
We define the `DensityEstimator` but pass the `binning_2x2` as the `projection_transform` argument
and use `solve` as usual. Note that the resulting detector shape is half the size in each
direction.
"""
den = reconstruct.DensityEstimator(
    Y_s, par, projection_transform=binning_2x2, non_negative_Z=False, dispperiod=50
)
den.solve(iterations=200)

Z_binned = binning_2x2(Z) / (2 * 2)  # over 4 to keep ground truth scale the same.
Z_hat_binned = den.Z

print(f"{Z.shape = }")
print(f"{Z_binned.shape = }")

"""
We plot the results below.
"""

plot_compare(
    Z_binned, "Binned Ground Truth", Z_hat_binned, "Reconstruction from Binned Measurement"
)
plt.show()


"""
Second we show an example of cropping data using the below defined `crop` function and the same
"""


def crop(Y):
    r"""Create center crop from 1/5 to 4/5 of the FOV."""
    N0 = Y.shape[0] // 5
    N1 = Y.shape[1] // 5
    Y = Y[N0 : 4 * N0, N1 : 4 * N1]
    return Y


den = reconstruct.DensityEstimator(
    Y_s, par, projection_transform=crop, non_negative_Z=False, dispperiod=50
)
den.solve(iterations=200)

Z_crop = crop(Z)
Z_hat_crop = den.Z

print(f"{Z_crop.shape = }")
print(f"{Z.shape = }")


plot_compare(Z_crop, "Cropped Ground Truth", Z_hat_crop, "Reconstruction from Cropped Measurement")
plt.show()
