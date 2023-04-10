r"""Example script for reconstruct submodule"""

import numpy as np

import matplotlib.pyplot as plt

from trinidi import cross_section, reconstruct, resolution, util


def plot_densities(fig, ax, Z, isotopes, vmax_list=None):
    r"""Show areal densities. `ax` must be list."""

    for i, isotope in enumerate(isotopes):
        z = Z[:, :, i]
        if vmax_list is None:
            vmax = np.percentile(z, 99.9)
        else:
            vmax = vmax_list[i]
        vmin = 0
        im = ax[i].imshow(z, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[i], format="%.1e")
        ax[i].set_title(f"{isotope}")
        ax[i].axis("off")

    fig.suptitle("Areal Densities [mol/cm²]")

    return fig, ax


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

    return Z, Y_o, Y_s, ground_truth_params, t_A, flight_path_length, N_b, kernels


isotopes = ["U-238", "Pu-239", "Ta-181"]
z = np.array([[0.005, 0.003, 0.004]]).T


Z, Y_o, Y_s, ground_truth_params, t_A, flight_path_length, N_b, kernels = generate_sample_data(
    isotopes, z, acquisition_time=10
)

fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
ax = np.atleast_1d(ax)
plot_densities(fig, ax, Z, isotopes, vmax_list=z * 1.5)
plt.show()


Ω_z = reconstruct.ProjectionRegion(np.prod(Z, axis=2, keepdims=True) > 0)
Ω_0 = reconstruct.ProjectionRegion(np.sum(Z, axis=2, keepdims=True) == 0)


fig, ax = plt.subplots(1, 3, figsize=[14, 4])
ax = np.atleast_1d(ax)

ax[0].imshow(np.sum(Y_s, axis=-1) / np.sum(Y_o, axis=-1), vmin=0)
ax[0].set_title("1Y_s / 1Y_o (TOF integrated measurement ratio)")
Ω_z.plot_contours(ax[0], color="red")
Ω_0.plot_contours(ax[0], color="blue")

Ω_z.imshow(ax[1], title="Ω_z")
Ω_0.imshow(ax[2], title="Ω_0")

plt.show()


R = resolution.ResolutionOperator(Y_s.shape, t_A, kernels=kernels)
t_F = R.t_F
D = cross_section.XSDict(isotopes, t_F, flight_path_length)

fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
ax = np.atleast_1d(ax)
ax[0].plot(t_A, Ω_z.average(Y_s).flatten(), label="Ω_z.average(Y_s)", alpha=0.75)
ax[0].plot(t_A, Ω_0.average(Y_s).flatten(), label="Ω_0.average(Y_s)", alpha=0.75)
D.plot(ax[1])
ax[0].legend(prop={"size": 8})
fig.suptitle("Average spectra in Ω_z and Ω_0")
plt.show()


from importlib import reload

import trinidi.reconstruct

reload(trinidi.reconstruct)


par = reconstruct.ParameterEstimator(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=N_b)


d = par.get()
print(d)

par.set(**d)
par.set(z=d["z"], α_1=d["α_1"], α_2=d["α_2"], θ=d["θ"])  # same as line above

par.save("par.npy")
par = reconstruct.ParameterEstimator(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=N_b)
par.load("par.npy")


par.plot_regions()
plt.show()

par.plot_results()
plt.show()


d = par.get()
par = reconstruct.ParameterEstimator(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=N_b)

if False:
    par.apgm_solve(iterations=100)
    fig, ax = par.apgm_plot_convergence(plot_residual=True, ground_truth=ground_truth_params)
    plt.show()

par.set(**d)


1 / 0


from importlib import reload

import trinidi.reconstruct

reload(trinidi.reconstruct)


def plot_compare(Z, str_Z, Z_hat, str_Z_hat):
    r"""Generate two plots comparing ground truth with reconstruction."""
    fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
    ax = np.atleast_1d(ax)
    plot_densities(fig, ax, Z, isotopes, vmax_list=z * 1.5)
    fig.suptitle(f"{str_Z} [mol/cm²]")

    fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
    ax = np.atleast_1d(ax)
    plot_densities(fig, ax, Z_hat, isotopes, vmax_list=z * 1.5)
    fig.suptitle(f"{str_Z_hat} [mol/cm²]")


den = reconstruct.DensityEstimator(Y_s, par, non_negative_Z=False)
Z_hat = den.solve(iterations=500)

den.plot_convergence(ground_truth=Z)
plt.show()


plot_compare(Z, "Ground Truth", Z_hat, "Reconstruction")
plt.show()


# How to handle binning or cropping data to different shape than in nuisance parameter estimation.


def binning_2x2(Y):
    r"""Bin an array of shape (N, M, ...) with the result being (N//2, M//2, ...)"""
    N0 = Y.shape[0] // 2
    N1 = Y.shape[1] // 2
    Y = Y[0::2][:N0] + Y[1::2][:N0]
    Y = Y[:, 0::2][:, :N1] + Y[:, 1::2][:, :N1]
    return Y


den = reconstruct.DensityEstimator(Y_s, par, projection_transform=binning_2x2, non_negative_Z=False)
den.solve(iterations=500)

Z_binned = binning_2x2(Z) / (2 * 2)  # over 4 to keep ground truth scale the same.
Z_hat_binned = den.Z

plot_compare(
    Z_binned, "Binned Ground Truth", Z_hat_binned, "Reconstruction from Binned Measurement"
)
plt.show()


def crop(Y):
    r"""Create center crop from 1/5 to 4/5 of the FOV."""
    N0 = Y.shape[0] // 5
    N1 = Y.shape[1] // 5
    Y = Y[N0 : 4 * N0, N1 : 4 * N1]
    return Y


den = reconstruct.DensityEstimator(Y_s, par, projection_transform=crop, non_negative_Z=False)
den.solve(iterations=500)

Z_crop = crop(Z)
Z_hat_crop = den.Z

plot_compare(Z_crop, "Cropped Ground Truth", Z_hat_crop, "Reconstruction from Cropped Measurement")
plt.show()

plt.show()
