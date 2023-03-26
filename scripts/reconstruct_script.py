r"""Example script for reconstruct submodule"""

import numpy as np

import matplotlib.pyplot as plt

from trinidi import cross_section, reconstruct, resolution, util


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


# fig, ax = plt.subplots(1, 1, figsize=[12, 8], sharex=True)
# ax = np.atleast_1d(ax)
# ax[0].plot(t_A, b.flatten(), label="b", alpha=0.75)
# ax[0].plot(t_A, (ϕ + b).flatten(), label="ϕ+b", alpha=0.75)
# ax[0].set_xlabel(util.TOF_LABEL)
# ax[0].legend(prop={"size": 8})


z = np.array([[0.005, 0.003, 0.004]]).T
Z = util.rose_phantom(projection_shape[0], num_circles=z.size, radius=2 / 3) * z.reshape(
    [1, 1, z.size]
)


# fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
# ax = np.atleast_1d(ax)
# plot_densities(fig, ax, Z, isotopes)


v = np.random.poisson(1000, size=projection_shape + (1,))
v = v / v.mean()

Φ = v @ ϕ.T
B = v @ b.T

Y_o_bar = Φ + B
Y_s_bar = α_1 * (Φ * R(np.exp(-Z @ D.values)) + α_2 * B)

Y_o = np.random.poisson(Y_o_bar)
Y_s = np.random.poisson(Y_s_bar)


Ω_z = reconstruct.ProjectionRegion(np.prod(Z, axis=2, keepdims=True) > 0)
Ω_0 = reconstruct.ProjectionRegion(np.sum(Z, axis=2, keepdims=True) == 0)


# fig, ax = plt.subplots(1, 3, figsize=[14, 4])
# ax = np.atleast_1d(ax)

# ax[0].imshow(np.sum(Y_s, axis=-1) / np.sum(Y_o, axis=-1), vmin=0)
# ax[0].set_title("1Y_s / 1Y_o")

# Ω_z.plot_contours(ax[0], color="red")
# Ω_0.plot_contours(ax[0], color="blue")

# Ω_z.imshow(ax[1], title="Ω_z")
# Ω_0.imshow(ax[2], title="Ω_0")

# plt.show()


from importlib import reload

import trinidi.reconstruct

reload(trinidi.reconstruct)


par = reconstruct.Parameters(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=3)

if False:
    par.plot_regions()

    par.solve(iterations=20)
    fig, ax = par.plot_convergence(
        plot_residual=True, ground_truth={"z": z, "α_1": α_1, "α_2": α_2, "θ": θ}
    )

    fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
    ax = np.atleast_1d(ax)
    par.plot_results(ax[0])
    D.plot(ax[1])

    par.save("par.npy")
    par = reconstruct.Parameters(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=3)
    par.load("par.npy")

    d = par.get_parameter_dict()
    par = reconstruct.Parameters(Y_o, Y_s, R, D, Ω_z, Ω_0=Ω_0, N_b=3)
    par.set_parameter_dict(**d)
    par.set_parameter_dict(z=d["z"], α_1=d["α_1"], α_2=d["α_2"], θ=d["θ"])  # same as line above

    plt.show()


par.set_parameter_dict(z=z, α_1=α_1, α_2=α_2, θ=θ)


from importlib import reload

import trinidi.reconstruct

reload(trinidi.reconstruct)


den = reconstruct.ArealDensityEstimator(Y_s, par)
den.solve(iterations=1000)

fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
ax = np.atleast_1d(ax)
plot_densities(fig, ax, Z, isotopes)
fig.suptitle("Z: Areal Densities [mol/cm²]")

fig, ax = plt.subplots(1, len(isotopes), figsize=[12, 3.3])
ax = np.atleast_1d(ax)
plot_densities(fig, ax, den.Z, isotopes)
fig.suptitle("Z_hat: Areal Densities [mol/cm²]")

plt.show()
