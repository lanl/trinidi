r"""Example script for estimating the time calibration."""
import numpy as np

import matplotlib.pyplot as plt

from trinidi import cross_section, util


def generate_measurement():
    r"""Generate measurements with `unknown` time calibration."""
    Δt = 0.30  # bin width [μs]
    t_0 = 72
    t_A = np.arange(t_0, 600, Δt)  # time-of-flight array [μs]
    print(f"Ground truth bin width:     {Δt = :.5f} μs")
    print(f"Ground truth starting time: {t_0 = :.2f} μs")

    t_F = t_A  # since R = Identity
    D = cross_section.XSDict(isotopes, t_F, flight_path_length)
    z = np.array([[0.005, 0.001]]).T

    ϕ, b, θ, α_1, α_2 = util.generate_spectra(t_A)
    y_s = α_1 * (ϕ.T * np.exp(-z.T @ D.values) + α_2 * b.T)
    return y_s.flatten()


######## Measurement + Cross sections

flight_path_length = 10  # [m]
isotopes = ["U-235", "U-238"]
y_s = generate_measurement()

D = cross_section.XSDict(isotopes, np.arange(60, 650, 0.2), flight_path_length)

fig, ax = plt.subplots(2, 1, figsize=[9, 9], sharex=False)
ax = np.atleast_1d(ax)
ax[0].plot(y_s, label="Measurement", alpha=0.75, color="tab:green")
ax[0].set_xlabel("Bin Index")
ax[0].set_title("Measurement Over Bin Index")
ax[0].legend()
D.plot(ax[1])
fig.suptitle("")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))
plt.show()


######## Selection of measurement bins and corresponding tof's


bin_index_list = [308.7, 692.9, 854.0, 1448.8]
time_list = [164.6, 279.9, 328.3, 506.6]

fig, ax = plt.subplots(2, 1, figsize=[9, 9], sharex=False)
ax = np.atleast_1d(ax)
ax[0].plot(y_s, label="Measurement", alpha=0.75, color="tab:green")
for x in bin_index_list:
    ax[0].axvline(x=x, c="r", alpha=0.6)
ax[0].set_title("Measurement Over Bin Index")
ax[0].set_xlabel("Bin Index")
ax[0].legend()
D.plot(ax[1])
for x in time_list:
    ax[1].axvline(x=x, c="b", alpha=0.6, ls="--")
fig.suptitle("Selected Bin Indices vs. Corresponding Times")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))
plt.show()


######## Calibration


def bin_number_to_time(bin_index_list, time_list):
    r"""Find affine linear mapping from bin index to time.

    Args:
        bin_index_list: list of bin indices.
        time_list: list of corresponding times.

    Returns:
        The mapping.
    """
    # f(x) = m*x + b
    mb = np.polyfit(bin_index_list, time_list, 1)
    Δt = mb[0]
    t_0 = mb[1]
    return lambda i: Δt * i + t_0


f = bin_number_to_time(bin_index_list, time_list)

N_A = y_s.size
t_A = f(np.arange(N_A))


fig, ax = plt.subplots(1, 1, figsize=[9, 5], sharex=False)
ax = np.atleast_1d(ax)
ax[0].plot(t_A, y_s, label="Measurement", alpha=0.75, color="tab:green")
for x in f(np.array(bin_index_list)):
    ax[0].axvline(x=x, c="r", alpha=0.6)
ax[0].set_xlabel("Time bin index")
ax[0].legend()
D.plot(ax[0])
for x in time_list:
    ax[0].axvline(x=x, c="b", alpha=0.6, ls="--")
ax[0].set_title("Measurement aligned with cross sections")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))
plt.show()


t_0 = t_A[0]  # = f(0)
Δt = t_A[1] - t_A[0]  # = f(1) - f(0)
print(f"Estitated bin width:        {Δt = :.5f} μs")
print(f"Estitated starting time:    {t_0 = :.2f} μs")


E = util.time2energy(t_A, flight_path_length)  # energy array [eV]
# Note that by our convention, the t_F (and t_A) arrays are increasing
# which forces the energy array to be decreasing :
print(f"{t_A[0]=:.2f},\t{t_A[1]=:.2f},\t..., {t_A[-1]=:.2f}\t[μs]")
print(f"{E[0]=:.2f},\t{E[1]=:.2f},\t..., {E[-1]=:.2f}\t\t[eV]")

print(f"t_A = [{t_A[0]:.2f}, {t_A[1]:.2f}, ..., {t_A[-1]:.2f}] [μs]")
print(f"E   = [{E[0]:.2f}, {E[1]:.2f}, ..., {E[-1]:.2f}] [eV]")
