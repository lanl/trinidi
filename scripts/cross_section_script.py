r"""Example script for cross_section submodule"""

import matplotlib.pyplot as plt
import numpy as np

from trinidi import cross_section, util

# Print out info of all isotopes.
cross_section.info()


# From all available isotopes pick the uranium isotopes.
available_isotopes = cross_section.avail()
isotopes_U = [iso for iso in available_isotopes if iso.split("-")[0] == "U"]
print("\nUranium Isotopes:")
cross_section.info(isotopes_U)


# Setup time-of-flight and energy arrays.
Δt = 0.30  # bin width [μs]
flight_path_length = 10  # [m]
t = np.arange(72, 720, Δt)  # time-of-flight array [μs]
E = util.time2energy(t, flight_path_length)  # energy array [eV]


# Generate cross section dictionary and plot it with respect to time-of-flight and energy.
isotopes = ["U-235", "U-238"]
D = cross_section.create_xsdict(isotopes, t, flight_path_length)

fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=False)
ax = np.atleast_1d(ax)
cross_section.plot_xsdict(ax[0], D, isotopes, t=t)
cross_section.plot_xsdict(ax[1], D, isotopes, E=E)
fig.suptitle("Plotting Cross Section Dictionary by TOF and Energy")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))


# Generate cross section dictionary and merge isotopes by natural abundance percentages
isotopes_full = ["Au-197", "W-180", "W-182", "W-183", "W-184", "W-186"]
isotopes_merged = ["Au", "W"]

C = np.array(
    [[1, 0, 0, 0, 0, 0], [0, 0.0012, 0.265, 0.143, 0.306, 0.284]]
)  # abundances

D_full = cross_section.create_xsdict(isotopes_full, t, flight_path_length)
D_merged = C @ D_full

fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
ax = np.atleast_1d(ax)
cross_section.plot_xsdict(ax[0], D_full, isotopes_full, t=t)
cross_section.plot_xsdict(ax[1], D_merged, isotopes_merged, t=t)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))
fig.suptitle("Plotting Full and Merged Cross Section Dictionaries")
plt.show()
