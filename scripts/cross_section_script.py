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
t_F = np.arange(72, 720, Δt)  # time-of-flight array [μs]
E = util.time2energy(t_F, flight_path_length)  # energy array [eV]


# Generate cross section dictionary and plot it with respect to time-of-flight and energy.
isotopes = ["U-235", "U-238"]

D = cross_section.XSDict(isotopes, t_F, flight_path_length, samples_per_bin=10)


fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=False)
ax = np.atleast_1d(ax)
D.plot(ax[0], function_of_energy=True)
D.plot(ax[1], function_of_energy=False)
fig.suptitle("Plotting Cross Section Dictionary by TOF and Energy")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))
plt.show()


# Generate cross section dictionary and merge isotopes by natural abundance percentages
isotopes_full = ["Au-197", "W-180", "W-182", "W-183", "W-184", "W-186"]
merge_isotopes = ["W-180", "W-182", "W-183", "W-184", "W-186"]
merge_weights = [0.0012, 0.265, 0.143, 0.306, 0.284]
new_key = "W"

D_full = cross_section.XSDict(isotopes_full, t_F, flight_path_length)

D_merged = cross_section.XSDict(isotopes_full, t_F, flight_path_length)
D_merged.merge(merge_isotopes, merge_weights, new_key)


fig, ax = plt.subplots(2, 1, figsize=[12, 8], sharex=True)
ax = np.atleast_1d(ax)
D_full.plot(ax[0])
D_merged.plot(ax[1])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=(0, 0, 1, 0.95))
fig.suptitle("Plotting Full and Merged Cross Section Dictionaries")
plt.show()
