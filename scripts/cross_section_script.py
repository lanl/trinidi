r"""Example script for cross_section submodule"""


import numpy as np

from trinidi import cross_section, util

cross_section.info()

available_isotopes = cross_section.avail()

# ['U-230', 'U-231', 'U-232', 'U-233', 'U-234', 'U-235', 'U-236', 'U-237', 'U-238', 'U-239', 'U-240', 'U-241']
uranium_isotopes = [
    isotope for isotope in available_isotopes if isotope.split("-")[0] == "U"
]

print("\nUranium Isotopes:")
cross_section.info(uranium_isotopes)


Δt = 0.30
t = np.arange(72, 720, Δt)
flight_path_length = 10

E = util.time2energy(t, flight_path_length)


D = cross_section.create_xsdict(uranium_isotopes, t, flight_path_length)
