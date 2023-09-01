# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

from double_gaussian_wake_model.wake_model import double_gaussian_deficit


# This script loads the calibration data (scaled G1-turbine) and re-generates Fig. 3 of:
# Schreiber, J., Balbaa, A., and Bottasso, C. L.: Brief communication: A double-Gaussian wake model,
# Wind Energ. Sci., 5, 237–244, https://doi.org/10.5194/wes-5-237-2020, 2020.

def get_points_for_line_plot(d0, downstream_D=6, resolution=0.01, limit_D=1.5):
    z = np.array([0])
    y = np.arange(-limit_D, limit_D, resolution) * d0
    x = np.array([downstream_D]) * d0
    return x, y, z


# Loading of epsilon table
if pathlib.Path('resources/fcn_epsilon.pickle').is_file():
    with open('resources/fcn_epsilon.pickle', 'rb') as f:
        fcn_epsilon = pickle.load(f)
    recompute_epsilon = False
else:
    print("Epsilon look-up table could not be found. "
          "Run 'generate_epsilon.py' to create it."
          "Using the slower and 'recompute_epsilon' option for now.")
    fcn_epsilon = None
    recompute_epsilon = True

# load calibration data
with open('resources/G1_calibration_data.pickle', 'rb') as f:
    data = pickle.load(f)

# Parameters:
kr, k, x0 = 0.534719991282902, 0.010379302460442, 1.103128529641920

Ct = 0.75
v = 1.0
d0 = 1.1

fig, axes = plt.subplots(1, len(data["distance_D"]), sharex=True, sharey=True, figsize=(12, 6))
for i, downstream_D in enumerate(data["distance_D"]):
    ax = axes[i]

    ax.plot(data["CFD_x"][i], data["CFD_y"][i], label=f"CFD", linewidth=0, marker="x", color="r")
    ax.plot(data["Exp_x"][i], data["Exp_y"][i], label=f"Exp.", linewidth=0, marker="o", color="k", mfc='none')

    x, y, z = get_points_for_line_plot(d0, downstream_D)
    deficit = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, recompute_epsilon, fcn_epsilon)
    ax.plot(v * (1 - deficit), y / d0, label=f"DG", color="tab:blue")

    ax.set_title(f"x={downstream_D}D")

    if i == 0:
        ax.legend(loc="lower right")

    ax.set_ylabel('Y [D]')
    ax.set_xlabel('V [-]')
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlim([0.4, 1.1])
    ax.grid()

fig.tight_layout()

plt.show()
