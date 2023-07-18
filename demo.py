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

# Parameters:
kr, k, x0 = 0.534719991282902, 0.010379302460442, 1.103128529641920

Ct = 0.75
v = 5.0
d0 = 1.1

fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(8, 4))
for downstream_D in [2, 4, 6, 8, 10]:
    x, y, z = get_points_for_line_plot(d0, downstream_D)
    deficit = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, recompute_epsilon, fcn_epsilon)
    ax.plot(y / d0, v * (1 - deficit), label=f"velocity {downstream_D} D downstream")
ax.set_xlabel('y [D]')
ax.set_ylabel('v [m/s]')
ax.legend(loc="upper right")
ax.set_title(f"Ct={Ct}, kr={kr:.5}, k={k:.5}, x0={x0:.5}D")
ax.grid()
fig.tight_layout()
plt.show()
