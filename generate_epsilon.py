# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from double_gaussian_wake_model.compute_epsilon import compute_epsilon

# epsilon does not depend on diameter, but it needs to be
d0 = 1.0

Cts = np.arange(0.01, 1, 0.01)
krs = np.arange(0.0, 1.01, 0.01)

epsilon = np.zeros(shape=(len(Cts), len(krs))) * np.NaN
for i, Ct in enumerate(Cts):
    for j, kr in enumerate(krs):
        r0 = kr * d0 / 2
        epsilon[i, j] = compute_epsilon(d0, Ct, r0)

    print(f"finished by: {100 * (i + 1) / len(Cts):.2f} %")

fcn_epsilon = RegularGridInterpolator((Cts, krs), epsilon)

with open('resources/fcn_epsilon.pickle', 'wb') as f:
    pickle.dump(fcn_epsilon, f)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
r0_grid, Ct_grid = np.meshgrid(krs / 2 / d0, Cts)
ax.plot_surface(r0_grid, Ct_grid, epsilon, cmap=cm.jet)
ax.set_xlabel('r0 [D]')
ax.set_ylabel('Ct [-]')
ax.set_zlabel('epsilon [D]')
ax.view_init(30, -135)
plt.show()
