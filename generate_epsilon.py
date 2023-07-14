# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from model.compute_epsilon import compute_epsilon

# epsilon does not depend on diameter, but it needs to be
d0 = 1.0

Cts = np.arange(0.01, 1, 0.01)
krs = np.arange(0.0, 1.01, 0.01)

Cts = np.arange(0.0, 1, 0.1)
krs = np.arange(0.0, 1.01, 0.1)

epsilon = np.zeros(shape=(len(Cts), len(krs))) * np.NaN
for i, Ct in enumerate(Cts):
    for j, kr in enumerate(krs):
        r0 = kr * d0 / 2
        epsilon[i, j] = compute_epsilon(d0, Ct, r0)

    print(f"finished by: {100 * (i + 1) / len(Cts):.2f} %")

fcn_epsilon = RegularGridInterpolator((Cts, krs), epsilon)

with open('resources/fcn_epsilon.pickle', 'wb') as f:
    pickle.dump(fcn_epsilon, f)
