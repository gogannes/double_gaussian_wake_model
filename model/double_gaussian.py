# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np

from model.compute_epsilon import compute_epsilon
from model.fcn_MNC import fnc_M, fnc_N, fnc_Cm


def double_gaussian_deficit(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            Ct: float, d0: float, kr: float, x0: float, k: float,
                            recompute_epsilon: bool, fcn_epsilon: callable = None) -> np.ndarray:
    """
       Computes the double gaussian wake deficit at given locations.

       :param x: downstream x-locations to be evaluated [m]
       :param y: downstream y-locations to be evaluated [m]
       :param z: downstream z-locations to be evaluated [m]
       :param Ct: thrust coefficient [-]
       :param d0: rotor diameter [m]
       :param kr: wake parameter, position of the Gaussian extrema (0: wake center, 1: blade tip) [-]
       :param x0: wake parameter, position stream tube outlet [D]
       :param k: wake parameter, slope of wake expansion [-]
       :param recompute_epsilon: flag, whether epsilon shall be computed on-the-fly (true) or whether a pre-defined function shall be evaluated (false)
       :param fcn_epsilon: pre-computed epsilon function with parameters Ct [-] and kr [D], only used if recompute_epsilon=False
       :return deficit: wake deficit at each location x, y, z (wake velocity = free-stream velocity v * (1 - deficit) [-]
       """

    # r0, span-wise location of the Gaussian extrema
    r0_D = kr / 2  # [D]
    r0 = r0_D * d0  # [m]

    if recompute_epsilon:
        # solve epsilon function on-the-fly
        epsilon = compute_epsilon(d0, Ct, r0)  # [D]
    else:
        # use precomputed epsilon function
        epsilon = fcn_epsilon((Ct, kr))  # [D]

    # sig, wake expansion
    sig = k * (x - x0 * d0) + epsilon * d0  # [m]

    M = fnc_M(sig, r0)
    N = fnc_N(sig, r0)
    Cm = fnc_Cm(M, N, Ct, d0)
    r = np.sqrt(y ** 2.0 + z ** 2.0)  # [m]
    deficit = 0.5 * Cm * (
            np.exp(-1.0 * ((r + r0) ** 2.0) / (2.0 * (sig ** 2.0))) +
            np.exp(-1.0 * ((r - r0) ** 2.0) / (2.0 * (sig ** 2.0)))
    )  # [-]

    if np.isnan(deficit).sum() > 0:
        warnings.warn("Deficit could not be computed in every location!")

    return deficit
