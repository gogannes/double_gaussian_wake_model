# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np

from model.compute_epsilon import compute_epsilon
from model.fcn_MNC import fnc_M, fnc_N, fnc_Cm


def double_gaussian_deficit(x: np.ndarray, y: np.ndarray, z: np.ndarray, Ct: float, d0: float,
                            kr: float, x0: float, k: float,
                            recompute_epsilon: bool, fcn_epsilon: callable = None):
    """
       Computes the double gaussian wake deficit at given locations.

       :param x: downstream x-locations to be evaluated [m]
       :param y: downstream y-locations to be evaluated [m]
       :param z: downstream z-locations to be evaluated [m]
       :param Ct: thrust coefficient [-]
       :param d0: rotor diameter [m]
       :param kr: wake parameter, position of the Gaussian extrema [D] (0: wake center, 1: blade tip)
       :param x0: wake parameter, position stream tube outlet [D]
       :param k: wake parameter, slope of wake expansion [-]
       :param recompute_epsilon: flag, whether epsilon shall be computed on-the-fly (true) or whether a pre-defined function shall be evaluated (false)
       :param fcn_epsilon: pre-computed epsilon function with parameters Ct [-] and kr [D], only used if recompute_epsilon=False
       :return deficit: wake deficit at each location [-]
       """

    # todo: is "[D]" correct? In "Table 1" we claimed it is "[-]"
    # "r0 is radial position of gaussian extrema" - this is in SI units (meters)
    # but: "When kr=1 the curve extrema are located at the tip of the rotor blades, while for kr=0
    # the two Gaussian functions coincide at the wake center"
    # but: "r0=kr/2, or kr=2*r0" -> shouldn't this be "kr = r0/(0.5*d), or r0=kr/2 * D" (DIAMETER MISSING IN EQN in 3.2)]

    # r0, span-wise location of the Gaussian extrema [m]
    r0 = kr * d0 / 2.0  # [m]

    if recompute_epsilon:
        # solve epsilon function on-the-fly
        epsilon = compute_epsilon(d0, Ct, r0)
    else:
        # use precomputed epsilon function
        epsilon = fcn_epsilon((Ct, kr))  # [D]

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
        # warnings.warn("Deficit could not be computed in every location! Setting respective deficits to 0.")
        # deficit[np.isnan(deficit)] = 0.0

    return deficit
