# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
from scipy.optimize import minimize

from model.fcn_MNC import fnc_M, fnc_N, fnc_Cm


def residual_of_mass_flow_deficits(epsilon: float, d0: float, Ct: float, r0: float) -> float:
    """
    Computes the residual between the mass flow deficits from gaussian wake and Frandsen wake.

    :param epsilon: wake expansion at x0 [D]
    :param d0: rotor diameter [m]
    :param Ct: thrust coefficient [-]
    :param r0: span-wise location of the Gaussian extrema [m]
    :return residual
    """

    sig0 = epsilon * d0
    M = fnc_M(sig0, r0)
    N = fnc_N(sig0, r0)
    Cm = fnc_Cm(M, N, Ct, d0)
    mDot_dg = np.pi * M * Cm  # ignoring air density
    beta = (1 / 2) * ((1 + np.sqrt(1 - Ct)) / (np.sqrt(1 - Ct)))
    mDot_frandsen = (np.pi / 8) * (d0 ** 2) * beta * (1 - np.sqrt(1 - ((2 / beta) * Ct)))  # ignoring air density
    residual = (mDot_dg - mDot_frandsen) ** 2  # ignoring air density
    return residual


def compute_epsilon(d0: float, Ct: float, r0: float) -> float:
    """
    Computes epsilon as function of diameter, thrust coefficent and spanwise location of the Gaussian extrema.

    :param d0: rotor diameter [m]
    :param Ct: thrust coefficient [-]
    :param r0: span-wise location of the Gaussian extrema [m]
    :return epsilon: wake expansion at x0 [D]
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        res = minimize(lambda epsilon: residual_of_mass_flow_deficits(epsilon, d0, Ct, r0),
                       x0=d0 / 2,
                       method='nelder-mead', options={'xatol': 1e-8, 'disp': False},
                       bounds=[(1E-5, 10 * d0)])

        epsilon = res.x[0]
        if not res.success:
            warnings.warn(f"Epsilon could not be found for d0={d0}, Ct={Ct}, r0={r0})")
            epsilon = np.NaN

    return epsilon
