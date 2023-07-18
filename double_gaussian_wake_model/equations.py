# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import special


def fnc_M(sig: float, r0: float) -> float:
    """
    Computes 'M' according to Eqn. 7a

    :param sig: sigma [m]
    :param r0: spanwise location of the Gaussian extrema [m]
    :return M: M [m^2]
    """
    return 2 * (sig ** 2) * np.exp((-(r0 ** 2)) / (2 * (sig ** 2))) + \
        np.sqrt(2 * np.pi) * r0 * sig * special.erf(r0 / (sig * np.sqrt(2)))


def fnc_N(sig: float, r0: float) -> float:
    """
    Computes 'N' according to Eqn. 7b

    :param sig: sigma [m]
    :param r0: spanwise location of the Gaussian extrema [m]
    :return: N [m^2]
    """
    return (sig ** 2) * np.exp((-(r0 ** 2)) / ((sig ** 2))) + \
        (1 / 2) * np.sqrt(np.pi) * r0 * sig * special.erf(r0 / sig)


def fnc_Cm(M: float, N: float, Ct: float, d0: float) -> float:
    """
    Computes amplitude 'Cm' according to Eqn. 8 (only C-minus, Cm, is computed)

    :param M: M [m^2]
    :param N: N [m^2]
    :param Ct: thrust coefficient [-]
    :param d0: rotor diameter [m]
    :return Cm: amplitude [-]
    """
    below_sqrt = (M ** 2.0) - 0.5 * N * Ct * (d0 ** 2)
    return (M - np.sqrt(below_sqrt)) / (2.0 * N)
