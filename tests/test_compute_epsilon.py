# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from model.compute_epsilon import compute_epsilon


def test_solution_independent_of_diameter():
    """
    Test whether epsilon is independent of turbine diameter.
    """

    # Ct: thrust coefficient
    Ct = 0.75  # [-]

    # kr: position of the Gaussian extrema (0: wake center, 1: blade tip) [-]
    kr = 0.534720  # [-]

    d0 = 1.1  # [m]
    r0_D = kr / 2  # [D]
    r0 = r0_D * d0  # [m]
    epsilon_1 = compute_epsilon(d0, Ct, r0)

    d0 = 119  # [m]
    r0_D = kr / 2  # [D]
    r0 = r0_D * d0  # [m]
    epsilon_2 = compute_epsilon(d0, Ct, r0)

    assert np.abs((epsilon_1 - epsilon_2) / epsilon_1) < 0.001
