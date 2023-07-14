# Copyright (c) 2023, Johannes Schreiber; Amr Balbaa; Carlo L. Bottasso
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import numpy as np

from model.double_gaussian import double_gaussian_deficit


def test_simple_with_recomputing_epsilon():
    """
    Test whether a deficit is obtained when recomputing epsilon.
    """
    x = np.array([5])
    y = np.array([0])
    z = np.array([0])
    Ct = 0.75
    d0 = 1.1

    kr = 0.534720
    x0 = 1.1031
    k = 0.0103793
    recompute_epsilon = True

    deficit = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, recompute_epsilon)
    assert deficit[0] > 0


def test_simple_with_epsilon_function():
    """
    Test whether a deficit is obtained when using pre-computed epsilon look-up function.
    """
    with open('../fcn_epsilon.pickle', 'rb') as f:
        fcn_epsilon = pickle.load(f)
    x = np.array([5])
    y = np.array([0])
    z = np.array([0])
    Ct = 0.75
    d0 = 1.1

    kr = 0.534720
    x0 = 1.1031
    k = 0.0103793
    recompute_epsilon = False

    deficit_1 = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, recompute_epsilon, fcn_epsilon)

    deficit_2 = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, True)

    assert np.abs((deficit_1[0] - deficit_2[0]) / deficit_1[0]) < 0.001


def test_solution_independent_of_diameter():
    """
    Test whether the deficit is independent of turbine diameter.
    """
    Ct = 0.75
    kr = 0.534720
    x0 = 1.1031
    k = 0.0103793
    recompute_epsilon = True
    y = np.array([0])
    z = np.array([0])

    d0 = 1.1
    x = np.array([5]) * d0
    deficit_1 = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, recompute_epsilon)

    d0 = 190.0
    x = np.array([5]) * d0
    deficit_2 = double_gaussian_deficit(x, y, z, Ct, d0, kr, x0, k, recompute_epsilon)

    assert np.abs((deficit_1[0] - deficit_2[0]) / deficit_1[0]) < 0.001


def test_conservation_of_momentum():
    """
    Test whether the momentum is conserved.
    """

    Ct = 0.75
    d0 = 2.3
    v = 5
    rho = 1
    downstream_D = 1.5

    kr = 0.534720
    x0 = 1.1031
    k = 0.0103793
    recompute_epsilon = False
    with open('../fcn_epsilon.pickle', 'rb') as f:
        fcn_epsilon = pickle.load(f)

    resolution_D = 0.1
    lateral_limit = 2 * d0
    x = np.array([downstream_D]) * d0
    y = np.arange(-lateral_limit, lateral_limit, resolution_D * d0)
    z = np.arange(-lateral_limit, lateral_limit, resolution_D * d0)

    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    deficit = double_gaussian_deficit(x_grid, y_grid, z_grid, Ct, d0, kr, x0, k, recompute_epsilon, fcn_epsilon)

    # thrust according to Eqn. 3 (rho * integral (U(x,r)*(U_inf-U(x,r)) dA)
    U_inf = np.zeros(shape=deficit.shape) + v
    U_wake = U_inf - (deficit * U_inf)
    dA = (z[1] - z[0]) * (y[1] - y[0])
    integral = np.sum(U_wake * (U_inf - U_wake)) * dA
    T = rho * integral

    # thrust according to Eqn. 4:
    T_rotor = 0.5 * rho * np.pi * (d0 / 2) ** 2 * v ** 2 * Ct

    deviation = (T_rotor / T) - 1

    # print(f"Rotor thrust: {T_rotor:.2f}, momentum deficit: {T:.2f}, a/b-1: {deviation * 100:.10f}%")
    assert np.abs(
        deviation) < 0.00001, f"Momentum not conserved. Rotor thrust: {T_rotor:.2f}, momentum deficit: {T:.2f}, a/b-1: {deviation * 100:.10f}%"

    if 0:
        # plotting for debugging
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        shape = (y_grid.shape[0], y_grid.shape[2])
        ax.plot_surface(y_grid.reshape(shape), z_grid.reshape(shape), deficit.reshape(shape), cmap=cm.jet)
        ax.set_xlabel('y [m]')
        ax.set_ylabel('z [m]')
        ax.set_zlabel('deficit [m/s]')
        ax.view_init(30, -135)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        shape = (y_grid.shape[0], y_grid.shape[2])
        ax.plot_surface(y_grid.reshape(shape), z_grid.reshape(shape), U_wake.reshape(shape), cmap=cm.jet)
        ax.set_xlabel('y [m]')
        ax.set_ylabel('z [m]')
        ax.set_zlabel('v [m/s]')
        ax.view_init(30, -135)
