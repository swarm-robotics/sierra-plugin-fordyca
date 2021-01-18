# Copyright 2021 Angel Sylvester, John Harwell, All rights reserved.
#
#  This file is part of SIERRA.
#
#  SIERRA is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  SIERRA is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with
#  SIERRA.  If not, see <http://www.gnu.org/licenses/

"""
Intra- and inter-experiment models for the time it takes a single robot to return to the nest after
picking up an object.
"""
# Core packages
import typing as tp

# 3rd party packages
import numpy as np
import scipy.integrate as si

# Project packages


class CRWSolver():
    """
    Solves the steady state counts of # reactive robots {searching, avoiding, homing} running the
    CRW controller in a foraging task.

    If N=1, requires the following parameters:
    - N - How many robots are in the swarm.
    - T - Length of simulation in timesteps.
    - n_datapoints - How many datapoints were taken during simulation.
    - tau_h1 - Average homing time for 1 robot.
    - tau_av1 - Average collision avoidance time for 1 robot.
    - alpha_ca1 - Rate of entering collision avoidance for 1 robot.
    - alpha_b - Rate of robots encountering blocks.

    If N > 1, the following addition parameters are needed:
    - tau_avN - Average collision avoidance time for a robot in a N robot swarm.
    - alpha_caN - Rate of entering collision avoidance for 1 robot.
    """

    def __init__(self, params: tp.Dict[str, float]):
        self.params = params

    def solve(self, z0: tp.Dict[str, float]):

        # initial conditions (can be changed)
        # z0 = [1, 0, 0, 20]
        # z0 = initial_conditions

        # time points
        t = np.linspace(0, self.params['T'], self.params['n_datapoints'])
        z0_arr = [z0['N_s0'], z0['N_h0'],  z0['N_avs0'], z0['B0']]
        z = si.odeint(self.kernel, z0_arr, t, args=(self.params,))

        return z

    @staticmethod
    def kernel(z: tp.Dict[str, float], t: np.array, params: tp.Dict[str, float]):
        N = params['N']
        N_s = z[0]
        N_h = z[1]
        N_avs = z[2]
        N_avh = N - N_s - N_h - N_avs
        B = z[3]

        if N == 1:
            tau_av = params['tau_av1']
            alpha_ca = params['alpha_ca1']
            tau_h = params['tau_h1']
            alpha_b = params['alpha_b0']

        else:
            tau_av = params['tau_avN']
            alpha_ca = params['alpha_caN']
            tau_h = params['tau_hN']
            alpha_b = params['alpha_bN']

        #
        # ODE terms: dN_s, dN_h, dN_avs, dB. dN_avh is NOT computed here, as it can be obtained from
        # conservation of robots.
        #
        dN_s =\
            - alpha_b\
            - (alpha_ca) \
            + (N_avs / tau_av)  \
            + (N_h / tau_h)

        dN_h = \
            alpha_b\
            + (N_avh / tau_av) \
            - (N_h / tau_h)

        # dN_avh = alpha_ca - (N_avh / tau_av)

        dN_avs = (alpha_ca) - (N_avs / tau_av)

        dB = (N_h / tau_h) - alpha_b

        return [dN_s, dN_h, dN_avs, dB]
