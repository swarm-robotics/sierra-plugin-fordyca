# Copyright 2021 John Harwell, All rights reserved.
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

# Core packages
import math

# 3rd party packages

# Project packages
import core.variables.time_setup as ts


def calc_crwD(N: float, wander_speed: float) -> float:
    """
    Approximates the diffusion constant in a swarm of N CRW robots for bounded arena geometry. From
    :xref:`Harwell2021a`, inspired by the results in :xref:`Codling2010`.

    """
    tick_len = 1.0 / ts.kTICKS_PER_SECOND
    characteristic = math.pow(2, 0.25)
    return (wander_speed ** 2 * math.sqrt(2) / (4 * tick_len)) * math.pow(N, characteristic)
