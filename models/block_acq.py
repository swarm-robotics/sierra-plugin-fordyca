# Copyright 2020 John Harwell, All rights reserved.
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
import scipy.integrate as si

# Project packages
import core.utils
import projects.fordyca.models.block_cluster as block_cluster


class BlockAcqDensity():
    """
    Block acquisition probability density calculations.
    """

    def __init__(self, nest: core.utils.ArenaExtent, cluster: block_cluster.BlockCluster):
        self.nest = nest

        rho_b = cluster.cum_avg_blocks / (cluster.extent.x() * cluster.extent.y())

        dist = math.sqrt((self.nest.xcenter - cluster.extent.xcenter) ** 2 +
                         (self.nest.ycenter - cluster.extent.ycenter) ** 2)
        dist_factor = math.log(dist ** 2)

        # Cluster contains nest -> RN distribution
        if dist <= 1.0:
            dist_factor = 1.0

        self.rho = math.pow(2 * rho_b, rho_b) / dist_factor

        print(dist, dist_factor, rho_b, self.rho)

        # TODO: The correct way would be to determine this analytically, which I can't figure out
        # yet. BUT the numeric approximation is pretty close.
        self.norm_factor = 1.0
        total = self.for_region(cluster.extent.xmin,
                                cluster.extent.xmax,
                                cluster.extent.ymin,
                                cluster.extent.ymax)
        self.norm_factor = 1.0 / total

    def at_point(self, x: float, y: float):
        r"""
        Calculate the block acquisition probability density at an (X,Y) point within the arena.

        .. math::
           \frac{1}{e^{{z}{e^{-\rho_b}}}}

        Arguments:
            z: Distance of (X,Y) point to the nest.
            rho_b: The block density at (X,Y).
        """
        # No acquisitions possible inside the nest.
        if self.nest.contains((x, y, 0.0)):
            return 0.0

        r = math.sqrt((x - self.nest.xcenter) ** 2 + (y - self.nest.ycenter) ** 2)

        return 1.0 / (math.exp(r * self.rho)) * self.norm_factor

    def for_region(self,
                   xmin: float,
                   xmax: float,
                   ymin: float,
                   ymax: float):
        r"""
        Calculate the cumulative block acquisition probability density within a region defined by
        [xmin, xmax], [ymin, ymax] using :method:`at_point`.

        """
        res, _ = si.nquad(self.at_point, [[xmin, xmax], [ymin, ymax]])
        return res

    def evx_for_region(self,
                       xmin: float,
                       xmax: float,
                       ymin: float,
                       ymax: float):
        """
        Calculate the expected value of the X coordinate of the average acquisition location within
        the region defined by [xmin, xmax], [ymin, ymax].
        """
        res, _ = si.nquad(self._evx, [[xmin, xmax], [ymin, ymax]])
        return res

    def evy_for_region(self,
                       xmin: float,
                       xmax: float,
                       ymin: float,
                       ymax: float):
        """
        Calculate the expected value of the Y coordinate of the average acquisition location within
        the region defined by [xmin, xmax], [ymin, ymax].
        """
        res, _ = si.nquad(self._evy, [[xmin, xmax], [ymin, ymax]])
        return res

    def _evx(self, x: float, y: float):
        """
        Calculate the expected value of the X coordinate of the average acquisition location in the
        cluster.
        """
        return self.at_point(x, y) * x

    def _evy(self, x: float, y: float):
        """
        Calculate the expected value of the Y coordinate of the average acquisition location in the
        cluster.
        """
        return self.at_point(x, y) * y
