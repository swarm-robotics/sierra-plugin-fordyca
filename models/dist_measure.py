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
from core.vector import Vector3D
from projects.fordyca.models.representation import Nest


class DistanceMeasure2D():
    """
    Defines how the distance between two (X,Y) points in the plane should be measured. This is
    necessary in order to handle different block distributions within the same model.
    """

    def __init__(self, scenario: str, nest: Nest):
        self.scenario = scenario
        self.nest = nest

        if 'RN' in self.scenario or 'PL' in self.scenario:
            # Our model assumes all robots finish foraging EXACTLY at the nest center, and the
            # implementation has robots pick a random point between where they enter the nest and
            # the center, in order to reduce congestion.
            #
            # This has the effect of making the expected distance the robots travel after entering
            # the nest but before dropping their object LESS than the distance the model
            # assumes. So, we calculate the average distance from any point in the square defined by
            # HALF the nest span in X,Y (HALF being a result of uniform random choice in X,Y) to the
            # nest center:
            # https://math.stackexchange.com/questions/15580/what-is-average-distance-from-center-of-square-to-some-point
            edge = nest.extent.xsize() / 2.0
            self.nest_factor = edge / 6.0 * (math.sqrt(2.0) + math.log(1 + math.sqrt(2.0)))

        elif 'SS' in self.scenario:
            res, _ = si.nquad(lambda x, y: (self.nest.extent.center - Vector3D(x, y)).length(),
                              [[self.nest.extent.center.x,
                                self.nest.extent.center.x + self.nest.extent.xsize() / 2.0],
                               [self.nest.extent.center.y - self.nest.extent.ysize() / 4.0,
                                self.nest.extent.center.y + self.nest.extent.ysize() / 4.0, ]],
                              opts={'limit': 100})
            self.nest_factor = res / (nest.extent.area())
        elif 'DS' in self.scenario:
            res, _ = si.nquad(lambda x, y: (self.nest.extent.center - Vector3D(x, y)).length(),
                              [[self.nest.extent.center.x - self.nest.extent.xsize() / 4.0,
                                self.nest.extent.center.x + self.nest.extent.xsize() / 4.0],
                               [self.nest.extent.center.y - self.nest.extent.ysize() / 8.0,
                                self.nest.extent.center.y + self.nest.extent.ysize() / 8.0, ]],
                              opts={'limit': 100})
            eff_area = nest.extent.xsize() / 2.0 * nest.extent.ysize() / 4.0
            self.nest_factor = res / eff_area

    def to_nest(self, pt: Vector3D):
        return (self.nest.extent.center - pt).length() - self.nest_factor
