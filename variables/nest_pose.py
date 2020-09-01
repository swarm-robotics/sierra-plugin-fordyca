# Copyright 2018 John Harwell, All rights reserved.
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

import typing as tp

import implements

from core.variables.base_variable import IBaseVariable
from core.utils import ArenaExtent as ArenaExtent


@implements.implements(IBaseVariable)
class NestPose(IBaseVariable):

    """
    Defines the position/size of the nest based on block distribution type. Exactly ONE nest is
    required for FORDYCA foraging.

    Attributes:
      dist_type: The block distribution type. Valid values are [single_source, dual_source,
                                                                quad_source, random, powerlaw].
      extents: List of arena extents to generation nest poses for.
    """

    def __init__(self, dist_type: str, extents: tp.List[ArenaExtent]):
        self.dist_type = dist_type
        self.extents = extents

    def gen_attr_changelist(self) -> list:
        """
        Generate list of sets of changes necessary to make to the input file to correctly set up the
        simulation for the specified block distribution/nest.

        """
        if self.dist_type == "single_source":
            return [set([
                (".//arena_map/nests/nest",
                 "dims",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.1, s.ymax * 0.8)),
                (".//arena_map/nests/nest",
                 "center",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.1, s.ymax / 2.0)),
                (".//block_sel_matrix",
                 "nest",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.1, s.ymax / 2.0)),
            ]) for s in self.extents]
        elif self.dist_type == "dual_source":
            return [set([
                (".//arena_map/nests/nest",
                 "dims",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.1, s.ymax * 0.8)),
                (".//arena_map/nests/nest",
                 "center",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.5, s.ymax * 0.5)),
                (".//block_sel_matrix",
                 "nest",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.5, s.ymax * 0.5)),
            ]) for s in self.extents]
        elif (self.dist_type == "powerlaw" or self.dist_type == "random" or
              self.dist_type == "quad_source"):
            return [set([
                (".//arena_map/nests/nest",
                 "dims",
                 "{0:.9f}, {1:.9f}".format(s.xmax * 0.20, s.xmax * 0.20)),
                (".//arena_map/nests/nest", "center", "{0:.9f}, {0:.9f}".format(s.xmax * 0.5)),
                (".//block_sel_matrix", "nest", "{0:.9f}, {0:.9f}".format(s.xmax * 0.5)),
            ])
                for s in self.extents]
        else:
            # Eventually, I might want to have definitions for the other block distribution types
            raise NotImplementedError

    def gen_tag_rmlist(self) -> list:
        return []

    def gen_tag_addlist(self) -> list:
        return []
