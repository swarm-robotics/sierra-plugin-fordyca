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
class StaticCache(IBaseVariable):

    """
    Defines the size and capacity of a static cache with.

    Attributes:
        sizes: List of the # of blocks the cache should have each time the simulation respawns
               it.
        extents: List of the extents within the arena to generate definitions for.
    """

    def __init__(self, sizes: tp.List[int], extents: tp.List[ArenaExtent]):
        self.sizes = sizes
        self.extents = extents

    def gen_attr_changelist(self) -> list:
        """
        Generate list of sets of changes necessary to make to the input file to correctly set up the
        simulation for the list of static cache sizes specified in constructor.

        - Disables dynamic caches
        - Enables static caches
        - Sets static cache size (initial # blocks upon creation) and its dimensions in the arena
          during its existence.
        """
        return [set([
            (".//loop_functions/caches/dynamic", "enable", "false"),
            (".//loop_functions/caches/static", "enable", "true"),
            (".//loop_functions/caches/static", "size", "{0:.9f}".format(s)),
            (".//loop_functions/caches", "dimension", "{0:.9f}".format(max(e.xmax * 0.20,
                                                                           e.ymax * 0.20)))
        ])
            for e in self.extents for s in self.sizes]

    def gen_tag_rmlist(self) -> list:
        return []

    def gen_tag_addlist(self) -> list:
        return []
