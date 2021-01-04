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

# Core packages
import typing as tp

# 3rd party packages
import implements

# Project packages
from core.variables.base_variable import IBaseVariable
from core.utils import ArenaExtent as ArenaExtent
from core.xml_luigi import XMLAttrChangeSet, XMLTagRmList, XMLTagAddList, XMLTagRm, XMLTagAdd, XMLAttrChange


@implements.implements(IBaseVariable)
class StaticCache():

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
        self.attr_changes = None

    def gen_attr_changelist(self) -> tp.List[XMLAttrChangeSet]:
        """
        Generate list of sets of changes necessary to make to the input file to correctly set up the
        simulation for the list of static cache sizes specified in constructor.

        - Disables dynamic caches
        - Enables static caches
        - Sets static cache size (initial # blocks upon creation) and its dimensions in the arena
          during its existence.
        """
        if self.attr_changes is None:
            self.attr_changes = [XMLAttrChangeSet(
                XMLAttrChange(".//loop_functions/caches/dynamic", "enable", "false"),
                XMLAttrChange(".//loop_functions/caches/static", "enable", "true"),
                XMLAttrChange(".//loop_functions/caches/static", "size", "{0:.9f}".format(s)),
                XMLAttrChange(".//loop_functions/caches", "dimension", "{0:.9f}".format(max(e.ur.x * 0.20,
                                                                                            e.ur.y * 0.20)))
            ) for e in self.extents for s in self.sizes]
        return self.attr_changes

    def gen_tag_rmlist(self) -> tp.List[XMLTagRmList]:
        return []

    def gen_tag_addlist(self) -> tp.List[XMLTagAddList]:
        return []
