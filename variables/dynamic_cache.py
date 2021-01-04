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
from core.xml_luigi import XMLAttrChangeSet, XMLAttrChange, XMLTagRmList, XMLTagAddList


@implements.implements(IBaseVariable)
class DynamicCache():

    """
    Defines the size and capacity of a dynamic cache to test with. Only really applicable to single
    source foraging scenarios, but will work with only types as well.

    Attributes:
        extents: List of the extents within the arena to generate definitions for.
    """

    def __init__(self, extents: tp.List[ArenaExtent]):
        self.extents = extents
        self.attr_changes = None

    def gen_attr_changelist(self) -> tp.List[XMLAttrChangeSet]:
        """
        Generate list of sets of changes necessary to make to the input file to correctly set up the
        simulation for the list of static cache sizes specified in constructor.

        - Disables dynamic caches
        - Enables static caches
        """
        if self.attr_changes is None:
            self.attr_changes = [XMLAttrChangeSet(
                XMLAttrChange(".//loop_functions/caches/dynamic", "enable", "true"),
                XMLAttrChange(".//loop_functions/caches/static", "enable", "false"),
                XMLAttrChange(".//loop_functions/caches/dynamic", "min_dist", "{0:.9f}".format(min(e.ur.x * 0.20,
                                                                                                   e.ur.y * 0.20))),

                XMLAttrChange(".//loop_functions/caches", "dimension", "{0:.9f}".format(max(e.ur.x * 0.20,
                                                                                            e.ur.y * 0.20))),

                # Set to dimensions of cache to ensure that caches will not be created such that they
                # overlap
                XMLAttrChange(".//cache_sel_matrix", "cache_prox_dist", "{0:.9f}".format(max(e.ur.x * 0.20,
                                                                                             e.ur.y * 0.20))),

                (".//cache_sel_matrix", "nest_prox_dist", "{0:.9f}".format(max(e.ur.x * 0.20,
                                                                               e.ur.y * 0.20))),

                (".//cache_sel_matrix", "block_prox_dist", "{0:.9f}".format(max(e.ur.x * 0.20,
                                                                                e.ur.y * 0.20))),

                (".//cache_sel_matrix", "site_xrange", "{0}:{1:.9f}".format(max(e.ur.x * 0.20,
                                                                                e.ur.y * 0.20) / 2.0,
                                                                            e.ur.x - max(e.ur.x * 0.20,
                                                                                         e.ur.y * 0.20) / 2.0)),
                (".//cache_sel_matrix", "site_yrange", "{0}:{1:.9f}".format(max(e.ur.x * 0.20,
                                                                                e.ur.y * 0.20) / 2.0,
                                                                            e.ur.y - max(e.ur.x * 0.20,
                                                                                         e.ur.y * 0.20) / 2.0)),
            ) for e in self.extents]
        return self.attr_changes

    def gen_tag_rmlist(self) -> tp.List[XMLTagRmList]:
        return []

    def gen_tag_addlist(self) -> tp.List[XMLTagAddList]:
        return []
