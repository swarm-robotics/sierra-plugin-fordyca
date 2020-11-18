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

import math
import typing as tp
from core.variables.base_variable import IBaseVariable
from core.utils import ArenaExtent


class Convergence(IBaseVariable):

    """
    Maps a list of desired arena dimensions to a list of sets of XML changes to set up convergence
    XML configuration for the TITARRA project. This includes setup for the following C++ TITARRA
    components:

    - cpp:class:`~cosm::convergence::convergence_calculator`.

    Attributes:
        extents: List of :class:`~core.utils.ArenaExtent` arena dimensions.
        attr_changes: List of sets of XML changes to apply to a template input file.
    """

    def __init__(self, extents: tp.List[ArenaExtent]) -> None:
        self.extents = extents
        self.attr_changes = []  # type: tp.List

    def gen_attr_changelist(self) -> list:
        """
        Generate list of sets of changes necessary to make to the input file to correctly set up the
        simulation with the specified area size/shape.
        """
        if not self.attr_changes:
            self.attr_changes = [set([(".//convergence/positional_entropy",
                                       "horizon",
                                       "0:{0:.9f}".format(math.sqrt(extent.xsize() ** 2 + extent.ysize() ** 2))),
                                      (".//convergence/positional_entropy",
                                       "horizon_delta",
                                       "{0:.9f}".format(math.sqrt(extent.xsize() ** 2 + extent.ysize() ** 2) / 10.0)),
                                      ])
                                 for extent in self.extents]

        return self.attr_changes

    def gen_tag_rmlist(self) -> list:
        return []

    def gen_tag_addlist(self) -> list:
        return []


__api__ = [
    'Convergence',
]
