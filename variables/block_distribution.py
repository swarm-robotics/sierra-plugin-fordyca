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
#

# core packages
import math
import typing as tp

# 3rd party packages
import implements
import logging

# Project packages
from core.variables.base_variable import IBaseVariable
from core.utils import ArenaExtent
from core.xml_luigi import XMLAttrChangeSet, XMLTagRmList, XMLTagAddList, XMLTagRm, XMLTagAdd, XMLAttrChange


@implements.implements(IBaseVariable)
class BaseDistribution():

    """
    Defines the type of distribution of objects in the arena.

    Attributes:
        dist_type: [single_source, dual_source, quad_source, powerlaw, random].
    """

    def __init__(self, dist_type: str) -> None:
        self.dist_type = dist_type
        self.attr_changes = []  # type: tp.List
        self.logger = logging.getLogger(__name__)

    def gen_attr_changelist(self) -> tp.List[XMLAttrChangeSet]:
        """
        Generate a list of sets of changes necessary to make to the input file to correctly set up
        the simulation with the specified block distribution
        """
        if not self.attr_changes:
            self.attr_changes = [XMLAttrChangeSet(XMLAttrChange(".//arena_map/blocks/distribution",
                                                                "dist_type",
                                                                "{0}".format(self.dist_type)))]
        return self.attr_changes

    def gen_tag_rmlist(self) -> tp.List[XMLTagRmList]:
        return []

    def gen_tag_addlist(self) -> tp.List[XMLTagAddList]:
        return []


class SingleSourceDistribution(BaseDistribution):
    def __init__(self) -> None:
        super().__init__("single_source")


class DualSourceDistribution(BaseDistribution):
    def __init__(self) -> None:
        super().__init__("dual_source")


class QuadSourceDistribution(BaseDistribution):
    def __init__(self) -> None:
        super().__init__("quad_source")


class PowerLawDistribution(BaseDistribution):
    def __init__(self, arena_dim: ArenaExtent) -> None:
        super().__init__("powerlaw")
        self.arena_dim = arena_dim

    def gen_attr_changelist(self):
        r"""
        Generate a list of sets of changes necessary to make to the input file to correctly set up
        the simulation for the powerlaw block distribution.

        2020/7/29: As a first guess, I've set the following parameters:

        - Min :math:`2`
        - Max :math:`2^k` where :math:`k=\lceil\sqrt{X}\rceil`
        - # clusters to :math:`X`

        2020/12/14: Update parameters, because with a 504x504 arena, you get 504 clusters of
        2^23x2^23 each, which causes a segfault. Now setting:

        2021/01/08: Update parameters, because with a 1448x1448 arena, you get a maximum of 7
        clusters of 2^7, which is way too few blocks for a large 10,000 robot swarm. Now setting:

        - Min :math:`2^\floor\ln{X}\rfloor`
        - Max :math:`2^k` where :math:`k=\lfloor{X^{1/3}}\rfloor`
        - # clusters to :math:`\lfloor{X^{1/2}}\rfloor`

        where :math:`X` is the arena dimension (assumed to be square). Not all of the clusters will
        be able to be placed in all likelihood for many arena layouts, but this is a good
        starting point.

        """
        changes = super().gen_attr_changelist()
        pwr_min = math.floor(math.log(self.arena_dim.xsize()))
        pwr_max = math.floor(math.pow(self.arena_dim.xsize(), 1.0 / 3.0))
        n_clusters = math.floor(math.pow(self.arena_dim.xsize(), 1.0 / 2.0))
        self.logger.debug("pwr_min=%s,pwr_max=%s,n_clusters=%s", pwr_min, pwr_max, n_clusters)

        for c in changes:
            c |= XMLAttrChangeSet(XMLAttrChange(".//arena_map/blocks/distribution/powerlaw",
                                                "pwr_min",
                                                "{0}".format(pwr_min)),
                                  XMLAttrChange(".//arena_map/blocks/distribution/powerlaw",
                                                "pwr_max",
                                                "{0}".format(pwr_max)),
                                  XMLAttrChange(".//arena_map/blocks/distribution/powerlaw",
                                                "n_clusters",
                                                "{0}".format(n_clusters)))
        return changes


class RandomDistribution(BaseDistribution):
    def __init__(self) -> None:
        super().__init__("random")


__api__ = [
    'BaseDistribution',
    'SingleSourceDistribution',
    'DualSourceDistribution',
    'QuadSourceDistribution',
    'PowerLawDistribution',
    'RandomDistribution'


]
