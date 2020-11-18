# Copyright 2019 John Harwell, All rights reserved.
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
Extensions to :class:`core.generators.BaseScenarioGenerator` common to all FORDYCA scenarios.
"""
# Core packages
import typing as tp
import pickle

# 3rd party packages

# Project packages
from core.utils import ArenaExtent as ArenaExtent
from core.xml_luigi import XMLLuigi
from core.generators.scenario_generator import ARGoSScenarioGenerator
from core.variables import block_distribution
from projects.fordyca.variables import nest_pose, arena
from core.variables import block_quantity


class CommonScenarioGenerator(ARGoSScenarioGenerator):
    def __init__(self, *args, **kwargs) -> None:
        ARGoSScenarioGenerator.__init__(self, *args, **kwargs)

    def generate_arena_map(self, exp_def: XMLLuigi, arena: arena.RectangularArena) -> None:
        """
        Generate XML changes for the specified arena map configuration.

        Writes generated changes to the simulation definition pickle file.
        """
        chgs = arena.gen_attr_changelist()[0]
        for a in chgs:
            exp_def.attr_change(a[0], a[1], a[2])

        with open(self.spec.exp_def_fpath, 'ab') as f:
            pickle.dump(chgs, f)

        rms = arena.gen_tag_rmlist()
        if rms:  # non-empty
            for a in rms[0]:
                exp_def.tag_remove(a[0], a[1])

    @staticmethod
    def generate_block_dist(exp_def: XMLLuigi,
                            block_dist: block_distribution.BaseDistribution):
        """
        Generate XML changes for the specified block distribution.

        Does not write generated changes to the simulation definition pickle file.
        """
        for a in block_dist.gen_attr_changelist()[0]:
            exp_def.attr_change(a[0], a[1], a[2])

        rms = block_dist.gen_tag_rmlist()
        if rms:  # non-empty
            for a in rms[0]:
                exp_def.tag_remove(a[0], a[1])

    def generate_block_count(self, exp_def: XMLLuigi):
        """
        Generates XML changes for # blocks in the simulation. If specified on the cmdline, that
        quantity is used (split evenly between ramp and cube blocks).

        Writes generated changes to the simulation definition pickle file.
        """
        if self.cmdopts['n_blocks'] is not None:
            n_blocks = self.cmdopts['n_blocks']
            chgs1 = block_quantity.BlockQuantity.gen_attr_changelist_from_list([n_blocks / 2],
                                                                               'cube')
            chgs2 = block_quantity.BlockQuantity.gen_attr_changelist_from_list([n_blocks / 2],
                                                                               'ramp')
        else:
            # This may have already been set by the batch criteria, but we can't know for sure, and
            # we need block quantity definitions to always be written to the pickle file for later
            # retrieval.
            n_blocks1 = int(exp_def.attr_get('.//manifest', 'n_cube'))
            n_blocks2 = int(exp_def.attr_get('.//manifest', 'n_ramp'))

            chgs1 = block_quantity.BlockQuantity.gen_attr_changelist_from_list([n_blocks1],
                                                                               'cube')
            chgs2 = block_quantity.BlockQuantity.gen_attr_changelist_from_list([n_blocks2],
                                                                               'ramp')
        chgs = [chgs1, chgs2]

        for chgl in chgs:
            for chg in chgl[0]:
                exp_def.attr_change(chg[0], chg[1], chg[2])

            with open(self.spec.exp_def_fpath, 'ab') as f:
                pickle.dump(chgl[0], f)


class SSGenerator(CommonScenarioGenerator):
    """
    Generates XML changes for single source foraging.

    This includes:

    - Rectangular 2x1 arena
    - Single source block distribution
    """

    def __init__(self, *args, **kwargs) -> None:
        CommonScenarioGenerator.__init__(self, *args, **kwargs)

    def generate(self):
        exp_def = super().generate()

        # Generate physics engine definitions
        self.generate_physics(exp_def,
                              self.cmdopts,
                              self.cmdopts['physics_engine_type2D'],
                              self.cmdopts['physics_n_engines'],
                              [self.spec.arena_dim])

        # Generate arena definitions
        assert self.spec.arena_dim.xsize() == 2 * self.spec.arena_dim.ysize(),\
            "FATAL: SS distribution requires a 2x1 arena: xdim={0},ydim={1}".format(self.spec.arena_dim.xsize(),
                                                                                    self.spec.arena_dim.ysize())

        arena_map = arena.RectangularArenaTwoByOne(x_range=[self.spec.arena_dim.xsize()],
                                                   y_range=[
                                                   self.spec.arena_dim.ysize()],
                                                   z=self.spec.arena_dim.zsize(),
                                                   dist_type='SS')
        self.generate_arena_map(exp_def, arena_map)

        # Generate and apply block distribution type definitions
        self.generate_block_dist(exp_def, block_distribution.SingleSourceDistribution())

        # Generate and apply # blocks definitions
        self.generate_block_count(exp_def)

        # Generate and apply robot count definitions
        self.generate_n_robots(exp_def)

        return exp_def


class DSGenerator(CommonScenarioGenerator):
    """
    Generates XML changes for dual source foraging.

    This includes:

    - Rectangular 2x1 arena
    - Dual source block distribution

    """

    def __init__(self, *args, **kwargs) -> None:
        CommonScenarioGenerator.__init__(self, *args, **kwargs)

    def generate(self):
        exp_def = super().generate()

        # Generate physics engine definitions
        self.generate_physics(exp_def,
                              self.cmdopts,
                              self.cmdopts['physics_engine_type2D'],
                              self.cmdopts['physics_n_engines'],
                              [self.spec.arena_dim])

        # Generate arena definitions
        assert self.spec.arena_dim.xsize() == 2 * self.spec.arena_dim.ysize(),\
            "FATAL: DS distribution requires a 2x1 arena: xdim={0},ydim={1}".format(self.spec.arena_dim.xsize(),
                                                                                    self.spec.arena_dim.ysize())

        arena_map = arena.RectangularArenaTwoByOne(x_range=[self.spec.arena_dim.xsize()],
                                                   y_range=[
                                                       self.spec.arena_dim.ysize()],
                                                   z=self.spec.arena_dim.zsize(),
                                                   dist_type='DS')
        self.generate_arena_map(exp_def, arena_map)

        # Generate and apply block distribution type definitions
        super().generate_block_dist(exp_def, block_distribution.DualSourceDistribution())

        # Generate and apply # blocks definitions
        self.generate_block_count(exp_def)

        # Generate and apply robot count definitions
        self.generate_n_robots(exp_def)

        return exp_def


class QSGenerator(CommonScenarioGenerator):
    """
    Generates XML changes for quad source foraging.

    This includes:

    - Square arena
    - Quad source block distribution

    """

    def __init__(self, *args, **kwargs) -> None:
        CommonScenarioGenerator.__init__(self, *args, **kwargs)

    def generate(self):
        exp_def = super().generate()

        # Generate physics engine definitions
        self.generate_physics(exp_def,
                              self.cmdopts,
                              self.cmdopts['physics_engine_type2D'],
                              self.cmdopts['physics_n_engines'],
                              [self.spec.arena_dim])

        # Generate arena definitions
        assert self.spec.arena_dim.xsize() == self.spec.arena_dim.ysize(),\
            "FATAL: QS distribution requires a square arena: xdim={0},ydim={1}".format(self.spec.arena_dim.xsize(),
                                                                                       self.spec.arena_dim.ysize())

        arena_map = arena.SquareArena(sqrange=[self.spec.arena_dim.xsize()],
                                      z=self.spec.arena_dim.zsize(),
                                      dist_type='QS')
        self.generate_arena_map(exp_def, arena_map)

        # Generate and apply block distribution type definitions
        source = block_distribution.QuadSourceDistribution()
        super().generate_block_dist(exp_def, source)

        # Generate and apply # blocks definitions
        self.generate_block_count(exp_def)

        # Generate and apply robot count definitions
        self.generate_n_robots(exp_def)

        return exp_def


class RNGenerator(CommonScenarioGenerator):
    """
    Generates XML changes for random foraging.

    This includes:

    - Square arena
    - Random block distribution

    """

    def __init__(self, *args, **kwargs) -> None:
        CommonScenarioGenerator.__init__(self, *args, **kwargs)

    def generate(self):
        exp_def = self.common_defs.generate()

        # Generate physics engine definitions
        self.generate_physics(exp_def,
                              self.cmdopts,
                              self.cmdopts['physics_engine_type2D'],
                              self.cmdopts['physics_n_engines'],
                              [self.spec.arena_dim])

        # Generate arena definitions
        assert self.spec.arena_dim.xsize() == self.spec.arena_dim.ysize(),\
            "FATAL: RN distribution requires a square arena: xdim={0},ydim={1}".format(self.spec.arena_dim.xsize(),
                                                                                       self.spec.arena_dim.ysize())
        arena_map = arena.SquareArena(sqrange=[self.spec.arena_dim.xsize()],
                                      z=self.spec.arena_dim.zsize(),
                                      dist_type='RN')
        self.generate_arena_map(exp_def, arena_map)

        # Generate and apply block distribution type definitions
        super().generate_block_dist(exp_def, block_distribution.RandomDistribution())

        # Generate and apply # blocks definitions
        self.generate_block_count(exp_def)

        # Generate and apply robot count definitions
        self.generate_n_robots(exp_def)

        return exp_def


class PLGenerator(CommonScenarioGenerator):
    """
    Generates XML changes for powerlaw source foraging.

    This includes:

    - Square arena
    - Powerlaw block distribution

    """

    def __init__(self, *args, **kwargs) -> None:
        CommonScenarioGenerator.__init__(self, *args, **kwargs)

    def generate(self):
        exp_def = super().generate()

        # Generate physics engine definitions
        self.generate_physics(exp_def,
                              self.cmdopts,
                              self.cmdopts['physics_engine_type2D'],
                              self.cmdopts['physics_n_engines'],
                              [self.spec.arena_dim])

        # Generate arena definitions
        assert self.spec.arena_dim.xsize() == self.spec.arena_dim.ysize(),\
            "FATAL: PL distribution requires a square arena: xdim={0},ydim={1}".format(self.spec.arena_dim.xsize(),
                                                                                       self.spec.arena_dim.ysize())

        arena_map = arena.SquareArena(sqrange=[self.spec.arena_dim.xsize()],
                                      z=self.spec.arena_dim.zsize(),
                                      dist_type='PL')
        self.generate_arena_map(exp_def, arena_map)

        # Generate and apply block distribution type definitions
        super().generate_block_dist(exp_def, block_distribution.PowerLawDistribution(self.spec.arena_dim))

        # Generate and apply # blocks definitions
        self.generate_block_count(exp_def)

        # Generate and apply robot count definitions
        self.generate_n_robots(exp_def)

        return exp_def


__api__ = [
    'SSGenerator',
    'DSGenerator',
    'QSGenerator',
    'PLGenerator',
    'RNGenerator',
]
