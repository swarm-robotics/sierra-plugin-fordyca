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

# Project packages
import projects.fordyca.generators.common as gc
from projects.fordyca.variables import dynamic_cache, static_cache
from core.utils import ArenaExtent as ArenaExtent
from core.xml_luigi import XMLLuigi


def generate_dynamic_cache(exp_def: XMLLuigi, extent: ArenaExtent):
    """
    Generate XML changes for dynamic cache usage (depth2 simulations only).

    Does not write generated changes to the simulation definition pickle file.
    """
    cache = dynamic_cache.DynamicCache([extent])

    [exp_def.attr_change(a[0], a[1], a[2]) for a in cache.gen_attr_changelist()[0]]
    rms = cache.gen_tag_rmlist()
    if rms:  # non-empty
        [exp_def.tag_remove(a) for a in rms[0]]


def generate_static_cache(exp_def: XMLLuigi,
                          extent: ArenaExtent,
                          cmdopts: tp.Dict[str, str]):
    """
    Generate XML changes for static cache usage (depth1 simulations only).

    Does not write generated changes to the simulation definition pickle file.
    """

    # If they specified how many blocks to use for static cache respawn, use that.
    # Otherwise use the floor of 2.
    if cmdopts['static_cache_blocks'] is None:
        cache = static_cache.StaticCache([2], [extent])
    else:
        cache = static_cache.StaticCache([cmdopts['static_cache_blocks']],
                                         [extent])

    [exp_def.attr_change(a[0], a[1], a[2]) for a in cache.gen_attr_changelist()[0]]
    rms = cache.gen_tag_rmlist()
    if rms:  # non-empty
        [exp_def.tag_remove(a) for a in rms[0]]


class SSGenerator(gc.SSGenerator):
    """
    FORDYCA extensions to the single source foraging generator
    :class:`~core.generators.scenario_generator.SSGenerator`.

    This includes:

    - Static caches
    - Dynamic caches

    """

    def __init__(self, *args, **kwargs):
        gc.SSGenerator.__init__(self, *args, **kwargs)

    def generate(self):

        exp_def = super().generate()

        if "depth1" in self.controller:
            generate_static_cache(exp_def, self.spec.arena_dim, self.cmdopts)
        if "depth2" in self.controller:
            generate_dynamic_cache(exp_def, self.spec.arena_dim)

        return exp_def


class DSGenerator(gc.DSGenerator):
    """
    FORDYCA extensions to the single source foraging generator
    :class:`~core.generators.single_source.DSGenerator`.

    This includes:

    - Static caches
    - Dynamic caches

    """

    def __init__(self, *args, **kwargs):
        gc.DSGenerator.__init__(self, *args, **kwargs)

    def generate(self):

        exp_def = super().generate()

        if "depth1" in self.controller:
            generate_static_cache(exp_def, self.spec.arena_dim, self.cmdopts)
        if "depth2" in self.controller:
            generate_dynamic_cache(exp_def, self.spec.arena_dim)

        return exp_def


class QSGenerator(gc.QSGenerator):
    """
    FORDYCA extensions to the single source foraging generator
    :class:`~core.generators.scenario_generator.QSGenerator`.

    This includes:

    - Static caches
    - Dynamic caches

    """

    def __init__(self, *args, **kwargs):
        gc.QSGenerator.__init__(self, *args, **kwargs)

    def generate(self):

        exp_def = super().generate()

        if "depth1" in self.controller:
            generate_static_cache(exp_def, self.spec.arena_dim, self.cmdopts)
        if "depth2" in self.controller:
            generate_dynamic_cache(exp_def, self.spec.arena_dim)

        return exp_def


class RNGenerator(gc.RNGenerator):
    """
    FORDYCA extensions to the single source foraging generator
    :class:`~core.generators.scenario_generator.RNGenerator`.

    This includes:

    - Static caches
    - Dynamic caches

    """

    def __init__(self, *args, **kwargs):
        gc.RNGenerator.__init__(self, *args, **kwargs)

    def generate(self):

        exp_def = super().generate()

        if "depth1" in self.controller:
            generate_static_cache(exp_def, self.spec.arena_dim, self.cmdopts)
        if "depth2" in self.controller:
            generate_dynamic_cache(exp_def, self.spec.arena_dim)

        return exp_def


class PLGenerator(gc.PLGenerator):
    """
    FORDYCA extensions to the single source foraging generator
    :class:`~core.generators.scenario_generator.PLGenerator`.

    This includes:

    - Static caches
    - Dynamic caches

    """

    def __init__(self, *args, **kwargs):
        gc.PLGenerator.__init__(self, *args, **kwargs)

    def generate(self):

        exp_def = super().generate()

        if "depth1" in self.controller:
            generate_static_cache(exp_def, self.spec.arena_dim, self.cmdopts)
        if "depth2" in self.controller:
            generate_dynamic_cache(exp_def, self.spec.arena_dim)

        return exp_def
