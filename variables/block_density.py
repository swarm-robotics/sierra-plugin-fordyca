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
"""
Classes for the block density batch criteria. See :ref:`ln-bc-block-density` for usage
documentation.
"""

# Core packages
import typing as tp
import os

# 3rd party packages
import implements

# Project packages
from core.variables import constant_density as cd
import core.generators.scenario_generator_parser as sgp
import core.utils
from core.vector import Vector3D
from core.xml_luigi import XMLAttrChange, XMLAttrChangeSet


@implements.implements(core.variables.batch_criteria.IConcreteBatchCriteria)
class BlockConstantDensity(cd.ConstantDensity):
    """
    A univariate range specifiying the block density (ratio of block count to arena size) to hold
    constant as arena size is increased. This class is a base class which should (almost)
    never be used on its own. Instead, the ``factory()`` function should be used to dynamically
    create derived classes expressing the user's desired density.
    """

    def __init__(self,
                 cli_arg: str,
                 main_config: tp.Dict[str, str],
                 batch_input_root: str,
                 target_density: float,
                 dimensions: tp.List[core.utils.ArenaExtent],
                 dist_type: str) -> None:
        cd.ConstantDensity.__init__(self,
                                    cli_arg,
                                    main_config,
                                    batch_input_root,
                                    target_density,
                                    dimensions,
                                    dist_type)
        self.already_added = False

    def gen_attr_changelist(self) -> tp.List[XMLAttrChangeSet]:
        """
        Generate list of sets of changes to input file to set the # blocks for a set of arena
        sizes such that the blocks density is constant. Blocks are approximated as point masses.
        """
        if not self.already_added:
            for changeset in self.attr_changes:
                for c in changeset:
                    if c.path == ".//arena" and c.attr == "size":
                        x, y, z = c.value.split(',')
                        dims = Vector3D(float(x), float(y), float(z))
                        extent = core.utils.ArenaExtent(dims)

                        # Always need at least 1 block
                        n_blocks = max(2, extent.area() * (self.target_density / 100.0))

                        changeset.add(XMLAttrChange(".//arena_map/blocks/distribution/manifest",
                                                    "n_cube",
                                                    "{0}".format(int(n_blocks / 2.0))))
                        changeset.add(XMLAttrChange(".//arena_map/blocks/distribution/manifest",
                                                    "n_ramp",
                                                    "{0}".format(int(n_blocks / 2.0))))
                        break
            self.already_added = True

        return self.attr_changes

    def gen_exp_dirnames(self, cmdopts: dict) -> tp.List[str]:
        changes = self.gen_attr_changelist()
        return ['exp' + str(x) for x in range(0, len(changes))]

    def graph_xticks(self,
                     cmdopts: dict,
                     exp_dirs: tp.List[str] = None) -> tp.List[float]:
        if exp_dirs is None:
            exp_dirs = self.gen_exp_dirnames(cmdopts)

        areas = []
        for d in exp_dirs:
            pkl_path = os.path.join(self.batch_input_root,
                                    d,
                                    core.config.kPickleLeaf)
            exp_def = XMLAttrChangeSet.unpickle(pkl_path)
            areas.append(core.utils.extract_arena_dims(exp_def).area())

        return areas

    def graph_xticklabels(self,
                          cmdopts: dict,
                          exp_dirs: tp.List[str] = None) -> tp.List[str]:
        return [str(x) + r' $m^2$' for x in self.graph_xticks(cmdopts, exp_dirs)]

    def graph_xlabel(self, cmdopts: dict) -> str:
        return r"Block Density ({0}\%)".format(self.target_density)

    def pm_query(self, pm: str) -> bool:
        return pm in ['raw']

    def inter_exp_graphs_exclude_exp0(self) -> bool:
        return False


def factory(cli_arg: str,
            main_config: tp.Dict[str, str],
            batch_input_root: str,
            **kwargs) -> BlockConstantDensity:
    """
    Factory to create :class:`BlockConstantDensity` derived classes from the command line definition
    of batch criteria.

    """
    attr = cd.Parser()(cli_arg)
    kw = sgp.ScenarioGeneratorParser.reparse_str(kwargs['scenario'])

    if kw['dist_type'] == "SS" or kw['dist_type'] == "DS":
        r = range(kw['arena_x'],
                  kw['arena_x'] + attr['cardinality'] * attr['arena_size_inc'],
                  attr['arena_size_inc'])
        dims = [core.utils.ArenaExtent(Vector3D(x, x / 2.0, 0)) for x in r]
    elif kw['dist_type'] == "PL" or kw['dist_type'] == "RN":
        r = range(kw['arena_x'],
                  kw['arena_x'] + attr['cardinality'] * attr['arena_size_inc'],
                  attr['arena_size_inc'])
        dims = [core.utils.ArenaExtent(Vector3D(x, x, 0)) for x in r]
    else:
        raise NotImplementedError(
            "Unsupported block dstribution '{0}': Only SS,DS,QS,RN supported".format(kw['dist_type']))

    def __init__(self) -> None:
        BlockConstantDensity.__init__(self,
                                      cli_arg,
                                      main_config,
                                      batch_input_root,
                                      attr["target_density"],
                                      dims,
                                      kw['dist_type'])

    return type(cli_arg,  # type: ignore
                (BlockConstantDensity,),
                {"__init__": __init__})


__api__ = [
    'BlockConstantDensity'
]
