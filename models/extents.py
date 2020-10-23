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
import typing as tp
import os

# 3rd party packages

# Project packages
import core.utils
from core.experiment_spec import ExperimentSpec
import core.generators.scenario_generator_parser as sgp
import projects.fordyca.variables.nest_pose as np


def nest_extent_calc(cmdopts, criteria, exp_num: int) -> core.utils.ArenaExtent:
    # Get nest position
    spec = ExperimentSpec(criteria, cmdopts, exp_num)
    res = sgp.ScenarioGeneratorParser.reparse_str(cmdopts['scenario'])
    pose = np.NestPose(res['dist_type'], [spec.arena_dim])

    for path, attr, val in pose.gen_attr_changelist()[0]:
        if 'nest' in path and 'center' in attr:
            nest_xcenter, nest_ycenter = val.split(',')
        if 'nest' in path and 'dims' in attr:
            nest_xdim, nest_ydim = val.split(',')

    dims = (float(nest_xdim), float(nest_ydim), 0)
    offset = (float(nest_xcenter) - float(nest_xdim) / 2.0,
              float(nest_ycenter) - float(nest_ydim) / 2.0,
              0)
    return core.utils.ArenaExtent(dims, offset)


def cluster_extents_calc(main_config: dict,
                         cmdopts: dict,
                         sim_dir: str) -> tp.List[core.utils.ArenaExtent]:
    """
    Given a simulation directory within an experiment in a batch, calculate the
    :class:`~core.utils.ArenaExtent`s for all clusters within the arena.
    """
    clusters_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_output_root'],
                                                      sim_dir,
                                                      main_config['sim']['sim_metrics_leaf'],
                                                      'block-clusters.csv'))
    n_clusters = len([c for c in clusters_df.columns if 'xmin' in c])

    # Create extents from clusters
    extents = []
    for c in range(0, n_clusters):
        xmin = clusters_df.filter(regex='cluster' + str(c) + '_xmin').iloc[-1].values[0]
        xmax = clusters_df.filter(regex='cluster' + str(c) + '_xmax').iloc[-1].values[0]
        ymin = clusters_df.filter(regex='cluster' + str(c) + '_ymin').iloc[-1].values[0]
        ymax = clusters_df.filter(regex='cluster' + str(c) + '_ymax').iloc[-1].values[0]
        extents.append(core.utils.ArenaExtent((xmax - xmin, ymax - ymin, 0),
                                              (xmin, ymin, 0)))

    return extents
