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
import os
import typing as tp

# 3rd party packages
import pandas as pd

# Project packages
import core.utils
import core.variables.batch_criteria as bc
import core.generators.scenario_generator_parser as sgp
import projects.fordyca.variables.nest_pose as np
from core.experiment_spec import ExperimentSpec
from core.utils import ArenaExtent
from core.vector import Vector3D


class BlockCluster():
    """
    Representation of a block cluster object within the arena.
    """
    @classmethod
    def from_df(cls, clusters_df: pd.DataFrame, cluster_id: int) -> 'BlockCluster':
        col_stem = 'cluster' + str(cluster_id)

        xmin = clusters_df.filter(regex=col_stem + '_xmin').iloc[-1].values[0]
        xmax = clusters_df.filter(regex=col_stem + '_xmax').iloc[-1].values[0]
        ymin = clusters_df.filter(regex=col_stem + '_ymin').iloc[-1].values[0]
        ymax = clusters_df.filter(regex=col_stem + '_ymax').iloc[-1].values[0]

        avg_blocks = clusters_df.filter(regex='cum_avg_' + col_stem +
                                        '_block_count').iloc[-1].values[0]
        return BlockCluster(ll=Vector3D(xmin, ymin),
                            ur=Vector3D(xmax, ymax),
                            cluster_id=cluster_id,
                            avg_blocks=avg_blocks)

    def __init__(self, ll: Vector3D, ur: Vector3D, cluster_id: int, avg_blocks: float) -> None:
        self.extent = ArenaExtent.from_corners(ll=ll, ur=ur)
        self.avg_blocks = avg_blocks


class Nest():
    def __init__(self, cmdopts: dict, criteria: bc.IConcreteBatchCriteria, exp_num: int):
        # Get nest position
        spec = ExperimentSpec(criteria, exp_num, cmdopts)
        res = sgp.ScenarioGeneratorParser.reparse_str(cmdopts['scenario'])
        pose = np.NestPose(res['dist_type'], [spec.arena_dim])

        for path, attr, val in pose.gen_attr_changelist()[0]:
            if 'nest' in path and 'center' in attr:
                x, y = val.split(',')
                center = Vector3D(float(x), float(y), 0.0)
            if 'nest' in path and 'dims' in attr:
                x, y = val.split(',')
                dims = Vector3D(float(x), float(y), 0.0)

        self.extent = ArenaExtent(dims, center - dims / 2.0)


class BlockClusterSet():
    """
    Given a simulation directory within an experiment in a batch, calculate the
    :class:`BlockCluster`s for all clusters within the arena.

    Arguments:
       main_config: Main YAML configuration for project.
       cmdopts: Parsed cmdline parameters.
       sim_opath: Directory path in which the ``block-clusters.csv`` can be found.
    """

    def __init__(self,
                 main_config: dict,
                 cmdopts: dict,
                 nest: Nest,
                 sim_opath: str) -> None:

        clusters_df = core.utils.pd_csv_read(os.path.join(sim_opath, 'block-clusters.csv'))
        n_clusters = len([c for c in clusters_df.columns if 'xmin' in c])

        # Create extents from clusters
        self.clusters = set()

        # RN block distribution has a single cluster, but the nest is in the middle of it, which
        # makes density calculations much trickier when integrating across/up to the nest (modeled
        # as a single point). We break it into an equivalent set of 4 smaller clusters ringing the
        # nest to avoid computational issues.
        if 'RN' in cmdopts['scenario']:
            cluster = BlockCluster.from_df(clusters_df, 0)

            ll1 = cluster.extent.ll
            ur1 = Vector3D(nest.extent.ll.x, cluster.extent.ur.y)
            c1 = BlockCluster(ll=ll1,
                              ur=ur1,
                              cluster_id=0,
                              avg_blocks=cluster.avg_blocks / 4.0)

            ll2 = Vector3D(nest.extent.ll.x, cluster.extent.ll.y)
            ur2 = Vector3D(nest.extent.ur.x, nest.extent.ll.y)
            c2 = BlockCluster(ll=ll2,
                              ur=ur2,
                              cluster_id=1,
                              avg_blocks=cluster.avg_blocks / 4.0)

            ll3 = Vector3D(nest.extent.ll.x, nest.extent.ur.y)
            ur3 = Vector3D(nest.extent.ur.x, cluster.extent.ur.y)
            c3 = BlockCluster(ll=ll3,
                              ur=ur3,
                              cluster_id=2,
                              avg_blocks=cluster.avg_blocks / 4.0)

            ll4 = Vector3D(nest.extent.ur.x, cluster.extent.ll.y)
            ur4 = cluster.extent.ur
            c4 = BlockCluster(ll=ll4,
                              ur=ur4,
                              cluster_id=3,
                              avg_blocks=cluster.avg_blocks / 4.0)
            self.clusters = set([c1, c2, c3, c4])

        else:  # General case
            for c in range(0, n_clusters):
                self.clusters |= set([BlockCluster.from_df(clusters_df, c)])

    def __iter__(self):
        return iter(self.clusters)

    def __len__(self):
        return len(self.clusters)
