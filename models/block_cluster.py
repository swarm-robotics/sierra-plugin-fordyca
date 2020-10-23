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


class BlockCluster():
    """
    Representation of a block cluster object within the arena.
    """

    def __init__(self, clusters_df: pd.DataFrame, cluster_id: int):
        col_stem = 'cluster' + str(cluster_id)

        xmin = clusters_df.filter(regex=col_stem + '_xmin').iloc[-1].values[0]
        xmax = clusters_df.filter(regex=col_stem + '_xmax').iloc[-1].values[0]
        ymin = clusters_df.filter(regex=col_stem + '_ymin').iloc[-1].values[0]
        ymax = clusters_df.filter(regex=col_stem + '_ymax').iloc[-1].values[0]
        self.extent = core.utils.ArenaExtent((xmax - xmin, ymax - ymin, 0),
                                             (xmin, ymin, 0))
        self.cum_avg_blocks = clusters_df.filter(regex='int_avg_' + col_stem +
                                                 '_block_count').iloc[-1].values[0]


class BlockClusterSet():
    """
    Given a simulation directory within an experiment in a batch, calculate the
    :class:`BlockCluster`s for all clusters within the arena.

    Arguments:
       main_config: Main YAML configuration for project.
       cmdopts: Parsed cmdline parameters.
       sim_opath: Directory path in which the ``block-clusters.csv`` can be found.
    """

    def __init__(self, main_config: dict, cmdopts: dict, sim_opath: str) -> None:

        clusters_df = core.utils.pd_csv_read(os.path.join(sim_opath, 'block-clusters.csv'))
        n_clusters = len([c for c in clusters_df.columns if 'xmin' in c])

        # Create extents from clusters
        self.clusters = set()
        for c in range(0, n_clusters):
            self.clusters |= set([BlockCluster(clusters_df, c)])

    def __iter__(self):
        return iter(self.clusters)

    def __len__(self):
        return len(self.clusters)
