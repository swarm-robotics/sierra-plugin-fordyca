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

"""
Intra-experiment models for the swarm's spatial distribution in 2D space.
"""

# Core packages
import math
import os
import typing as tp
import copy

# 3rd party packages
import scipy.integrate as si
import implements
import pandas as pd

# Project packages
import models.interface
import core.utils
from core.experiment_spec import ExperimentSpec
import projects.fordyca.models.block_cluster as block_cluster
import core.variables.batch_criteria as bc
import projects.fordyca.models.extents as extents
from projects.fordyca.models.block_acq import BlockAcqDensity
from projects.fordyca.models.model_error import Model2DError


def available_models(category: str):
    if category == 'intra':
        return [  # 'IntraExpSearchingDistribution',
            'IntraExpAcqDistribution']
    elif category == 'inter':
        return ['InterExpAcqDistributionError']
    else:
        return None


@implements.implements(models.interface.IConcreteIntraExpModel2D)
class IntraExpSearchingDistribution(models.interface.IConcreteIntraExpModel2D):
    """
    Models swarm's steady state spatial distribution in 2D space when engaged in searching, assuming
    purely reactive robots.
    """

    def __init__(self, main_config: dict, config: dict):
        self.nest = None
        self.main_config = main_config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stem(self) -> str:
        return 'block-acq-explore-locs2D'

    def run(self,
            cmdopts: dict,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int) -> pd.DataFrame:

        # Calculate nest extent
        self.nest = extents.nest_extent_calc(cmdopts, criteria, exp_num)

        sim_dirs = [d for d in os.listdir(cmdopts['exp_output_root']) if
                    self.main_config['sierra']['avg_output_leaf'] not in d]

        # Get the arena dimensions in cells from the actual swarm distribution. This is OK because
        # we don't refer to the empirical results in any other way. We can't just use the dims in
        # cmdopts, because those are real-valued--not descretized.
        df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_output_root'],
                                                 sim_dirs[0],
                                                 self.main_config['sim']['sim_metrics_leaf'],
                                                 'block-acq-explore-locs2D.csv'))

        # Calculate arena resolution
        spec = ExperimentSpec(criteria, cmdopts, exp_num)
        resolution = spec.arena_dim.x() / len(df.index)

        # Calculate block density
        # exp_def = core.utils.unpickle_exp_def(spec.exp_def_fpath)
        # area = spec.arena_dim.x() * spec.arena_dim.y()
        # n_blocks = 0

        # for _, attr, value in exp_def:
        #     if 'n_cube' in attr:
        #         n_blocks = int(value)

        # block_density = n_blocks / 156.24

        res_df = pd.DataFrame(columns=[df.columns], index=df.index)

        # We calculate per-sim, rather than using the aggregate block cluster results, because for
        # power law distributions, different simulations have different cluster locations, which
        # affects the distribution via locality.
        for d in sim_dirs:
            self._calc_for_sim(resolution, res_df)

        # Average our results across all simulations
        res_df /= len(sim_dirs)

        # Normalize our results to get a true probability distribution
        res_df /= res_df.values.sum()

        # All done!
        return res_df

    def _calc_for_sim(self,
                      resolution: float,
                      res_df: pd.DataFrame):
        for i in range(0, len(res_df.index)):
            for j in range(0, len(res_df.columns)):
                res_df.iloc[i, j] += self._calc_searching_density(i, j, resolution)

        return res_df

    def _searching_linear_inv(self, y: float, x: float):
        nest_dist = math.sqrt((x - self.nest.xcenter) ** 2 + (y - self.nest.ycenter) ** 2)
        return 1.0 / math.pow(nest_dist, 1.0 / 4.0)

    def _calc_searching_density(self, i: int, j: int, resolution: float):
        density, _ = si.nquad(self._searching_linear_inv,
                              [[i * resolution, (i + 1) * resolution],
                               [j * resolution, (j + 1) * resolution]])
        return density


@implements.implements(models.interface.IConcreteInterExpModel1D)
class InterExpSearchingDistributionError(models.interface.IConcreteInterExpModel1D):
    """
    Runs :class:`IntraExpSearchingDistribution` for each experiment in the batch, and compute the average
    error between model prediction and empirical data as a single data point.

    In order for this model to run, all experiments in the batch must have 1 robot.

    The model is the same for reactive and cognitive robots, as robots use light sensors to return
    to the nest regardless of their memory model.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config
        self.nest = None

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return all([p == 1 for p in criteria.populations(cmdopts)])

    def target_csv_stem(self) -> str:
        return 'block-explore-locs2D-LN-model-error'

    def legend_name(self) -> str:
        return 'Model Error'

    def run(self,
            cmdopts: dict,
            criteria: bc.IConcreteBatchCriteria) -> pd.DataFrame:

        error = Model2DError('block-explore-locs2D.stddev',
                             IntraExpSearchingDistribution,
                             self.main_config,
                             self.config)
        return error.generate(cmdopts, criteria)


@implements.implements(models.interface.IConcreteIntraExpModel2D)
class IntraExpAcqDistribution(models.interface.IConcreteIntraExpModel2D):
    """
    Models the steady state spatial distribution of the locations which robots acquire blocks,
    assuming purely reactive robots.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stem(self) -> str:
        return 'block-acq-locs2D'

    def run(self,
            cmdopts: dict,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int) -> pd.DataFrame:

        # Calculate nest extent
        nest = extents.nest_extent_calc(cmdopts, criteria, exp_num)

        # We calculate per-sim, rather than using the averaged block cluster results, because for
        # power law distributions different simulations have different cluster locations, which
        # affects the distribution via locality.
        #
        # For all other block distributions, we can operate on the averaged results, because the
        # position of block clusters is the same in all simulations.
        if 'PL' in cmdopts['scenario']:
            result_opaths = [os.path.join(cmdopts['exp_output_root'],
                                          d,
                                          self.main_config['sim']['sim_metrics_leaf'])
                             for d in os.listdir(cmdopts['exp_output_root']) if
                             self.main_config['sierra']['avg_output_leaf'] not in d]
        else:
            result_opaths = [os.path.join(cmdopts['exp_avgd_root'])]

        # Calculate arena resolution
        acq_df = core.utils.pd_csv_read(os.path.join(result_opaths[0], 'block-acq-locs2D.csv'))
        spec = ExperimentSpec(criteria, cmdopts, exp_num)
        resolution = spec.arena_dim.x() / len(acq_df.index)

        # Acquisition distribution will be 0 outside of block clusters
        res_df = pd.DataFrame(columns=acq_df.columns, index=acq_df.index, dtype=float)
        res_df[:] = 0.0

        # Calculate distribution for each result path
        for path in result_opaths:
            self._calc_for_result(cmdopts, path, resolution, nest, res_df)

        # Average our results
        res_df /= len(result_opaths)

        # All done!
        return res_df

    def _calc_for_result(self,
                         cmdopts: dict,
                         result_opath: str,
                         resolution: float,
                         nest: core.utils.ArenaExtent,
                         res_df: pd.DataFrame):
        # Get clusters in the arena
        clusters = block_cluster.BlockClusterSet(self.main_config, cmdopts, result_opath)
        for cluster in clusters:
            rangex = range(int(cluster.extent.xmin / resolution),
                           int(cluster.extent.xmax / resolution))
            rangey = range(int(cluster.extent.ymin / resolution),
                           int(cluster.extent.ymax / resolution))

            density = BlockAcqDensity(nest=nest, cluster=cluster)
            for i in rangex:
                for j in rangey:
                    val = density.for_region(i * resolution,
                                             (i + 1) * resolution,
                                             j * resolution,
                                             (j + 1) * resolution)
                    res_df.iloc[i, j] += val

        print(res_df)
        return res_df


@implements.implements(models.interface.IConcreteInterExpModel1D)
class InterExpAcqDistributionError(models.interface.IConcreteInterExpModel1D):
    """
    Runs :class:`IntraExpAcqDistribution` for each experiment in the batch, and compute the average
    error between model prediction and empirical data as a single data point.

    In order for this model to run, all experiments in the batch must have 1 robot.

    The model is the same for reactive and cognitive robots, as robots use light sensors to return
    to the nest regardless of their memory model.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config
        self.nest = None

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return all([p == 1 for p in criteria.populations(cmdopts)])

    def target_csv_stem(self) -> str:
        return 'block-acq-locs2D-LN-model-error'

    def legend_name(self) -> str:
        return 'Model Error'

    def run(self,
            cmdopts: dict,
            criteria: bc.IConcreteBatchCriteria) -> pd.DataFrame:

        error = Model2DError('block-acq-locs2D.stddev',
                             IntraExpAcqDistribution,
                             self.main_config,
                             self.config)
        return error.generate(cmdopts, criteria)
