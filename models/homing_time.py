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
Intra- and inter-experiment models for the time it takes a single robot to return to the nest after
picking up an object.
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
import core.variables.time_setup as ts
import core.variables.batch_criteria as bc
import projects.fordyca.models.extents as extents
import projects.fordyca.models.block_cluster as block_cluster
from projects.fordyca.models.block_acq import BlockAcqDensity


def available_models(category: str):
    if category == 'intra':
        return ['IntraExpNestHomingTime1Robot']
    elif category == 'inter':
        return ['InterExpNestHomingTime1Robot']
    else:
        return None


@implements.implements(models.interface.IConcreteIntraExpModel1D)
class IntraExpNestHomingTime1Robot(models.interface.IConcreteIntraExpModel1D):
    """
    Models the time it takes a robot to return to the nest after it has picked up an object during
    foraging during a single experiment within a batch. That is, one model datapoint is computed for
    each metric collection interval in each simulation.

    In order to run for a given experiment in a batch, the swarm population size must be 1.

    The model is the same for reactive and cognitive robots, as robots use light sensors to return
    to the nest regardless of their memory model.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return criteria.populations(cmdopts)[i] == 1

    def target_csv_stem(self) -> str:
        return 'block-transport-time'

    def legend_name(self) -> str:
        return 'Predicted Homing Time'

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

        cluster_df = core.utils.pd_csv_read(os.path.join(result_opaths[0], 'block-clusters.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=cluster_df.index)
        res_df['model'] = 0.0

        for result in result_opaths:
            self._calc_for_result(cmdopts, result, nest, res_df)

        # Average our results
        res_df['model'] /= len(result_opaths)

        # All done!
        return res_df

    def _calc_for_result(self,
                         cmdopts: dict,
                         result_opath: str,
                         nest: core.utils.ArenaExtent,
                         res_df: pd.DataFrame):
        # Get clusters in the arena
        clusters = block_cluster.BlockClusterSet(self.main_config, cmdopts, result_opath)

        # Integrate to find average distance from nest to all clusters, weighted by acquisition
        # density.
        dist = 0.0
        for cluster in clusters:
            dist += self._calc_acq_edist(cluster, nest, cmdopts['scenario'])

        avg_dist = dist / len(clusters)
        spatial_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                         'spatial-movement.csv'))

        print(avg_dist)
        # Finally, calculate the average homing time for each interval in the simulation
        for idx in spatial_df.index:
            # Reported in cm/s, and we need m/s
            avg_vel = spatial_df.loc[idx, 'cum_avg_velocity_homing'] / 100.0

            # After getting the average distance to ANY block in ANY cluster in the arena, we can
            # compute the average time, in SECONDS, that robots spend returning to the nest.
            avg_homing_sec = avg_dist / avg_vel

            # Convert seconds to timesteps
            avg_homing_ts = avg_homing_sec * ts.kTICKS_PER_SECOND

            # All done!
            res_df.loc[idx, 'model'] += avg_homing_ts

    def _calc_acq_edist(self,
                        cluster: block_cluster.BlockCluster,
                        nest: core.utils.ArenaExtent,
                        scenario: str):
        density = BlockAcqDensity(nest=nest, cluster=cluster)

        # For random distributions, the nest in is in the center of the cluster extent, so we have
        # to integrate around it, otherwise the average X/Y acq location will be the center of the
        # nest.
        if 'RN' in scenario:
            # Compute expected value of X coordinate of average distance from nest to acquisition
            # location.
            evx1 = density.evx_for_region(cluster.extent.xmin, nest.xcenter,
                                          cluster.extent.ymin, cluster.extent.ymax)

            # Compute expected value of Y coordinate of average distance from nest to acquisition
            # location.
            evy1 = density.evy_for_region(cluster.extent.xmin, nest.xcenter,
                                          cluster.extent.ymin, cluster.extent.ymax)

            evx2 = density.evx_for_region(nest.xmin, nest.xmax,
                                          nest.ycenter, cluster.extent.ymax)

            evy2 = density.evy_for_region(nest.xmin, nest.xmax,
                                          nest.ycenter, cluster.extent.ymax)
            dist1 = math.sqrt((evx1 - nest.xcenter) ** 2 + (evy1 - nest.ycenter) ** 2)
            dist2 = math.sqrt((evx2 - nest.xcenter) ** 2 + (evy2 - nest.ycenter) ** 2)
            return (dist1 + dist2) / 4.0
        else:
            # Compute expected value of X coordinate of average distance from nest to acquisition
            # location.
            evx = density.evx_for_region(cluster.extent.xmin, cluster.extent.xmax,
                                         cluster.extent.ymin, cluster.extent.ymax)

            # Compute expected value of Y coordinate of average distance from nest to acquisition
            # location.
            evy = density.evy_for_region(cluster.extent.xmin, cluster.extent.xmax,
                                         cluster.extent.ymin, cluster.extent.ymax)

            # Compute expected distance from nest to block acquisitions
            dist = math.sqrt((evx - nest.xcenter) ** 2 + (evy - nest.ycenter) ** 2)
            return dist


@implements.implements(models.interface.IConcreteInterExpModel1D)
class InterExpNestHomingTime1Robot(models.interface.IConcreteInterExpModel1D):
    """
    Models the time it takes a robot to return to the nest after it has picked up an object during
    foraging across all experiments in the batch. That is, one model datapoint is computed for
    each experiment within the batch.

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
        return 'block-transport-time-cum-avg'

    def legend_name(self) -> str:
        return 'Predicted Homing Time'

    def run(self,
            cmdopts: dict,
            criteria: bc.IConcreteBatchCriteria) -> pd.DataFrame:
        dirs = criteria.gen_exp_dirnames(cmdopts)
        res_df = pd.DataFrame(columns=dirs, index=[0])

        for i, exp in enumerate(dirs):

            # Setup cmdopts for intra-experiment model
            cmdopts2 = copy.deepcopy(cmdopts)
            cmdopts2["exp_input_root"] = os.path.join(cmdopts['batch_input_root'], exp)
            cmdopts2["exp_output_root"] = os.path.join(cmdopts['batch_output_root'], exp)
            cmdopts2["exp_graph_root"] = os.path.join(cmdopts['batch_graph_root'], exp)
            cmdopts2["exp_avgd_root"] = os.path.join(cmdopts2["exp_output_root"],
                                                     self.main_config['sierra']['avg_output_leaf'])
            cmdopts2["exp_model_root"] = os.path.join(cmdopts['batch_model_root'], exp)
            core.utils.dir_create_checked(cmdopts2['exp_model_root'], exist_ok=True)

            intra_df = IntraExpNestHomingTime1Robot(self.main_config,
                                                    self.config).run(cmdopts2,
                                                                     criteria,
                                                                     i)
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return res_df
