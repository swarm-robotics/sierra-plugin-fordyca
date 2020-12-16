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
import os
import typing as tp
import copy
import math

# 3rd party packages
import implements
import pandas as pd

# Project packages
import core.models.interface
import core.utils
import core.variables.time_setup as ts
import core.variables.batch_criteria as bc
from core.vector import Vector3D
import projects.fordyca.models.representation as rep
from projects.fordyca.models.density import BlockAcqDensity
from projects.fordyca.models.dist_measure import DistanceMeasure2D
from projects.fordyca.models.interference import IntraExpRobotInterferenceRate, IntraExpRobotInterferenceTime


def available_models(category: str):
    if category == 'intra':
        return ['IntraExpNestHomingTime1Robot']
    elif category == 'inter':
        return ['InterExpNestHomingTime1Robot', 'InterExpNestHomingTimeNRobots']
    else:
        return None


################################################################################
# Intra-experiment models
################################################################################

@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpNestHomingTime1Robot():
    r"""
    Models the time it takes a robot in a swarm of size 1 to return to the nest after it has picked
    up an object during foraging during a single experiment within a batch. That is, one model
    datapoint is computed for each metric collection interval in each simulation.

    Only runs for swarms with :math:`\mathcal{N}=1`.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because it
       is strictly dependent on arena geometry, block distribution, etc., and must be :meth:`run()`
       to be used.

    From :xref:`Harwell2021a`.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return criteria.populations(cmdopts)[i] == 1

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-transport-time']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Homing Time']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        # Calculate nest extent
        nest = rep.Nest(cmdopts, criteria, exp_num)

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
        # er.intra_record_add(self.__class__.__name__, exp_num)
        return [res_df]

    def _calc_for_result(self,
                         cmdopts: dict,
                         result_opath: str,
                         nest: rep.Nest,
                         res_df: pd.DataFrame):
        # Get clusters in the arena
        clusters = rep.BlockClusterSet(self.main_config, cmdopts, nest, result_opath)

        # Integrate to find average distance from nest to all clusters, weighted by acquisition
        # density.
        dist = 0.0
        for cluster in clusters:
            dist += self._calc_acq_edist(cluster, nest, cmdopts['scenario'])

        avg_dist = dist / len(clusters)

        spatial_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                         'spatial-movement.csv'))

        # Finally, calculate the average homing time for each interval in the simulation
        for idx in spatial_df.index:
            # Reported in cm/s, and we need m/s
            avg_vel = spatial_df.loc[idx, 'cum_avg_velocity_homing'] / 100.0

            # After getting the average distance to ANY block in ANY cluster in the arena, we can
            # compute the average time, in SECONDS, that robots spend returning to the nest.
            avg_homing_sec = avg_dist / avg_vel

            # Convert seconds to timesteps for displaying on graphs
            avg_homing_ts = avg_homing_sec * ts.kTICKS_PER_SECOND

            # All done!
            res_df.loc[idx, 'model'] += avg_homing_ts

    def _calc_acq_edist(self,
                        cluster: rep.BlockCluster,
                        nest: core.utils.ArenaExtent,
                        scenario: str) -> float:
        dist_measure = DistanceMeasure2D(scenario, nest=nest)
        density = BlockAcqDensity(nest=nest, cluster=cluster, dist_measure=dist_measure)

        # Compute expected value of X coordinate of average distance from nest to acquisition
        # location.
        ll = cluster.extent.ll
        ur = cluster.extent.ur
        evx = density.evx_for_region(ll=ll, ur=ur)

        # Compute expected value of Y coordinate of average distance from nest to acquisition
        # location.
        evy = density.evy_for_region(ll=ll, ur=ur)

        # Compute expected distance from nest to block acquisitions
        dist = dist_measure.to_nest(Vector3D(evx, evy))

        return dist


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpNestHomingTimeNRobots():
    r"""
    Models the time it takes a robot in a swarm of :math:`\mathcal{N}` robots to return to the nest
    after it has picked up an object during foraging during a single experiment within a batch. That
    is, one model datapoint is computed for each metric collection interval in each simulation.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because it
       is strictly dependent on arena geometry, block distribution, etc., and must be :meth:`run()`
       to be used.

    From :xref:`Harwell2021a`.

    """
    @staticmethod
    def kernel(tau_h1: tp.Union[pd.DataFrame, float],
               alpha_rN: tp.Union[pd.DataFrame, float],
               tau_avN: tp.Union[pd.DataFrame, float],
               n_robots: int) -> tp.Union[pd.DataFrame, float]:
        r"""
        Perform the homing time calculation.

        .. math::
           \tau_h = \tau_{h}^1\big[1 + \alpha_r\tau_{av}\mathcal{N}\big]

        Args:
            tau_h1: Homing time of robots in the swarm at time :math:`t`: :math:`\tau_{h}^1`.

            alpha_rN: Robot encounter rate of robots in the swarm at time :math:`t`:
                      :math:`\alpha_{r}`.

            tau_avN: Average time each robot spends in the interference state beginning at time
                     :math:`t`: :math:`\tau_{av}`.

            n_robots: The number of robots in the swarm.

        Returns:
            Estimate of the steady state homing time for a swarm of :math:`\mathcal{N}` robots,
            :math:`\tau_h`.
        """
        return tau_h1 * (1.0 + alpha_rN * tau_avN / n_robots)

    @staticmethod
    def calc_kernel_args(criteria: bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict,
                         main_config: dict) -> dict:
        homing1 = IntraExpNestHomingTime1Robot(main_config, None)
        tau_h1 = homing1.run(criteria, exp_num, cmdopts)[0]

        av_rateN = IntraExpRobotInterferenceRate(main_config, None)
        alpha_rN = av_rateN.run(criteria, exp_num, cmdopts)[0]

        av_timeN = IntraExpRobotInterferenceTime(main_config, None)
        tau_avN = av_timeN.run(criteria, exp_num, cmdopts)[0]

        counts_df = core.utils.pd_csv_read(os.path.join(os.path.join(cmdopts['exp_avgd_root']),
                                                        'block-acq-counts.csv'))
        n_robots = criteria.populations(cmdopts)[exp_num]

        return {
            'tau_h1': tau_h1,
            'alpha_rN': alpha_rN,
            'tau_avN': tau_avN,
            'n_robots': n_robots
        }

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-transport-time']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Homing Time']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        cluster_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                         'block-clusters.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=cluster_df.index)
        kargs = self.calc_kernel_args(criteria, exp_num, cmdopts, self.main_config)
        res_df['model'] = self.kernel(**kargs)

        # All done!
        # er.intra_record_add(self.__class__.__name__, exp_num)
        return [res_df]


################################################################################
# Inter-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpNestHomingTime1Robot():
    r"""
    Models the time it takes a robot to return to the nest after it has picked up an object during
    foraging across all experiments in the batch. That is, one model datapoint is computed for
    each experiment within the batch.

    Only runs for swarms with :math:`\mathcal{N}=1`.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because
       it is a summary model, built on simpler intra-experiment models.

    From :xref:`Harwell2021a`.
    """

    def __init__(self, main_config: dict, config: dict) -> None:
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return all([p == 1 for p in criteria.populations(cmdopts)])

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-transport-time-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Homing Time']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            cmdopts: dict) -> tp.List[pd.DataFrame]:
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

            # Model only targets a single graph
            intra_df = IntraExpNestHomingTime1Robot(self.main_config,
                                                    self.config).run(criteria,
                                                                     i,
                                                                     cmdopts2)[0]
            # Last datapoint is the closest to the steady state value (presumably) so we select it
            # to use as our prediction for the experiment within the batch.
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return [res_df]


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpNestHomingTimeNRobots():
    r"""
    Models the time it takes a robot to return to the nest after it has picked up an object during
    foraging across all experiments in the batch. That is, one model datapoint is computed for
    each experiment within the batch.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because
       it is a summary model, built on simpler intra-experiment models.

    From :xref:`Harwell2021a`.
    """

    def __init__(self, main_config: dict, config: dict) -> None:
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-transport-time-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Homing Time']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            cmdopts: dict) -> tp.List[pd.DataFrame]:
        dirs = criteria.gen_exp_dirnames(cmdopts)
        res_df = pd.DataFrame(columns=dirs, index=[0])

        for i, exp in enumerate(dirs):
            # Setup cmdopts for intra-experiment model
            cmdopts2 = copy.deepcopy(cmdopts)

            cmdopts2["exp0_output_root"] = os.path.join(cmdopts2["batch_output_root"], dirs[0])
            cmdopts2["exp0_avgd_root"] = os.path.join(cmdopts2["exp0_output_root"],
                                                      self.main_config['sierra']['avg_output_leaf'])

            cmdopts2["exp_input_root"] = os.path.join(cmdopts['batch_input_root'], exp)
            cmdopts2["exp_output_root"] = os.path.join(cmdopts['batch_output_root'], exp)
            cmdopts2["exp_graph_root"] = os.path.join(cmdopts['batch_graph_root'], exp)
            cmdopts2["exp_avgd_root"] = os.path.join(cmdopts2["exp_output_root"],
                                                     self.main_config['sierra']['avg_output_leaf'])
            cmdopts2["exp_model_root"] = os.path.join(cmdopts['batch_model_root'], exp)
            core.utils.dir_create_checked(cmdopts2['exp_model_root'], exist_ok=True)

            # Model only targets a single graph
            intra_df = IntraExpNestHomingTimeNRobots(self.main_config,
                                                     self.config).run(criteria,
                                                                      i,
                                                                      cmdopts2)[0]
            # Last datapoint is the closest to the steady state value (presumably) so we select it
            # to use as our prediction for the experiment within the batch.
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return [res_df]
