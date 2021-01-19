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
Free block acquisition and block collection models for the FORDYCA project.
"""

# Core packages
import os
import copy
import typing as tp
import math

# 3rd party packages
import implements
import pandas as pd

# Project packages
import core.models.interface
import core.utils
import core.variables.time_setup as ts
from core.experiment_spec import ExperimentSpec
import projects.fordyca.models.representation as rep
import core.variables.batch_criteria as bc
from core.vector import Vector3D

from projects.fordyca.models.density import BlockAcqDensity
from projects.fordyca.models.dist_measure import DistanceMeasure2D
import projects.fordyca.models.diffusion as diffusion


def available_models(category: str):
    if category == 'intra':
        return ['IntraExp_AcqRate_NRobots', 'IntraExp_BlockCollectionRate_NRobots']
    elif category == 'inter':
        return ['InterExp_BlockCollectionRate_NRobots',
                'InterExp_AcqRate_NRobots']
    else:
        return None

################################################################################
# Intra-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExp_BlockAcqRate_NRobots():
    """
    Models the steady state block acquisition rate of a swarm of N CRW robots.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because
       it does not require ANY experimental data, and can be computed from first principles, so it
       is always OK to :method:`run()` it.

    From :xref:`Harwell2021a`.
    """

    def __init__(self, main_config: dict, config: dict) -> None:
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-manip-events-free-pickup']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Block Acquisition Rate']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        result_opath = os.path.join(cmdopts['exp_avgd_root'])

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

        nest = rep.Nest(cmdopts, criteria, exp_num)

        dist = 0.0
        for result in result_opaths:
            dist += ExpectedAcqDist()(cmdopts, result, nest)

        # Average our results
        avg_acq_dist = dist / len(result_opaths)
        n_robots = criteria.populations(cmdopts)[exp_num]

        alpha_b = self._kernel(N=n_robots,
                               wander_speed=float(self.config['wander_mean_speed']),
                               avg_acq_dist=avg_acq_dist)

        rate_df = core.utils.pd_csv_read(os.path.join(result_opath, 'block-manipulation.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=rate_df.index)
        res_df['model'] = alpha_b

        # All done!
        return [res_df]

    @staticmethod
    def _kernel(N: float, wander_speed: float, avg_acq_dist: float) -> float:
        """
        Calculates the CRW Diffusion constant in :xref:`Harwell2021a` for bounded arena geometry,
        inspired by the results in :xref:`Codling2010`.
        """
        D = diffusion.calc_crwD(N, wander_speed)

        # diffusion time = (diffusion dist) ^2 / (2 * D)
        diff_time = avg_acq_dist ** 2 / (2 * D)

        # Inverse of diffusion time from nest to expected acquisition location is alpha_b
        return 1.0 / diff_time


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExp_BlockCollectionRate_NRobots():
    """
    Models the steady state block collection rate :math:`L_{b}` of the swarm of CRW robots using
    Little's law and :class:`IntraExp_BlockAcqRate_NRobots`. Makes the following assumptions:

    - The reported homing time includes a non-negative penalty :math:`\mu_{b}` assessed in the nest
      which robots must serve before collection can complete. This models physical time taken to
      actually drop the block, and other environmental factors.

    - At most 1 robot can drop an object per-timestep (i.e. an M/M/1 queue).
    """

    @staticmethod
    def kernel(alpha_bN: tp.Union[pd.DataFrame, float],
               mu_bN: tp.Union[pd.DataFrame, float]) -> tp.Union[pd.DataFrame, float]:
        r"""
        Perform the block collection rate calculation using Little's Law. We want to find the
        average # of customers being served--this is the rate of robots leaving the homing queue as
        they deposit their blocks in the nest.

        .. math::
           L_{b} = \frac{\alpha_b}{\mu_b}

        where :math:`L_s` is the average number of customers being served each timestep :math:`t`.

        Args:
            alpha_bN: Rate of robots in the swarm encountering blocks at time :math:`t`:
                      :math:`\alpha_{b}`.

            mu_bN: The average penalty in timesteps that a robot from a swarm of size
                   :math:`\mathcal{N}` dropping an object in the nest at time :math:`t` will be
                   subjected to before collection occurs.

        Returns:
            Estimate of the steady state rate of block collection, :math:`L_{b}`.

        """
        return alpha_bN / mu_bN

    @staticmethod
    def calc_kernel_args(criteria:  bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict,
                         main_config: dict):
        block_manip_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                             'block-manipulation.csv'))

        # Calculate acquisition rate kernel args
        kargs = IntraExpAcqRate.calc_kernel_args(criteria, exp_num, cmdopts, main_config)
        alpha_bN = IntraExpAcqRate.kernel(**kargs)

        # FIXME: In the future, this will be another model, rather than being read from experimental
        # data.
        mu_bN = block_manip_df['cum_avg_free_drop_penalty']

        return {
            'alpha_bN': alpha_bN,
            'mu_bN': mu_bN
        }

    def __init__(self, main_config: dict, config: dict) -> None:
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-manip-events-free-drop']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Block Collection Rate']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:
        rate_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                      'block-manipulation.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=rate_df.index)
        kargs = self.calc_kernel_args(criteria, exp_num, cmdopts, self.main_config)
        res_df['model'] = self.kernel(**kargs)

        # All done!
        return [res_df]

################################################################################
# Inter-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExp_BlockAcqRate_NRobots():
    """
    Models the steady state block acquisition rate of the swarm, assuming purely reactive robots.
    That is, one model datapoint is computed for each experiment within the batch.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because
       it is a summary model, built on simpler intra-experiment models.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-manip-free-pickup-events-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Block Acquisition Rate']

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
            intra_df = IntraExp_BlockAcqRate_NRobots(self.main_config,
                                                     self.config).run(criteria,
                                                                      i,
                                                                      cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        # All done!
        return [res_df]


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExp_BlockCollectionRate_NRobots():
    """
    Models the steady state block collection rate of the CRW swarm.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because
       it is a summary model, built on simpler intra-experiment models.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['blocks-transported-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Block Collection Rate']

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
            intra_df = IntraExp_BlockCollectionRate_NRobots(self.main_config,
                                                            self.config).run(criteria,
                                                                             i,
                                                                             cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        # All done!
        return [res_df]

################################################################################
# Helper Classes
################################################################################


class ExpectedAcqDist():
    def __call__(self, cmdopts: dict, result_opath: str, nest: rep.Nest) -> float:

        # Get clusters in the arena
        clusters = rep.BlockClusterSet(cmdopts, nest, result_opath)

        # Integrate to find average distance from nest to all clusters, weighted by acquisition
        # density.
        dist = 0.0
        for cluster in clusters:
            dist += self._nest_to_cluster(cluster, nest, cmdopts['scenario'])

        return dist / len(clusters)

    def _nest_to_cluster(self,
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
