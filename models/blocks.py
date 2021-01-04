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

# 3rd party packages
import implements
import pandas as pd

# Project packages
import core.models.interface
import core.utils
from core.experiment_spec import ExperimentSpec
import projects.fordyca.models.representation as rep
import core.variables.batch_criteria as bc
from core.vector import Vector3D

from projects.fordyca.models.density import BlockAcqDensity
from projects.fordyca.models.model_error import Model2DError
from projects.fordyca.models.dist_measure import DistanceMeasure2D
from projects.fordyca.models.homing_time import IntraExpNestHomingTimeNRobots


def available_models(category: str):
    if category == 'intra':
        return ['IntraExpAcqRate', 'IntraExpCollectionRate']
    elif category == 'inter':
        return ['InterExpCollectionRate',
                'InterExpAcqRate']
    else:
        return None

################################################################################
# Intra-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpAcqRate():
    """
    Models the steady state block acquisition rate of the swarm, assuming purely reactive
    robots. Robots are in 1 of 3 states via their FSM: exploring, homing, or avoiding collision,
    which we model as a queueing network, in which robots enter the homing queue when they
    pick up a block, and exit it when they drop the block in the nest. We know:

    - The average amount of time a robot spends in the homing queue
      (:class:`IntraExpNestHomingTimeNRobots`).
    - The average number of robots in the homing queue from empirical data.

    From this, we can use Little's Law to compute the arrival rate, which is the block acquisition
    rate.
    """

    @staticmethod
    def kernel(exp_countN: tp.Union[pd.DataFrame, float],
               tau_hN: tp.Union[pd.DataFrame, float],
               n_robots: int) -> tp.Union[pd.DataFrame, float]:
        r"""
        Perform the block acquisition rate calculation using Little's Law, modeling CRW robots
        entering/exiting the homing state using a two state queueing network:
        robots are either homing or searching, with interference avoidance treated as part of
        each of those states.

        .. math::
           \alpha_{b}^1 = \frac{\tau_{h}}{\mathcal{N}_{h}(t)}

        Args:
            exp_countN: Number of robots in the swarm which are searching at time :math:`t`:
                        :math:`\mathcal{N}_{s}(t)`.

            int_count1: Number of robots in a swarm of size 1 which are experiencing interference at
                        time :math:`t`: :math:`\mathcal{N}_{av}(t)`.

            tau_hN: Average time each robot spends in the homing queue beginning at time
                    :math:`t`: :math:`\tau_{h}`.

        Returns:
            Estimate of the steady state rate of robots entering the homing queue,
            :math:`\alpha_{b}`.
        """
        homing_count = (n_robots - exp_countN)

        return homing_count / tau_hN

    @staticmethod
    def calc_kernel_args(criteria:  bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict,
                         main_config: dict):
        # Calculate homing_time kernel args
        kargs1 = IntraExpNestHomingTimeNRobots.calc_kernel_args(criteria,
                                                                exp_num,
                                                                cmdopts,
                                                                main_config)

        # Run kernel to get tau_hN
        tau_hN = IntraExpNestHomingTimeNRobots.kernel(**kargs1)['model']
        acq_counts_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                            'block-acq-counts.csv'))

        # We read these fractions directly from experimental data for the purposes of getting the
        # model correct. In the parent ODE model, these will be variables.
        exp_count = acq_counts_df['cum_avg_true_exploring_for_goal'] + \
            acq_counts_df['cum_avg_false_exploring_for_goal']

        return {
            'exp_countN': exp_count,
            'tau_hN': tau_hN,
            'n_robots': kargs1['n_robots'],
        }

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
        rate_df = core.utils.pd_csv_read(os.path.join(result_opath, 'block-manipulation.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=rate_df.index)
        kargs = self.calc_kernel_args(criteria, exp_num, cmdopts, self.main_config)
        res_df['model'] = self.kernel(**kargs)

        # All done!
        return [res_df]


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpCollectionRate():
    """
    Models the steady state block collection rate :math:`L_{b}` of the swarm using Little's law and
    :class:`IntraExpBlockAcqRate`. Makes the following assumptions:

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
class InterExpAcqRate():
    """
    Models the steady state block acquisition rate of the swarm, assuming purely reactive robots.
    That is, one model datapoint is computed for each experiment within the batch.
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
            intra_df = IntraExpAcqRate(self.main_config,
                                       self.config).run(criteria,
                                                        i,
                                                        cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        # All done!
        return [res_df]


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpCollectionRate():
    """
    Models the steady state block collection rate of the swarm, assuming purely reactive robots.
    That is, one model datapoint is computed for each experiment within the batch.
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
            intra_df = IntraExpCollectionRate(self.main_config,
                                              self.config).run(criteria,
                                                               i,
                                                               cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        # All done!
        return [res_df]
