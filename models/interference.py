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
Intra- and inter-experiment models for the time a robots spends avoiding interference and the rate
at which a robot experiences interference.
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
import core.variables.batch_criteria as bc


def available_models(category: str):
    if category == 'intra':
        return ['IntraExpWallInterferenceRate',
                'IntraExpRobotInterferenceRate',
                'IntraExpRobotInterferenceTime']
    elif category == 'inter':
        return ['InterExpRobotInterferenceRate',
                'InterExpRobotInterferenceTime']
    else:
        return None

################################################################################
# Intra-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpWallInterferenceRate():
    r"""
    Models the steady state interference rate of a swarm of size 1, assuming purely reactive
    robots. Robots are in 1 of 3 states via their FSM: exploring, homing, or avoiding collision,
    which we model as a queueing network, in which robots enter the interference queue when
    sense a nearby wall, and exit it sometime later. We know:

    - The average amount of time a robot spends in the interference queue from empirical data.
    - The average number of robots in the interference queue from empirical data.

    From this, we can use Little's Law to compute the arrival rate for the queue, which is the
    interference rate for the swarm.

    This model has a `:meth:`kernel()` function which computes the calculation, enabling this
    model to be used as a building block without necessarily needing to be :meth:`run()`.

    Only runs for swarms with :math:`\mathcal{N}=1`.

    From :xref:`Harwell2021a`.
    """
    @staticmethod
    def kernel(int_count1: tp.Union[pd.DataFrame, float],
               tau_av1: tp.Union[pd.DataFrame, float]) -> tp.Union[pd.DataFrame, float]:
        r"""
        Perform the interference rate calculation using Little's Law, modeling CRW robots
        entering/exiting an interference avoidance state using a two state queueing network: robots
        are either experiencing interference or are doing something else.

        .. math::
           \alpha_{r}^1 = \frac{\tau_{av}}{\mathcal{N}_{av}(t)}

        Args:
            int_count1: Number of robots in the swarm which are experiencing interference at time
                        :math:`t`: :math:`\mathcal{N}_{av}(t)`.

            tau_av1: Average time each robot spends in the interference state beginning at time
                     :math:`t`: :math:`\tau_{av}^1`.

        Returns:
            Estimate of the steady state rate of robots entering the interference queue,
            :math:`\alpha_{r}^1`.
        """
        # All robots can enter the avoidance queue, so we don't need to modify lambda according to
        # the # of contributing robots.
        return int_count1 / tau_av1

    @staticmethod
    def calc_kernel_args(exp_avgd_root: str) -> tp.Dict[str, pd.DataFrame]:
        fsm_counts_df = core.utils.pd_csv_read(os.path.join(exp_avgd_root,
                                                            'fsm-interference-counts.csv'))
        return {
            'int_count1': fsm_counts_df['cum_avg_exp_interference'],
            'tau_av1': fsm_counts_df['cum_avg_interference_duration']
        }

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return criteria.populations(cmdopts)[i] == 1

    def target_csv_stems(self) -> tp.List[str]:
        return ['fsm-interference-counts']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Interference Rate']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        result_opath = os.path.join(cmdopts['exp_avgd_root'])

        fsm_df = core.utils.pd_csv_read(os.path.join(result_opath, 'fsm-interference-counts.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=fsm_df.index)

        # Calculate kernel arguments
        kargs = self.calc_kernel_args(cmdopts['exp_avgd_root'])

        # Run kernel on our results
        res_df['model'] = self.kernel(**kargs)

        # All done!
        return [res_df]


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpRobotInterferenceRate():
    r"""
    Models the steady state inter-robot interference rate of a swarm of :math:`\mathcal{N}`
    robots. Robots are in 1 of 3 states via their FSM: exploring, homing, or avoiding collision,
    which we model as a queueing network, in which robots enter the interference queue when sense a
    nearby they sense a nearby wall, and exit it sometime later. We know:

    - The average amount of time a robot spends in the interference queue from empirical data.
    - The average number of robots in the interference queue from empirical data.

    From this, we can use Little's Law to compute the arrival rate for the queue, which is the
    interference rate for the swarm.

    This model has a `:meth:`kernel()` function which computes the calculation, enabling this
    model to be used as a building block without necessarily needing to be :meth:`run()`.

    From :xref:`Harwell2021a`.

    """
    @staticmethod
    def kernel(int_count1: tp.Union[pd.DataFrame, float],
               tau_av1: tp.Union[pd.DataFrame, float],
               int_countN: tp.Union[pd.DataFrame, float],
               tau_avN: tp.Union[pd.DataFrame, float],
               n_robots: int) -> tp.Union[pd.DataFrame, float]:
        r"""
        Perform the interference rate calculation using Little's Law, modeling CRW robots
        entering/exiting an interference avoidance state using a two state queueing network: robots
        are either experiencing interference or are doing something else.

        For 1 robot, the rate is the rate of a single robot experiencing interference near arena
        walls; we want the rate of robots encountering other robots, so we correct for this with a
        linear factor.

        .. math::
           \alpha_{r} = \frac{\tau_{av}}{\mathcal{N}_{av}(t)} - \alpha_{r}^1\mathcal{N}.

        Args:
            int_count1: Fraction of robots in a swarm of size 1 which are experiencing interference
                        at time :math:`t`: :math:`\mathcal{N}_{av}(t)`.

            tau_av1: Average time each robot in a swarm of size 1 spends in the interference queue
                     beginning at time :math:`t`: :math:`\tau_{av}^1`.

            int_countN: Number of robots in a swarm of size :math:`\mathcal{N}` which are
                        experiencing interference at time :math:`t`: :math:`\mathcal{N}_{av}(t)`.

            tau_avN: Average time each robot in a swarm of size :math:`\mathcal{N}` spends in the
                     interference state beginning at time :math:`t`: :math:`\tau_{av}`.

            n_robots: The number of robots in the swarm.

        Returns:
            Estimate of the steady state rate of robots from a swarm of :math:`\mathcal{N}` robots
            entering the interference queue, :math:`\alpha_{r}`.

        """
        alpha_r1 = IntraExpWallInterferenceRate.kernel(int_count1, tau_av1)

        # All robots can enter the avoidance queue, so we don't need to modify lambda according to
        # the # of contributing robots.
        return int_countN / tau_avN - alpha_r1 * int_countN

    @staticmethod
    def calc_kernel_args(criteria: bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict) -> tp.Dict[str, pd.DataFrame]:
        # Calculate kernel args for the 1 robot case
        kargs = IntraExpWallInterferenceRate.calc_kernel_args(cmdopts['exp0_avgd_root'])

        # Add additional args for N robot case
        resultN_opath = os.path.join(cmdopts['exp_avgd_root'])
        fsm_countsN_df = core.utils.pd_csv_read(os.path.join(resultN_opath,
                                                             'fsm-interference-counts.csv'))

        kargs['int_countN'] = fsm_countsN_df['cum_avg_exp_interference']
        kargs['tau_avN'] = fsm_countsN_df['cum_avg_interference_duration']
        kargs['n_robots'] = criteria.populations(cmdopts)[exp_num]

        return kargs

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['fsm-interference-counts']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Interference Rate']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        result_opath = os.path.join(cmdopts['exp_avgd_root'])
        fsm_df = core.utils.pd_csv_read(os.path.join(result_opath, 'fsm-interference-counts.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=fsm_df.index)

        # Calculate kernel arguments
        kargs = self.calc_kernel_args(criteria, exp_num, cmdopts)

        # Run kernel on our results
        res_df['model'] = self.kernel(**kargs)

        # All done!
        return [res_df]


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExpRobotInterferenceTime():
    r"""
    Models the steady state time a robot in a swarm of size :math:`\mathcal{N}` spends avoiding
    interference from other robots.  Uses Little's Law and
    :class:`IntraExpInterferenceRateNRobots`to perform the calculation.

    This model has a `:meth:`kernel()` function which computes the calculation, enabling this
    model to be used as a building block without necessarily needing to be :meth:`run()`.

    From :xref:`Harwell2021a`.

    """
    @staticmethod
    def kernel(int_count1: tp.Union[pd.DataFrame, float],
               tau_av1: tp.Union[pd.DataFrame, float],
               int_countN: tp.Union[pd.DataFrame, float],
               tau_avN: tp.Union[pd.DataFrame, float],
               n_robots: int) -> tp.Union[pd.DataFrame, float]:
        r"""Perform the interference time calculation.

        .. math::
           \tau_{av} = \big[\alpha_{r} + \alpha_{r}^1\mathcal{N}\big]\mathcal{N}_{av}(t).

        Args:
            int_count1: Fraction of robots in a swarm of size 1 which are experiencing interference
                        at time :math:`t`: :math:`\mathcal{N}_{av}(t)`.

            tau_av1: Average time each robot in a swarm of size 1 spends in the interference state
                     beginning at time :math:`t`: :math:`\tau_{av}^1`.

            int_countN: Number of robots in a swarm of size :math:`\mathcal{N}` which are
                        experiencing interference at time :math:`t`: :math:`\mathcal{N}_{av}(t)`.

            tau_avN: Average time each robot in a swarm of size :math:`\mathcal{N}` spends in the
                     interference state beginning at time :math:`t`: :math:`\tau_{av}`.

            n_robots: The number of robots in the swarm.

        Returns:
            Estimate of the steady state time robots from a swarm of :math:`\mathcal{N}` spend in
            the interference queue, :math:`\tau_{av}`.

        """
        alpha_r1 = IntraExpWallInterferenceRate.kernel(int_count1, tau_av1)
        alpha_rN = IntraExpRobotInterferenceRate.kernel(int_count1,
                                                        tau_av1,
                                                        int_countN,
                                                        tau_avN,
                                                        n_robots)
        # All robots can enter the avoidance queue, so we don't need to modify lambda according to
        # the # of contributing robots.
        if n_robots == 1:
            return int_count1 / alpha_r1
        else:
            return int_countN / (alpha_rN + alpha_r1 * int_countN)

    @staticmethod
    def calc_kernel_args(criteria: bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict) -> tp.Dict[str, pd.DataFrame]:
        return IntraExpRobotInterferenceRate.calc_kernel_args(criteria, exp_num, cmdopts)

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['fsm-interference-counts']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Interference Time']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        result_opath = os.path.join(cmdopts['exp_avgd_root'])
        fsm_df = core.utils.pd_csv_read(os.path.join(result_opath, 'fsm-interference-counts.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=fsm_df.index)

        # Calculate kernel arguments
        kargs = self.calc_kernel_args(criteria, exp_num, cmdopts)

        # Run kernel on our results
        res_df['model'] = self.kernel(**kargs)

        # All done!
        return [res_df]

################################################################################
# Inter-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpRobotInterferenceRate():
    r"""
    Models the rate at which a swarm experiences inter-robot interference during foraging across all
    experiments in the batch. That is, one model datapoint is computed for each experiment within
    the batch.

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
        return ['interference-entered-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Interference Rate']

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

            # Model only targets one graph
            intra_df = IntraExpRobotInterferenceRate(self.main_config,
                                                     self.config).run(criteria,
                                                                      i,
                                                                      cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return [res_df]


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpRobotInterferenceTime():
    r"""
    Models the steady state average time robots from a swarm of size :math:`\mathcal{N}` spend in
    the interference queue during foraging across all experiments in the batch. That is, one model
    datapoint is computed for each experiment within the batch.

    .. IMPORTANT::
       This model does not have a kernel() function which computes the calculation, because
       it is a summary model, built on simpler intra-experiment models.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return any([p > 1 for p in criteria.populations(cmdopts)])

    def target_csv_stems(self) -> tp.List[str]:
        return ['interference-duration-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Interference Time']

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

            # Model only targets one graph
            intra_df = IntraExpRobotInterferenceTime(self.main_config,
                                                     self.config).run(criteria,
                                                                      i,
                                                                      cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return [res_df]
