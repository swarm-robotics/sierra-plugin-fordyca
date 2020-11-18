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
import copy

# 3rd party packages
import implements
import pandas as pd

# Project packages
import models.interface
import core.utils
import core.variables.batch_criteria as bc


def available_models(category: str):
    if category == 'intra':
        return ['IntraExpInterferenceTime1Robot']
    elif category == 'inter':
        return ['InterExpInterferenceTime1Robot']
    else:
        return None


@implements.implements(models.interface.IConcreteIntraExpModel1D)
class IntraExpInterferenceRate1Robot(models.interface.IConcreteIntraExpModel1D):
    """
    Models the steady state interference rate of a swarm of size 1, assuming purely reactive
    robots. Robots are in 1 of 3 states via their FSM: exploring, homing, or avoiding collision,
    which we model as a queueing network, in which robots enter the interference queue when
    sense a nearby they sense a nearby wall, and exit it sometime later. We know:

    - The average amount of time a robot spends in the interference queue from empirical data.
    - The average number of robots in the interference queue from empirical data.

    From this, we can use Little's Law to compute the arrival rate, which is the interference rate.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return criteria.populations(cmdopts)[i] == 1

    def target_csv_stem(self) -> str:
        return 'fsm-interference-counts'

    def legend_name(self) -> str:
        return 'Predicted Interference Rate'

    def run(self,
            cmdopts: dict,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int) -> pd.DataFrame:

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

        fsm_df = core.utils.pd_csv_read(os.path.join(result_opaths[0],
                                                     'fsm-interference-counts.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=fsm_df.index)
        res_df['model'] = 0.0

        for result in result_opaths:
            self._calc_for_result(result, res_df)

        # Average our results
        res_df['model'] /= len(result_opaths)

        # All done!
        return res_df

    def _calc_for_result(self, result_opath: str, res_df: pd.DataFrame):

        fsm_counts_df = core.utils.pd_csv_read(os.path.join(result_opath,
                                                            'fsm-interference-counts.csv'))

        # Robots that are not exploring or homing are avoiding collision/experiencing interference
        # by definition. We read this directly from empirical data, because in the parent ODE model,
        # this will be available via a variable. Here we are just attempting to get the model
        # correct.
        int_frac = fsm_counts_df['cum_avg_exp_interference']
        int_time = fsm_counts_df['cum_avg_interference_duration']

        # L = lambda / (mu - lambda), solving for lambda
        lam = int_frac / (int_time + 1)

        # Only robots experiencing interference contribute to the interference rate
        res_df['model'] = lam * int_frac

        return res_df


@implements.implements(models.interface.IConcreteInterExpModel1D)
class InterExpInterferenceRate1Robot(models.interface.IConcreteInterExpModel1D):
    """
    Models the rate at which a swarm of size one experiences interference during foraging across all
    experiments in the batch. That is, one model datapoint is computed for each experiment within
    the batch.

    In order for this model to run, all experiments in the batch must have 1 robot.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config
        self.nest = None

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return all([p == 1 for p in criteria.populations(cmdopts)])

    def target_csv_stem(self) -> str:
        return 'interference-entered-cum-avg'

    def legend_name(self) -> str:
        return 'Predicted Interference Rate'

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

            intra_df = IntraExpInterferenceRate1Robot(self.main_config,
                                                      self.config).run(cmdopts2,
                                                                       criteria,
                                                                       i)
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return res_df
