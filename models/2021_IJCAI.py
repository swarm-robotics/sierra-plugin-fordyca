# Copyright 2021 Angel Sylvester, John Harwell, All rights reserved.
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
from functools import reduce

# 3rd party packages
import implements
import pandas as pd

# Project packages
import core.models.interface
import core.utils
import core.variables.batch_criteria as bc
import projects.fordyca.models.representation as rep
from projects.fordyca.models.interference import IntraExpRobotInterferenceRate, IntraExpWallInterferenceRate
from projects.fordyca.models.homing_time import IntraExpHomingTimeNRobots, IntraExpHomingTime1Robot
import projects.fordyca.models.ode_solver as ode
from projects.fordyca.models.blocks import IntraExpAcqRate
from core.experiment_spec import ExperimentSpec
import core.variables.time_setup as ts
from core.xml_luigi import XMLAttrChangeSet


def available_models(category: str):
    if category == 'intra':
        return ['IntraCRW_N_Robots']
    elif category == 'inter':
        return ['InterCRW_N_Robots']
    else:
        return None


################################################################################
# Intra-experiment models
################################################################################
@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraCRW_1_Robot():
    r"""
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return criteria.populations(cmdopts)[i] == 1

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-acq-counts', 'block-transporter-homing-nest', 'fsm-interference-counts']

    def legend_names(self) -> tp.List[str]:
        return ['ODE Solution for Searching Counts (1 Robot)',
                'ODE Solution for Homing Counts (1 Robot)',
                'ODE Solution for Interference Counts (1 Robot)']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        model_params = self._calc_model_params(criteria, exp_num, cmdopts)

        nest = rep.Nest(cmdopts, criteria, exp_num)
        clusters = rep.BlockClusterSet(cmdopts, nest, cmdopts['exp0_avgd_root'])
        n_blocks = reduce(lambda accum, cluster: accum + cluster.avg_blocks, clusters, 0)
        z0 = {
            'N_s0': model_params['N'],
            'N_h0': 0,
            'N_avh0': 0,
            'N_avs0': 0,
            'B0': n_blocks
        }
        soln = ode.CRWSolver(model_params).solve(z0)

        res = {
            'searching': [soln[:, 0]][0],
            'homing': [soln[:, 1]][0]
        }

        res['avoiding'] = z0['N_s0'] - res['searching'] - res['homing']
        res_df = pd.DataFrame(res)

        return [res_df['searching'], res_df['homing'], res_df['avoiding']]

    def _calc_model_params(self,
                           criteria: bc.IConcreteBatchCriteria,
                           exp_num: int,
                           cmdopts: dict) -> tp.Dict[str, float]:
        """
        Calculate parameters for :class`CRWSolver`: N, tau_h, tau_av, alpha_ca, alpha_b.

        - N - # of robots in the swarm; direct simulation input parameter.

        - T - Length of simulation in timesteps; direct simulation input parameter.

        - n_datapoints - How many datapoints were taken during simultion;  direct simulation input
          parameter.

        - tau_h1 - Computed a priori using only arena geometry and block distribution information.

        - tau_av1 - Currently read from experiment data, though it should eventually be computed
          via a model.

        - alpha_ca1 - Currently computed from experiment data, though it should eventually be
          computed via a model.

        - alpha_b - Currently computed from experiment data, though it CAN and WILL be replaced
          with a biased random walk/diffusion calculation soon.
        """
        print("--------------------------------------------------------------------------------")
        print("Start ODE_1_robot param calc, exp_num=", exp_num)
        print("--------------------------------------------------------------------------------")

        fsm_counts_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp0_avgd_root'],
                                                            'fsm-interference-counts.csv'))

        tau_av1 = fsm_counts_df['cum_avg_interference_duration'].iloc[-1]

        n_robots = criteria.populations(cmdopts)[0]
        tau_h = IntraExpHomingTime1Robot(self.main_config, self.config).run(criteria,
                                                                            0,
                                                                            cmdopts)[0]

        alpha_ca1 = IntraExpWallInterferenceRate(self.main_config, self.config).run(criteria,
                                                                                    0,
                                                                                    cmdopts)[0]

        alpha_b = IntraExpAcqRate(self.main_config, self.config).run(criteria,
                                                                     0,
                                                                     cmdopts)[0]

        spec = ExperimentSpec(criteria, exp_num, cmdopts)

        T_in_secs = ts.TimeSetup.extract_explen(XMLAttrChangeSet.unpickle(spec.exp_def_fpath))
        T = T_in_secs * ts.kTICKS_PER_SECOND

        params = {
            'N': n_robots,
            'T': T,
            'tau_av1': tau_av1,
            'tau_h1': tau_h['model'].iloc[-1],
            'alpha_b0': alpha_b['model'].iloc[-1],
            'alpha_ca1': alpha_ca1['model'].iloc[-1],
            'n_datapoints': len(fsm_counts_df.index)
        }

        print("--------------------------------------------------------------------------------")
        print("Calculated ODE_1_robot params:")
        print(params)
        print("--------------------------------------------------------------------------------")
        return params


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraCRW_N_Robots():
    r"""
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-acq-counts', 'block-transporter-homing-nest', 'fsm-interference-counts']

    def legend_names(self) -> tp.List[str]:
        return ['ODE Solution for Searching Counts (N Robots)',
                'ODE Solution for Homing Counts (N Robots)',
                'ODE Solution for Interference Counts (N Robots)']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        n_robots = criteria.populations(cmdopts)[exp_num]

        model1_robot = IntraCRW_1_Robot(self.main_config, self.config)
        print(criteria.populations(cmdopts), exp_num)
        if n_robots == 1:
            return model1_robot.run(criteria, 1, cmdopts)

        model_params = model1_robot._calc_model_params(criteria, 1, cmdopts)
        model_params.update(self._calc_model_params(criteria, exp_num, cmdopts))
        print(model_params)
        nest = rep.Nest(cmdopts, criteria, exp_num)
        clusters = rep.BlockClusterSet(cmdopts, nest, cmdopts['exp_avgd_root'])
        n_blocks = reduce(lambda accum, cluster: accum + cluster.avg_blocks, clusters, 0)
        z0 = {
            'N_s0': model_params['N'],
            'N_h0': 0,
            'N_avh0': 0,
            'N_avs0': 0,
            'B0': n_blocks
        }
        soln = ode.CRWSolver(model_params).solve(z0)

        res = {
            'searching': [soln[:, 0]][0],
            'homing': [soln[:, 1]][0]
        }

        res['avoiding'] = z0['N_s0'] - res['searching'] - res['homing']
        res_df = pd.DataFrame(res)

        return [res_df['searching'], res_df['homing'], res_df['avoiding']]

    def _calc_model_params(self,
                           criteria: bc.IConcreteBatchCriteria,
                           exp_num: int,
                           cmdopts: dict) -> tp.Dict[str, float]:
        """
        Calculate parameters for :class`CRWSolver`: N, tau_h, tau_av, alpha_ca, alpha_b.

        - N - # of robots in the swarm; direct simulation input parameter.

        - T - Length of simulation in timesteps; direct simulation input parameter.

        - n_datapoints - How many datapoints were taken during simultion;  direct simultion input
          parameter.

        - tau_hN - Computed a priori using only arena geometry and block distribution information.

        - tau_avN - Currently read from experiment data, though it should eventually be computed
          via a model.

        - alpha_caN - Currently computed from experiment data, though it should eventually be
          computed via a model.

        - alpha_b - Currently computed from experiment data, though it CAN and WILL be replaced
          with a biased random walk/diffusion calculation soon.
        """
        print("--------------------------------------------------------------------------------")
        print("Start ODE_N_robot param calc")
        print("--------------------------------------------------------------------------------")
        fsm_counts_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                            'fsm-interference-counts.csv'))

        tau_avN = fsm_counts_df['cum_avg_interference_duration'].iloc[-1]

        n_robots = criteria.populations(cmdopts)[exp_num]

        tau_hN = IntraExpHomingTimeNRobots(self.main_config, self.config).run(criteria,
                                                                              exp_num,
                                                                              cmdopts)[0]

        alpha_caN = IntraExpRobotInterferenceRate(self.main_config, self.config).run(criteria,
                                                                                     exp_num,
                                                                                     cmdopts)[0]

        alpha_b = IntraExpAcqRate(self.main_config, self.config).run(criteria,
                                                                     exp_num,
                                                                     cmdopts)[0]

        spec = ExperimentSpec(criteria, exp_num, cmdopts)

        T_in_secs = ts.TimeSetup.extract_explen(XMLAttrChangeSet.unpickle(spec.exp_def_fpath))
        T = T_in_secs * ts.kTICKS_PER_SECOND

        params = {
            'N': n_robots,
            'T': T,
            'tau_avN': tau_avN,
            'tau_hN': tau_hN['model'].iloc[-1],
            'alpha_bN': alpha_b['model'].iloc[-1],
            'alpha_caN': alpha_caN['model'].iloc[-1],
            'n_datapoints': len(fsm_counts_df.index)
        }

        print("--------------------------------------------------------------------------------")
        print("Calculated ODE_N_robot params:")
        print(params)
        print("--------------------------------------------------------------------------------")

        return params


################################################################################
# Inter-experiment models
################################################################################

@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterCRW_N_Robots():
    r"""
    """

    def __init__(self, main_config: dict, config: dict) -> None:
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['interference-in-cum-avg',
                'block-acq-exploring-cum-avg',
                'block-transporter-homing-nest-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['ODE Solution for Interference Counts',
                'ODE Solution for Exploring Counts',
                'ODE Solution for Homing Counts']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> tp.List[pd.DataFrame]:

        dirs = criteria.gen_exp_dirnames(cmdopts)
        res_df_avoiding = pd.DataFrame(columns=dirs, index=[0])
        res_df_searching = pd.DataFrame(columns=dirs, index=[0])

        # attempting to get one model datapoint from batch to be representative of ODE solution

        for i, exp in enumerate(dirs):
            # Setup cmdopts for intra-experiment model
            cmdopts2 = copy.deepcopy(cmdopts)
            cmdopts2["exp_input_root"] = os.path.join(cmdopts['batch_input_root'], exp)
            cmdopts2["exp_output_root"] = os.path.join(cmdopts['batch_output_root'], exp)
            cmdopts2["exp_graph_root"] = os.path.join(cmdopts['batch_graph_root'], exp)
            cmdopts2["exp_avgd_root"] = os.path.join(cmdopts2["exp_output_root"],
                                                     self.main_config['sierra']['avg_output_leaf'])
            cmdopts2["exp_model_root"] = os.path.join(cmdopts['batch_model_root'], exp)

            cmdopts2["exp0_output_root"] = os.path.join(cmdopts2["batch_output_root"], dirs[0])
            cmdopts2["exp0_avgd_root"] = os.path.join(
                cmdopts2["exp0_output_root"], self.main_config['sierra']['avg_output_leaf'])

            core.utils.dir_create_checked(cmdopts2['exp_model_root'], exist_ok=True)

            # Model only targets a single graph   (will be N robot case)
            intra_df = IntraODE_N_Robots(self.main_config,
                                         self.config).run(criteria,
                                                          i,
                                                          cmdopts2)

            # gets steady state solution for avoiding and searching counts
            avoiding_sol = float(intra_df[0].iloc[-1])
            searching_sol = float(intra_df[1].iloc[-1])

            res_df_avoiding[exp] = avoiding_sol
            res_df_searching[exp] = searching_sol

            print('avoiding and searching df ---------------')
            print(res_df_avoiding)
            print(res_df_searching)

            # res_df[exp] = float(avoiding_df.iloc[-1])
            # searching_df.loc[searching_df.index[-1]]]
            # res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

        return [res_df_avoiding, res_df_searching]
