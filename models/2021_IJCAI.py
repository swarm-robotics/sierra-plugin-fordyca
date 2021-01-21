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

r"""
Models of the steady state collective foraging behavior of a swarm of :math:`\mathcal{N}` CRW
robots. Used in the :xref:`Harwell2021a` paper.

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
from core.experiment_spec import ExperimentSpec
import core.variables.time_setup as ts
from core.xml_luigi import XMLAttrChangeSet

import projects.fordyca.models.representation as rep
from projects.fordyca.models.interference import IntraExp_RobotInterferenceRate_NRobots, IntraExp_WallInterferenceRate_1Robot
from projects.fordyca.models.homing_time import IntraExp_HomingTime_NRobots, IntraExp_HomingTime_1Robot
import projects.fordyca.models.ode_solver as ode
from projects.fordyca.models.blocks import IntraExp_BlockAcqRate_NRobots
from projects.fordyca.models.perf_measures import InterExp_RawPerf_NRobots, InterExp_Scalability_NRobots, InterExp_SelfOrg_NRobots
import projects.fordyca.models.diffusion as diffusion


def available_models(category: str):
    if category == 'intra':
        return ['IntraExp_ODE_NRobots']
    elif category == 'inter':
        return ['InterExp_ODEWrapper_NRobots']
    else:
        return None


################################################################################
# Intra-experiment models
################################################################################
@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExp_ODE_1Robot():
    r"""
    Calculates the following steady state quantities of a swarm of :math:`\mathcal{1}` foraging CRW
    robots operating under SS,DS,RN,PL block distributions in the arena:

    - :math:`\mathcal{N}_s` - Number of searching robots
    - :math:`\mathcal{N}_{av}` - Number of robots avoiding collision
    - :math:`\mathcal{N}_h` - Number of homing robots
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

        model_params = self._ode_params_calc(criteria, exp_num, cmdopts)

        nest = rep.Nest(cmdopts, criteria, exp_num)
        clusters = rep.BlockClusterSet(cmdopts, nest, cmdopts['exp0_avgd_root'])
        n_blocks = reduce(lambda accum, cluster: accum + cluster.avg_blocks, clusters, 0)
        z0 = {
            'N_s0': 1,
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

    def _ode_params_calc(self,
                         criteria: bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict) -> tp.Dict[str, float]:
        fsm_counts_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp0_avgd_root'],
                                                            'fsm-interference-counts.csv'))

        # T,n_datapoints are directly from simulation inputs
        spec = ExperimentSpec(criteria, exp_num, cmdopts)
        T_in_secs = ts.TimeSetup.extract_explen(XMLAttrChangeSet.unpickle(spec.exp_def_fpath))
        T = T_in_secs * ts.kTICKS_PER_SECOND

        n_datapoints = len(fsm_counts_df.index)

        # This is OK to read from experimental data, per the paper.
        tau_av1 = fsm_counts_df['int_avg_interference_duration'].iloc[-1]

        # tau_h, alpha_b are computed directly from simulation inputs/configuration, so we can run()
        # them here.
        tau_h = IntraExp_HomingTime_1Robot(self.main_config, self.config).run(criteria,
                                                                              exp_num,
                                                                              cmdopts)[0]

        alpha_b = IntraExp_BlockAcqRate_NRobots(self.main_config, self.config).run(criteria,
                                                                                   exp_num,
                                                                                   cmdopts)[0]

        # FIXME: This currently reads alpha_ca1 from experimental data
        alpha_ca1 = IntraExp_WallInterferenceRate_1Robot(self.main_config, self.config).run(criteria,
                                                                                            exp_num,
                                                                                            cmdopts)[0]
        return {
            'N': 1,
            'T': T,
            'tau_av1': tau_av1,
            'tau_h1': tau_h['model'].iloc[-1],
            'alpha_b0': alpha_b['model'].iloc[-1],
            'alpha_ca1': alpha_ca1['model'].iloc[-1],
            'n_datapoints': n_datapoints
        }


@implements.implements(core.models.interface.IConcreteIntraExpModel1D)
class IntraExp_ODE_NRobots():
    r"""
    Calculates the following steady state quantities of a swarm of :math:`\mathcal{N}` foraging CRW
    robots operating under SS,DS,RN,PL block distributions in the arena:

    - :math:`\mathcal{N}_s` - Number of searching robots
    - :math:`\mathcal{N}_{av}` - Number of robots avoiding collision
    - :math:`\mathcal{N}_h` - Number of homing robots
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

        model1_robot = IntraExp_ODE_1Robot(self.main_config, self.config)

        if n_robots == 1:
            return model1_robot.run(criteria, 0, cmdopts)

        model_params = model1_robot._ode_params_calc(criteria, 0, cmdopts)
        model_params.update(self._ode_params_calc(criteria, exp_num, cmdopts))

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

    def _ode_params_calc(self,
                         criteria: bc.IConcreteBatchCriteria,
                         exp_num: int,
                         cmdopts: dict) -> tp.Dict[str, float]:
        fsm_counts_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp_avgd_root'],
                                                            'fsm-interference-counts.csv'))

        # N,T,n_datapoints are directly from simulation inputs
        N = criteria.populations(cmdopts)[exp_num]

        spec = ExperimentSpec(criteria, exp_num, cmdopts)
        T_in_secs = ts.TimeSetup.extract_explen(XMLAttrChangeSet.unpickle(spec.exp_def_fpath))
        T = T_in_secs * ts.kTICKS_PER_SECOND
        n_datapoints = len(fsm_counts_df.index)

        # This is OK to read from experimental data, per the paper.
        tau_avN = fsm_counts_df['int_avg_interference_duration'].iloc[-1]

        # tau_h, alpha_b are computed directly from simulation inputs/configuration, so we can run()
        # them here.
        tau_hN = IntraExp_HomingTime_NRobots(self.main_config, self.config).run(criteria,
                                                                                exp_num,
                                                                                cmdopts)[0]

        # FIXME: N_av1 COULD be computed a priori, but I don't have time to do it right now, so I
        # just read it from simulation results.
        fsm_counts1_df = core.utils.pd_csv_read(os.path.join(cmdopts['exp0_avgd_root'],
                                                             'fsm-interference-counts.csv'))

        N_av1 = fsm_counts1_df['int_avg_exp_interference'].iloc[-1]

        # crwD calculated directly from simulation inputs/configuration
        acq = IntraExp_BlockAcqRate_NRobots(self.main_config, self.config)
        alpha_b = acq.run(criteria, exp_num, cmdopts)[0]
        crwD = diffusion.calc_crwD(N, float(self.config['wander_mean_speed']))

        params = {
            'N': N,
            'T': T,
            'tau_avN': tau_avN,
            'N_av1': N_av1,
            'tau_hN': tau_hN['model'].iloc[-1],
            'alpha_bN': alpha_b['model'].iloc[-1],
            'crwD': crwD,
            'n_datapoints': n_datapoints
        }

        print("--------------------------------------------------------------------------------")
        print("Calculated ODE params for N robots:")
        print(params)
        print("--------------------------------------------------------------------------------")

        return params


################################################################################
# Inter-experiment models
################################################################################

@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExp_ODE_NRobots():
    r"""
    Models the behavior of a swarm of :math:`\mathcal{N}` foraging robots using
    :class:`IntraExp_ODE_NRobots` across all experiments in the batch.

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
        return ['block-acq-counts-true-exploring-int-avg',
                'block-transporter-homing-nest-int-avg',
                'interference-in-int-avg']

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
        res_df_homing = pd.DataFrame(columns=dirs, index=[0])

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
            cmdopts2["exp0_avgd_root"] = os.path.join(cmdopts2["exp0_output_root"],
                                                      self.main_config['sierra']['avg_output_leaf'])

            core.utils.dir_create_checked(cmdopts2['exp_model_root'], exist_ok=True)

            intra_dfs = IntraExp_ODE_NRobots(self.main_config,
                                             self.config).run(criteria,
                                                              i,
                                                              cmdopts2)

            # gets steady state solution for avoiding and searching counts
            res_df_searching[exp] = intra_dfs[0].iloc[-1]
            res_df_homing[exp] = intra_dfs[1].iloc[-1]
            res_df_avoiding[exp] = intra_dfs[2].iloc[-1]

            # print(res_df_searching)
            # print(res_df_homing)
            # print(res_df_avoiding)

        return [res_df_searching, res_df_homing, res_df_avoiding]


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExp_ODEWrapper_NRobots():
    r"""
    Thin wrapper class around :class:`InterExp_ODE_NRobots` which runs the ODE model across all
    experiments in a batch, and then uses the results to predict performance:

    - Raw performance
    - Scalability
    - Emergent self-organization

    .. IMPORTANT::
        This model does not have a kernel() function which computes the calculation, because
        it is a summary model, built on simpler inter-experiment models.

    From :xref:`Harwell2021a`.

    """

    def __init__(self, main_config: dict, config: dict) -> None:
        self.main_config = main_config
        self.config = config

        # Performance measures
        self.raw_perf = InterExp_RawPerf_NRobots(self.main_config,
                                                 self.config)
        self.scalability = InterExp_Scalability_NRobots(self.main_config,
                                                        self.config)
        self.self_org = InterExp_SelfOrg_NRobots(self.main_config,
                                                 self.config)

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return [self.raw_perf.target_csv_stems()[0],
                self.scalability.target_csv_stems()[0],
                self.self_org.target_csv_stems()[0]]

    def legend_names(self) -> tp.List[str]:
        return [self.raw_perf.legend_names()[0],
                self.scalability.legend_names()[0],
                self.self_org.legend_names()[0]]

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> tp.List[pd.DataFrame]:

        dirs = criteria.gen_exp_dirnames(cmdopts)

        ode = InterExp_ODE_NRobots(self.main_config, self.config)
        dfs = ode.run(criteria, cmdopts)

        perf_df = self.raw_perf.run(criteria, cmdopts)[0]
        sc_df = self.scalability.kernel(criteria, cmdopts, perf_df)
        so_df = self.self_org.kernel(criteria, cmdopts, perf_df, dfs[2])

        return [perf_df, sc_df, so_df]
