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
Free block acquisition models for the FORDYCA project.
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
from core.models.execution_record import ExecutionRecord
import core.utils
from core.experiment_spec import ExperimentSpec
import projects.fordyca.models.representation as rep
import core.variables.batch_criteria as bc
from core.vector import Vector3D

from projects.fordyca.models.density import BlockAcqDensity
from projects.fordyca.models.model_error import Model2DError
from projects.fordyca.models.dist_measure import DistanceMeasure2D
from projects.fordyca.models.homing_time import IntraExpNestHomingTime1Robot


def available_models(category: str):
    if category == 'intra':
        # return ['IntraExpAcqSpatialDist']
        return ['IntraExpAcqRate']
    elif category == 'inter':
        return ['InterExpAcqRate']
        return ['InterExpAcqSpatialDistError']
    else:
        return None

################################################################################
# Intra-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteIntraExpModel2D)
class IntraExpAcqSpatialDist():
    """
    Models the steady state spatial distribution of the locations which robots acquire blocks,
    assuming purely reactive robots.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-acq-locs2D']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> tp.List[pd.DataFrame]:
        er = ExecutionRecord()
        if er.intra_record_exists(self.__class__.__name__, exp_num):
            return [core.utils.pd_csv_read(os.path.join(cmdopts['exp_model_root'],
                                                        self.target_csv_stems()[0] + '.model'))]

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

        # Calculate arena resolution
        acq_df = core.utils.pd_csv_read(os.path.join(result_opaths[0], 'block-acq-locs2D.csv'))
        spec = ExperimentSpec(criteria, exp_num, cmdopts)
        resolution = spec.arena_dim.xsize() / len(acq_df.index)

        # Acquisition distribution will be 0 outside of block clusters
        res_df = pd.DataFrame(columns=acq_df.columns, index=acq_df.index, dtype=float)
        res_df[:] = 0.0

        # Calculate distribution for each result path
        for path in result_opaths:
            self._calc_for_result(cmdopts, path, resolution, nest, res_df)

        # Average our results
        res_df /= len(result_opaths)

        # All done!
        er.intra_record_add(self.__class__.__name__, exp_num)
        return [res_df]

    def _calc_for_result(self,
                         cmdopts: dict,
                         result_opath: str,
                         resolution: float,
                         nest: rep.Nest,
                         res_df: pd.DataFrame):
        # Get clusters in the arena
        clusters = rep.BlockClusterSet(self.main_config, cmdopts, result_opath)
        dist_measure = DistanceMeasure2D(cmdopts['scenario'])
        for cluster in clusters:
            rangex = range(int(cluster.extent.ll.x / resolution),
                           int(cluster.extent.ur.x / resolution))
            rangey = range(int(cluster.extent.ll.y / resolution),
                           int(cluster.extent.ur.y / resolution))

            density = BlockAcqDensity(nest=nest,
                                      cluster=cluster,
                                      clusters=clusters,
                                      dist_measure=dist_measure)
            for i in rangex:
                for j in rangey:
                    val = density.for_region(ll=Vector3D(i * resolution,
                                                         j * resolution),
                                             ur=Vector3D((i + 1) * resolution,
                                                         (j + 1) * resolution))
                    res_df.iloc[i, j] += val

        return res_df


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
        er = ExecutionRecord()
        if er.intra_record_exists(self.__class__.__name__, exp_num):
            return [core.utils.pd_csv_read(os.path.join(cmdopts['exp_model_root'],
                                                        self.target_csv_stems()[0] + '.model'))]

        homing = IntraExpNestHomingTime1Robot(self.main_config, self.config)
        homing_df = homing.run(cmdopts, criteria, exp_num)[0]

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

        rate_df = core.utils.pd_csv_read(os.path.join(result_opaths[0], 'block-manipulation.csv'))

        # We calculate 1 data point for each interval
        res_df = pd.DataFrame(columns=['model'], index=rate_df.index)
        res_df['model'] = 0.0

        # Calculate density for each result path
        for path in result_opaths:
            self._calc_for_result(cmdopts, exp_num, path, homing_df, res_df)

        # Average our results
        res_df['model'] /= len(result_opaths)

        # All done!
        er.intra_record_add(self.__class__.__name__, exp_num)
        return [res_df]

        def calc_kernel(self,
                        exp_count: tp.Union[pd.DataFrame, float],
                        homing_time: tp.Union[pd.DataFrame, float]) -> tp.Union[pd.DataFrame, float]:
            r"""
            Perform the block acquisition rate calculation using Little's Law, modeling CRW robots
            entering/exiting the homing state using a two state queueing network:
            robots are either homing or searching, with interference avoidance treated as part of
            each of those states.

            .. math::
               \alpha_{b}^1 = \frac{\tau_{h}}{\mathcal{N}_{h}(t)}

            Args:
                exp_count: Number of robots in the swarm which are searching at time :math:`t`:
                           :math:`\mathcal{N}_{s}(t)`.

                homing_time: Average time each robot spends in the homing queue beginning at time
                             :math:`t`: :math:`\tau_{h}`.

            Returns:
                Estimate of the steady state rate of robots entering the homing queue,
                :math:`\alpha_{b}^1`.
            """
            homing_count = 1.0 - exp_count  # only 1 robot so this is OK to hardcode

            # Only searching robots contribute to the encounter rate
            return homing_count / homing_time * exp_count

    def _calc_for_result(self,
                         cmdopts: dict,
                         exp_num: int,
                         result_opath: str,
                         homing_df: pd.DataFrame,
                         res_df: pd.DataFrame):
        homing_time = homing_df['model']

        acq_counts_df = core.utils.pd_csv_read(os.path.join(result_opath, 'block-acq-counts.csv'))

        # We read these fractions directly from experimental data for the purposes of getting the
        # model correct. In the parent ODE model, these will be variables.
        exp_frac = acq_counts_df['cum_avg_true_exploring_for_goal'] + \
            acq_counts_df['cum_avg_false_exploring_for_goal']

        # Robots that are not exploring are homing by definition (in the queueing network view of
        # FSM avoiding collision is not a state)
        homing_frac = (1.0 - exp_frac)

        # Little's law: L=lambda* W, solving for lambda
        lam = homing_frac / homing_time

        # Only searching robots contribute to the encounter rate
        res_df['model'] = lam * exp_frac
        return res_df

################################################################################
# Inter-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpAcqSpatialDistError():
    """
    Runs :class:`IntraExpAcqSpatialDist` for each experiment in the batch, and compute the average
    error between model prediction and empirical data as a single data point.

    In order for this model to run, all experiments in the batch must have 1 robot.

    The model is the same for reactive and cognitive robots, as robots use light sensors to return
    to the nest regardless of their memory model.

    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return all([p == 1 for p in criteria.populations(cmdopts)])

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-acq-locs2D-LN-model-error']

    def legend_names(self) -> tp.List[str]:
        return ['Model Error']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        error = Model2DError('block-acq-locs2D.stddev',
                             IntraExpAcqSpatialDist,
                             self.main_config,
                             self.config)
        return error.generate(criteria, cmdopts)


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpAcqRate():
    """
    Models the steady state block acquisition rate of the swarm, assuming purely reactive robots.
    That is, one model datapoint is computed for each experiment within the batch.

    In order for this model to run, all experiments in the batch must have 1 robot.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_batch(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict) -> bool:
        return all([p == 1 for p in criteria.populations(cmdopts)])

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-manip-free-pickup-events-cum-avg']

    def legend_names(self) -> tp.List[str]:
        return ['Predicted Block Acquisition Rate']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            cmdopts: dict) -> tp.List[pd.DataFrame]:

        er = ExecutionRecord()
        if er.inter_record_exists(self.__class__.__name__):
            return [core.utils.pd_csv_read(os.path.join(cmdopts['exp_model_root'],
                                                        self.target_csv_stems()[0] + '.model'))]
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
            intra_df = IntraExpAcqRate(self.main_config,
                                       self.config).run(criteria,
                                                        i,
                                                        cmdopts2)[0]
            res_df[exp] = intra_df.loc[intra_df.index[-1], 'model']

            print(res_df)

        # All done!
        er.inter_record_add(self.__class__.__name__)
        return [res_df]
