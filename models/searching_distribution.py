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
import os
import typing as tp

# 3rd party packages
import implements
import pandas as pd

# Project packages
import core.models.interface
import core.utils
from core.experiment_spec import ExperimentSpec
import core.variables.batch_criteria as bc
from core.vector import Vector3D

from projects.fordyca.models.density import ExplorationDensity
from projects.fordyca.models.model_error import Model2DError
from projects.fordyca.models.representation import BlockClusterSet
from projects.fordyca.models.representation import Nest
from projects.fordyca.models.dist_measure import DistanceMeasure2D


def available_models(category: str):
    if category == 'intra':
        return ['IntraExpSearchingDistribution']
    elif category == 'inter':
        return ['InterExpSearchingDistributionError']
    else:
        return None

################################################################################
# Intra-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteIntraExpModel2D)
class IntraExpSearchingDistribution():
    """
    Models swarm's steady state spatial distribution in 2D space when engaged in searching, assuming
    purely reactive robots.
    """

    def __init__(self, main_config: dict, config: dict):
        self.main_config = main_config
        self.config = config

    def run_for_exp(self, criteria: bc.IConcreteBatchCriteria, cmdopts: dict, i: int) -> bool:
        return True

    def target_csv_stems(self) -> tp.List[str]:
        return ['block-acq-explore-locs2D']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            exp_num: int,
            cmdopts: dict) -> pd.DataFrame:

        # Calculate nest extent
        nest = Nest(cmdopts, criteria, exp_num)

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

        # Get the arena dimensions in cells from the actual swarm distribution. This is OK because
        # we don't refer to the empirical results in any other way. We can't just use the dims in
        # cmdopts, because those are real-valued--not descretized.
        exp_df = core.utils.pd_csv_read(os.path.join(result_opaths[0],
                                                     'block-acq-explore-locs2D.csv'))

        # Calculate arena resolution
        spec = ExperimentSpec(criteria, exp_num, cmdopts)
        resolution = spec.arena_dim.xsize() / len(exp_df.index)

        res_df = pd.DataFrame(columns=exp_df.columns, index=exp_df.index, dtype=float)
        res_df[:] = 0.0

        for path in result_opaths:
            self._calc_for_result(cmdopts, path, resolution, nest, spec.arena_dim, res_df)

        # Average our results across all simulations
        res_df /= len(result_opaths)

        # All done!
        return res_df

    def _calc_for_result(self,
                         cmdopts: dict,
                         result_opath: str,
                         resolution: float,
                         nest: core.utils.ArenaExtent,
                         arena: core.utils.ArenaExtent,
                         res_df: pd.DataFrame):

        # Get clusters in the arena
        clusters = BlockClusterSet(cmdopts, result_opath)

        dist_measure = DistanceMeasure2D(cmdopts['scenario'])
        density = ExplorationDensity(nest=nest,
                                     arena=arena,
                                     clusters=clusters,
                                     dist_measure=dist_measure)

        for i in range(0, len(res_df.index)):
            for j in range(0, len(res_df.columns)):
                val = density.for_region(Vector3D(i * resolution,
                                                  j * resolution),
                                         Vector3D((i + 1) * resolution,
                                                  (j + 1) * resolution))

                res_df.iloc[i, j] += val

        return res_df

################################################################################
# Inter-experiment models
################################################################################


@implements.implements(core.models.interface.IConcreteInterExpModel1D)
class InterExpSearchingDistributionError():
    """
    Runs :class:`IntraExpSearchingDistribution` for each experiment in the batch, and compute the
    average error between model prediction and empirical data as a single data point.

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
        return ['block-acq-explore-locs2D-LN-model-error']

    def legend_names(self) -> tp.List[str]:
        return ['Model Error']

    def __repr__(self) -> str:
        return self.__class__.__name__

    def run(self,
            criteria: bc.IConcreteBatchCriteria,
            cmdopts: dict) -> pd.DataFrame:

        error = Model2DError('block-acq-explore-locs2D.stddev',
                             IntraExpSearchingDistribution,
                             self.main_config,
                             self.config)
        return error.generate(criteria, cmdopts)
