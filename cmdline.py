# Copyright 2019 John Harwell, All rights reserved.
#
#  This file is part of SIERRA.
#
#  SIERRA is free software: you can redistribute it and/or modify it under the terms of the GNU
#  General Public License as published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  SIERRA is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with
#  SIERRA.  If not, see <http://www.gnu.org/licenses/
#
"""
Command line parsing and validation for the :xref:`FORDYCA` project.
"""

import typing as tp

import core.cmdline


class Cmdline(core.cmdline.CoreCmdline):
    """
    Defines FORDYCA extensions to the core command line arguments defined in
    :class:`~core.cmdline.CoreCmdline`.
    """

    def __init__(self, super_scaffold: bool = False) -> None:
        super().__init__(super_scaffold)

        self.parser.add_argument("--controller",
                                 metavar="{depth0, depth1, depth2}.<controller>",
                                 help="""

                                 Which controller footbot robots will use in the foraging experiment. All robots use the
                                 same controller (homogeneous swarms).

                                 Valid controllers:

                                 - d0.{CRW, DPO, MDPO},
                                 - d1.{BITD_DPO, OBITD_DPO},
                                 - d2.{BIRTD_DPO, OBIRTD_DPO}

                                 Head over to the :xref:`FORDYCA` docs for the descriptions of these controllers.


                                 """ + self.stage_usage_doc([1, 2, 3, 4, 5],
                                                            "Only required for stage 5 if ``--scenario-comp`` is passed."))

        self.stage1.add_argument("--static-cache-blocks",
                                 help="""

                                 # of blocks used when the static cache is respawned (depth1 controllers only).


                                 """ + self.stage_usage_doc([1]),
                                 default=None)

    @staticmethod
    def cmdopts_update(cli_args, cmdopts: tp.Dict[str, str]):
        """
        Updates the core cmdopts dictionary with (key,value) pairs from the FORDYCA-specific cmdline options.
        """
        # Stage1
        updates = {
            'controller': cli_args.controller,
            'static_cache_blocks': cli_args.static_cache_blocks
        }
        cmdopts.update(updates)


class CmdlineValidator(core.cmdline.CoreCmdlineValidator):
    pass


def sphinx_cmdline():
    """
    Return a handle to the :xref:`FORDYCA` cmdline extensions to autogenerate nice documentation
    from it.

    """
    return Cmdline(True).parser
