# Copyright 2018 John Harwell, All rights reserved.
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
See :ref:`ln-var-ts` for documentation and usage.
"""

# Core packages
import typing as tp

# 3rd party packages
import implements

# Project packages
from core.variables.base_variable import IBaseVariable
from core.xml_luigi import XMLAttrChangeSet, XMLAttrChange, XMLTagRmList, XMLTagAddList
import core.variables.time_setup as ts


@implements.implements(IBaseVariable)
class TimeSetup():
    def __init__(self, duration: int, metric_interval: int) -> None:
        self.duration = duration
        self.metric_interval = metric_interval
        self.attr_changes = []

    def gen_attr_changelist(self) -> tp.List[XMLAttrChangeSet]:
        if not self.attr_changes:
            self.attr_changes = [XMLAttrChangeSet(XMLAttrChange(".//output/metrics/append",
                                                                "output_interval",
                                                                "{0}".format(self.metric_interval)),
                                                  XMLAttrChange(".//output/metrics/truncate",
                                                                "output_interval",
                                                                "{0}".format(self.metric_interval)),
                                                  XMLAttrChange(".//output/metrics/create",
                                                                "output_interval",
                                                                "{0}".format(max(1, self.metric_interval / ts.kND_DATA_DIVISOR))))]

        return self.attr_changes

    def gen_tag_rmlist(self) -> tp.List[XMLTagRmList]:
        return []

    def gen_tag_addlist(self) -> tp.List[XMLTagAddList]:
        return []


class Parser(ts.Parser):
    pass


def factory(cmdline: str) -> TimeSetup:
    """
    Factory to create :class:`TimeSetup` derived classes from the command line definition.

    Parameters:
       cmdline: The value of ``--time-setup``
    """
    attr = Parser()(cmdline.split(".")[1])

    def __init__(self) -> None:
        TimeSetup.__init__(self,
                           attr["duration"],
                           int(attr["duration"] * ts.kTICKS_PER_SECOND / attr["n_datapoints"]))

    return type(cmdline,  # type: ignore
                (TimeSetup,),
                {"__init__": __init__})


__api__ = [
    'k1D_DATA_POINTS',
    'kND_DATA_DIVISOR',
    'TimeSetup',
]
