from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.EM.TDEM.RxTDEM import BaseRx as BaseRxTDEM
from .Rx import BaseRx
from SimPEG.EM.TDEM.SrcTDEM import BaseTDEMSrc
from SimPEG.EM.TDEM import Survey as BaseTDEMSurvey
from SimPEG import Utils
from SimPEG.Utils import Zero, closestPoints
import uuid


class Survey(BaseTDEMSurvey):
    """
    Time domain electromagnetic survey
    """

    srcPair = BaseTDEMSrc
    rxPair = BaseRx

    def eval(self, u):
        data = SimPEG.Survey.Data(self)
        for i_src, src in enumerate(self.srcList):
            for rx in src.rxList:
                data[src, rx] = rx.eval(
                    i_src, self.mesh, self.prob.timeMesh, u
                )
        return data


class SurveyLinear(BaseEMSurvey):
    srcPair = BaseTDEMSrc
    rxPair = BaseRxTDEM
    times = None

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)
        self.getUniqueTimes()

    def getUniqueTimes(self):
        time_rx = []
        for src in self.srcList:
            for rx in src.rxList:
                time_rx.append(rx.times)
        self.times = np.unique(np.hstack(time_rx))

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        return self.prob.forward(m, f=f)


class Survey_singletime(Survey):

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        return self.prob.forward(m, f=f)

    @property
    def nD(self):
        return int(self.vnD.sum() / self.times.size)
