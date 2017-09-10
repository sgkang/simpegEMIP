from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.EM.TDEM.RxTDEM import BaseRx
from SimPEG.EM.TDEM.SrcTDEM import BaseTDEMSrc
from SimPEG import Utils
from SimPEG.Utils import Zero, closestPoints
import uuid


class Survey(BaseEMSurvey):
    rxPair = BaseRx
    srcPair = BaseTDEMSrc
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
