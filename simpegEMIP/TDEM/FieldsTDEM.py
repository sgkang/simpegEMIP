from __future__ import division
import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG import Utils, EM
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, Identity


class Fields3D_e(EM.TDEM.Fields3D_e):

    def startup(self):
        self._MeSigmaInfI = self.survey.prob.MeSigmaInfI
        self._MeSigmaInfIDeriv = self.survey.prob.MeSigmaInfIDeriv
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MfMui = self.survey.prob.MfMui
        self._times = self.survey.prob.times
