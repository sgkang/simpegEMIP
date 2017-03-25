from __future__ import division
import numpy as np
import scipy.sparse as sp
from SimPEG import Utils, EM

class Fields3D_e(EM.TDEM.Fields3D_e):

    def startup(self):
        self._MeSigmaInfI = self.survey.prob.MeSigmaInfI
        self._MeSigmaInfIDeriv = self.survey.prob.MeSigmaInfIDeriv
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MfMui = self.survey.prob.MfMui
        self._times = self.survey.prob.times        
        self._nodalGrad = self.survey.prob.mesh.nodalGrad

class Fields3D_phi(Fields3D_e):

    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'phiSolution': 'N'}
    aliasFields = {
                    'e': ['phiSolution', 'E', '_e']
                  }

    def _phi(self, phiSolution, srcList, tInd):
        return phiSolution

    def _e(self, phiSolution, srcList, tInd):
        return - self._nodalGrad * phiSolution
