from __future__ import division
import numpy as np
import scipy.sparse as sp
from SimPEG import Utils, EM
from SimPEG.EM.Static.DC import FieldsDC


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


class Fields3D_e_Inductive(FieldsDC):
    knownFields = {'eref': 'E', 'phiIP': 'N'}
    aliasFields = {
        'e': ['eref', 'E', '_e']
    }
    # primary - secondary
    # N variables

    def __init__(self, mesh, survey, **kwargs):
        FieldsDC.__init__(self, mesh, survey, **kwargs)

    def startup(self):
        self.prob = self.survey.prob

    def _GLoc(self, fieldType):
        if fieldType == 'e' or fieldType == 'j':
            return 'E'
        else:
            raise Exception('Field type must be e, j')

    def _e(self, eSolution, srcList):
        return eSolution

    def _eDeriv_u(self, src, v, adjoint=False):
        return Identity()*v

    def _eDeriv_m(self, src, v, adjoint=False):
        return Zero()
