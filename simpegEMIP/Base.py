from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
from scipy.constants import mu_0

from SimPEG import Survey
from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Props
from SimPEG import Solver as SimpegSolver
from SimPEG.EM.Base import BaseEMProblem


class BaseEMIPProblem(BaseEMProblem):

    sigmaInf, sigmaInfMap, sigmaInfDeriv = Props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    eta, etaMap, etaDeriv = Props.Invertible(
        "Cole-Cole chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Cole-Cole time constant (s)"
    )

    c, cMap, cDeriv = Props.Invertible(
        "Cole-Cole frequency dependency"
    )

    surveyPair = Survey.BaseSurvey  #: The survey to pair with.
    dataPair = Survey.Data  #: The data to pair with.

    mapPair = Maps.IdentityMap  #: Type of mapping to pair with

    Solver = SimpegSolver  #: Type of solver to pair with
    solverOpts = {}  #: Solver options

    verbose = False

    ####################################################
    # Mass Matrices
    ####################################################
    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaInfMap is not None:
            toDelete += ['_MeSigmaInf', '_MeSigmaInfI', '_MeSigma0', '_MeSigma0I']

        if hasattr(self, 'muMap') or hasattr(self, 'muiMap'):
            if self.muMap is not None or self.muiMap is not None:
                toDelete += ['_MeMu', '_MeMuI', '_MfMui', '_MfMuiI']
        return toDelete


    ####################################################
    # Electrical Conductivity
    ####################################################
    @property
    def MeSigmaInf(self):
        """
        Edge inner product matrix for \\(\\sigmaInf\\).
        Used in the E-B formulation
        """
        if getattr(self, '_MeSigmaInf', None) is None:
            self._MeSigmaInf = self.mesh.getEdgeInnerProduct(self.sigmaInf)
        return self._MeSigmaInf

    # TODO: This should take a vector
    def MeSigmaInfDeriv(self, u):
        """
        Derivative of MeSigmaInf with respect to the model
        """
        if self.sigmaInfMap is None:
            return Utils.Zero()

        return (
            self.mesh.getEdgeInnerProductDeriv(self.sigmaInf)(u) *
            self.sigmaInfDeriv
        )

    @property
    def MeSigmaInfI(self):
        """
        Inverse of the edge inner product matrix for \\(\\sigmaInf\\).
        """
        if getattr(self, '_MeSigmaInfI', None) is None:
            self._MeSigmaInfI = self.mesh.getEdgeInnerProduct(self.sigmaInf, invMat=True)
        return self._MeSigmaInfI

    # TODO: This should take a vector
    def MeSigmaInfIDeriv(self, u):
        """
        Derivative of :code:`MeSigmaInfI` with respect to the model
        """
        if self.sigmaInfMap is None:
            return Utils.Zero()

        if len(self.sigmaInf.shape) > 1:
            if self.sigmaInf.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MeSigmaInfIDeriv."
                )

        dMeSigmaInfI_dI = -self.MeSigmaInfI**2
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(self.sigmaInf)(u)
        return dMeSigmaInfI_dI * (dMe_dsig * self.sigmaInfDeriv)

    @property
    def MeSigma0(self):
        """
        Edge inner product matrix for \\(\\sigma0\\).
        Used in the E-B formulation
        """
        if getattr(self, '_MeSigma0', None) is None:
            self._MeSigma0 = self.mesh.getEdgeInnerProduct(self.sigmaInf*(1.-self.eta))
        return self._MeSigma0

    @property
    def MeSigma0(self):
        """
        Edge inner product matrix for \\(\\sigma0\\).
        Used in the E-B formulation
        """
        if getattr(self, '_MeSigma0', None) is None:
            self._MeSigma0 = self.mesh.getEdgeInnerProduct(self.sigmaInf*(1.-self.eta))    

if __name__ == '__main__':
    pass
