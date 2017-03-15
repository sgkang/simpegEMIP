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

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        sigmaInf, sigmaInfMap, sigmaInfDeriv = Props.Invertible(
            "Electrical conductivity at infinite frequency (S/m)"
        )

        rhoInf, rhoInfMap, rhoInfDeriv = Props.Invertible(
            "Electrical resistivity at infinite frequency (Ohm m)"
        )

        Props.Reciprocal(sigmaInf, rhoInf)

        mu = Props.PhysicalProperty(
            "Magnetic Permeability (H/m)",
            default=mu_0
        )
        mui = Props.PhysicalProperty(
            "Inverse Magnetic Permeability (m/H)"
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

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete += ['_MeSigmaInf', '_MeSigmaInfI', '_MfRhoInf', '_MfRhoInfI']

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
    def MfRhoInf(self):
        """
        Face inner product matrix for \\(\\rhoInf\\). Used in the H-J
        formulation
        """
        if getattr(self, '_MfRhoInf', None) is None:
            self._MfRhoInf = self.mesh.getFaceInnerProduct(self.rhoInf)
        return self._MfRhoInf

    # TODO: This should take a vector
    def MfRhoInfDeriv(self, u):
        """
        Derivative of :code:`MfRhoInf` with respect to the model.
        """
        if self.rhoInfMap is None:
            return Utils.Zero()

        return (
            self.mesh.getFaceInnerProductDeriv(self.rhoInf)(u) * self.rhoInfDeriv
        )

    @property
    def MfRhoInfI(self):
        """
        Inverse of :code:`MfRhoInf`
        """
        if getattr(self, '_MfRhoInfI', None) is None:
            self._MfRhoInfI = self.mesh.getFaceInnerProduct(self.rhoInf, invMat=True)
        return self._MfRhoInfI

    # TODO: This should take a vector
    def MfRhoInfIDeriv(self, u):
        """
            Derivative of :code:`MfRhoInfI` with respect to the model.
        """
        if self.rhoInfMap is None:
            return Utils.Zero()

        if len(self.rhoInf.shape) > 1:
            if self.rhoInf.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MfRhoInfIDeriv."
                )

        dMfRhoInfI_dI = -self.MfRhoInfI**2
        dMf_drhoInf = self.mesh.getFaceInnerProductDeriv(self.rhoInf)(u)
        return dMfRhoInfI_dI * (dMf_drhoInf * self.rhoInfDeriv)

if __name__ == '__main__':
    pass