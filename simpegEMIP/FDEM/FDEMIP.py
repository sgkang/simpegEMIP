from scipy.constants import mu_0
from SimPEG import Maps, Utils, Problem, Props
from SimPEG.EM.FDEM import Survey as SurveyFDEM
from SimPEG.EM.FDEM import Fields3D_e, FieldsFDEM
from SimPEG.EM.FDEM import Problem3D_e as Problem3DEM_e
from SimPEG.EM.Utils import omega
from .Utils import ColeColePelton, ColeColeSeigel


class Problem3D_e(Problem3DEM_e):

    _solutionType = 'eSolution'
    _formulation = 'EB'
    fieldsPair = Fields3D_e

    sigmaInf, sigmaInfMap, sigmaInfDeriv = Props.Invertible(
        "Electrical conductivity at infinite frequency(S/m)"
    )

    chi = Props.PhysicalProperty(
        "Magnetic susceptibility",
        default=0.
    )

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical chargeability (V/V), 0 <= eta < 1",
        default=0.
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Time constant (s)",
        default=1.
    )

    c, cMap, cDeriv = Props.Invertible(
        "Frequency Dependency, 0 < c < 1",
        default=0.5
    )

    h, hMap, hDeriv = Props.Invertible(
        "Receiver Height (m), h > 0",
    )

    def __init__(self, mesh, **kwargs):
        Problem3DEM_e.__init__(self, mesh, **kwargs)

    def MeSigma(self, freq):
        """
            Edge inner product matrix for \\(\\sigma\\). Used in the E-B formulation
        """
        sigma = ColeColePelton(freq, self.sigmaInf, self.eta, self.tau, self.c)
        return self.mesh.getEdgeInnerProduct(sigma)

    def MeSigmaI(self, freq):
        """
            Inverse of the edge inner product matrix for \\(\\sigma\\).
        """
        sigma = ColeColePelton(freq, self.sigmaInf, self.eta, self.tau, self.c)
        return self.mesh.getEdgeInnerProduct(sigma, invMat=True)

    def getA(self, freq):
        """
        System matrix

        .. math ::
            \mathbf{A} = \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C}
            + i \omega \mathbf{M^e_{\sigma}}

        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A
        """

        MfMui = self.MfMui
        MeSigma = self.MeSigma(freq)
        C = self.mesh.edgeCurl

        return C.T*MfMui*C + 1j*omega(freq)*MeSigma
