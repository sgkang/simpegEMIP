
# from SimPEG import Problem, Utils, np, sp, Solver as SimpegSolver
# from scipy.constants import mu_0
# from SimPEG.EM.FDEM import Survey as SurveyFDEM
# from SimPEG.EM.FDEM import Fields_b, Fields
# from SimPEG.EM.Base import BaseEMProblem
# from SimPEG.EM.Utils import omega
# from BaseAFEMIP import BaseAFEMIPProblem
# from simpegAIP.FD.Utils import ColeColePelton, ColeColeSeigel

# class AFEMIPProblem_b(BaseAFEMIPProblem):
#     """
#         We eliminate \\\(\\\mathbf{e}\\\) using

#         .. math ::

#              \mathbf{e} = \mathbf{M^e_{\sigma}}^{-1} \\left(\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{s_e}\\right)

#         and solve for \\\(\\\mathbf{b}\\\) using:

#         .. math ::

#             \\left(\mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f}  + i \omega \\right)\mathbf{b} = \mathbf{s_m} + \mathbf{M^e_{\sigma}}^{-1}\mathbf{M^e}\mathbf{s_e}

#         .. note ::
#             The inverse problem will not work with full anisotropy
#     """

#     _fieldType = 'b'
#     _eqLocs    = 'FE'
#     fieldsPair = Fields_b
#     # addprimary = False

#     def __init__(self, mesh, **kwargs):
#         BaseAFEMIPProblem.__init__(self, mesh, **kwargs)

#     def getA(self, freq):
#         """
#             .. math ::
#                 \mathbf{A} = \mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f}  + i \omega

#             :param float freq: Frequency
#             :rtype: scipy.sparse.csr_matrix
#             :return: A
#         """

#         MfMui = self.MfMui
#         MeSigmaI = self.MeSigmaI(freq)
#         C = self.mesh.edgeCurl
#         iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

#         A = C * (MeSigmaI * (C.T * MfMui)) + iomega

#         if self._makeASymmetric is True:
#             return MfMui.T*A
#         return A

#     # TODOs: frequency-dependent ... conductivity

#     # def getADeriv_m(self, freq, u, v, adjoint=False):

#     #     MfMui = self.MfMui
#     #     C = self.mesh.edgeCurl
#     #     MeSigmaIDeriv = self.MeSigmaIDeriv
#     #     vec = C.T * (MfMui * u)

#     #     MeSigmaIDeriv = MeSigmaIDeriv(vec)

#     #     if adjoint:
#     #         if self._makeASymmetric is True:
#     #             v = MfMui * v
#     #         return MeSigmaIDeriv.T * (C.T * v)

#     #     if self._makeASymmetric is True:
#     #         return MfMui.T * ( C * ( MeSigmaIDeriv * v ) )
#     #     return C * ( MeSigmaIDeriv * v )


#     def getRHS(self, freq):
#         """
#             .. math ::
#                 \mathbf{RHS} = \mathbf{s_m} + \mathbf{M^e_{\sigma}}^{-1}\mathbf{s_e}

#             :param float freq: Frequency
#             :rtype: numpy.ndarray (nE, nSrc)
#             :return: RHS
#         """

#         S_m, S_e = self.getSourceTerm(freq)
#         C = self.mesh.edgeCurl
#         MeSigmaI = self.MeSigmaI(freq)

#         RHS = S_m + C * ( MeSigmaI * S_e )

#         if self._makeASymmetric is True:
#             MfMui = self.MfMui
#             return MfMui.T * RHS

#         return RHS

#     # TODOs: frequency-dependent ... conductivity

#     # def getRHSDeriv_m(self, freq, src, v, adjoint=False):
#     #     C = self.mesh.edgeCurl
#     #     S_m, S_e = src.eval(self)
#     #     MfMui = self.MfMui

#     #     if self._makeASymmetric and adjoint:
#     #         v = self.MfMui * v

#     #     MeSigmaIDeriv = self.MeSigmaIDeriv(S_e)
#     #     S_mDeriv, S_eDeriv = src.evalDeriv(self, adjoint)

#     #     if not adjoint:
#     #         RHSderiv = C * (MeSigmaIDeriv * v)
#     #         SrcDeriv = S_mDeriv(v) + C * (self.MeSigmaI * S_eDeriv(v))
#     #     elif adjoint:
#     #         RHSderiv = MeSigmaIDeriv.T * (C.T * v)
#     #         SrcDeriv = S_mDeriv(v) + self.MeSigmaI.T * (C.T * S_eDeriv(v))

#     #     if self._makeASymmetric is True and not adjoint:
#     #         return MfMui.T * (SrcDeriv + RHSderiv)

#     #     return RHSderiv + SrcDeriv
