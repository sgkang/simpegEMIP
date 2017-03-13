# from SimPEG import Utils, np, sp, Models, Maps, Solver as SimpegSolver
# from scipy.constants import mu_0
# from SimPEG.EM.FDEM import Survey as SurveyFDEM
# from SimPEG.EM.FDEM import Fields_b, Fields, BaseFDEMProblem
# from SimPEG.EM.Utils import omega
# from simpegAIP.FD.Utils import ColeColePelton, ColeColeSeigel
# from simpegAIP.Base import ColeColePropMap

# class BaseAFEMIPProblem(BaseFDEMProblem):
#     """
#         We start by looking at Maxwell's equations in the electric
#         field \\\(\\\mathbf{e}\\\) and the magnetic flux
#         density \\\(\\\mathbf{b}\\\)

#         .. math ::

#             \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
#             {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{M^e} \mathbf{s_e}}

#         if using the E-B formulation (:code:`Problem_e`
#         or :code:`Problem_b`) or the magnetic field
#         \\\(\\\mathbf{h}\\\) and current density \\\(\\\mathbf{j}\\\)

#         .. math ::

#             \mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{j} + i \omega \mathbf{M_{\mu}^e} \mathbf{h} = \mathbf{M^e} \mathbf{s_m} \\\\
#             \mathbf{C} \mathbf{h} - \mathbf{j} = \mathbf{s_e}

#         if using the H-J formulation (:code:`Problem_j` or :code:`Problem_h`).

#         The problem performs the elimination so that we are solving the system for \\\(\\\mathbf{e},\\\mathbf{b},\\\mathbf{j} \\\) or \\\(\\\mathbf{h}\\\)
#     """
#     surveyPair = SurveyFDEM
#     fieldsPair = Fields
#     PropMap = ColeColePropMap

#     def __init__(self, mesh,**kwargs):
#         BaseFDEMProblem.__init__(self, mesh, **kwargs)

#     @property
#     def deleteTheseOnModelUpdate(self):
#         toDelete = []
#         # Later put somethings...
#         return toDelete
#     # @property
#     # def curModel(self):
#     #     """
#     #         Sets the current model, and removes dependent mass matrices.
#     #     """
#     #     return getattr(self, '_curModel', None)

#     # @curModel.setter
#     # def curModel(self, value):

#     #     if value is self.curModel:
#     #         return # it is the same!
#     #     if self.PropMap is not None:
#     #         self._curModel = self.mapping(value)
#     #     else:
#     #         self._curModel = Models.Model(value, self.mapping)
#     #         self._curModel.mui = 1./mu_0
#     #         self._curModel.mu = mu_0
#     #         m = value.reshape((self.mesh.nC, 4), order="F")
#     #         # Set Cole-Cole parameters
#     #         self.sigmaInf = m[:,0]
#     #         self.eta = m[:,1]
#     #         self.tau = m[:,2]
#     #         self.c = m[:,3]
#     #         self.mu = mu_0

#     #     for prop in self.deleteTheseOnModelUpdate:
#     #         if hasattr(self, prop):
#     #             delattr(self, prop)

#     def MeSigma(self, freq):
#         """
#             Edge inner product matrix for \\(\\sigma\\). Used in the E-B formulation
#         """
#         sigma = ColeColePelton(freq, self.curModel.sigmaInf, self.curModel.eta, self.curModel.tau, self.curModel.c)
#         return self.mesh.getEdgeInnerProduct(sigma)



#     def MeSigmaI(self, freq):
#         """
#             Inverse of the edge inner product matrix for \\(\\sigma\\).
#         """
#         sigma = ColeColePelton(freq, self.curModel.sigmaInf, self.curModel.eta, self.curModel.tau, self.curModel.c)
#         return self.mesh.getEdgeInnerProduct(sigma, invMat=True)


#     def fields(self, m=None):
#         """
#             Solve the forward problem for the fields.
#         """

#         self.curModel = m
#         F = self.fieldsPair(self.mesh, self.survey)

#         for freq in self.survey.freqs:
#             A = self.getA(freq)
#             rhs = self.getRHS(freq)
#             Ainv = self.Solver(A, **self.solverOpts)
#             sol = Ainv * rhs
#             Srcs = self.survey.getSrcByFreq(freq)
#             ftype = self._fieldType + 'Solution'
#             F[Srcs, ftype] = sol
#             Ainv.clean()
#         return F

# if __name__ == '__main__':

#     from SimPEG import *

#     expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
#     iMap = Maps.IdentityMap(nP=3)
#     assert expMap.nP == 3

#     mapping = {'maps':[('sigmaInf', expMap), ('c', iMap)],'slices':{'c':slice(1,4)}}

#     PM = ColeColePropMap(mapping)

#     model = PM(np.r_[1.,2,3,4])

#     print model.sigmaInf
#     print model.rhoInf
#     print model.rhoInfDeriv
#     print model.tau
#     print model.c
#     print model.eta
