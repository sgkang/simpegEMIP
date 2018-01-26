from SimPEG import Problem, Survey, Utils, Maps
# from pymatsolver import MumpsSolver
from . import BiotSavart
import numpy as np
from sys import stdout
from time import sleep

# class LinearProblem(Problem.BaseProblem):
#     surveyPair = Survey.BaseSurvey
#     P = None
#     J = None
#     emax = None
#     sigma = None

#     def __init__(self, mesh, mapping, **kwargs):
#         Problem.BaseProblem.__init__(self, mesh, mapping, **kwargs)

#     @Utils.requires('survey')
#     def getJ(self, sigma, emax, senseoption=True):
#         """
#             Computing sensitivity for airbonre time domain EM IP inversion.
#         """
#         Meinv = self.mesh.getEdgeInnerProduct(invMat=True)
#         MeSig = self.mesh.getEdgeInnerProduct(sigma)
#         Asiginf = self.mesh.nodalGrad.T*MeSig*self.mesh.nodalGrad
#         MeDeriv = self.mesh.getEdgeInnerProductDeriv(np.ones(self.mesh.nC))

#         print ">>Factorizing ADC matrix"
#         if senseoption == True:
#             # ctx = DMumpsContext()
#             # if ctx.myid == 0:
#             #     ctx.set_icntl(14, 60)
#             #     ctx.set_centralized_sparse(Asiginf)
#             # ctx.run(job=4) # Factorization
#             Ainv = MumpsSolver(Asiginf)
#         ntx = self.survey.xyz_tx.shape[0]
#         J = np.zeros((ntx, self.mesh.nC))

#         J1_temp = np.zeros(self.mesh.nC)
#         J2_temp = np.zeros(self.mesh.nC)

#         print ">>Computing sensitivity"
#         for i in range (ntx):
#             stdout.write( (" \r%d/%d transmitter") % (i, ntx) )
#             stdout.flush()
#             S_temp = MeDeriv((emax[i,:]))*Utils.sdiag(sigma)
#             G_temp = BiotSavart.BiotSavartFun(self.mesh, self.survey.xyz_tx[i,:], component = 'z')


#             # Considering eIP term in jIP
#             if senseoption == True:

#                 rhs = self.mesh.nodalGrad.T*MeSig*Meinv*self.mesh.aveE2CCV.T*G_temp.T
#                 # if ctx.myid == 0:
#                 #     x = rhs.copy()
#                 #     ctx.set_rhs(x)
#                 # ctx.run(job=3) # Solve
#                 x = Ainv*rhs
#                 J1_temp = Utils.mkvc((S_temp.T*self.mesh.nodalGrad*x).T)
#                 J2_temp = Utils.mkvc(G_temp*self.mesh.aveE2CCV*Meinv*S_temp)

#                 J[i,:] = J1_temp - J2_temp

#             # Only consider polarization current
#             else:

#                 J2_temp = Utils.mkvc(G_temp*self.mesh.aveE2CCV*Meinv*S_temp)
#                 J[i,:] = - J2_temp
#         stdout.write("\n")
#         if senseoption == True:
#             ctx.destroy()

#         self.J = J
#         self.sigma = sigma
#         self.emax = emax

#     def fields(self, m, u=None):
#         return self.J.dot(self.mapping*m)

#     def Jvec(self, m, v, u=None):
#         P = self.mapping.deriv(m)
#         jvec = self.J.dot(P*v)
#         return jvec

#     def Jtvec(self, m, v, u=None):
#         P = self.mapping.deriv(m)
#         jtvec =P.T*(self.J.T.dot(v))
#         return jtvec



# class AirbornSurvey(Survey.BaseSurvey):
#     xyz_tx = None
#     def __init__(self, **kwargs):
#         Survey.BaseSurvey.__init__(self, **kwargs)

#     @Utils.requires('prob')
#     def dpred(self, m, u=None):
#         return self.prob.fields(m)

#     def residual(self, m, u=None):
#         if self.dobs.size ==1:
#             return Utils.mkvc(np.r_[self.dpred(m, u=u) - self.dobs])
#         else:
#             return Utils.mkvc(self.dpred(m, u=u) - self.dobs)

#     def residualWeighted(self, m, u=None):
#         if self.dobs.size ==1:
#             return Utils.mkvc(np.r_[self.Wd*self.residual(m, u=u)])
#         else:
#             return Utils.mkvc(self.Wd*self.residual(m, u=u))

class WeightMap(Maps.IdentityMap):
    """Weighted Map for distributed parameters"""

    def __init__(self, weight, **kwargs):
        Maps.IdentityMap.__init__(self)
        self.mesh = mesh
        self.weight = weight

    def _transform(self, m):
        return m*self.weight

    def deriv(self, m):
        return Utils.sdiag(self.weight)
