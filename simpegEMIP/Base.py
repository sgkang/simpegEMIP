# from SimPEG.EM.Base import BaseEMProblem, EMPropMap
# from SimPEG import Maps
# from scipy.constants import mu_0
# import numpy as np

# class ColeColePropMap(Maps.PropMap):
#     """
#         Property Map for EM Problems. The electrical conductivity (\\(\\sigma\\)) is the default inversion property, and the default value of the magnetic permeability is that of free space (\\(\\mu = 4\\pi\\times 10^{-7} \\) H/m)
#     """

#     sigmaInf = Maps.Property("Electrical Conductivity", defaultInvProp=True, propertyLink=('rhoInf', Maps.ReciprocalMap))
#     rhoInf   = Maps.Property("Electrical Resistivity", propertyLink=('sigmaInf', Maps.ReciprocalMap))

#     eta = Maps.Property("Electrical Conductivity", defaultVal=0)
#     tau = Maps.Property("Electrical Conductivity", defaultVal=0.001)
#     c   = Maps.Property("Electrical Conductivity", defaultVal=0.5)

#     mu  = Maps.Property("Inverse Magnetic Permeability", defaultVal=mu_0, propertyLink=('mui', Maps.ReciprocalMap))
#     mui = Maps.Property("Inverse Magnetic Permeability", defaultVal=1./mu_0, propertyLink=('mu', Maps.ReciprocalMap))


# class BaseAEMProblem(BaseEMProblem):

#     def __init__(self, mesh, **kwargs):
#         Problem.BaseProblem.__init__(self, mesh, **kwargs)

#     @property
#     def MeS(self):
#         """
#             Current term: MeS = Me(js)
#         """
#         if getattr(self, '_MeS', None) is None:
#             self._MeS = np.zeros((self.mesh.nE,len(self.survey.srcList)))
#             for isrc, src in enumerate(self.survey.srcList):
#                 self._MeS[:,isrc] = src.getMeS(self.mesh, self.MfMui)
#         return self._MeS

# # class ColeColeMap(IdentityMap):
# #     """
# #         Takes a vector of [sigmaInf, eta, tau, c]
# #         and sets

# #             - sigmaInf: Conductivity at infinite frequency (S/m)
# #             - eta: Chargeability (V/V)
# #             - tau: time constant (sec)
# #             - c: Frequency dependency (Dimensionless)
# #     """
# #     sigmaInf = None
# #     eta = None
# #     tau = None
# #     c = None

# #     def __init__(self, mesh, nC=None):
# #         self.mesh = mesh
# #         self.nC = nC or mesh.nC

# #     def _transform(self, m):
# #         m = m.reshape((self.mesh.nC, 4), order = "F")
# #         self.sigmaInf = m[:,0]
# #         self.eta = m[:,1]
# #         self.tau = m[:,2]
# #         self.c = m[:,3]
# #         return Utils.mkvc(m)

# #     @property
# #     def shape(self):
# #         return (self.nP, self.nP)

# #     @property
# #     def nP(self):
# #         """Number of parameters in the model."""
# #         return self.nC*4


# if __name__ == '__main__':
#     from simpegAIP.FD.Utils import ColeColePelton, ColeColeSeigel

#     maps = [("sigmaInf", Maps.IdentityMap(nP=5))]
#     PM = ColeColePropMap(maps)



#     m = PM(np.ones(5))
#     print m.mu

#     print ColeColePelton(1, m.sigmaInf, m.eta, m.tau, m.c)
