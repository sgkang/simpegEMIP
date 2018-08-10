# import numpy as np
# import scipy.sparse as sp
# from simpegEMIP.StretchedExponential import SEMultiInvProblem, SEMultiSurvey
# from SimPEG import Mesh, Utils, Maps
# import unittest
# import matplotlib.pyplot as plt


# class problem_e_forward(unittest.TestCase):

#     def setUp(self):

#         time = np.logspace(-3, 0, 21)
#         n_loc = 10
#         wires = Maps.Wires(('eta', n_loc), ('tau', n_loc), ('c', n_loc))
#         taumap = Maps.ExpMap(nP=n_loc)*wires.tau
#         etamap = Maps.ExpMap(nP=n_loc)*wires.eta
#         cmap = Maps.ExpMap(nP=n_loc)*wires.c
#         eta0, tau0, c0 = 0.1, 10., 0.5
#         m0 = np.log(
#             np.r_[eta0*np.ones(n_loc), tau0*np.ones(n_loc), c0*np.ones(n_loc)]
#         )
#         survey = SEMultiSurvey(time=time, locs=np.zeros((n_loc, 3)))
#         m1D = Mesh.TensorMesh([np.ones(int(n_loc*3))])
#         prob = SEMultiInvProblem(m1D, etaMap=etamap, tauMap=taumap, cMap=cmap)
#         prob.pair(survey)

#         self.survey = survey
#         self.m0 = m0
#         self.time = time

#     def test_SEMultiInvProblem_step_off(self, plotIt=False):

#         self.survey.n_pulse = 0
#         pred = self.survey.dpred(self.m0)[:self.survey.n_time]
#         true = get_peta_stepoff(self.time)
#         err = abs(pred-true) / abs(true)
#         if np.all(err < 0.01):
#             passed = True
#         else:
#             passed = False

#         if plotIt:
#             plt.plot(pred)
#             plt.plot(true[:self.survey.n_time])
#             plt.show()

#         self.assertTrue(passed)

#     def test_SEMultiInvProblem_pulse(self, plotIt=False):

#         self.survey.n_pulse = 2
#         pred = self.survey.dpred(self.m0)[:self.survey.n_time]
#         true = get_peta_off(self.time, self.survey.n_pulse)
#         err = abs(pred-true) / abs(true)
#         if np.all(err < 0.01):
#             passed = True
#         else:
#             passed = False

#         if plotIt:
#             plt.plot(pred)
#             plt.plot(true[:self.survey.n_time])
#             plt.show()

#         self.assertTrue(passed)

#     def test_SEMultiInvProblem_pulse(self, plotIt=False):

#         self.survey.n_pulse = 4
#         pred = self.survey.dpred(self.m0)[:self.survey.n_time]
#         true = get_peta_off(self.time, self.survey.n_pulse)
#         err = abs(pred-true) / abs(true)
#         if np.all(err < 0.01):
#             passed = True
#         else:
#             passed = False

#         if plotIt:
#             plt.plot(pred)
#             plt.plot(true[:self.survey.n_time])
#             plt.show()

#         self.assertTrue(passed)

# if __name__ == '__main__':
#     unittest.main()
