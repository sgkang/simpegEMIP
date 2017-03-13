# from SimPEG import *
# from simpegAIP.FD import AFEMIPProblem_b
# from SimPEG import EM
# from pymatsolver import MumpsSolver
# from scipy.constants import mu_0
# from simpegAIP.FD.Utils import hzAnalyticDipoleF_CC, hzAnalyticCentLoopF_CC
# from scipy.constants import mu_0
# import matplotlib.pyplot as plt

# import unittest

# def halfSpaceProblemAnaDiff(showIt=False, srcType="VMD", radius=13.):
#     cs, ncx, ncz, npad = 20., 25, 20, 20
#     hx = [(cs,ncx), (cs,npad,1.3)]
#     hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
#     mesh = Mesh.CylMesh([hx,1,hz], '00C')
#     sighalf = 1e-3
#     siginf = np.ones(mesh.nC)*1e-8
#     siginf[mesh.gridCC[:,-1]<0.] = sighalf
#     eta = np.ones(mesh.nC)*0.1
#     tau = np.ones(mesh.nC)*0.0005
#     c = np.ones(mesh.nC)*0.7
#     m = np.r_[siginf, eta, tau, c]

#     offset = 8.
#     frequency = np.logspace(1, 4, 41)
#     srcLists = []
#     nfreq = frequency.size
#     if srcType == "VMD":
#         rx0 = EM.FDEM.Rx(np.array([[offset, 0., 0.]]), 'bzr_sec')
#         rx1 = EM.FDEM.Rx(np.array([[offset, 0., 0.]]), 'bzi_sec')
#         bz_ana = mu_0*hzAnalyticDipoleF_CC(offset, frequency, sigmaInf=sighalf, eta=eta[0], tau=tau[0], c=c[0])
#         for ifreq in range(nfreq):
#             src = EM.FDEM.Src.MagDipole([rx0, rx1], loc = np.array([[0., 0., 0.]]), freq=frequency[ifreq])
#             srcLists.append(src)
#     elif srcType == "CircularLoop":
#         rx0 = EM.FDEM.Rx(np.array([[0., 0., 0.]]), 'bzr_sec')
#         rx1 = EM.FDEM.Rx(np.array([[0., 0., 0.]]), 'bzi_sec')
#         bz_ana = mu_0*hzAnalyticCentLoopF_CC(radius, frequency, sigmaInf=sighalf, eta=eta[0], tau=tau[0], c=c[0])
#         for ifreq in range(nfreq):
#             src = EM.FDEM.Src.CircularLoop([rx0, rx1], frequency[ifreq], np.array([[0., 0., 0.]]), radius=radius)
#             srcLists.append(src)

#     survey = EM.FDEM.Survey(srcLists)
#     iMap = Maps.IdentityMap(nP=int(mesh.nC))
#     maps = [('sigmaInf', iMap), ('eta', iMap), ('tau', iMap), ('c', iMap)]

#     prob = AFEMIPProblem_b(mesh, mapping = maps)

#     prob.Solver = MumpsSolver
#     survey.pair(prob)
#     obs = survey.dpred(m)
#     out = obs.reshape((nfreq, 2))[:,0]+1j*obs.reshape((nfreq, 2))[:,1]


#     err = np.linalg.norm(bz_ana-out)/np.linalg.norm(bz_ana)
#     print '>> Relative error = ', err

#     if showIt:
#         plt.loglog(frequency, abs(bz_ana.real), 'k')
#         plt.loglog(frequency, abs(out.real), 'b')
#         plt.loglog(frequency, abs(bz_ana.imag), 'k--')
#         plt.loglog(frequency, abs(out.imag), 'b--')
#         plt.show()
#     return err


# class FDEM_bTests(unittest.TestCase):

#     def test_analytic_VMD(self):
#         self.assertTrue(halfSpaceProblemAnaDiff(showIt=False, srcType="VMD") < 0.3)

#     def test_analytic_CircularLoop(self):
#         self.assertTrue(halfSpaceProblemAnaDiff(showIt=False, srcType="CircularLoop") < 0.3)

# if __name__ == '__main__':
#     unittest.main()
