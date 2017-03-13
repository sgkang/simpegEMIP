from scipy.constants import mu_0
from simpegAIP.TD import ProblemATEMIP_b
import SimPEG.EM as EM
from SimPEG import Mesh, np, Maps
import EMTD
from EMTD.Utils import hzAnalyticDipoleT_CC
from pymatsolver import MumpsSolver
import matplotlib.pyplot as plt

import unittest

def halfSpaceProblemAnaVMDDiff(showIt=False, waveformType="STEPOFF"):
	cs, ncx, ncz, npad = 20., 25, 25, 15
	hx = [(cs,ncx), (cs,npad,1.3)]
	hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
	mesh = Mesh.CylMesh([hx,1,hz], '00C')   
	sighalf = 1e-2
	siginf = np.ones(mesh.nC)*1e-8
	siginf[mesh.gridCC[:,-1]<0.] = sighalf
	eta = np.ones(mesh.nC)*0.2
	tau = np.ones(mesh.nC)*0.005
	c = np.ones(mesh.nC)*0.7
	m = np.r_[siginf, eta, tau, c]
	iMap = Maps.IdentityMap(nP=int(mesh.nC))
	maps = [('sigmaInf', iMap), ('eta', iMap), ('tau', iMap), ('c', iMap)]
	prb = ProblemATEMIP_b(mesh, mapping = maps)	

	if waveformType =="GENERAL":
		# timeon = np.cumsum(np.r_[np.ones(10)*1e-3, np.ones(10)*5e-4, np.ones(10)*1e-4])
		timeon = np.cumsum(np.r_[np.ones(10)*1e-3, np.ones(10)*5e-4, np.ones(10)*1e-4])
		timeon -= timeon.max()
		timeoff = np.cumsum(np.r_[np.ones(20)*1e-5, np.ones(20)*1e-4, np.ones(20)*1e-3])
		time = np.r_[timeon, timeoff]
		current_on = np.ones_like(timeon)
		current_on[[0,-1]] = 0.
		current = np.r_[current_on, np.zeros_like(timeoff)]
		wave = np.c_[time, current]		
		prb.waveformType = "GENERAL"
		prb.currentwaveform(wave)
		prb.t0 = time.min()
	elif waveformType =="STEPOFF":
		prb.timeSteps = [(1e-5, 20), (1e-4, 20), (1e-3, 10)]
	offset = 20.
	tobs = np.logspace(-4, -2, 21)
	rx = EM.TDEM.RxTDEM(np.array([[offset, 0., 0.]]), tobs, "bz")
	src = EM.TDEM.SrcTDEM_VMD_MVP([rx], np.array([[0., 0., 0.]]), waveformType=waveformType)
	survey = EM.TDEM.SurveyTDEM([src])
	prb.Solver = MumpsSolver
	prb.pair(survey)
	out = survey.dpred(m)
	bz_ana = mu_0*hzAnalyticDipoleT_CC(offset, rx.times, sigmaInf=sighalf, eta=eta[0], tau=tau[0], c=c[0])
	err = np.linalg.norm(bz_ana-out)/np.linalg.norm(bz_ana)
	print '>> Relative error = ', err

	if showIt:
		plt.loglog(rx.times, abs(bz_ana), 'k')
		plt.loglog(rx.times, abs(out), 'b.')
		plt.show()	
	return err

class TDEM_bTests(unittest.TestCase):

	def test_analytic_STEPOFF(self):
		self.assertTrue(halfSpaceProblemAnaVMDDiff(showIt=False,waveformType="STEPOFF") < 0.3)	
		
	def test_analytic_GENERAL(self):
		self.assertTrue(halfSpaceProblemAnaVMDDiff(showIt=False, waveformType="GENERAL") < 0.6)	        

if __name__ == '__main__':
    unittest.main()
