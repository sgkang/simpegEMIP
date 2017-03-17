from SimPEG import Mesh, Survey, Problem, Utils, Models, Maps, PropMaps, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SimPEG.EM.Base import BaseEMProblem, EMPropMap
from SimPEG.EM.TDEM.BaseTDEM import FieldsTDEM
from SimPEG.Problem import BaseTimeProblem
from simpegAIP.TD.BaseATEM import BaseATEMProblem
from simpegAIP.TD import ProblemATEM_b
import SimPEG.EM as EM
import EMTD
from EMTD.Utils import hzAnalyticDipoleT
from pymatsolver import MumpsSolver
import matplotlib.pyplot as plt

import unittest

def halfSpaceProblemAnaVMDDiff(showIt=False, waveformType="STEPOFF"):
	cs, ncx, ncz, npad = 20., 25, 25, 15
	hx = [(cs,ncx), (cs,npad,1.3)]
	hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
	mesh = Mesh.CylMesh([hx,1,hz], '00C')    
	prb = ProblemATEM_b(mesh)
	if waveformType =="GENERAL":
		timeon = np.cumsum(np.r_[np.ones(10)*1e-3, np.ones(10)*5e-4, np.ones(10)*1e-4])
		timeon -= timeon.max()
		timeoff = np.cumsum(np.r_[np.ones(10)*5e-5, np.ones(10)*1e-4, np.ones(10)*5e-4, np.ones(10)*1e-3, np.ones(10)*5e-3])
		time = np.r_[timeon, timeoff]
		current_on = np.ones_like(timeon)
		current_on[[0,-1]] = 0.
		current = np.r_[current_on, np.zeros_like(timeoff)]
		wave = np.c_[time, current]		
		prb.waveformType = "GENERAL"
		prb.currentwaveform(wave)
		prb.t0 = time.min()
	elif waveformType =="STEPOFF":
		prb.timeSteps = [(1e-5, 10), (5e-5, 10), (1e-4, 10), (5e-4, 10), (1e-3, 10),(5e-3, 10)]
	offset = 20.
	tobs = np.logspace(-4, -2, 21)
	rx = EM.TDEM.RxTDEM(np.array([[offset, 0., 0.]]), tobs, "bz")
	src = EM.TDEM.SrcTDEM_VMD_MVP([rx], np.array([[0., 0., 0.]]), waveformType=waveformType)
	survey = EM.TDEM.SurveyTDEM([src])
	prb.Solver = MumpsSolver
	sigma = np.ones(mesh.nC)*1e-8
	active = mesh.gridCC[:,2]<0.
	sig_half = 1e-2
	sigma[active] = sig_half
	prb.pair(survey)

	out = survey.dpred(sigma)
	bz_ana = mu_0*hzAnalyticDipoleT(offset, rx.times, sig_half)
	err = np.linalg.norm(bz_ana-out)/np.linalg.norm(bz_ana)
	print '>> Relative error = ', err

	if showIt:
		plt.loglog(rx.times, bz_ana, 'k')
		plt.loglog(rx.times, out, 'b.')
		plt.show()	
	return err


class TDEM_bTests(unittest.TestCase):

	def test_analytic_STEPOFF(self):
		self.assertTrue(halfSpaceProblemAnaVMDDiff(showIt=False,waveformType="STEPOFF") < 0.3)	
		
	def test_analytic_GENERAL(self):
		self.assertTrue(halfSpaceProblemAnaVMDDiff(showIt=False, waveformType="GENERAL") < 0.3)	        

if __name__ == '__main__':
    unittest.main()
