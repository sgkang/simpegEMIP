from __future__ import print_function
import unittest
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import DataMisfit
from SimPEG import regularization
from SimPEG import Optimization
from SimPEG import Inversion
from SimPEG import InvProblem
from SimPEG import Tests
import numpy as np
from simpegEMIP.StretchedExponential import SEMultiInvProblem, SEMultiSurvey
np.random.seed(30)


class stretched_exponential_derivatives(unittest.TestCase):

    def setUp(self):

        time = np.logspace(-3, 0, 21)
        n_loc = 5
        wires = Maps.Wires(('eta', n_loc), ('tau', n_loc), ('c', n_loc))
        taumap = Maps.ExpMap(nP=n_loc)*wires.tau
        etamap = Maps.ExpMap(nP=n_loc)*wires.eta
        cmap = Maps.ExpMap(nP=n_loc)*wires.c
        survey = SEMultiSurvey(time=time, locs=np.zeros((n_loc, 3)), n_pulse=0)
        mesh = Mesh.TensorMesh([np.ones(int(n_loc*3))])
        prob = SEMultiInvProblem(mesh, etaMap = etamap, tauMap = taumap, cMap=cmap)
        prob.pair(survey)

        eta0, tau0, c0 = 0.1, 10., 0.5
        m0 = np.log(np.r_[eta0*np.ones(n_loc), tau0*np.ones(n_loc), c0*np.ones(n_loc)])
        survey.makeSyntheticData(m0)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = prob
        self.survey = survey
        self.m0 = m0
        self.dmis = dmis
        self.mesh = mesh

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3,
            dx=self.m0*2
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
