import unittest

import numpy as np

from scipy.constants import mu_0
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import DataMisfit
from SimPEG import Regularization
from SimPEG import Optimization
from SimPEG import Inversion
from SimPEG import InvProblem
from SimPEG import Tests
from SimPEG import EM
from SimPEG import Directives

from pymatsolver import PardisoSolver

from simpegEM1D import DigFilter
from simpegEMIP.StretchedExponential import SEInvImpulseProblem, SESurvey
from simpegEMIP.TDEM import Problem3D_e
from simpegEMIP.TDEM import Survey as SurveyEMIP
from simpegEMIP.TDEM import Rx as RxEMIP

import matplotlib.pyplot as plt


class problem_e_forward(unittest.TestCase):

    def setUp(self):

        # Obtained SE parameters from fitting
        mopt = self.fit_colecole_with_se()
        eta, tau, c = mopt[0], np.exp(mopt[1]), mopt[2]

        # Step2: Set up EMIP problem and Survey
        csx, csz, ncx, ncz, npadx, npadz = 6.5, 5., 10, 20, 20, 20
        hx = [(csx, ncx), (csx, npadx, 1.3)]
        hz = [(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)]
        mesh = Mesh.CylMesh([hx, 1, hz], '00C')
        sig_half = 0.1
        sigmaInf = np.ones(mesh.nC) * sig_half
        airind = mesh.gridCC[:, 2] > 0.
        sigmaInf[airind] = 1e-8
        etavec = np.ones(mesh.nC) * eta
        etavec[airind] = 0.
        tauvec = np.ones(mesh.nC) * tau
        cvec = np.ones(mesh.nC) * c
        wiresEM = Maps.Wires(
            ('sigmaInf', mesh.nC),
            ('eta', mesh.nC),
            ('tau', mesh.nC),
            ('c', mesh.nC)
        )
        tauvecmap = Maps.ExpMap(nP=mesh.nC) * wiresEM.tau
        src_z = 30.
        rxloc = np.array([0., 0., src_z])
        srcloc = np.array([0., 0., src_z])
        rx = RxEMIP.Point_dbdt(
            rxloc, np.logspace(np.log10(1e-5), np.log10(0.009), 51), 'z'
        )
        src = EM.TDEM.Src.CircularLoop(
            [rx],
            waveform=EM.TDEM.Src.StepOffWaveform(),
            loc=srcloc,
            radius=13.
        )
        survey = SurveyEMIP([src])
        prb_em = Problem3D_e(
            mesh,
            sigmaInfMap=wiresEM.sigmaInf,
            etaMap=wiresEM.eta,
            tauMap=tauvecmap,
            cMap=wiresEM.c
        )
        prb_em.verbose = False
        prb_em.timeSteps = [
            (1e-06, 5), (2.5e-06, 5), (5e-06, 5),
            (1e-05, 10), (2e-05, 10), (4e-05, 10),
            (8e-05, 10), (1.6e-04, 10), (3.2e-04, 20)
        ]
        prb_em.Solver = PardisoSolver
        prb_em.pair(survey)
        m = np.r_[sigmaInf, etavec, np.log(tauvec), cvec]
        self.survey = survey
        self.src_z = src_z
        self.m = m
        self.sig_half = sig_half
        self.eta = eta
        self.tau = tau
        self.c = c
        self.time = rx.times

    def fit_colecole_with_se(self, eta_cc=0.8, tau_cc=0.003, c_cc=0.6):

        def ColeColeSeigel(f, sigmaInf, eta, tau, c):
            w = 2*np.pi*f
            return sigmaInf*(1 - eta/(1 + (1j*w*tau)**c))

        # Step1: Fit Cole-Cole with Stretched Exponential function
        time = np.logspace(-6, np.log10(0.01), 41)
        wt, tbase, omega_int = DigFilter.setFrequency(time)
        frequency = omega_int / (2*np.pi)
        # Cole-Cole parameters
        siginf = 1.
        self.eta_cc = eta_cc
        self.tau_cc = tau_cc
        self.c_cc = c_cc

        sigma = ColeColeSeigel(frequency, siginf, eta_cc, tau_cc, c_cc)
        sigTCole = DigFilter.transFiltImpulse(
            sigma, wt, tbase, omega_int, time, tol=1e-12
        )
        wires = Maps.Wires(('eta', 1), ('tau', 1), ('c', 1))
        taumap = Maps.ExpMap(nP=1)*wires.tau
        survey = SESurvey()
        dtrue = -sigTCole
        survey.dobs = dtrue
        m1D = Mesh.TensorMesh([np.ones(3)])
        prob = SEInvImpulseProblem(
            m1D, etaMap=wires.eta, tauMap=taumap, cMap=wires.c
        )
        update_sens = Directives.UpdateSensitivityWeights()
        prob.time = time
        prob.pair(survey)
        m0 = np.r_[eta_cc, np.log(tau_cc), c_cc]
        perc = 0.05
        dmisfitpeta = DataMisfit.l2_DataMisfit(survey)
        dmisfitpeta.W = 1/(abs(survey.dobs)*perc)
        reg = Regularization.Simple(m1D)
        opt = Optimization.ProjectedGNCG(maxIter=10)
        invProb = InvProblem.BaseInvProblem(dmisfitpeta, reg, opt)
        # Create an inversion object
        target = Directives.TargetMisfit()
        invProb.beta = 0.
        inv = Inversion.BaseInversion(invProb, directiveList=[target])
        reg.mref = 0.*m0
        prob.counter = opt.counter = Utils.Counter()
        opt.LSshorten = 0.5
        opt.remember('xc')
        opt.tolX = 1e-20
        opt.tolF = 1e-20
        opt.tolG = 1e-20
        opt.eps = 1e-20
        mopt = inv.run(m0)
        return mopt

    def get_analytic(self, sigma, eta, tau, c, src_z, time):
        from simpegEM1D import (
            EM1D, EM1DSurveyTD
        )
        mesh1D = Mesh.TensorMesh([1])
        TDsurvey = EM1DSurveyTD(
            rx_location=np.array([0., 0., src_z]),
            src_location=np.array([0., 0., src_z]),
            topo=np.r_[0., 0., 0.],
            depth=np.r_[0.],
            rx_type='dBzdt',
            wave_type='stepoff',
            src_type='CircularLoop',
            a=13.,
            I=1.,
            time=time,
            half_switch=True
        )
        # Convert to Pelton's model
        tau_p = tau / (1-eta)**(1./c)
        expmap = Maps.ExpMap(mesh1D)
        prob = EM1D(
            mesh1D, sigma=np.r_[sigma], eta=np.r_[eta],
            tau=np.r_[tau_p], c=np.r_[c]
        )
        if prob.ispaired:
            prob.unpair()
        if TDsurvey.ispaired:
            TDsurvey.unpair()
        prob.pair(TDsurvey)
        prob.chi = np.zeros(TDsurvey.n_layer)
        dhzdt = TDsurvey.dpred([])
        return dhzdt

    def test_Problem_e_step_off(self, plotIt=False):
        data = self.survey.dpred(self.m)
        data_analytic = self.get_analytic(
            self.sig_half, self.eta_cc, self.tau_cc, self.c_cc,
            self.src_z, self.time
        )
        uncert = abs(data_analytic)*0.2 + 1e-10
        chi_factor = np.linalg.norm(
            (data - data_analytic)/(uncert)
        )**2 / self.survey.nD

        if chi_factor < 1:
            passed = True
            print (
                ("Probelm_e test is passed: chi factor %.1e") % (chi_factor)
            )
        else:
            passed = False
            print (
                ("Probelm_e test is failed: chi factor %.1e") % (chi_factor)
            )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
