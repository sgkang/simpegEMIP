import numpy as np
import scipy.sparse as sp
from simpegEMIP.StretchedExponential import SEMultiInvProblem, SEMultiSurvey
from SimPEG import Mesh, Utils, Maps
import unittest
import matplotlib.pyplot as plt

def CausalConv(array1, array2, time):
    """
        Evaluate convolution for two causal functions.
        Input

        * array1: array for \\\\(\\\\ f_1(t)\\\\)
        * array2: array for \\\\(\\\\ f_2(t)\\\\)
        * time: array for time

        .. math::

            Out(t) = \int_{0}^{t} f_1(a) f_2(t-a) da

    """

    if array1.shape == array2.shape == time.shape:
        out = np.convolve(array1, array2)
        # print time[1]-time[0]
        return out[0:np.size(time)]*(time[1]-time[0])
    else:
        print ("Give me same size of 1D arrays!!")


def get_peta_impulse(time, eta=0.1, tau=10., c=0.5):
    m = eta*c/(tau**c)
    peta = m*time**(c-1.)*np.exp(-(time/tau)**c)
    return peta

def get_peta_stepoff(time, eta=0.1, tau=10., c=0.5):
    peta = eta*np.exp(-(time/tau)**c)
    return peta

def getpeta_pulse_off(time, eta=0.1, tau=10., c=0.5, T=8.):
    peta = eta*(np.exp(-((time)/tau)**c) - np.exp(-((time+T/4.)/tau)**c))
    return peta

def getpeta_pulse_on(time, eta=0.1, tau=10., c=0.5, T=8.):
    peta = eta*(1-np.exp(-(time/tau)**c))
    return peta

def get_peta_off(time, n_pulse, eta=0.1, tau=10., c=0.5, T=8.):
    peta = np.zeros_like(time)
    for i_pulse in range (n_pulse):
        factor = (-1)**i_pulse * (n_pulse-i_pulse)
        peta += getpeta_pulse_off(time+T/2*i_pulse, eta=eta, tau=tau, c=c) * factor
    return peta/n_pulse

def get_peta_on(time, n_pulse, eta=0.1, tau=10., c=0.5, T=8.):
    peta = getpeta_pulse_on(time, eta=eta, tau=tau, c=c) * n_pulse
    for i_pulse in range (1, n_pulse):
        factor = (-1)**(i_pulse) * (n_pulse-i_pulse)
        peta += getpeta_pulse_off(time + T/4. + T/2*(i_pulse-1), eta=eta, tau=tau, c=c) * factor
    return peta/n_pulse

def generate_rectangular_waveform(dt=1e-3, n_time=int(4e3), n_pulse=2):
    time = np.arange(n_time*n_pulse) * dt + dt
    current_single_pulse = np.zeros(n_time)
    current_single_pulse[:int(n_time/2)] = 1.

    current_pulse = current_single_pulse.copy()
    if n_pulse > 1:
        for i in range(n_pulse-1):
            current_pulse = np.r_[current_pulse, (-1)**(i-1)*current_single_pulse]
    return time, current_pulse

def stack(time, data, n_pulse, T=8.):
    n_time = int(time.size/n_pulse)
    data_stack = np.zeros(n_time)
    i_start = 0
    i_end = n_time
    for i in range(n_pulse):
        data_stack += ((-1)**i)*data[i_start:i_end]
        i_start += n_time
        i_end += n_time
    return time[:n_time], data_stack/n_pulse

class stretched_exponential_forward(unittest.TestCase):

    def setUp(self):

        time = np.logspace(-3, 0, 21)
        n_loc = 10
        wires = Maps.Wires(('eta', n_loc), ('tau', n_loc), ('c', n_loc))
        taumap = Maps.ExpMap(nP=n_loc)*wires.tau
        etamap = Maps.ExpMap(nP=n_loc)*wires.eta
        cmap = Maps.ExpMap(nP=n_loc)*wires.c
        eta0, tau0, c0 = 0.1, 10., 0.5
        m0 = np.log(np.r_[eta0*np.ones(n_loc), tau0*np.ones(n_loc), c0*np.ones(n_loc)])
        survey = SEMultiSurvey(time=time, locs=np.zeros((n_loc, 3)))
        m1D = Mesh.TensorMesh([np.ones(int(n_loc*3))])
        prob = SEMultiInvProblem(m1D, etaMap = etamap, tauMap = taumap, cMap=cmap)
        prob.pair(survey)

        self.survey = survey
        self.m0 = m0
        self.time = time

    def test_SEMultiInvProblem_step_off(self, plotIt=False):

        self.survey.n_pulse = 0
        pred = self.survey.dpred(self.m0)[:self.survey.n_time]
        true = get_peta_stepoff(self.time)
        err = abs(pred-true) / abs(true)
        if np.all(err < 0.01):
            passed = True
        else:
            passed = False

        if plotIt:
            plt.plot(pred)
            plt.plot(true[:self.survey.n_time])
            plt.show()

        self.assertTrue(passed)

    def test_SEMultiInvProblem_pulse(self, plotIt=False):

        self.survey.n_pulse = 2
        pred = self.survey.dpred(self.m0)[:self.survey.n_time]
        true = get_peta_off(self.time, self.survey.n_pulse)
        err = abs(pred-true) / abs(true)
        if np.all(err < 0.01):
            passed = True
        else:
            passed = False

        if plotIt:
            plt.plot(pred)
            plt.plot(true[:self.survey.n_time])
            plt.show()

        self.assertTrue(passed)

    def test_SEMultiInvProblem_pulse(self, plotIt=False):

        self.survey.n_pulse = 4
        pred = self.survey.dpred(self.m0)[:self.survey.n_time]
        true = get_peta_off(self.time, self.survey.n_pulse)
        err = abs(pred-true) / abs(true)
        if np.all(err < 0.01):
            passed = True
        else:
            passed = False

        if plotIt:
            plt.plot(pred)
            plt.plot(true[:self.survey.n_time])
            plt.show()

        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
