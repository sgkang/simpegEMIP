import numpy as np
from SimPEG import Survey, Problem, Mesh, Utils
from SimPEG.Regularization import BaseRegularization
import scipy.sparse as sp


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
        return out[0:np.size(time)]*(time[1]-time[0])
    else:
        raise Exception("Input same size of 1D arrays!!")


class ExpFitSurvey(Survey.BaseSurvey):
    nD = None
    xyz = None
    uncert = None

    def __init__(self, time, xyz, **kwargs):
        self.time = time
        self.xyz = xyz
        self.ntime = time.size
        self.ntx = self.xyz.shape[0]
        self.nD = self.ntime*self.ntx

    def eval(self, f):
        return f

    def setUncertainty(self, dobs, perc=0.1, floor=0., floorIP=0.):
        self.uncert = np.zeros((self.ntime, self.ntx))
        self.dobs = dobs
        dobs = dobs.reshape((self.ntime, self.ntx), order='F')
        for itx in range(self.ntx):
            ipind = dobs[:, itx] < 0.

            # Set different uncertainty for stations having negative transients
            if (ipind).sum() > 3:
                ip = dobs[ipind, itx]
                self.uncert[:, itx] = perc*abs(dobs[:, itx]) + abs(ip).max()
            else:
                self.uncert[:, itx] = perc*abs(dobs[:, itx])+floor
        self.uncert = Utils.mkvc(self.uncert)

        return self.uncert


class ExpFitProblem(Problem.BaseProblem):

    surveyPair = ExpFitSurvey
    tau = None
    AofT_type = "Impulse"   # Type: "Impulse" or "Impulse_Stack"
    Frequency = 25.         # Frequency
    T = 1./25.              # Period
    currentderiv = None
    current = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)
        self.setTau()

    def getCur_from_Curderiv(self):
        self.current = CausalConv(
            self.currentderiv, np.ones_like(self.currentderiv), time_conv
            )

    def getAofT(self, timeconv, tau, dt=None):

        if self.AofT_type == "Step":
            AofT = np.exp(-timeconv/tau)
        elif self.AofT_type == "Impulse":
            AofT = 1./tau*np.exp(-timeconv/tau)
        elif self.AofT_type == "Step_Stack":
            AofT = np.exp(-timeconv/tau)
        elif self.AofT_type == "Impulse_Stack":
            AofT = np.exp(-timeconv/tau)/(1.+np.exp(-self.T/(2*taui)))
        elif self.AofT_type == "Impulse_Conv":
            AofT = CausalConv(
                1./tau*np.exp(-timeconv/tau), -self.currentderiv, timeconv
                )
        return AofT

    # TODO: make general (not it works for db/dt)
    def getG(self, timeconv, t0=0.):
        dt = timeconv[1]-timeconv[0]
        ntime = timeconv.size
        meshtime = Mesh.TensorMesh([dt*np.ones(ntime)], x0=[-dt/2.])
        P = meshtime.getInterpolationMat(self.survey.time+t0, 'CC')
        self.G = np.zeros((self.survey.ntime, self.ntau))
        for j in range(self.ntau):
            self.G[:, j] = P*self.getAofT(timeconv, self.tau[j])

    def setTau(self, minlogtau=-5, maxlogtau=-2, ntau=31):
        self.tau = np.logspace(minlogtau, maxlogtau, ntau)
        self.ntau = ntau

    def fields(self, m, f=None):
        m = m.reshape((self.ntau, self.survey.ntx), order='F')
        pred = self.G.dot(m)
        return Utils.mkvc(pred)

    def Jvec(self, m, v, f=None):
        v = v.reshape((self.ntau, self.survey.ntx), order='F')
        jvec = self.G.dot(v)
        return Utils.mkvc(jvec)

    def Jtvec(self, m, v, f=None):
        v = v.reshape((self.survey.ntime, self.survey.ntx), order='F')
        jtvec = self.G.T.dot(v)
        return Utils.mkvc(jtvec)


class LineRegularization(BaseRegularization):
    xyz_line = None
    alpha_s = 1e0
    alpha_x = 1e-1
    ntau = None

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseRegularization.__init__(self, mesh, mapping=mapping, **kwargs)
        if self.ntau is None:
            raise Exception("Input parameter ntau must be initiated!")
    @property
    def W(self):
        """Regularization matrix W"""
        if getattr(self, '_W', None) is None:
            ntx = self.xyz_line.shape[0]
            meshline = Mesh.TensorMesh([ntx])
            Gx = meshline.cellGradx
            Wx = np.sqrt(self.alpha_x)*sp.kron(Gx, Utils.speye(self.ntau))
            Ws = np.sqrt(self.alpha_s)*Utils.speye(self.ntau*ntx)
            wlist = (Wx, Ws)
            self._W = sp.vstack(wlist)
        return self._W
