from SimPEG import Problem, Utils, Maps, Props, Mesh, Tests
from SimPEG import Problem, Survey, Utils, Maps
import numpy as np
import scipy.sparse as sp

class SEInvProblem(Problem.BaseProblem):

    sigmaInf, sigmaInfMap, sigmaInfDeriv = Props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    eta, etaMap, etaDeriv = Props.Invertible(
        "Cole-Cole chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Cole-Cole time constant (s)"
    )

    c, cMap, cDeriv = Props.Invertible(
        "Cole-Cole frequency dependency"
    )

    P = None
    J = None
    time = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fields(self, m=None, f=None):
        if m is not None:
            self.model = m
        self.J = self.ColeSEJfun(self.time)
        return self.ColeSEfun(self.time)

    def Jvec(self, m, v, f=None):
        jvec = self.J.dot(v)
        return jvec

    def Jtvec(self, m, v, f=None):
        jtvec = self.J.T.dot(v)
        return jtvec

    def ColeSEfun(self, time):
        return elf.eta*np.exp(-(time/self.tau)**self.c)

    def ColeSEJfun(self, time):
        kerneleta = lambda t, eta: np.exp(-(time/tau)**c)
        kerneltau = lambda t, tau: (c*eta/tau)*((t/tau)**c)*np.exp(-(t/tau)**c)
        kernelc = lambda t, c: -eta*((t/tau)**c)*np.exp(-(t/tau)**c)*np.log(t/tau)

        tempeta = kerneleta(time, self.eta).reshape([-1,1])
        temptau = kerneltau(time, self.tau).reshape([-1,1])
        tempc = kernelc(time, self.c).reshape([-1,1])
        J = (self.etaDeriv) + temptau.dot(self.tauDeriv) + tempc.dot(self.cDeriv)
        return J


class SEInvImpulseProblem(SEInvProblem):

    def __init__(self, mesh, **kwargs):
        SEInvProblem.__init__(self, mesh, **kwargs)

    def fields(self, m, f=None):
        if m is not None:
            self.model = m
        self.J = self.ColeSEJImpulsefun(self.time)
        return self.ColeSEImpulsefun(self.time)

    def ColeSEImpulsefun(self, time):
        return self.eta*self.c/time*((time/self.tau)**self.c)*np.exp(-(time/self.tau)**self.c)

    def ColeSEJImpulsefun(self, time):
        kerneleta = lambda t, eta, tau, c: c/t*((t/tau)**c)*np.exp(-(t/tau)**c)
        kerneltau = lambda t, eta, tau, c: -c**2 * eta * ((t/tau)**c) / (t*tau) * np.exp(-(t/tau)**c) * (-(t/tau)**c+1.)
        kernelc = lambda t, eta, tau, c: -eta/t*((t/tau)**c)*np.exp(-(t/tau)**c)*( c*((t/tau)**c)*np.log(t/tau)-c*np.log(t/tau)-1.)

        tempeta = kerneleta(time, self.eta, self.tau, self.c).reshape([-1,1])
        temptau = kerneltau(time, self.eta, self.tau, self.c).reshape([-1,1])
        tempc = kernelc(time, self.eta, self.tau, self.c).reshape([-1,1])
        J = tempeta * self.etaDeriv + temptau * self.tauDeriv + tempc * self.cDeriv
        return J

class SESurvey(Survey.BaseSurvey):


    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)
    @property
    def nD(self):
        self._nD = self.dobs.size
        return self._nD

    @Utils.requires('prob')

    def dpred(self, m, f=None):
        return self.prob.fields(m)

    def residual(self, m, f=None):
        if self.dobs.size ==1:
            return Utils.mkvc(np.r_[self.dpred(m, f=f) - self.dobs])
        else:
            return Utils.mkvc(self.dpred(m, f=f) - self.dobs)

class SE_MultiSurvey(Survey.BaseSurvey):

    nD = None
    locs = None
    uncert = None
    nLoc = None

    def __init__(self, time, locs, **kwargs):
        self.time = time
        self.locs = locs
        self.ntime = time.size
        self.nLoc = self.locs.shape[0]
        self.nD = self.ntime*self.nLoc

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


class SE_MultiInvProblem(Problem.BaseProblem):

    eta, etaMap, etaDeriv = Props.Invertible(
        "Cole-Cole chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Cole-Cole time constant (s)"
    )

    c, cMap, cDeriv = Props.Invertible(
        "Cole-Cole frequency dependency"
    )

    P = None
    J = None
    surveyPair = SE_MultiSurvey

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fields(self, m, f=None):
        if m is not None:
            self.model = m
        return self.ColeSEfun(self.survey.time)

    @property
    def J(self):
        if getattr(self, '_J', None) is None:
            self._J = self.ColeSEJfun(self.survey.time)
        return self._J

    def Jvec(self, m, v, f=None):
        jvec = self.J * v
        return jvec

    def Jtvec(self, m, v, f=None):
        jtvec = self.J.T * v
        return jtvec

    def ColeSEfun(self, time):
        # Vectorize or Cythonize
        d = []
        for iloc in range(self.survey.nLoc):
            d.append(
                self.eta[iloc]*np.exp(-(time/self.tau[iloc])**self.c[iloc])
                )
        return np.hstack(d)

    def ColeSEJfun(self, time):
        # Vectorize or Cythonize
        kerneleta = lambda t, eta, tau, c: np.exp(-(time/tau)**c)
        kerneltau = lambda t, eta, tau, c: (c*eta/tau)*((t/tau)**c)*np.exp(-(t/tau)**c)
        kernelc = lambda t, eta, tau, c: -eta*((t/tau)**c)*np.exp(-(t/tau)**c)*np.log(t/tau)

        Jeta = []
        Jtau = []
        Jc = []

        for iloc in range(self.survey.nLoc):
            Jeta.append(kerneleta(time, self.eta[iloc], self.tau[iloc], self.c[iloc]).reshape([-1, 1]))
            Jtau.append(kerneltau(time, self.eta[iloc], self.tau[iloc], self.c[iloc]).reshape([-1, 1]))
            Jc.append(kernelc(time, self.eta[iloc], self.tau[iloc], self.c[iloc]).reshape([-1, 1]))

        Jeta = sp.block_diag(Jeta)
        Jtau = sp.block_diag(Jtau)
        Jc = sp.block_diag(Jc)

        J = (
            Jeta*self.etaDeriv + Jtau*self.tauDeriv +
            Jc*self.cDeriv
            )
        return J

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = ['_J']
        return toDelete


