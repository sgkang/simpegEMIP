from SimPEG import Problem, Utils, Maps, Props, Mesh, Tests
from SimPEG.Survey import BaseSurvey
import numpy as np

class DebyeDecProblem(Problem.BaseProblem):

    sigmaInf, sigmaInfMap, sigmaInfDeriv = Props.Invertible(
        "Scalar, Conductivity at infinite frequency (S/m)"
    )
    eta, etaMap, etaDeriv = Props.Invertible(
        "Array, Chargeability (V/V)"
    )

    tau = Props.PhysicalProperty(
        "Array, Time constant (s)",
        default=0.1
    )

    nfreq = None
    ntau = None
    G = None
    frequency = None
    tau = None
    f = None
    InvertOnlyEta = False

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)
        self.omega = 2*np.pi*self.frequency
        self.nfreq = self.frequency.size
        self.tau = self.mesh.gridN
        self.ntau = self.mesh.nN
        if self.sigmaInfMap == None:
            self.InvertOnlyEta = True
            print ("Assume sigmaInf is known")

    @property
    def X(self):
        """
        Denominator in Cole-Cole model
        """
        if getattr(self, '_X', None) is None:
            X = np.zeros((self.nfreq, self.ntau), dtype=complex)
            for itau in range(self.ntau):
                X[:,itau] = 1./(1.+(1.-self.eta[itau])*(1j*self.omega*self.tau[itau]))
            self._X = X
        return self._X


    def fields(self, m=None):
        if m is not None:
            self._X = None
            self.model = m
        f = self.sigmaInf - self.sigmaInf*(np.dot(self.X, self.eta))
        self.f = f
        return f

    def get_petaImpulse(self, time, m):
        etas = m
        taus = self.tau
        b = -1. / ((1.-etas)*taus)
        a = -etas*b
        t_temp = np.atleast_2d(time).T
        temp = np.exp(np.dot(t_temp, np.atleast_2d(b)))
        out = np.dot(temp, a)

        return out

    def get_petaStepon(self, time, m):
        etas = m
        taus = self.tau
        b = -1. / ((1.-etas)*taus)
        t_temp = np.atleast_2d(time).T
        temp = np.exp(np.dot(t_temp, np.atleast_2d(b)))
        out = np.dot(temp, etas)
        return out

    def get_Expb(self, time, m):
        etas = m
        taus = self.tau
        b = -1. / ((1.-etas)*taus)
        e = etas / b
        t_temp = np.atleast_2d(time).T
        temp = 1.-np.exp(np.dot(t_temp, np.atleast_2d(b)))
        out = np.dot(temp, e)
        return out

    def dsig_dm(self, v, adjoint=False):

        if not adjoint:

            deta_dm_v = self.etaDeriv*v
            if self.InvertOnlyEta:
                return self.dsig_deta(deta_dm_v)
            else:
                dsigmaInf_dm_v = self.sigmaInfDeriv*v
                return self.dsig_dsigmaInf(dsigmaInf_dm_v) + self.dsig_deta(deta_dm_v)

        elif adjoint:

            dsig_detaT_v = self.dsig_deta(v, adjoint=adjoint)
            dsig_dm_v = self.etaDeriv.T * dsig_detaT_v

            if not self.InvertOnlyEta:

                dsig_dsigmaInfT_v = self.dsig_dsigmaInf(v, adjoint=adjoint)
                dsig_dm_v += self.sigmaInfDeriv.T * dsig_dsigmaInfT_v

            return dsig_dm_v

    def dsig_deta(self, v, adjoint=False):
        """
            NxM matrix vec
            I*eta*omega*tau/(I*omega*tau*(-eta + 1) + 1)**2 + 1/(I*omega*tau*(-eta + 1) + 1)
        """
        if not adjoint:
            dsig_deta_v = -self.sigmaInf*np.dot(self.X, v)
            temp_v = Utils.sdiag(self.tau*self.eta) * v
            dsig_deta_v -= np.dot(Utils.sdiag(self.sigmaInf*1j*self.omega)*self.X**2, temp_v)
            return dsig_deta_v

        elif adjoint:
            dsig_detaT_v = -self.sigmaInf*np.dot(self.X.conj().T, v)
            tempa_v = Utils.sdiag(self.sigmaInf*1j*self.omega).conj() * v
            temp_v = np.dot((self.X**2).conj().T, tempa_v)
            dsig_detaT_v -= Utils.sdiag(self.tau*self.eta) * temp_v
            return dsig_detaT_v.real

    def dsig_dsigmaInf(self, v, adjoint=False):
        """
            Nx1 matrix vec
        """
        if not adjoint:
            dsig_dsigmaInf_v = (1.-np.dot(self.X, self.eta)) * v
            return dsig_dsigmaInf_v

        elif adjoint:
            e = np.ones_like(v)
            dsig_dsigmaInfT_v = np.dot(e, v) - np.dot(self.eta, np.dot(self.X.conj().T, v))
            return np.r_[dsig_dsigmaInfT_v.real]

    def Jvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m=m)
        dsig_dm_v = self.dsig_dm(v)
        Jv = self.survey.evalDeriv(dsig_dm_v, f, adjoint=False)
        return Jv

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m=m)
        dP_dsigT_v = self.survey.evalDeriv(v, f, adjoint=True)
        Jtv = self.dsig_dm(dP_dsigT_v, adjoint=True)
        return Jtv


def getTau(taumin, taumax, ntau):
    tau = np.logspace(np.log10(taumin), np.log10(taumax), ntau)
    mesh = Mesh.TensorMesh([np.diff(tau)], x0=[tau[0]])
    return mesh

class DebyeDecSurvey(BaseSurvey):

    def eval(self, f):
        """
            f: complex array

            Set data as log(|f|) and its phase

            .. math::

                f=fexp(\imath \psi)
                log(f) = log(|f|) + 1\imath \psi

        """
        return np.r_[np.log(f).real, np.log(f).imag]

    def evalDeriv_logf_v(self, v, f, adjoint=False):
        if not adjoint:
            return (1./f*v).real
        elif adjoint:
            return (1./f).conj()*v

    def evalDeriv_psi_v(self, v, f, adjoint=False):
        if not adjoint:
            return (np.zeros(f.size) * v).real
        elif adjoint:
            return np.zeros(f.size) * v

    def evalDeriv(self, v, f, adjoint=False):
        if not adjoint:
            return np.r_[self.evalDeriv_logf_v(v, f, adjoint=adjoint), self.evalDeriv_psi_v(v, f, adjoint=adjoint)]
        elif adjoint:
            v = v.reshape((self.prob.nfreq, 2), order="F")
            return self.evalDeriv_logf_v(v[:,0], f, adjoint=adjoint) + self.evalDeriv_psi_v(v[:,1], f, adjoint=adjoint)

    @property
    def nD(self):
        return self.prob.nfreq*2
