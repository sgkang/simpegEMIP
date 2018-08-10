from SimPEG import Problem, Utils, Maps, Props, Mesh, Tests
from SimPEG import Problem, Survey, Utils, Maps
import numpy as np
import scipy.sparse as sp


# TODO: deprecate this later
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
        self.J = self.get_peta_deriv(self.time)
        return self.get_peta(self.time)

    def Jvec(self, m, v, f=None):
        jvec = self.J.dot(v)
        return jvec

    def Jtvec(self, m, v, f=None):
        jtvec = self.J.T.dot(v)
        return jtvec

    def get_peta(self, time):
        return self.eta*np.exp(-(time/self.tau)**self.c)

    def get_peta_deriv(self, time):
        kerneleta = lambda t, eta, tau, c: np.exp(-(time/tau)**c)
        kerneltau = lambda t, eta, tau, c: (c*eta/tau)*((t/tau)**c)*np.exp(-(t/tau)**c)
        kernelc = lambda t, eta, tau, c: -eta*((t/tau)**c)*np.exp(-(t/tau)**c)*np.log(t/tau)

        tempeta = kerneleta(time, self.eta, self.tau, self.c).reshape([-1,1])
        temptau = kerneltau(time, self.eta, self.tau, self.c).reshape([-1,1])
        tempc = kernelc(time, self.eta, self.tau, self.c).reshape([-1,1])
        J = tempeta * self.etaDeriv + temptau * self.tauDeriv + tempc * self.cDeriv
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


class SEMultiSurvey(Survey.BaseSurvey):

    nD = None
    locs = None
    uncert = None
    n_location = None
    n_pulse = 2     # Number of pulses
    T = 8.          # Period of the input current waveform (s)

    def __init__(self, time, locs, **kwargs):
        self.time = time
        self.locs = np.atleast_2d(locs)
        Utils.setKwargs(self, **kwargs)

    @property
    def n_time(self):
        return self.time.size

    @property
    def nD(self):
        return self.n_time*self.n_location

    @property
    def n_location(self):
        return self.locs.shape[0]

    def dpred(self, m, f=None):
        return self.prob.fields(m)


class SEMultiInvProblem(Problem.BaseProblem):

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
    surveyPair = SEMultiSurvey
    time_over_tau_vec = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fields(self, m, f=None):
        if m is not None:
            self.model = m
            self._eta_vec = self.eta.repeat(self.survey.time.size)
            self._tau_vec = self.tau.repeat(self.survey.time.size)
            self._c_vec = self.c.repeat(self.survey.time.size)

        return self.get_peta()


    @property
    def eta_vec(self):
        if getattr(self, '_eta_vec', None) is None:
            self._eta_vec = self.eta.repeat(self.survey.time.size)
        return self._eta_vec

    @property
    def tau_vec(self):
        if getattr(self, '_tau_vec', None) is None:
            self._tau_vec = self.tau.repeat(self.survey.time.size)
        return self._tau_vec

    @property
    def c_vec(self):
        if getattr(self, '_c_vec', None) is None:
            self._c_vec = self.c.repeat(self.survey.time.size)
        return self._c_vec

    @property
    def time_vec(self):
        if getattr(self, '_time_vec', None) is None:
            self._time_vec = np.repeat(
            self.survey.time.reshape([-1, 1]), self.survey.n_location, axis=1
        ).flatten(order='F')

        return self._time_vec

    @property
    def J(self):
        if getattr(self, '_J', None) is None:
            self._J = self.get_peta_deriv()
        return self._J

    def Jvec(self, m, v, f=None):
        jvec = self.J * v
        return jvec

    def Jtvec(self, m, v, f=None):
        jtvec = self.J.T * v
        return jtvec

    def get_peta_step_off(self, exponent):
        """
            Compute pseudo-chargeability from a step-off waveform
        """
        eta_vec = self.eta_vec
        return  eta_vec*np.exp(-exponent)

    def get_peta_pulse_off(self, time_vec):
        """
            Compute pseudo-chargeability from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(time_vec)
        exponent_1 = self.get_exponent(time_vec+T/4.)
        peta = self.get_peta_step_off(exponent_0) - self.get_peta_step_off(exponent_1)
        return peta

    def get_exponent(self, time_vec):
        tau_vec = self.tau_vec
        c_vec = self.c_vec
        # TODO: make this as a property later
        # self.time_over_tau_vec is updated only when def get_exponent is called
        return (time_vec/tau_vec)**c_vec

    def get_time_over_tau(self, time_vec):
        tau_vec = self.tau_vec
        c_vec = self.c_vec
        return time_vec/tau_vec

    def get_peta(self):
        n_pulse = self.survey.n_pulse
        T = self.survey.T
        time_vec = self.time_vec.copy()
        peta = np.zeros_like(time_vec)
        if n_pulse == 0:
            exponent = self.get_exponent(time_vec)
            return self.get_peta_step_off(exponent)
        else:
            for i_pulse in range (n_pulse):
                factor = (-1)**i_pulse * (n_pulse-i_pulse)
                peta += self.get_peta_pulse_off(time_vec+T/2*i_pulse) * factor
            return peta/n_pulse

    def get_eta_deriv_step_off(self, exponent):
        return np.exp(-exponent)

    def get_tau_deriv_step_off(self, exponent):
        eta_vec = self.eta_vec
        tau_vec = self.tau_vec
        c_vec = self.c_vec
        return (c_vec*eta_vec/tau_vec)*(exponent)*np.exp(-exponent)

    def get_c_deriv_step_off(self, exponent, time_over_tau):
        eta_vec = self.eta_vec
        return -eta_vec*(exponent)*np.exp(-exponent)*np.log(time_over_tau)

    def get_eta_deriv_pulse_off(self, time_vec):
        """
            Compute derivative of pseudo-chargeability w.r.t eta from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(time_vec)
        exponent_1 = self.get_exponent(time_vec+T/4.)
        eta_deriv = self.get_eta_deriv_step_off(exponent_0) - self.get_eta_deriv_step_off(exponent_1)
        return eta_deriv

    def get_tau_deriv_pulse_off(self, time_vec):
        """
            Compute derivative of pseudo-chargeability w.r.t tau from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(time_vec)
        exponent_1 = self.get_exponent(time_vec+T/4.)
        tau_deriv = self.get_tau_deriv_step_off(exponent_0) - self.get_tau_deriv_step_off(exponent_1)
        return tau_deriv

    def get_c_deriv_pulse_off(self, time_vec):
        """
            Compute derivative of pseudo-chargeability w.r.t c from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(time_vec)
        exponent_1 = self.get_exponent(time_vec+T/4.)
        time_over_tau_0 = self.get_time_over_tau(time_vec)
        time_over_tau_1 = self.get_time_over_tau(time_vec+T/4.)

        c_deriv = (
            self.get_c_deriv_step_off(exponent_0, time_over_tau_0) -
            self.get_c_deriv_step_off(exponent_1, time_over_tau_1)
        )
        return c_deriv


    def get_eta_deriv(self):
        n_pulse = self.survey.n_pulse
        T = self.survey.T
        time_vec = self.time_vec.copy()
        eta_deriv = np.zeros_like(time_vec, order='F')
        if n_pulse == 0:
            exponent = self.get_exponent(time_vec)
            return self.get_eta_deriv_step_off(exponent)
        else:
            for i_pulse in range (n_pulse):
                factor = (-1)**i_pulse * (n_pulse-i_pulse)
                eta_deriv += self.get_eta_deriv_pulse_off(time_vec+T/2*i_pulse) * factor
            return eta_deriv/n_pulse

    def get_tau_deriv(self):
        n_pulse = self.survey.n_pulse
        T = self.survey.T
        time_vec = self.time_vec.copy()
        tau_deriv = np.zeros_like(time_vec, order='F')
        if n_pulse == 0:
            exponent = self.get_exponent(time_vec)
            return self.get_tau_deriv_step_off(exponent)
        else:
            for i_pulse in range (n_pulse):
                factor = (-1)**i_pulse * (n_pulse-i_pulse)
                tau_deriv += self.get_tau_deriv_pulse_off(time_vec+T/2*i_pulse) * factor
            return tau_deriv/n_pulse


    def get_c_deriv(self):
        n_pulse = self.survey.n_pulse
        T = self.survey.T
        time_vec = self.time_vec.copy()
        c_deriv = np.zeros_like(time_vec, order='F')
        if n_pulse == 0:
            exponent = self.get_exponent(time_vec)
            time_over_tau = self.get_time_over_tau(time_vec)
            return self.get_c_deriv_step_off(exponent, time_over_tau)
        else:
            for i_pulse in range (n_pulse):
                factor = (-1)**i_pulse * (n_pulse-i_pulse)
                c_deriv += self.get_c_deriv_pulse_off(time_vec+T/2*i_pulse) * factor
            return c_deriv/n_pulse

    @property
    def column_index(self):
        if getattr(self, '_column_index', None) is None:
            self._column_index = np.arange(self.survey.nD)
        return self._column_index

    @property
    def row_index(self):
        if getattr(self, '_row_index', None) is None:
            self._row_index = np.repeat(
                np.arange(self.survey.n_location), self.survey.n_time
            )
        return self._row_index

    def get_sparse_matrix_from_vector(self, vector):
        shape = (self.survey.nD, self.survey.n_location)
        sparse_matrix = sp.coo_matrix(
            (vector, (self.column_index, self.row_index)), shape=shape
        )
        return sparse_matrix.tocsr()

    def get_peta_deriv(self):

        Jeta = self.get_sparse_matrix_from_vector(self.get_eta_deriv())
        Jtau = self.get_sparse_matrix_from_vector(self.get_tau_deriv())
        Jc = self.get_sparse_matrix_from_vector(self.get_c_deriv())

        J = (
            Jeta*self.etaDeriv + Jtau*self.tauDeriv +
            Jc*self.cDeriv
        )

        return J

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = [
            '_J',
            '_eta_vec', '_tau_vec', '_c_vec'
        ]
        return toDelete
