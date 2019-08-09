import numpy as np
import scipy.sparse as sp
from SimPEG import Survey, Problem, Mesh, Utils
from SimPEG.regularization import BaseRegularization
from simpegEM1D.Waveforms import piecewise_pulse_fast
import properties


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


class ExpFitSurvey(Survey.BaseSurvey, properties.HasProperties):

    time = properties.Array(
        "Time channels (s) at current off-time",
        dtype=float
    )

    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general", "general_conv"]
    )

    moment_type = properties.StringChoice(
        "Source moment type",
        default="single",
        choices=["single", "dual"]
    )

    n_pulse = properties.Integer(
        "The number of pulses", default=1
    )

    base_frequency = properties.Float(
        "Base frequency (Hz)"
    )

    time_input_currents = properties.Array(
        "Time for input currents", dtype=float
    )

    input_currents = properties.Array(
        "Input currents", dtype=float
    )

    t0 = properties.Float(
        "End of the ramp"
    )

    use_lowpass_filter = properties.Bool(
        "Switch for low pass filter", default=False
    )

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)", default=1e5
    )

    # Predicted data
    _pred = None

    # ------------- For dual moment ------------- #

    time_dual_moment = properties.Array(
        "Off-time channels (s) for the dual moment", dtype=float
    )

    time_input_currents_dual_moment = properties.Array(
        "Time for input currents (dual moment)", dtype=float
    )

    input_currents_dual_moment = properties.Array(
        "Input currents (dual moment)", dtype=float
    )

    t0_dual_moment = properties.Array(
        "End of the ramp", dtype=float
    )

    base_frequency_dual_moment = properties.Float(
        "Base frequency for the dual moment (Hz)"
    )

    xyz = properties.Array(
        "sounding locations", dtype=float,
        shape=('*', '*')
    )

    uncert = None

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def n_time(self):
        n_time = self.time.size
        if self.moment_type == 'dual':
            n_time += self.time_dual_moment.size
        return n_time

    @property
    def n_sounding(self):
        return self.xyz.shape[0]

    @property
    def nD(self):
        """
            # of data
        """
        return self.n_time * self.n_sounding

    def eval(self, f):
        return f

    def set_uncertainty(self, dobs, perc=0.1, floor=0., floorIP=0.):
        # TODO: need to consider dual moment
        self.uncert = np.zeros((self.n_time, self.n_sounding))
        self.dobs = dobs
        dobs = dobs.reshape((self.n_time, self.n_sounding), order='F')
        for itx in range(self.n_sounding):
            ipind = dobs[:, itx] < 0.
            # Set different uncertainty for stations having negative transients
            if (ipind).sum() > 3:
                ip = dobs[ipind, itx]
                self.uncert[:, itx] = (
                    perc*abs(dobs[:, itx]) + abs(ip).max() * 10
                )
                self.uncert[ipind, itx] =  np.Inf
            else:
                self.uncert[:, itx] = perc*abs(dobs[:, itx])+floor
        self.uncert = Utils.mkvc(self.uncert)

        return self.uncert


class ExpFitProblem(Problem.BaseProblem):

    surveyPair = ExpFitSurvey
    tau = None
    currentderiv = None
    Jmatrix = None
    timeconv = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)
        self.setTau()

    def getCur_from_Curderiv(self):
        self.current = CausalConv(
            self.currentderiv, np.ones_like(self.currentderiv), self.timeconv
        )

    def getAofT(self, tau, dt=None):
        # AofT = np.exp(-self.survey.time/tau)/(1.+np.exp(-self.T/(2*tau)))

        if self.survey.wave_type == "stepoff":
            AofT = np.exp(-self.survey.time/tau)
        elif self.survey.wave_type == "general_conv":
            # Dual moment is not considered here yet
            AofT = CausalConv(
                np.exp(-self.timeconv/tau),
                -self.currentderiv, self.timeconv
            )
        elif self.survey.wave_type == "general":
            def step_func(time):
                return np.exp(-time/tau)
            AofT = piecewise_pulse_fast(
                step_func,
                self.survey.time,
                self.survey.time_input_currents,
                self.survey.input_currents,
                self.survey.base_frequency,
                n_pulse=self.survey.n_pulse
            )
            if self.survey.moment_type == 'dual':
                AofT_dual = piecewise_pulse_fast(
                    step_func,
                    self.survey.time_dual_moment,
                    self.survey.time_input_currents_dual_moment,
                    self.survey.input_currents_dual_moment,
                    self.survey.base_frequency_dual_moment,
                    n_pulse=self.survey.n_pulse
                 )
                AofT = np.r_[AofT, AofT_dual]
        return AofT

    # TODO: make general (not it works for db/dt)
    def getJ(self, m):
        if self.Jmatrix is not None:
            return self.Jmatrix
        else:
            self.Jmatrix = np.zeros((self.survey.n_time, self.ntau))

            if "conv" not in self.survey.wave_type.split("_"):
                for j in range(self.ntau):
                    self.Jmatrix[:, j] = self.getAofT(self.tau[j])
            else:
                # Dual moment is not considered here yet
                dt = self.timeconv[1]-self.timeconv[0]
                ntime = self.timeconv.size
                meshtime = Mesh.TensorMesh([dt*np.ones(ntime)], x0=[-dt/2.])
                P = meshtime.getInterpolationMat(
                    self.survey.time+self.survey.t0, 'CC'
                )
                for j in range(self.ntau):
                    self.Jmatrix[:, j] = P*self.getAofT(self.tau[j])
        return self.Jmatrix

    def setTau(self, minlogtau=-5, maxlogtau=-2, ntau=31):
        self.tau = np.logspace(minlogtau, maxlogtau, ntau)
        self.ntau = ntau

    def fields(self, m, f=None):
        m = m.reshape((self.ntau, self.survey.n_sounding), order='F')
        # can be parallelized using dask
        pred = self.Jmatrix.dot(m)
        return Utils.mkvc(pred)

    def Jvec(self, m, v, f=None):
        v = v.reshape((self.ntau, self.survey.n_sounding), order='F')
        # can be parallelized using dask
        jvec = self.Jmatrix.dot(v)
        return Utils.mkvc(jvec)

    def Jtvec(self, m, v, f=None):
        v = v.reshape((self.survey.n_time, self.survey.n_sounding), order='F')
        # can be parallelized using dask
        jtvec = self.Jmatrix.T.dot(v)
        return Utils.mkvc(jtvec)

    def getJtJdiag(self, m):
        weightone = np.diag(np.dot(self.Jmatrix.T, self.Jmatrix))
        JtJdiag = (
            np.repeat(
                weightone.reshape([1, -1]),
                self.survey.n_sounding, axis=0
            )
        ).flatten()
        return JtJdiag


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
            n_sounding = self.xyz_line.shape[0]
            meshline = Mesh.TensorMesh([n_sounding])
            Gx = meshline.cellGradx
            Wx = np.sqrt(self.alpha_x)*sp.kron(Gx, Utils.speye(self.ntau))
            Ws = np.sqrt(self.alpha_s)*Utils.speye(self.ntau*n_sounding)
            wlist = (Wx, Ws)
            self._W = sp.vstack(wlist)
        return self._W
