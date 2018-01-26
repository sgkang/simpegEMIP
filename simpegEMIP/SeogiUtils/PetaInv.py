from SimPEG import Problem, Survey, Utils, Maps
from . import Convolution
import numpy as np


class PetaInvProblem(Problem.BaseProblem):
    surveyPair = Survey.BaseSurvey
    P = None
    J = None
    time = None
    we = None

    def __init__(self, mesh, mapping, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, mapping, **kwargs)

    def fields(self, m, f=None):
        mtemp = self.mapping*m
        self.J = petaJconvfun(m[0], m[1], self.we, self.time, self.P)
        return petaconvfun(mtemp[0], mtemp[1], self.we, self.time, self.P)

    def Jvec(self, m, v, f=None):
        P = self.mapping.deriv(m)
        jvec = self.J.dot(P*v)
        return jvec

    def Jtvec(self, m, v, f=None):
        P = self.mapping.deriv(m)
        jtvec = P.T*(self.J.T.dot(v))
        return jtvec


def petaconvfun(a, b, we, time, P, ColeCole="Debye"):
    if ColeCole == "Debye":
        kernel = lambda x: a*np.exp(-b*x)
    elif ColeCole == "Warburg":
        kernel = lambda x: a*(
            1./np.sqrt(np.pi*x) - b*np.exp(b**2*x)*erfc(b*np.sqrt(x))
            )
    temp = kernel(time)
    temp = Convolution.CausalConvIntSingle(we, time, kernel)
    out = P*temp
    return out


def petaJconvfun(a, b, we, time, P):

    if ColeCole == "Debye":
        kernela = lambda x: np.exp(-b*x)
        kernelb = lambda x: -a*x*np.exp(-b*x)

    elif ColeCole == "Warburg":
        kernela = lambda x: 1./np.sqrt(np.pi*t) - b*np.exp(b**2*t)*erfc(b*np.sqrt(t))
        kernelb = lambda x: a*(
            2*b*np.sqrt(t)/np.sqrt(np.pi)
            - 2*b**2*t*np.exp(b**2*t)*erfc(b*np.sqrt(t))
            - np.exp(b**2*t)*erfc(b*np.sqrt(t))
            )

    tempa = kernela(time)
    tempb = kernelb(time)
    tempa = Convolution.CausalConvIntSingle(we, time, kernela)
    tempb = Convolution.CausalConvIntSingle(we, time, kernelb)
    J = np.c_[P*tempa, P*tempb]
    return J


class PetaSurvey(Survey.BaseSurvey):

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)

    @Utils.requires('prob')
    def dpred(self, m, f=None):
        return self.prob.fields(m)

    def residual(self, m, f=None):
        if self.dobs.size ==1:
            return Utils.mkvc(np.r_[self.dpred(m, f=u) - self.dobs])
        else:
            return Utils.mkvc(self.dpred(m, f=u) - self.dobs)

