from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
from SimPEG import Maps, Props, Problem, Utils, Solver as SimpegSolver
from SimPEG.EM.Static.DC import FieldsDC
from SimPEG.EM.TDEM import FieldsTDEM
from SimPEG.Problem import BaseTimeProblem
from simpegEMIP.Base import BaseEMIPProblem
from simpegEMIP.TDEM.Survey import SurveyLinear
from simpegEMIP.TDEM.FieldsTDEMIP import Fields3D_e_Inductive
import time
from scipy.constants import mu_0
import sys
# from profilehooks import profile


# @profile
def geteref(e, mesh, option=None, tInd=0):
    """
    geteref(e, mesh, option=None, tInd=0)

    Return a reference electric field of given option.

    Parameters
    ----------
    e : array of electric field [nE x nSrc x ntime]
        or [nE x ntime] for a single source.
    mesh: SimPEG mesh object.
    option: "None" or "max"
        "None" requires tInd, and simply choose that time index
        "max" quarrry electric field at maximum.
    tInd: int, time index for the choice of the reference electric field

    """
    # Single source
    if e.ndim == 2:
        ntime = e.shape[1]
        if option == "max":
            inds = np.argmax(abs(e), axis=1)
            inds_max = Utils.sub2ind(
                e.shape, np.c_[np.arange(mesh.nE), inds]
                )
            eref = Utils.mkvc(e)[inds_max]
        else:
            eref = e[:, tInd]

    # Multi source
    elif e.ndim == 3:
        ntime = e.shape[2]
        nSrc = e.shape[1]
        eref = []
        if option == "max":
            for iSrc in range (nSrc):
                inds = np.argmax(abs(e[:, iSrc, :]), axis=1)
                inds_max = Utils.sub2ind(
                    e[:, iSrc, :].shape, np.c_[np.arange(mesh.nE), inds]
                    )
                eref.append(Utils.mkvc(e[:, iSrc, :])[inds_max])
            eref = np.vstack(eref).T
        else:
            eref = e[:, :, tInd]
    else:
        raise Exception("Dimension of e should be either 1 or 2")
    return eref


# @profile
def getwe(e, eref, mesh, ndim=1):
    """
    getwe(e, eref, mesh)

    Return a time history of electric field, we for a single source.

    Parameters
    ----------
    e : array of electric field [nE x ntime] for a single source
    mesh: SimPEG mesh object.

    See also
    --------

    get_we_eff: Return an effective we from multi-sources.
    geteref: Return a reference electric field of given option.

    """
    e_eref = Utils.sdiag(eref) * e
    eref_eref = eref**2
    we_cc = Utils.sdiag(1./(mesh.aveE2CC*eref_eref)) * (mesh.aveE2CC*e_eref)
    we_cc[we_cc < 0.] = 0.
    return we_cc


# @profile
def get_we_eff(e, eref, J, mesh, actinds):
    # Here we assume dimension of J: [nSrc x nC]
    nSrc = e.shape[1]
    ntime = e.shape[2]
    we_eff = np.zeros((mesh.nE, nSrc))
    J_sum_src = (J**2).sum(axis=0)
    we_cc_eff = np.zeros((mesh.nC, ntime))
    for iSrc in range(nSrc):
        we_cc = getwe(e[:, iSrc, :], eref[:, iSrc], mesh)
        a_ik = J[iSrc, :]**2 / J_sum_src
        we_cc_eff[actinds] += Utils.sdiag(a_ik) * we_cc[actinds, :]
    return we_cc_eff


class LinearIPProblem(BaseEMIPProblem, BaseTimeProblem):
    """
    Linnear IP Problem class
    """

    surveyPair = SurveyLinear
    fieldsPair = FieldsDC
    Ainvdc = None
    f = None
    phi = None
    actinds = None
    storeJ = False
    J = None
    actMap = None    # Surjection Map
    galvanicterm = False
    wave_option = "impulse_ramp"
    tlags = None
    # Options are:
    # impulse_ramp
    # step_ramp
    we = None

    def __init__(self, mesh, **kwargs):
        BaseEMIPProblem.__init__(self, mesh, **kwargs)

        if self.actinds is None:
            print ("You did not put Active indices")
            print ("So, set actMap = IdentityMap(mesh)")
            self.actinds = np.ones(mesh.nC, dtype=bool)

        self.actMap = Maps.InjectActiveCells(mesh, self.actinds, 0.)

    def BiotSavartFun(self, rxlocs, component='z'):
        """
            Compute systematrix G using Biot-Savart Law

            G = np.vstack((G1,G2,G3..,Gnpts)

            .. math::

        """
        if rxlocs.ndim == 1:
            npts = 1
        else:
            npts = rxlocs.shape[0]
        e = np.ones((self.mesh.nC, 1))
        o = np.zeros((self.mesh.nC, 1))
        const = mu_0/4/np.pi
        if self.mesh._meshType == "CYL":
            G = np.zeros((npts, self.mesh.nC))
        else:
            G = np.zeros((npts, self.mesh.nC*3))

        # TODO: this works fine, but potential improvement by analyticially
        # evaluating integration.
        for i in range(npts):

            if npts == 1:
                r_rx = np.repeat(
                    Utils.mkvc(rxlocs).reshape([1, -1]), self.mesh.nC, axis = 0
                    )
            else:
                r_rx = np.repeat(
                    rxlocs[i, :].reshape([1, -1]), self.mesh.nC, axis = 0
                    )
            r_CC = self.mesh.gridCC
            r = r_rx-r_CC
            r_abs = np.sqrt((r**2).sum(axis=1))
            rxind = r_abs == 0.
            # r_abs[rxind] = self.mesh.vol.min()**(1./3.)*0.5
            r_abs[rxind] = 1e20
            Sx = const*Utils.sdiag(self.mesh.vol*r[:, 0]/r_abs**3)
            Sy = const*Utils.sdiag(self.mesh.vol*r[:, 1]/r_abs**3)
            Sz = const*Utils.sdiag(self.mesh.vol*r[:, 2]/r_abs**3)

            if self.mesh._meshType == "CYL":
                if component == 'x':
                    G_temp = np.hstack((e.T*Sz))
                elif component == 'z':
                    G_temp = np.hstack((-e.T*Sx))
            else:
                if component == 'x':
                    G_temp = np.hstack((o.T, e.T*Sz, -e.T*Sy))
                elif component == 'y':
                    G_temp = np.hstack((-e.T*Sz, o.T, e.T*Sx))
                elif component == 'z':
                    G_temp = np.hstack((e.T*Sy, -e.T*Sx, o.T))
            G[i, :] = G_temp
        return G

    def getPeta(self, t):
        signs = [1., -1., -1., 1]
        peta = np.zeros_like(self.eta)

        if self.wave_option == "impulse_ramp":
            m = self.eta*self.c/(self.tau**self.c)
            for i, tlag in enumerate(self.tlags):
                peta += (
                    signs[i]*m*(t+tlag)**(self.c-1.) *
                    np.exp(-((t+tlag)/self.tau)**self.c)
                    )
            # dt = self.tlags[1]

        elif self.wave_option == "step_ramp":
            for tlag in self.tlags:
                peta += signs[i]*self.eta*(
                    np.exp(-((t+tlag)/self.tau)**self.c)
                    )
        else:
            raise Exception("Input known wave_option!")

        return peta

    # This evaluates convolution, and takes most of time
    # TODO: Use cython

    @property
    def Peta(self):
        if getattr(self, '_Peta', None) is None:
            ntimes = self.times.size
            dt_temp = np.diff(self.times)
            dt = np.r_[self.times[0], dt_temp]
            self._Peta = np.zeros((self.mesh.nC, ntimes))
            # Ignore first step since we = 0 when t=0
            for i in range(1, ntimes):
                temp = np.zeros(self.mesh.nC)
                for k in range(1, i):
                    temp += self.we[:, k-1] * self.getpetaI(self.times[i]-self.times[k-1]) * dt[k] * 0.5
                    temp += self.we[:, k] * self.getpetaI(self.times[i]-self.times[k]) * dt[k] * 0.5
                self.Peta[:, i] = (
                    temp + self.getKappa(dt[i]) * self.we[:, i-1] + self.getGamma(dt[i]) * self.we[:, i]
                )
        return self._Peta

    @property
    def Pt(self):
        if getattr(self, '_Pt', None) is None:
            self._Pt = self.timeMesh.getInterpolationMat(
                self.survey.times, "N"
                )
            return self._Pt

    @property
    def Pdt(self):
            self._Pdt = self.timeMesh.getInterpolationMat(
                self.survey.times, "CC"
                ) * self.timeMesh.faceDiv
            return self._Pdt

    def getpetaI(self, time):
        m = self.eta*self.c/(self.tau**self.c)
        peta = m*time**(self.c-1.)*np.exp(-(time/self.tau)**self.c)
        return peta

    def getGamma(self, dt):
        m = self.eta*self.c/(self.tau**self.c)
        gamma = m / (self.c*(self.c+1.)) * (dt) ** self.c
        - m / (2*self.c*(2*self.c+1.) * self.tau**self.c) * (dt) ** (2*self.c)
        return - self.sigmaInf * gamma

    def getKappa(self, dt):
        m = self.eta*self.c/(self.tau**self.c)
        kappa = m / (self.c+1.) * (dt) ** self.c
        - m / ((2*self.c+1.)*self.tau ** self.c) * (dt) ** (2*self.c)
        return - self.sigmaInf * kappa

    def fields(self, m):
        return None

    # @profile
    def getJ(self, f=None):
        """
            Generate Sensitivity matrix
        """

        print (">> Compute Sensitivity matrix")

        J = []
        MeDeriv = self.mesh.getEdgeInnerProductDeriv(np.ones(self.mesh.nC))
        Grad = self.mesh.nodalGrad

        if self.galvanicterm:
            A = self.getAdc()
            self.Ainvdc = self.Solver(A, **self.solverOpts)
            print (">> Consider Galvanic term")
        for isrc, src in enumerate(self.survey.srcList):
            for rx in src.rxList:
                eref = self.f[src, 'eref'].copy()
                S_temp = MeDeriv(eref)*Utils.sdiag(self.sigmaInf)*self.actMap.P
                G_temp = self.BiotSavartFun(rx.locs, component=rx.projComp)
                if self.galvanicterm:
                    rhs = (
                        Grad.T*self.MeSigmaInf*self.MeI*self.mesh.aveE2CCV.T*G_temp.T
                    )
                    x = self.Ainvdc*rhs
                    J.append(
                        - Utils.mkvc(G_temp*self.mesh.aveE2CCV*self.MeI*S_temp)
                        + Utils.mkvc((S_temp.T*self.mesh.nodalGrad*x).T)
                    )
                else:
                    J.append(
                        Utils.mkvc(- G_temp*self.mesh.aveE2CCV*self.MeI*S_temp)
                    )
            sys.stdout.write(("\r %d / %d") % (isrc+1, self.survey.nSrc))
            sys.stdout.flush()
        return -np.vstack(J)

    # @profile
    def forward(self, m, f=None):

        self.model = m
        Jv = []
        if self.J is None:
            self.J = self.getJ(f=f)

        ntime = len(self.survey.times)

        self.model = m

        # both impulse and step requires convolution
        if self.wave_option == "impulse":
            return - self.J.dot(
                self.actMap.P.T *
                (self.Peta * self.Pdt.T)
                ).flatten()
        elif self.wave_option == "step":
            return self.J.dot(
                self.actMap.P.T *
                (self.Peta * self.Pt.T)
                ).flatten()
        else:
            for tind in range(ntime):
                Jv.append(
                    self.J.dot(
                        self.actMap.P.T*self.getPeta(self.survey.times[tind]))
                    )
            return np.hstack(Jv)

    def Jvec(self, m, v, f=None):

        self.model = m

        Jv = []

        if self.J is None:
            self.J = self.getJ(f=f)

        ntime = len(self.survey.times)

        for tind in range(ntime):

            t = self.survey.times[tind]
            v0 = self.PetaEtaDeriv(t, v)
            v1 = self.PetaTauiDeriv(t, v)
            v2 = self.PetaCDeriv(t, v)
            PTv = self.actMap.P.T*(v0+v1+v2)
            Jv.append(self.J.dot(PTv))

        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):

        self.model = m

        if self.J is None:
            self.J = self.getJ(f=f)

        ntime = len(self.survey.times)
        Jtvec = np.zeros(m.size)
        v = v.reshape((int(self.survey.nD/ntime), ntime), order="F")

        for tind in range(ntime):
            t = self.survey.times[tind]
            Jtv = self.actMap.P*self.J.T.dot(v[:, tind])
            Jtvec += (
                self.PetaEtaDeriv(t, Jtv, adjoint=True) +
                self.PetaTauiDeriv(t, Jtv, adjoint=True) +
                self.PetaCDeriv(t, Jtv, adjoint=True)
                )

        return Jtvec

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    def MfRhoIDeriv(self, u):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """

        dMfRhoI_dI = -self.MfRhoI**2
        dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
        if self.storeJ:
            drho_dlogrho = Utils.sdiag(self.rho)*self.actMap.P
        else:
            drho_dlogrho = Utils.sdiag(self.rho)
        return dMfRhoI_dI * (dMf_drho * drho_dlogrho)

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """
        if self.storeJ:
            dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.actMap.P
        else:
            dsigma_dlogsigma = Utils.sdiag(self.sigma)
        return (
            self.mesh.getEdgeInnerProductDeriv(self.sigma)(u)
            * dsigma_dlogsigma
            )


# ------------------------------- Problem3D_e ------------------------------- #
class Problem3DIP_Linear(LinearIPProblem,):
    """
        Solve the EB-formulation of Maxwell's equations for the electric field, e.

        Starting with

        .. math::

            \\nabla \\times \\mathbf{e} + \\frac{\\partial \\mathbf{b}}{\\partial t} = \\mathbf{s_m} \\
            \\nabla \\times \mu^{-1} \\mathbf{b} - \\mathbf{j} = \\mathbf{s_e}


        we eliminate :math:`\\frac{\\partial b}{\\partial t}` using


        taking the time-derivative of Ampere's law, we see


        which gives us


    """

    _fieldType = 'e'
    _formulation = 'EB'
    fieldsPair = Fields3D_e_Inductive  #: A Fields3D_e
    Adcinv = None

    def __init__(self, mesh, **kwargs):
        LinearIPProblem.__init__(self, mesh, **kwargs)

    def set_eref(self, eref):
        self.f = self.fieldsPair(self.mesh, self.survey)
        self.f[:, 'eref'] = eref

    def getAdc(self):
        MeSigma0 = self.MeSigma0
        Grad = self.mesh.nodalGrad
        Adc = Grad.T * MeSigma0 * Grad
        # Handling Null space of A
        Adc[0, 0] = Adc[0, 0] + 1.
        return Adc

    def getInitialFields(self):
        """
        Ask the sources for initial fields
        """

        Srcs = self.survey.srcList

        ifields = np.zeros((self.mesh.nE, len(Srcs)))

        if self.verbose:
            print ("Calculating Initial fields")

        for i, src in enumerate(Srcs):
            # Check if the source is grounded
            if src.srcType == "Galvanic" and src.waveform.hasInitialFields:
                # Check self.Adcinv and clean
                if self.Adcinv is not None:
                    self.Adcinv.clean()
                # Factorize Adc matrix
                if self.verbose:
                    print ("Factorize system matrix for DC problem")
                Adc = self.getAdc()
                self.Adcinv = self.Solver(Adc)

            ifields[:, i] = (
                ifields[:, i] + getattr(
                    src, '{}Initial'.format(self._fieldType), None
                )(self)
            )

        return ifields

    def clean(self):
        """
        Clean factors
        """
        if self.Adcinv is not None:
            self.Adcinv.clean()


class Problem3DIP_Linear_singletime(Problem3DIP_Linear):
    peta, petaMap, petaDeriv = Props.Invertible(
        "Peudo-chargeability"
    )

    def forward(self, m, f=None):
        self.model = m
        if self.J is None:
            self.getJ(f=f)

        ntime = len(self.survey.times)
        self.model = m
        return self.J.dot(self.actMap.P.T * self.peta)

    def Jvec(self, m, v, f=None):
        return self.J.dot(self.actMap.P.T * self.petaDeriv * v)

    def Jtvec(self, m, v, f=None):
        return self.petaDeriv.T * (self.actMap.P*(self.J.T.dot(v)))

