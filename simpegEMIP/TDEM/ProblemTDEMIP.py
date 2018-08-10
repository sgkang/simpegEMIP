from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
from SimPEG import Props, Problem, Utils, Solver as SimpegSolver
from SimPEG.EM.TDEM.SurveyTDEM import Survey as SurveyTDEM
from simpegEMIP.TDEM.FieldsTDEMIP import Fields3D_e, Fields3D_phi
from SimPEG.EM.TDEM import FieldsTDEM
from simpegEMIP.Base import BaseEMIPProblem
import time
from . import getJpol as pyx
# from . import getJpol_py as pyx
# from profilehooks import profile
from scipy.sparse import coo_matrix


def sdiag(h):
    """Sparse diagonal matrix"""
    N = h.size
    row = np.arange(N, dtype='intc')
    return coo_matrix((h, (row, row)), shape=(N, N)).tocsr()


class BaseTDEMIPProblem(Problem.BaseTimeProblem, BaseEMIPProblem):
    """
    We start with the first order form of Maxwell's equations, eliminate and
    solve the second order form. For the time discretization, we use backward
    Euler.
    """

    surveyPair = SurveyTDEM  #: A SimPEG.EM.TDEM.SurveyTDEM Class
    fieldsPair = FieldsTDEM  #: A SimPEG.EM.TDEM.FieldsTDEM Class

    jpol = None
    jpoln1 = None

    def __init__(self, mesh, **kwargs):
        BaseEMIPProblem.__init__(self, mesh, **kwargs)

    # @profile
    def fields(self, m):
        """
        Solve the forward problem for the fields.

        :param numpy.array m: inversion model (nP,)
        :rtype: SimPEG.EM.TDEM.FieldsTDEM
        :return F: fields object
        """

        tic = time.time()
        self.model = m

        F = self.fieldsPair(self.mesh, self.survey)

        # set initial fields
        F[:, self._fieldType+'Solution', 0] = self.getInitialFields()

        self.jpol = np.zeros((F[:, 'e', 0].shape))
        self.jpoln1 = self.getJpol(-1, F)

        # timestep to solve forward
        if self.verbose:
            print('{}\nCalculating fields(m)\n{}'.format('*'*50, '*'*50))
        Ainv = None
        for tInd, dt in enumerate(self.timeSteps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Ainv is not None and (
                tInd > 0 and dt != self.timeSteps[tInd - 1]
            ):
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(tInd)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose:
                    print('Done')

            # Compute polarization urrents at current step
            # self.jpol = self.getJpol(tInd, F)
            # Cythonize
            MeK = self.MeK(dt)
            MeDsigOff_0 = self.MeDsigOff(0)
            MeDsigOff_n = self.MeDsigOff(tInd)
            MeCnk = self.getMeCnk(tInd+1, tInd)

            if F[:, 'e', :].ndim == 2:
                n, m = F[:, 'e', :].shape
                self.jpol = pyx.getJpol(
                    self.timeSteps, tInd,
                    F[:, 'e', :].reshape(n, 1, m, order="F"),
                    MeK, MeCnk,
                    MeDsigOff_0, MeDsigOff_n
                    )
            else:
                self.jpol = pyx.getJpol(
                    self.timeSteps, tInd, F[:, 'e', :],
                    MeK, MeCnk,
                    MeDsigOff_0, MeDsigOff_n
                    )

            rhs = self.getRHS(tInd+1)  # this is on the nodes of the time mesh

            Asubdiag = self.getAsubdiag(tInd)

            if self.verbose:
                print('    Solving...   (tInd = {:d})'.format(tInd+1))
            # taking a step
            sol = Ainv * (rhs - Asubdiag * F[:, (self._fieldType + 'Solution'),
                                             tInd])
            # Store polarization currents at this step
            self.jpoln1 = self.jpol.copy()

            if self.verbose:
                print('    Done...')

            if sol.ndim == 1:
                sol.shape = (sol.size, 1)
            F[:, self._fieldType+'Solution', tInd+1] = sol
        if self.verbose:
            print('{}\nDone calculating fields(m)\n{}'.format('*'*50, '*'*50))
        Ainv.clean()
        return F

    def getSourceTerm(self, tInd):
        """
        Assemble the source term. This ensures that the RHS is a vector / array
        of the correct size
        """

        Srcs = self.survey.srcList

        if self._formulation == 'EB':
            s_m = np.zeros((self.mesh.nF, len(Srcs)))
            s_e = np.zeros((self.mesh.nE, len(Srcs)))
        elif self._formulation == 'HJ':
            s_m = np.zeros((self.mesh.nE, len(Srcs)))
            s_e = np.zeros((self.mesh.nF, len(Srcs)))

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self, self.times[tInd])
            s_m[:, i] = s_m[:, i] + smi
            s_e[:, i] = s_e[:, i] + sei

        return s_m, s_e

    def getInitialFields(self):
        """
        Ask the sources for initial fields
        """

        Srcs = self.survey.srcList

        if self._fieldType in ['b', 'j']:
            ifields = np.zeros((self.mesh.nF, len(Srcs)))
        elif self._fieldType in ['e', 'h']:
            ifields = np.zeros((self.mesh.nE, len(Srcs)))

        for i, src in enumerate(Srcs):
            ifields[:, i] = (
                ifields[:, i] + getattr(
                    src, '{}Initial'.format(self._fieldType), None
                )(self)
            )

        return ifields

    def getpetaI(self, time):
        m = self.eta*self.c/(self.tau**self.c)
        peta = m*time**(self.c-1.)*np.exp(-(time/self.tau)**self.c)
        return peta

    def getpetaOff(self, time):
        peta = self.eta*np.exp(-(time/self.tau)**self.c)
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


# ------------------------------- Problem3D_e ------------------------------- #
class Problem3D_e(BaseTDEMIPProblem):
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
    fieldsPair = Fields3D_e  #: A Fields3D_e
    surveyPair = SurveyTDEM
    Adcinv = None

    def __init__(self, mesh, **kwargs):
        BaseTDEMIPProblem.__init__(self, mesh, **kwargs)

    # TODO: Cythonize
    def getJpol(self, tInd, F):
        """
            Computation of polarization currents
        """

        dt = self.timeSteps[tInd]

        # Handling when jpol at t = 0
        if tInd < 0:
            jpol = sdiag(self.MeDsigOff(0))*F[:, 'e', 0]
            return jpol

        jpol = self.MeK(dt)*F[:, 'e', tInd]

        for k in range(1, tInd):
            dt = self.timeSteps[k]
            jpol += (dt/2)*self.MeCnk(tInd+1, k)*F[:, 'e', k]
            jpol += (dt/2)*self.MeCnk(tInd+1, k+1)*F[:, 'e', k+1]

        # Handling when jpol at t < 0
        jpol += self.MeDsigOff(tInd+1)*F[:, 'e', 0]
        return jpol

    def MeA(self, dt):
        gamma = self.getGamma(dt)
        val = self.sigmaInf + gamma
        return self.mesh.aveE2CC.T * (self.mesh.vol*val)
        # return self.mesh.getEdgeInnerProduct(val)

    def MeK(self, dt):
        kappa = self.getKappa(dt)
        return self.mesh.aveE2CC.T * (self.mesh.vol*kappa)
        # return self.mesh.getEdgeInnerProduct(kappa)

    def MeCnk(self, n, k):
        tn = self.times[n]
        tk = self.times[k]
        val = -self.sigmaInf * self.getpetaI(tn-tk)
        return self.mesh.aveE2CC.T * (self.mesh.vol*val)
        # return self.mesh.getEdgeInnerProduct(val)

    def getMeCnk(self, n, k):
        return np.hstack(
            [self.MeCnk(n, i).reshape([-1, 1]) for i in range(k+1)]
        )

    def MeDsigOff(self, n):
        tn = self.times[n]
        val = -self.sigmaInf * self.getpetaOff(tn)
        # return self.mesh.getEdgeInnerProduct(val)
        return self.mesh.aveE2CC.T * (self.mesh.vol*val)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        return C.T * (MfMui * C) + 1./dt * sdiag(self.MeA(dt))

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        dtn1 = self.timeSteps[tInd-1]
        return - 1./dt * sdiag(self.MeA(dtn1))

    def getRHS(self, tInd):
        """
        right hand side
        """
        # if tInd == len(self.timeSteps):
        #     tInd = tInd - 1

        dt = self.timeSteps[tInd-1]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd-1)
        # return (- 1./dt * (s_e - s_en1)
        #         + self.mesh.edgeCurl.T * self.MfMui * s_m
        #         - 1./dt * (self.jpol-self.jpoln1))
        return (- 1./dt * (s_e - s_en1)
                - 1./dt * (self.jpol-self.jpoln1))

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


class Problem3D_phi(Problem3D_e):
    """
        Solve the EB-formulation of Maxwell's equations for the electric potential, e.

        Starting with

        .. math::

            \\nabla \\times \\mathbf{e} + \\frac{\\partial \\mathbf{b}}{\\partial t} = \\mathbf{s_m} \\
            \\nabla \\times \mu^{-1} \\mathbf{b} - \\mathbf{j} = \\mathbf{s_e}


        we eliminate :math:`\\frac{\\partial b}{\\partial t}` using


        taking the time-derivative of Ampere's law, we see


        which gives us


    """

    _fieldType = 'phi'
    _formulation = 'EB'
    fieldsPair = Fields3D_phi  #: A Fields3D_phi

    def __init__(self, mesh, **kwargs):
        Problem3D_e.__init__(self, mesh, **kwargs)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        G = self.mesh.nodalGrad
        A = G.T * sdiag(self.MeA(dt)) * G
        A[0, 0] = A[0, 0] + 1.
        return A

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        return Utils.Zero()

    def getRHS(self, tInd):
        """
        right hand side
        """
        s_m, s_e = self.getSourceTerm(tInd)
        G = self.mesh.nodalGrad
        return G.T * (s_e+self.jpol)

    def getInitialFields(self):
        """
        Ask the sources for initial fields
        """

        Srcs = self.survey.srcList

        ifields = np.zeros((self.mesh.nN, len(Srcs)))

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
