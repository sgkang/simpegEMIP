from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
from SimPEG import Problem, Utils, Solver as SimpegSolver
from SimPEG.EM.TDEM.SurveyTDEM import Survey as SurveyTDEM
from SimPEG.EM.TDEM.FieldsTDEM import (
    FieldsTDEM, Fields3D_b, Fields3D_e, Fields3D_h, Fields3D_j, Fields_Derivs
)
from simpegEMIP.Base import BaseEMIPProblem
from simpegEMIP import 

class BaseTDEMIPProblem(Problem.BaseTimeProblem, BaseEMIPProblem):
    """
    We start with the first order form of Maxwell's equations, eliminate and
    solve the second order form. For the time discretization, we use backward
    Euler.
    """
    surveyPair = SurveyTDEM  #: A SimPEG.EM.TDEM.SurveyTDEM Class
    fieldsPair = FieldsTDEM  #: A SimPEG.EM.TDEM.FieldsTDEM Class

    def __init__(self, mesh, **kwargs):
        BaseEMIPProblem.__init__(self, mesh)
        Debye = []
    
    def getDebye():

        return
        
    def MeA(dt):
        gamma = getGamma(dt)
        val = sigmaInf + gamma
        return mesh.getEdgeInnerProduct(val)      

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

            rhs = self.getRHS(tInd+1)  # this is on the nodes of the time mesh
            Asubdiag = self.getAsubdiag(tInd)

            if self.verbose:
                print('    Solving...   (tInd = {:i})'.format(tInd+1))
            # taking a step
            sol = Ainv * (rhs - Asubdiag * F[:, (self._fieldType + 'Solution'),
                                             tInd])

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

    def __init__(self, mesh, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, **kwargs)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        
        return C.T * ( MfMui * C ) + 1./dt * MeA(dt)

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Deriv of ADiag with respect to electrical conductivity
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        MeSigmaInfDeriv = self.MeSigmaInfDeriv(u)

        if adjoint:
            return 1./dt * MeSigmaInfDeriv.T * v

        return 1./dt * MeSigmaInfDeriv * v

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]

        return - 1./dt * self.MeSigmaInf

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of the matrix below the diagonal with respect to electrical
        conductivity
        """
        dt = self.timeSteps[tInd]

        if adjoint:
            return - 1./dt * self.MeSigmaInfDeriv(u).T * v

        return - 1./dt * self.MeSigmaInfDeriv(u) * v

    def getRHS(self, tInd):
        """
        right hand side
        """
        if tInd == len(self.timeSteps):
            tInd = tInd - 1
        dt = self.timeSteps[tInd]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd-1)
        return (-1./dt * (s_e - s_en1) +
                self.mesh.edgeCurl.T * self.MfMui * s_m)

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        # right now, we are assuming that s_e, s_m do not depend on the model.
        return Utils.Zero()


if __name__ == '__main__':
    from SimPEG import Mesh
    cs, ncx, ncz, npad = 5., 25, 15, 15
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')    
    prob = BaseTDEMIPProblem(mesh)
