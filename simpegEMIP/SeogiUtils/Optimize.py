import SimPEG.Utils as Utils
from SimPEG.Optimization import Minimize, Remember
from SimPEG.Utils.SolverUtils import *
import numpy as np
import scipy.sparse as sp

norm = np.linalg.norm


class ProjectedGNCG(Minimize, Remember):

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    name = 'Projected GNCG'

    maxIterCG = 5
    tolCG = 1e-1

    lower = -np.inf
    upper = np.inf

    @Utils.count
    def projection(self, x):
        """projection(x)

            Make sure we are feasible.

        """
        return np.median(np.c_[self.lower,x,self.upper],axis=1)

    @Utils.count
    def activeSet(self, x):
        """activeSet(x)

            If we are on a bound

        """
        return np.logical_or(x <= self.lower, x >= self.upper)

    @property
    def approxHinv(self):
        """
            The approximate Hessian inverse is used to precondition CG.

            Default uses BFGS, with an initial H0 of *bfgsH0*.

            Must be a scipy.sparse.linalg.LinearOperator
        """
        _approxHinv = getattr(self,'_approxHinv',None)
        if _approxHinv is None:
            M = sp.linalg.LinearOperator( (self.xc.size, self.xc.size), self.bfgs, dtype=self.xc.dtype )
            return M
        return _approxHinv

    @approxHinv.setter
    def approxHinv(self, value):
        self._approxHinv = value

	@Utils.timeIt
	def findSearchDirection(self):

		"""
			findSearchDirection()
			Finds the search direction based on either CG or steepest descent.
	    """
		Active = self.activeSet(self.xc)
		temp = sum((np.ones_like(self.xc.size)-Active))
		allBoundsAreActive =  temp == self.xc.size
		print 'kang'

		if allBoundsAreActive:
			Hinv = SolverICG(self.H, M=self.approxHinv, tol=self.tolCG, maxiter=self.maxIterCG)
			p = Hinv * (-self.g)
			return p
		else:


			delx = zeros(size(self.g))
			resid = -(1-Active) * self.g

			# Begin CG iterations.
			cgiter = 0
			cgFlag = 0
			normResid0 = norm(resid)

			while cgFlag == 0:

				cgiter = cgiter + 1
				dc = (1-Active)*(self.approxHinv*resid)
				rd = np.dot(resid, dc)

				#  Compute conjugate direction pc.
				if cgiter == 1:
					pc = dc
				else:
					betak = rd / rdlast
					pc = dc + betak * pc

				#  Form product Hessian*pc.
				Hp = self.H*pc
				Hp = (1-Active)*Hp

				#  Update delx and residual.
				alphak = rd / np.dot(pc, Hp)
				delx = delx + alphak*pc
				resid = resid - alphak*Hp
				rdlast = rd

				if np.logical_or(norm(resid)/normResid0 <= cgTol, cgiter == maxCG):
					cgFlag = 1
	    		# End CG Iterations

			return delx




