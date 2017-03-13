from SimPEG import Mesh, Survey, Problem, Utils, Models, Maps, PropMaps, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SimPEG.EM.TDEM.BaseTDEM import FieldsTDEM
from simpegAIP.Base import BaseAEMProblem
from SimPEG.Problem import BaseTimeProblem


class BaseATEMProblem(BaseAEMProblem,BaseTimeProblem):
    """docstring for BaseTDEMProblem"""    

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTimeProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    _FieldsForward_pair = FieldsTDEM  #: used for the forward calculation only

    def fields(self, m):
        if self.verbose: print '%s\nCalculating fields(m)\n%s'%('*'*50,'*'*50)
        self.curModel = m
        # Create a fields storage object
        F = self._FieldsForward_pair(self.mesh, self.survey)
        for src in self.survey.srcList:
            # Set the initial conditions
            F[src,:,0] = src.getInitialFields(self.mesh)
        F = self.forward(m, self.getRHS, F=F)
        if self.verbose: print '%s\nDone calculating fields(m)\n%s'%('*'*50,'*'*50)
        return F

    def forward(self, m, RHS, F=None):
        self.curModel = m
        F = F or FieldsTDEM(self.mesh, self.survey)
        dtFact = None
        Ainv   = None
        for tInd, dt in enumerate(self.timeSteps):
            if dt != dtFact:
                dtFact = dt
                if Ainv is not None:
                    Ainv.clean()
                A = self.getA(tInd)
                if self.verbose: print 'Factoring...   (dt = %e)'%dt
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose: print 'Done'
            rhs = RHS(tInd, F)
            if self.verbose: print '    Solving...   (tInd = %d)'%tInd
            sol = Ainv * rhs
            if self.verbose: print '    Done...'
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,self.solType,tInd+1] = sol
        Ainv.clean()
        return F

if __name__ == '__main__':
    cs, ncx, ncz, npad = 5., 25, 15, 15
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')    
    prob = BaseATEMProblem(mesh)
