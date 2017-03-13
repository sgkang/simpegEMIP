from mumps import DMumpsContext

class Mumps():
    A = None
    ctx = None
    x = None

    def __init__(self, A, **kwagrs):
        
        self.ctx = DMumpsContext()  
    
        self.A = A.tocsc()
        self.ctx.set_icntl(14, 60)            
        self.ctx.set_centralized_sparse(A)
        # print 'Factoring'
        self.ctx.set_silent()    
        # print 'Done'
        self.ctx.run(job=4) # Factorization            

    def solve(self,b):               
        # print 'Solving'
        self.x = b.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.run(job=3) # Solve 
        # print 'Done'
        return self.x

    def clean(self):
    	self.ctx.destroy()