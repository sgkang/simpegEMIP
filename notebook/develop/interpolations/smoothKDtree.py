""" smoothKDtree.py: distance based smoothing using KDTree
"""
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree as KDTree
    # http://docs.scipy.org/doc/scipy/reference/spatial.html


#...............................................................................
class smoothKDtree:
    """ 
    Smooth model with nearest neighbours

    """


    def __init__( self, X, z, leafsize=3):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, eps=0, p=3, distance = 500):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        self.distance = distance
        # self.ix = self.tree.query.( q, k=100, eps=eps, distance_upper_bound=distance )
        # interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        interpol = np.zeros( (len(q),) + np.shape(self.z[0]) )
        jinterpol = 0
        # for dist, ix in zip( self.distances, self.ix ):
        for v in q:
            # weight z s by 1/dist --
            # w = 1 / dist**p
            # w /= np.sum(w)
            # wz = np.dot( w, self.z[ix] )
            ix = self.tree.query_ball_point( v , eps=eps, p=p, r=distance )
            wzlog = np.mean(np.log10(self.z[ix]))
            interpol[jinterpol] = np.power(10,wzlog)
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]

#...............................................................................
if __name__ == "__main__":

    import smoothKDtree as sKDt 
    import SimPEG as simpeg
    # Make the mesh. 
    xLim = np.array([0,5000])
    yLim = np.array([0,6000])
    zLim = np.array([-1600,0])
    cellSize = [200,200,40]

    # Define padding
    horPad = np.round(np.cumprod(np.ones(10)*1.4)*cellSize[0],-2)
    airPad = np.round(np.cumprod(np.ones(12)*1.4)*cellSize[2],-1)
    belPad = np.round(np.cumprod(np.ones(15)*1.4)*cellSize[2],-1)[::-1]

    dx = np.ascontiguousarray(np.concatenate((horPad[::-1],np.ones(np.diff(xLim)/cellSize[0])*cellSize[0],horPad)))
    dy = np.ascontiguousarray(np.concatenate((horPad[::-1],np.ones(np.diff(yLim)/cellSize[1])*cellSize[1],horPad)))
    dz = np.ascontiguousarray(np.concatenate((belPad,np.ones(np.diff(zLim)/cellSize[2])*cellSize[2],airPad)))
    x0 = (xLim[0] - np.sum(horPad),yLim[0] - np.sum(horPad),zLim[0] - np.sum(belPad))
    mesh = simpeg.Mesh.TensorMesh([dx,dy,dz],x0) 
    # Make the model parameters
    top = - 500
    bot = -1500
    p0arr = np.array([[1000,0,bot],[2000,1500,bot]])
    p1arr = np.array([[2000,5000,top],[4000,3500,top]])

    pMin = np.array([mesh.vectorNx.min(),mesh.vectorNy.min(),mesh.vectorNz.min()])
    pMax = np.array([mesh.vectorNx.max(),mesh.vectorNy.max(),0])
    modelT = simpeg.Utils.ModelBuilder.defineBlock(mesh.gridCC,pMin,pMax,[1e-2,1e-8])
    # Define the structure
    for p0,p1 in zip(p0arr,p1arr):
        temp = simpeg.Utils.ModelBuilder.defineBlock(mesh.gridCC,p0,p1,[1,0])
        modelT += temp

    # Smooth all the cells in the model with 500m distance kernel
    smAllKern = sKDt.smoothKDtree(mesh.gridCC,modelT)
    allSmoothValue = smAllKern(mesh.gridCC,distance=500)

    # Only smooth the earth cell
    indEarth = modelT < 1e-8
    smEarthKern = sKDt.smoothKDtree(mesh.gridCC[indEarth,:],modelT[indEarth])
    earthSmoothValue = smEarthKern(mesh.gridCC[indEarth,:],distance=500)