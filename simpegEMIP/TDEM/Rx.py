import SimPEG
from SimPEG import Utils
import scipy.sparse as sp


class BaseRx(SimPEG.Survey.BaseTimeRx):
    """
    Time domain receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        assert(orientation in ['x', 'y', 'z']), (
            "Orientation {0!s} not known. Orientation must be in "
            "'x', 'y', 'z'. Arbitrary orientations have not yet been "
            "implemented.".format(orientation)
        )
        self.projComp = orientation
        SimPEG.Survey.BaseTimeRx.__init__(self, locs, times, rxType=None)

    def getSpatialP(self, mesh):
        """
            Returns the spatial projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        if not "Ps" in self._Ps:
            self._Ps['Ps'] = mesh.getInterpolationMat(self.locs, 'Fz')
        return self._Ps['Ps']

    def getTimeP(self, timeMesh):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        if not "Pt" in self._Ps:
            self._Ps['Pt'] = timeMesh.getInterpolationMat(self.times, 'N')
        return self._Ps['Pt']

    def eval(self, i_src, mesh, timeMesh, e):
        """
        Project fields to receivers to get data.

        :param SimPEG.EM.TDEM.SrcTDEM.BaseSrc src: TDEM source
        :param BaseMesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        dbdt = -mesh.edgeCurl * e[:, i_src, :]
        Pt = self.getTimeP(timeMesh)
        Ps = self.getSpatialP(mesh)
        return Utils.mkvc((Ps*dbdt) * Pt.T)


class Point_dbdt(BaseRx):
    """
    dbdt TDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        self.projField = 'dbdt'
        super(Point_dbdt, self).__init__(locs, times, orientation)
