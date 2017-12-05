import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import torch
from   torch.autograd import Variable
import torch.optim as optim
from pydeformetrica.src.support.utilities.kernel_types import KernelType
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel


class Diffeomorphism:
    """
    Control-point-based LDDMM diffeomophism, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    # Constructor.
    def __init__(self):
        self.KernelType = KernelType.Torch
        self.KernelWidth = None
        if self.KernelType == KernelType.Torch:
            self.Kernel = TorchKernel()
        self.NumberOfTimePoints = 10
        self.T0 = 0.
        self.TN = 1.
        #Initial position of control points
        self.StartPositions = None
        #Initial momenta
        self.StartMomenta = None
        #Control points trajectort
        self.PositionsT = None
        #Momenta trajectory
        self.MomentaT = None
        #Flow of voxel positions (with backward integration i.e. \f$\phi_t^{-1}\f$)
        self.MapsT = None
        #Flow of voxel positions (with true inverse flow i.e. \f$\phi_t\circ\phi_1^{-1}\f$).
        self.InverseMapsT = None
        #Velocity of landmark points
        self.LandmarkPointsVelocity = None
        #Trajectory of the whole vertices of Landmark type at different time steps.
        self.m_LandmarkPointsT = None

    def SetStartPositions(StartPositions):
        """
        InitialControlPoints is a numpy array containing the initial cp positions, shape (num_cp, dim)
        """
        if self.KernelType == KernelType.Torch:
            self.StartPositions = Variable(torch.from_numpy(StartPositions))
        else:
            print "Diffeo not yet implemented for kernels other than pytorch"

    def SetKernelWidth(kernelWidth):
        self.KernelWidth = kernelWidth
        self.Kernel.KernelWidth = kernelWidth

    def SetStartMomenta(StartMomenta):
        """
        InitialMomenta is a numpy array containing the initial momenta, shape (num_cp, dim)
        """
        if self.KernelType == KernelType.Torch:
            self.StartMomenta = Variable(torch.from_numpy(StartMomenta))
        else:
            print "Diffeo not yet implemented for kernels other than pytorch"

    def Shoot():
        """
        Computes the flow of momenta and control points
        """
        numCP = len(self.StartPositions)
        assert numCP > 0, "Control points not initialized in shooting"
        assert len(self.StartMomenta) > 0, "Momenta not initialized in shooting"
        self.PositionsT = []
        self.MomentaT = []
        self.PositionsT.append(self.StartPositions)
        self.MomentaT.append(self.MomentaT)
        dt = (self.TN - self.T0)/(self.NumberOfTimePoints - 1.)
        for i in range(self.NumberOfTimePoints):
            dPos = self.Kernel.Convolve(self.PositionsT[i], self.MomentaT[i], self.PositionsT[i])
            dMom = self.Kernel.ConvolveGradient(self.PositionsT[i], self.MomentaT[i], self.PositionsT[i])
            self.PositionsT.append(self.PositionsT[i] + dt * dPos)
            self.MomentaT.append(self.MomentaT[i] - dt * dMom)


controlPoints = np.array([[0.,0.],[0.5,0.]])
momenta = np.array([[-1.,0.],[1.,0.]])
diffeo = Diffeomorphism()
diffeo.SetKernelWidth(0.2)
diffeo.SetStartPositions(controlPoints)
diffeo.SetStartMomenta(momenta)
diffeo.Shoot()
