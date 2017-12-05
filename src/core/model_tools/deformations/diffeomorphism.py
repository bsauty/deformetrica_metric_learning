import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
import torch
from   torch.autograd import Variable
import torch.optim as optim
from pydeformetrica.src.support.utilities.kernel_types import KernelType
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel
# import matplotlib

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
        #Velocity of landmark points
        self.LandmarkPointsVelocity = None
        #Trajectory of the whole vertices of Landmark type at different time steps.
        self.LandmarkPointsT = None
        #Initial landmark points
        self.LandmarkPoints = None

    def SetStartPositions(self, StartPositions):
        self.StartPositions = StartPositions

    def SetStartMomenta(self, StartMomenta):
        self.StartMomenta = StartMomenta

    def SetLandmarkPoints(self, LandmarkPoints):
        self.LandmarkPoints = LandmarkPoints

    def GetLandmarkPoints(self, pos=None):
        """
        Returns the position of the landmark points, at the given pos in the Trajectory
        """
        if pos == None:
            return self.LandmarkPointsT[self.NumberOfTimePoints]
        return self.LandmarkPointsT[pos]

    def SetKernelWidth(self, kernelWidth):
        self.KernelWidth = kernelWidth
        self.Kernel.KernelWidth = kernelWidth

    def Shoot(self):
        """
        Computes the flow of momenta and control points
        """
        #TODO : not shoot if small momenta norm
        assert len(self.StartPositions) > 0, "Control points not initialized in shooting"
        assert len(self.StartMomenta) > 0, "Momenta not initialized in shooting"
        self.PositionsT = []
        self.MomentaT = []
        self.PositionsT.append(self.StartPositions)
        self.MomentaT.append(self.StartMomenta)
        dt = (self.TN - self.T0)/(self.NumberOfTimePoints - 1.)
        for i in range(self.NumberOfTimePoints):
            dPos = self.Kernel.Convolve(self.PositionsT[i], self.MomentaT[i], self.PositionsT[i])
            dMom = self.Kernel.ConvolveGradient(self.PositionsT[i], self.MomentaT[i], self.PositionsT[i])
            self.PositionsT.append(self.PositionsT[i] + dt * dPos)
            self.MomentaT.append(self.MomentaT[i] - dt * dMom)

    def Flow(self):
        """
        Flow The trajectory of the landmark points
        """
        #TODO : not flow if small momenta norm
        assert len(self.PositionsT)>0, "Shoot before flow"
        assert len(self.MomentaT)>0, "Something went wrong, how can this be ?"
        assert len(self.LandmarkPoints)>0, "Please give landmark points to flow"
        dt = (self.TN - self.T0)/(self.NumberOfTimePoints - 1.)
        self.LandmarkPointsT = []
        self.LandmarkPointsT.append(self.LandmarkPoints)
        for i in range(self.NumberOfTimePoints):
            dPos = self.Kernel.Convolve(self.LandmarkPointsT[i], self.MomentaT[i], self.PositionsT[i])
            print(dPos)
            self.LandmarkPointsT.append(self.LandmarkPointsT[i] + dt * dPos)



# controlPoints = Variable(torch.from_numpy(np.array([[0.,0.],[0.5,0.]])))
# momenta = Variable(torch.from_numpy(np.array([[-1.,0.],[1.,0.]])), requires_grad=True)
#
# diffeo = Diffeomorphism()
# diffeo.SetKernelWidth(0.2)
# diffeo.SetStartPositions(controlPoints)
# diffeo.SetStartMomenta(momenta)
#
# landmarkPoints = np.zeros((100,2))
# x = np.linspace(-1,1,10)
# y = np.linspace(-1,1,10)
# for i in range(10):
#     for j in range(10):
#         landmarkPoints[i*10+j] = np.array([x[i], y[j]])
# landmarkPointsTorch = Variable(torch.from_numpy(landmarkPoints))
#
# diffeo.SetLandmarkPoints(landmarkPointsTorch)
#
# diffeo.Shoot()
# diffeo.Flow()
# endLandmarkPoints = diffeo.GetLandmarkPoints().data.numpy()
# for i in range(100):
#     print(landmarkPoints[i], endLandmarkPoints[i])
