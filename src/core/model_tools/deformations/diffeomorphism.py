import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
import torch
from   torch.autograd import Variable
import torch.optim as optim
from pydeformetrica.src.support.utilities.kernel_types import KernelType
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
import copy

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

    def GetLandmarkPoints(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if time_index == None:
            return self.LandmarkPointsT[self.NumberOfTimePoints]
        return self.LandmarkPointsT[time_index]

    def SetKernelWidth(self, kernelWidth):
        self.KernelWidth = kernelWidth
        self.Kernel.KernelWidth = kernelWidth

    def GetNorm(self):
        return torch.dot(self.StartMomenta.view(-1), self.Kernel.Convolve(self.StartPositions,self.StartMomenta,self.StartPositions).view(-1))

    def Shoot(self):
        """
        Computes the flow of momenta and control points
        """
        #TODO : not shoot if small momenta norm
        assert len(self.StartPositions) > 0, "Control points not initialized in shooting"
        assert len(self.StartMomenta) > 0, "Momenta not initialized in shooting"
        # if torch.norm(self.StartMomenta)<1e-20:
        #     self.PositionsT = [self.StartPositions for i in range(self.NumberOfTimePoints)]
        #     self.StartMomenta = [self.StartPositions for i in range(self.NumberOfTimePoints)]
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
        #TODO : no flow if small momenta norm
        assert len(self.PositionsT)>0, "Shoot before flow"
        assert len(self.MomentaT)>0, "Control points given but no momenta"
        assert len(self.LandmarkPoints)>0, "Please give landmark points to flow"
        # if torch.norm(self.StartMomenta)<1e-20:
        #     self.LandmarkPointsT = [self.LandmarkPoints for i in range(self.StartMomenta)]
        dt = (self.TN - self.T0)/(self.NumberOfTimePoints - 1.)
        self.LandmarkPointsT = []
        self.LandmarkPointsT.append(self.LandmarkPoints)
        for i in range(self.NumberOfTimePoints):
            dPos = self.Kernel.Convolve(self.LandmarkPointsT[i], self.MomentaT[i], self.PositionsT[i])
            self.LandmarkPointsT.append(self.LandmarkPointsT[i] + dt * dPos)

    def WriteFlow(self, objects_names, objects_extensions, template):
        for i in range(self.NumberOfTimePoints):
            names = []
            for j,elt in enumerate(objects_names):
                names.append(elt+"_t="+str(i)+objects_extensions[j])
            deformedPoints = self.LandmarkPointsT[i]
            aux_points = template.GetData()
            template.SetData(deformedPoints.data.numpy())
            template.Write(names)
            #restauring state of the template object for further computations
            template.SetData(aux_points.Concatenate())
