import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch


class Diffeomorphism:
    """
    Control-point-based LDDMM diffeomophism, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.Kernel = None
        self.NumberOfTimePoints = 10
        self.T0 = 0.
        self.TN = 1.
        # Initial position of control points
        self.InitialControlPoints = None
        # Initial momenta
        self.InitialMomenta = None
        # Control points trajectory
        self.PositionsT = None
        # Momenta trajectory
        self.MomentaT = None
        # Trajectory of the whole vertices of Landmark type at different time steps.
        self.LandmarkPointsT = None
        # Initial landmark points
        self.LandmarkPoints = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def SetInitialControlPoints(self, InitialControlPoints):
        self.InitialControlPoints = InitialControlPoints

    def SetInitialMomenta(self, InitialMomenta):
        self.InitialMomenta = InitialMomenta

    def SetLandmarkPoints(self, LandmarkPoints):
        self.LandmarkPoints = LandmarkPoints

    def GetLandmarkPoints(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if time_index == None:
            return self.LandmarkPointsT[self.NumberOfTimePoints]
        return self.LandmarkPointsT[time_index]

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def GetNorm(self):
        return torch.dot(self.InitialMomenta.view(-1), self.Kernel.Convolve(
            self.InitialControlPoints, self.InitialControlPoints, self.InitialMomenta).view(-1))

    def Shoot(self):
        """
        Computes the flow of momenta and control points
        """
        # TODO : not shoot if small momenta norm
        assert len(self.InitialControlPoints) > 0, "Control points not initialized in shooting"
        assert len(self.InitialMomenta) > 0, "Momenta not initialized in shooting"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.PositionsT = [self.InitialControlPoints for i in range(self.NumberOfTimePoints)]
        #     self.InitialMomenta = [self.InitialControlPoints for i in range(self.NumberOfTimePoints)]
        self.PositionsT = []
        self.MomentaT = []
        self.PositionsT.append(self.InitialControlPoints)
        self.MomentaT.append(self.InitialMomenta)
        dt = (self.TN - self.T0) / (self.NumberOfTimePoints - 1.)
        # REPLACE with an hamiltonian (e.g. une classe hamiltonien)
        for i in range(self.NumberOfTimePoints):
            dPos = self.Kernel.Convolve(self.PositionsT[i], self.PositionsT[i], self.MomentaT[i])
            dMom = self.Kernel.ConvolveGradient(self.MomentaT[i], self.PositionsT[i])
            self.PositionsT.append(self.PositionsT[i] + dt * dPos)
            self.MomentaT.append(self.MomentaT[i] - dt * dMom)

    def Flow(self):
        """
        Flow The trajectory of the landmark points
        """
        # TODO : no flow if small momenta norm
        assert len(self.PositionsT) > 0, "Shoot before flow"
        assert len(self.MomentaT) > 0, "Control points given but no momenta"
        assert len(self.LandmarkPoints) > 0, "Please give landmark points to flow"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.LandmarkPointsT = [self.LandmarkPoints for i in range(self.InitialMomenta)]
        dt = (self.TN - self.T0) / (self.NumberOfTimePoints - 1.)
        self.LandmarkPointsT = []
        self.LandmarkPointsT.append(self.LandmarkPoints)
        for i in range(self.NumberOfTimePoints):
            dPos = self.Kernel.Convolve(self.LandmarkPointsT[i], self.PositionsT[i], self.MomentaT[i])
            self.LandmarkPointsT.append(self.LandmarkPointsT[i] + dt * dPos)

    def WriteFlow(self, objects_names, objects_extensions, template):
        for i in range(self.NumberOfTimePoints):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for j, elt in enumerate(objects_names):
                names.append(elt + "_t=" + str(i) + objects_extensions[j])
            deformedPoints = self.LandmarkPointsT[i]
            aux_points = template.GetData()
            template.SetData(deformedPoints.data.numpy())
            template.Write(names)
            # restauring state of the template object for further computations
            template.SetData(aux_points)
