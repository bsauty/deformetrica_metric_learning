import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from os.path import splitext
import numpy as np
import math

import torch
from torch.autograd import Variable

from pydeformetrica.src.core.models.abstract_statistical_model import AbstractStatisticalModel
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel
from pydeformetrica.src.in_out.utils import *

class DeterministicAtlas(AbstractStatisticalModel):

    """
    Deterministic atlas object class.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.Template = DeformableMultiObject()
        self.ObjectsName = []
        self.ObjectsNameExtension = []
        self.ObjectsNoiseVariance = []
        self.ObjectsNorm = []
        self.ObjectsNormKernelType = []
        self.ObjectsNormKernelWidth = []

        self.Diffeomorphism = Diffeomorphism()

        self.SmoothingKernelWidth = None
        self.InitialCpSpacing = None
        self.NumberOfSubjects = None
        self.NumberOfObjects = None
        self.NumberOfControlPoints = None
        self.BoundingBox = None

        self.FixedEffects = {}
        self.FixedEffects['TemplateData'] = None
        self.FixedEffects['ControlPoints'] = None
        self.FixedEffects['Momenta'] = None


    ################################################################################
    ### Encapsulation methods:
    ################################################################################

    def GetControlPoints(self): return self.FixedEffects['ControlPoints']
    def SetControlPoints(self, cp): self.FixedEffects['ControlPoints'] = cp

    def GetMomenta(self): return self.FixedEffects['Momenta']
    def SetMomenta(self, mom): self.FixedEffects['Momenta'] = mom

    def GetTemplateData(self): return self.FixedEffects['TemplateData']
    def SetTemplateData(self, td): self.FixedEffects['TemplateData'] = td

    ################################################################################
    ### Public methods:
    ################################################################################

    # Final initialization steps.
    def Update(self):

        self.Template.Update()

        self.NumberOfObjects = len(self.Template.ObjectList)
        self.BoundingBox = self.Template.BoundingBox

        self.FixedEffects['TemplateData'] = self.Template.GetData()

        if self.FixedEffects['ControlPoints'] is None: self.InitializeControlPoints()
        else: self.InitializeBoundingBox()
        if self.FixedEffects['Momenta'] is None: self.InitializeMomenta()

    # Compute the complete log-likelihood first mode, given an input random effects realization.
    def ComputeCompleteLogLikelihood(self, dataset, popRER, indRER, logLikelihoodTerms):

        # Initialization -----------------------------------------------------------
        logLikelihoodTerms = np.zeros((2, 1))
        controlPoints = Variable(torch.from_numpy(self.GetControlPoints()), requires_grad=True)
        momenta = Variable(torch.from_numpy(self.GetMomenta()), requires_grad=True)
        templateData = Variable(torch.from_numpy(self.GetTemplateData()), requires_grad=True)

        oob, residuals = self.ComputeResiduals(dataset, controlPoints, momenta, templateData)

        # Data (residuals) term ----------------------------------------------------
        for i in range(self.NumberOfSubjects):
            for k in range(self.NumberOfObjects):
                logLikelihoodTerms[0] -= residuals[i][k] / self.ObjectsNoiseVariance[k]

        # Regularity term (RKHS norm) ----------------------------------------------
        kernel = self.Diffeomorphism.Kernel

        for i in range(self.NumberOfSubjects):
            kMom = kernel.Convolve(controlPoints, momenta[i], controlPoints)



    # Same method than ComputeCompleteLogLikelihood for the DeterministicAtlas model.
    def UpdateFixedEffectsAndComputeCompleteLogLikelihood(self, dataset, popRER, indRER, logLikelihoodTerms):
        return self.ComputeCompleteLogLikelihood(dataset, popRER, indRER, logLikelihoodTerms)


    ################################################################################
    ### Private methods:
    ################################################################################

    # Sets the Template, TemplateObjectsName, TemplateObjectsNameExtension, TemplateObjectsNorm,
    # TemplateObjectsNormKernelType and TemplateObjectsNormKernelWidth attributes.
    def InitializeTemplateAttributes(self, templateSpecifications):

        for object_id, object in templateSpecifications.iteritems():
            filename = object['Filename']
            objectType = object['DeformableObjectType'].lower()

            root, extension = splitext(filename)
            reader = DeformableObjectReader()

            self.Template.ObjectList.append(reader.CreateObject(filename, objectType))
            self.ObjectsName.append(object_id)
            self.ObjectsNameExtension.append(extension)
            self.ObjectsNoiseVariance.append(object['NoiseStd']**2)

            if objectType == 'OrientedSurfaceMesh'.lower():
                self.ObjectsNorm.append('Current')
                self.ObjectsNormKernelType.append(object['KernelType'])
                self.ObjectsNormKernelWidth.append(float(object['KernelWidth']))
            elif objectType == 'NonOrientedSurfaceMesh'.lower():
                self.ObjectsNorm.append('Varifold')
                self.ObjectsNormKernelType.append(object['KernelType'])
                self.ObjectsNormKernelWidth.append(float(object['KernelWidth']))
            else:
                raise RuntimeError('In DeterminiticAtlas.InitializeTemplateAttributes: '
                                   'unknown object type: ' + objectType)

        self.SetTemplateData(self.Template.GetData())


    # Initialize the control points fixed effect.
    def InitializeControlPoints(self):
        dimension = GeneralSettings.Instance().Dimension

        axis = []
        for d in range(dimension):
            min = self.BoundingBox[d, 0]
            max = self.BoundingBox[d, 1]
            length = max - min
            assert(length > 0)

            offset = 0.5 * (length - self.InitialCpSpacing * math.floor(length)) / self.InitialCpSpacing
            axis.append(np.arange(min + offset, max, self.InitialCpSpacing))

        if dimension == 2:
            xAxis, yAxis = np.meshgrid(axis[0], axis[1])

            assert (xAxis.shape == yAxis.shape)
            self.NumberOfControlPoints = xAxis.flatten().shape[0]
            controlPoints = np.zeros((self.NumberOfControlPoints, dimension))

            controlPoints[:, 0] = xAxis.flatten()
            controlPoints[:, 1] = yAxis.flatten()

        elif dimension == 3:
            xAxis, yAxis, zAxis = np.meshgrid(axis[0], axis[1], axis[2])

            assert (xAxis.shape == yAxis.shape)
            assert (xAxis.shape == zAxis.shape)
            self.NumberOfControlPoints = xAxis.flatten().shape[0]
            controlPoints = np.zeros((self.NumberOfControlPoints, dimension))

            controlPoints[:, 0] = xAxis.flatten()
            controlPoints[:, 1] = yAxis.flatten()
            controlPoints[:, 2] = zAxis.flatten()

        else:
            raise RuntimeError('In DeterministicAtlas.InitializeControlPoints: invalid ambient space dimension.')

        self.FixedEffects['ControlPoints'] = controlPoints
        print('>> Set of ' + str(self.NumberOfControlPoints) + ' control points defined.')

    # Initialize the momenta fixed effect.
    def InitializeMomenta(self):
        assert(self.NumberOfSubjects > 0)
        self.FixedEffects['Momenta'] = np.zeros((self.NumberOfSubjects, self.NumberOfControlPoints, GeneralSettings.Instance().Dimension))
        print('>> Deterministic atlas momenta initialized to zero, for ' + str(self.NumberOfSubjects) + ' subjects.')

    # Initialize the bounding box. which tightly encloses all template objects and the atlas control points.
    # Relevant when the control points are given by the user.
    def InitializeBoundingBox(self):
        assert(self.NumberOfControlPoints > 0)

        dimension = GeneralSettings.Instance().Dimension
        controlPoints = self.GetControlPoints()

        for k in range(self.NumberOfControlPoints):
            for d in range(dimension):
                if controlPoints[k, d] < self.BoundingBox[d, 0]: self.BoundingBox[d, 0] = controlPoints[k, d]
                elif controlPoints[k, d] > self.BoundingBox[d, 1]: self.BoundingBox[d, 1] = controlPoints[k, d]

    def WriteTemplate(self):
        self.Template.SetPoints(self.GetTemplateData())#because it's not automatic !
        templateNames = []
        for i in range(len(self.ObjectsName)):
            aux = "Atlas_" + self.ObjectsName[i] + self.ObjectsNameExtension[i]
            aux = os.path.join(GeneralSettings.Instance().OutputDir, aux)
            templateNames.append(aux)
        self.Template.Write(templateNames)

    def WriteControlPoints(self):
        saveArray(self.GetControlPoints(), "Atlas_ControlPoints.txt")

    def WriteMomenta(self):
        saveMomenta(self.GetMomenta(), "Atlas_Momenta.txt")

    def WriteTemplateToSubjectsTrajectories(self):
        pass

    def Write(self):
        #We save the template, the cp, the mom and the trajectories
        self.WriteTemplate()
        self.WriteControlPoints()
        self.WriteMomenta()
        self.WriteTemplateToSubjectsTrajectories()
