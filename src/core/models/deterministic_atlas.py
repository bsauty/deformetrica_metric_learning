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
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import ComputeMultiObjectWeightedDistance

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

        # Numpy arrays.
        self.FixedEffects = {}
        self.FixedEffects['TemplateData'] = None
        self.FixedEffects['ControlPoints'] = None
        self.FixedEffects['Momenta'] = None

        self.FreezeTemplate = False
        self.FreezeControlPoints = False


    ################################################################################
    ### Encapsulation methods:
    ################################################################################

    # Those methods do the numpy/torch conversion.
    def GetTemplateData(self):
        if self.FreezeTemplate:
            return Variable(torch.from_numpy(self.FixedEffects['TemplateData']))
        else:
            return Variable(torch.from_numpy(self.FixedEffects['TemplateData']), requires_grad=True)
    def SetTemplateData(self, td):
        self.FixedEffects['TemplateData'] = td.data.numpy()
        self.Template.SetData(self.FixedEffects['TemplateData'])
    # def SetTemplateData_Numpy(self, td):
    #     self.FixedEffects['TemplateData'] = td
    #     self.Template.SetPoints(td)

    def GetControlPoints(self):
        if self.FreezeControlPoints:
            return Variable(torch.from_numpy(self.FixedEffects['ControlPoints']), requires_grad = False)
        else:
            return Variable(torch.from_numpy(self.FixedEffects['ControlPoints']), requires_grad = True)
    def SetControlPoints(self, cp):
        self.FixedEffects['ControlPoints'] = cp.data.numpy()
    def GetControlPointsNumpy(self):
        return self.FixedEffects['ControlPoints']

    def GetMomenta(self):
        return Variable(torch.from_numpy(self.FixedEffects['Momenta']), requires_grad = True)
    def SetMomenta(self, mom):
        self.FixedEffects['Momenta'] = mom.data.numpy()
    def GetMomentaNumpy(self):
        return self.FixedEffects['Momenta']

    # From vectorized torch tensor.
    def SetFixedEffects(self, fixedEffects):
        td, cp, mom = self.UnvectorizeFixedEffects(fixedEffects)
        self.SetTemplateData(td)
        self.SetControlPoints(cp)
        self.SetMomenta(mom)


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

    # Compute the functional. Fully torch function.
    def ComputeLogLikelihood(self, dataset, fixedEffects, popRER, indRER):

        # Initialize ---------------------------------------------------------------
        templateData, controlPoints, momenta = self.UnvectorizeFixedEffects(fixedEffects)
        targets = dataset.DeformableObjects
        targets = [target[0] for target in targets] # Cross-sectional data.
        targetsData = [Variable(torch.from_numpy(target.GetData())) for target in targets]

        # Deform -------------------------------------------------------------------
        regularity = 0.
        attachment = 0.

        self.Diffeomorphism.SetLandmarkPoints(templateData)
        for i, targetData in enumerate(targetsData):
            self.Diffeomorphism.SetStartPositions(controlPoints)
            self.Diffeomorphism.SetStartMomenta(momenta[i])
            self.Diffeomorphism.Shoot()
            self.Diffeomorphism.Flow()
            deformedPoints = self.Diffeomorphism.GetLandmarkPoints()
            regularity -= self.Diffeomorphism.GetNorm()
            attachment -= ComputeMultiObjectWeightedDistance(
                deformedPoints, targetData, self.Template, targets[i],
                self.ObjectsNormKernelWidth, self.ObjectsNoiseVariance)
        return attachment, regularity

    # Numpy input, torch output.
    def GetVectorizedFixedEffects(self):
        # Numpy arrays.
        templateData = self.GetTemplateData()
        controlPoints = self.GetControlPoints()
        momenta = self.GetMomenta()

        return [templateData, controlPoints, momenta]

    # Fully torch method.
    def UnvectorizeFixedEffects(self, fixedEffects):
        return fixedEffects[0], fixedEffects[1], fixedEffects[2]


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

        # self.SetTemplateData_Numpy(self.Template.GetData().Concatenate())


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
        momenta = np.zeros((self.NumberOfSubjects, self.NumberOfControlPoints, GeneralSettings.Instance().Dimension))
        self.FixedEffects['Momenta'] = momenta
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
<<<<<<< HEAD
        self.Template.SetData(self.GetTemplateData().data.numpy())#because it's not automatic !
=======
        # self.Template.SetData(self.GetTemplateData()) #because it's not automatic !
>>>>>>> d385e3646e2824214e5f905d584ff06e131f77de
        templateNames = []
        for i in range(len(self.ObjectsName)):
            aux = "Atlas_" + self.ObjectsName[i] + self.ObjectsNameExtension[i]
            templateNames.append(aux)
        self.Template.Write(templateNames)

    def WriteControlPoints(self):
        saveArray(self.GetControlPointsNumpy(), "Atlas_ControlPoints.txt")

    def WriteMomenta(self):
        saveMomenta(self.GetMomentaNumpy(), "Atlas_Momenta.txt")

    def WriteTemplateToSubjectsTrajectories(self, dataset):
        td = self.GetTemplateData()
        cp = self.GetControlPoints()
        mom = self.GetMomenta()

        self.Diffeomorphism.SetStartPositions(cp)
        self.Diffeomorphism.SetLandmarkPoints(td)
        for i,subject in enumerate(dataset.DeformableObjects):
            names = [elt + "_to_subject_"+str(i) for elt in self.ObjectsName]
            self.Diffeomorphism.SetStartMomenta(mom[i])
            self.Diffeomorphism.Shoot()
            self.Diffeomorphism.Flow()
            self.Diffeomorphism.WriteFlow(names, self.ObjectsNameExtension, self.Template)

    def Write(self, dataset):
        #We save the template, the cp, the mom and the trajectories
        self.WriteTemplate()
        self.WriteControlPoints()
        self.WriteMomenta()
        self.WriteTemplateToSubjectsTrajectories(dataset)
