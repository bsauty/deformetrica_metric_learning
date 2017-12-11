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
from pydeformetrica.src.support.utilities.general_settings import *
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

        # Torch arrays.
        self.FixedEffects = {}
        self.FixedEffects['TemplateData'] = None
        self.FixedEffects['ControlPoints'] = None
        self.FixedEffects['Momenta'] = None

        self.FreezeTemplate = False
        self.FreezeControlPoints = False


    ################################################################################
    ### Encapsulation methods:
    ################################################################################

    # Template data ----------------------------------------------------------------
    def GetTemplateData(self):
        return self.FixedEffects['TemplateData']

    def SetTemplateData(self, td):
        self.FixedEffects['TemplateData'] = td
        self.Template.SetData(td.data.numpy())
    def SetTemplateData_FromNumpy(self, td):
        if self.FreezeTemplate:
            self.FixedEffects['TemplateData'] = Variable(
                torch.from_numpy(td).type(Settings().TensorType), requires_grad=False)
        else:
            self.FixedEffects['TemplateData'] = Variable(
                torch.from_numpy(td).type(Settings().TensorType), requires_grad=True)

    # Control points ---------------------------------------------------------------
    def GetControlPoints(self):
        return self.FixedEffects['ControlPoints']
    def GetControlPoints_ToNumpy(self):
        return self.FixedEffects['ControlPoints'].data.numpy()

    def SetControlPoints(self, cp):
        self.FixedEffects['ControlPoints'] = cp
    def SetControlPoints_FromNumpy(self, cp):
        if self.FreezeControlPoints:
            self.FixedEffects['ControlPoints'] = Variable(
                torch.from_numpy(cp).type(Settings().TensorType), requires_grad=False)
        else:
            self.FixedEffects['ControlPoints'] = Variable(
                torch.from_numpy(cp).type(Settings().TensorType), requires_grad=True)

    # Momenta ----------------------------------------------------------------------
    def GetMomenta(self):
        return self.FixedEffects['Momenta']
    def GetMomenta_ToNumpy(self):
        return self.FixedEffects['Momenta'].data.numpy()

    def SetMomenta(self, mom):
        self.FixedEffects['Momenta'] = mom
    def SetMomenta_FromNumpy(self, mom):
        self.FixedEffects['Momenta'] = Variable(torch.from_numpy(mom).type(Settings().TensorType), requires_grad=True)

    # Full fixed effects ------------------------------------------------------------
    def GetFixedEffects(self):
        return self.FixedEffects
    def SetFixedEffects(self, fixedEffects):
        self.SetTemplateData(fixedEffects['TemplateData'])
        self.SetControlPoints(fixedEffects['ControlPoints'])
        self.SetMomenta(fixedEffects['Momenta'])

    ################################################################################
    ### Public methods:
    ################################################################################

    # Final initialization steps.
    def Update(self):

        self.Template.Update()

        self.NumberOfObjects = len(self.Template.ObjectList)
        self.BoundingBox = self.Template.BoundingBox

        self.SetTemplateData_FromNumpy(self.Template.GetData())
        if self.FixedEffects['ControlPoints'] is None: self.InitializeControlPoints()
        else: self.InitializeBoundingBox()
        if self.FixedEffects['Momenta'] is None: self.InitializeMomenta()

    # # Compute the functional. Fully torch function.
    # def ComputeLogLikelihood(self, dataset, fixedEffects, popRER, indRER):
    #
    #     # Initialize ---------------------------------------------------------------
    #     templateData = fixedEffects['TemplateData']
    #     controlPoints = fixedEffects['ControlPoints']
    #     momenta = fixedEffects['Momenta']
    #
    #     targets = dataset.DeformableObjects
    #     targets = [target[0] for target in targets] # Cross-sectional data.
    #     # targetsData = [Variable(torch.from_numpy(target.GetData()).type(Settings().TensorType)) for target in targets]
    #
    #     # Deform -------------------------------------------------------------------
    #     regularity = 0.
    #     attachment = 0.
    #
    #     self.Diffeomorphism.SetLandmarkPoints(templateData)
    #     self.Diffeomorphism.SetInitialControlPoints(controlPoints)
    #     for i, target in enumerate(targets):
    #         self.Diffeomorphism.SetInitialMomenta(momenta[i])
    #         self.Diffeomorphism.Shoot()
    #         self.Diffeomorphism.Flow()
    #         deformedPoints = self.Diffeomorphism.GetLandmarkPoints()
    #         regularity -= self.Diffeomorphism.GetNorm()
    #         attachment -= ComputeMultiObjectWeightedDistance(
    #             deformedPoints, self.Template, target,
    #             self.ObjectsNormKernelWidth, self.ObjectsNoiseVariance, self.ObjectsNorm)
    #     return attachment, regularity

    # Compute the functional. Numpy input/outputs.
    def ComputeLogLikelihood(self, dataset, theta, z_pop, z_ind, with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters theta and random effects realizations
        z_pop and z_ind.

        :param dataset: LongitudinalDataset instance
        :param theta: Dictionary of linearized fixed effects.
        :param z_pop: Dictionary of linearized population random effects realizations.
        :param z_ind: Dictionary of linearized individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: unvectorize + conversion from numpy to torch -----------------------------------------------------
        templateData = theta['TemplateData'].reshape(self.FixedEffects['TemplateData'].shape)
        if with_grad and not(self.FreezeTemplate):
            templateData = Variable(torch.from_numpy(templateData), requires_grad=True)
        else:
            templateData = torch.from_numpy(templateData)

        controlPoints = theta['ControlPoints'].reshape(self.FixedEffects['ControlPoints'].shape)
        if with_grad and not(self.FreezeControlPoints):
            controlPoints = Variable(torch.from_numpy(controlPoints), requires_grad=True)
        else:
            controlPoints = torch.from_numpy(controlPoints)

        momenta = theta['Momenta'].reshape(self.FixedEffects['Momenta'].shape)
        if with_grad:
            momenta = Variable(torch.from_numpy(momenta), requires_grad=True)
        else:
            momenta = torch.from_numpy(momenta)

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.DeformableObjects
        targets = [target[0] for target in targets]

        # Deform -------------------------------------------------------------------------------------------------------
        regularity = 0.
        attachment = 0.

        self.Diffeomorphism.SetLandmarkPoints(templateData)
        self.Diffeomorphism.SetInitialControlPoints(controlPoints)
        for i, target in enumerate(targets):
            self.Diffeomorphism.SetInitialMomenta(momenta[i])
            self.Diffeomorphism.Shoot()
            self.Diffeomorphism.Flow()
            deformedPoints = self.Diffeomorphism.GetLandmarkPoints()
            regularity -= self.Diffeomorphism.GetNorm()
            attachment -= ComputeMultiObjectWeightedDistance(
                deformedPoints, self.Template, target,
                self.ObjectsNormKernelWidth, self.ObjectsNoiseVariance, self.ObjectsNorm)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {
                'TemplateData': templateData.grad if templateData.requires_grad,
                'ControlPoints': controlPoints.grad if controlPoints.requires_grad,
                'Momenta': momenta.grad if momenta.requires_grad
            }

            return attachment, regularity, gradient

        else:
            return attachment, regularity, None


    def ConvolveGradTemplate(gradTemplate):
        """
        Smoothing of the template gradient (for landmarks)
        """
        gradTemplateSob = []

        kernel = TorchKernel()
        kernel.KernelWidth = self.SmoothingKernelWidth
        tempData = self.GetTemplateData()
        pos = 0
        for elt in tempData:
            #TODO : assert if data is image or not.
            gradTemplateSob.append(kernel.Convolve(tempData, gradTemplate[pos:pos+len(tempData)], tempData))
            pos += len(tempData)
        return gradTemplate

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


    # Initialize the control points fixed effect.
    def InitializeControlPoints(self):
        dimension = Settings().Dimension

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

        self.SetControlPoints_FromNumpy(controlPoints)
        print('>> Set of ' + str(self.NumberOfControlPoints) + ' control points defined.')

    # Initialize the momenta fixed effect.
    def InitializeMomenta(self):
        assert(self.NumberOfSubjects > 0)
        momenta = np.zeros((self.NumberOfSubjects, self.NumberOfControlPoints, GeneralSettings.Instance().Dimension))
        self.SetMomenta_FromNumpy(momenta)
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
        self.Template.SetData(self.GetTemplateData().data.numpy())#because it's not automatic !
        templateNames = []
        for i in range(len(self.ObjectsName)):
            aux = "Atlas_" + self.ObjectsName[i] + self.ObjectsNameExtension[i]
            templateNames.append(aux)
        self.Template.Write(templateNames)

    def WriteControlPoints(self):
        saveArray(self.GetControlPoints_ToNumpy(), "Atlas_ControlPoints.txt")

    def WriteMomenta(self):
        saveMomenta(self.GetMomenta_ToNumpy(), "Atlas_Momenta.txt")

    def WriteTemplateToSubjectsTrajectories(self, dataset):
        td = self.GetTemplateData()
        cp = self.GetControlPoints()
        mom = self.GetMomenta()

        self.Diffeomorphism.SetInitialControlPoints(cp)
        self.Diffeomorphism.SetLandmarkPoints(td)
        for i, subject in enumerate(dataset.DeformableObjects):
            names = [elt + "_to_subject_"+str(i) for elt in self.ObjectsName]
            self.Diffeomorphism.SetInitialMomenta(mom[i])
            self.Diffeomorphism.Shoot()
            self.Diffeomorphism.Flow()
            self.Diffeomorphism.WriteFlow(names, self.ObjectsNameExtension, self.Template)

    def Write(self, dataset):
        #We save the template, the cp, the mom and the trajectories
        self.WriteTemplate()
        self.WriteControlPoints()
        self.WriteMomenta()
        self.WriteTemplateToSubjectsTrajectories(dataset)
