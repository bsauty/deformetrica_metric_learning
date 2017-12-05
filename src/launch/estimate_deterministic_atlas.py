import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism
from pydeformetrica.src.core.model_tools.attachments.landmarks_attachments import OrientedSurfaceDistance
import torch
from torch.autograd import Variable

"""
Basic info printing.

"""

print('')
print('##############################')
print('##### PyDeformetrica 1.0 #####')
print('##############################')
print('')

print('[ estimate_deterministic_atlas function ]')
print('')


"""
Read command line, read xml files, create dataset object.

"""

assert len(sys.argv) >= 4, "Usage: " + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
datasetXmlPath = sys.argv[2]
optimizationParametersXmlPath = sys.argv[3]

xmlParameters = XmlParameters()
xmlParameters.ReadAllXmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

datasetCreator = DatasetCreator()
dataset = datasetCreator.CreateDataset(xmlParameters.DatasetFilenames, xmlParameters.VisitAges,
                                       xmlParameters.SubjectIds, xmlParameters.TemplateSpecifications)

"""
Create the model object.

"""

model = DeterministicAtlas()

model.Diffeomorphism.KernelType = xmlParameters.DeformationKernelType
model.Diffeomorphism.KernelWidth = xmlParameters.DeformationKernelWidth
model.Diffeomorphism.NumberOfTimePoints = xmlParameters.NumberOfTimePoints

model.InitializeTemplateAttributes(xmlParameters.TemplateSpecifications)

model.SmoothingKernelWidth = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.InitialCpSpacing = xmlParameters.InitialCpSpacing
model.NumberOfSubjects = dataset.NumberOfSubjects

model.Update()

# """
# Create the estimator object.
#
# """
#
# estimator = GradientAscent()
#
# estimator.MaxIterations = xmlParameters.MaxIterations
# estimator.PrintEveryNIters = xmlParameters.PrintEveryNIters
# estimator.SaveEveryNIters = xmlParameters.SaveEveryNIters
# estimator.InitialStepSize = xmlParameters.InitialStepSize
# estimator.MaxLineSearchIterations = xmlParameters.MaxLineSearchIterations
# estimator.LineSearchShrink = xmlParameters.LineSearchShrink
# estimator.LineSearchExpand = xmlParameters.LineSearchExpand
# estimator.ConvergenceTolerance = xmlParameters.ConvergenceTolerance
#
# estimator.Dataset = dataset
# estimator.StatisticalModel = model
#
# """
# Launch.
#
# """
#
# model.Name = 'DeterministicAtlas'
# estimator.Update()

cp = Variable(torch.from_numpy(model.GetControlPoints()), requires_grad=True)
mom = Variable(torch.from_numpy(model.GetMomenta()), requires_grad=True)
templatePoints = Variable(torch.from_numpy(model.GetTemplateData().RawMatrixList[0]), requires_grad = True)#matrix list
templateObject = model.Template.ObjectList[0]
subjects = dataset.DeformableObjects
subjectsData = [Variable(torch.from_numpy(elt[0][0].GetData())) for elt in subjects]
subjects = [elt[0][0] for elt in subjects]


def cost(cp, mom, templatePoints):
    #for each subject, get phi(cp[i], mom[i], template)
    #get the attachment and the deformation norm
    #add the two
    penalty = 0.
    attachment = 0.
    for i,elt in enumerate(subjectsData):
        diffeo = Diffeomorphism()
        diffeo.SetKernelWidth(xmlParameters.DeformationKernelWidth)
        diffeo.SetStartPositions(cp)
        diffeo.SetStartMomenta(mom[i])
        diffeo.SetLandmarkPoints(templatePoints)
        diffeo.Shoot()
        diffeo.Flow()
        deformedPoints = diffeo.GetLandmarkPoints()
        penalty += diffeo.GetNorm()
        print("a")
        attachment += OrientedSurfaceDistance(deformedPoints, elt, templateObject, subjects[i], kernel_width=10.)
        print("b")
    return penalty + model.ObjectsNoiseVariance[0] * attachment

c = cost(cp, mom, templatePoints)
print(torch.autograd.grad(c, mom))
