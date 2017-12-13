import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import warnings
import time

from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import *
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator
from pydeformetrica.src.in_out.template_creator import TemplateCreator
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism

"""
Basic info printing.

"""

print('')
print('##############################')
print('##### PyDeformetrica 1.0 #####')
print('##############################')
print('')

print('[ shooting function ]')
print('')



"""
Read command line, read xml files, set general settings.

"""

assert len(sys.argv) >= 4, "Usage: " + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
datasetXmlPath = sys.argv[2]
optimizationParametersXmlPath = sys.argv[3]

xmlParameters = XmlParameters()
xmlParameters.ReadAllXmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

Settings().Dimension = xmlParameters.Dimension




"""
Create the template object
"""

templateCreator = TemplateCreator()
t_list, t_name, t_name_extension, t_noise_variance, t_norm, t_norm_kernel_type, t_norm_kernel_width = \
    templateCreator.create_template(xmlParameters.TemplateSpecifications)

template = DeformableMultiObject()
template.ObjectList = t_list
template.Update()

"""
Reading Control points and momenta
"""

if not (os.path.exists(Settings().OutputDir)):
    Settings().OutputDir


if not xmlParameters.InitialControlPoints is None:
    control_points = read_2D_array(xmlParameters.InitialControlPoints)
else:
    raise ArgumentError('Please specify a path to control points to perform a shooting')

if not xmlParameters.InitialMomenta is None:
    momenta = read_momenta(xmlParameters.InitialMomenta)
else:
    raise ArgumentError('Please specify a path to momenta to perform a shooting')

templateDataNumpy = template.GetData()
templateDataTorch = Variable(torch.from_numpy(self.FixedEffects['ControlPoints']))

momenta_torch = Variable(torch.from_numpy(control_points))
control_points_torch = Variable(torch.from_numpy(momenta))

diffeo = Diffeomorphism()
diffeo.SetInitialControlPoints(controlPoints)
diffeo.SetLandmarkPoints(templateDataTorch)
diffeo.SetKernelWidth(xmlParameters.DeformationKernelWidth)
diffeo.SetKernelType(xmlParameters.DeformationKernelType)


for i in range(len(momenta)):
    self.Diffeomorphism.SetInitialMomenta(momenta[i])
    self.Diffeomorphism.Shoot()
    self.Diffeomorphism.Flow()
    deformedPoints = self.Diffeomorphism.GetLandmarkPoints()
    names = [elt + "_"+ str(i) for elt in self.ObjectsName]
    diffeo.WriteFlow(names, t_name_extension, template)














