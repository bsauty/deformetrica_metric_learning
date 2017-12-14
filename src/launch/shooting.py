import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import warnings
import time

from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel

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
xmlParameters.read_all_xmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

Settings().dimension = xmlParameters.Dimension




"""
Create the template object
"""

templateCreator = TemplateCreator()
t_list, t_name, t_name_extension, t_noise_variance, t_norm, t_norm_kernel_type, t_norm_kernel_width = \
    templateCreator.create_template(xmlParameters.TemplateSpecifications)

template = DeformableMultiObject()
template.ObjectList = t_list
template.update()

"""
Reading Control points and momenta
"""

if not (os.path.exists(Settings().output_dir)):
    Settings().output_dir


if not xmlParameters.InitialControlPoints is None:
    control_points = read_2D_array(xmlParameters.InitialControlPoints)
else:
    raise ArgumentError('Please specify a path to control points to perform a shooting')

if not xmlParameters.InitialMomenta is None:
    momenta = read_momenta(xmlParameters.InitialMomenta)
else:
    raise ArgumentError('Please specify a path to momenta to perform a shooting')

templateDataNumpy = template.get_data()
templateDataTorch = Variable(torch.from_numpy(self.FixedEffects['ControlPoints']))

momenta_torch = Variable(torch.from_numpy(control_points))
control_points_torch = Variable(torch.from_numpy(momenta))

diffeo = Exponential()
model.diffeomorphism.kernel = create_kernel(xml_parameters.deformation_kernel_type,
                                            xml_parameters.deformation_kernel_width)
diffeo.set_initial_control_points(controlPoints)
diffeo.set_initial_template_data(templateDataTorch)


for i in range(len(momenta)):
    self.diffeomorphism.initial_momenta = momenta[i]
    self.Diffeomorphism.shoot()
    self.Diffeomorphism.flow()
    deformedPoints = self.Diffeomorphism.get_template_data()
    names = [elt + "_"+ str(i) for elt in self.ObjectsName]
    diffeo.write_flow(names, t_name_extension, template)














