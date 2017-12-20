import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import warnings
import time

from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import *
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.in_out.utils import *
from torch.autograd import Variable
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel


def run_shooting(xml_parameters):
    
    print('[ run_shooting function ]')
    print('')
    
    """
    Create the template object
    """
    
    t_list, t_name, t_name_extension, t_noise_variance, multi_object_attachment = \
        create_template_metadata(xml_parameters.template_specifications)
    
    print("Object list:", t_list)
    
    template = DeformableMultiObject()
    template.object_list = t_list
    template.update()
    
    """
    Reading Control points and momenta
    """
    
    # if not (os.path.exists(Settings().output_dir)): Settings().output_dir
    
    
    if not xml_parameters.initial_control_points is None:
        control_points = read_2D_array(xml_parameters.initial_control_points)
    else:
        raise ArgumentError('Please specify a path to control points to perform a shooting')
    
    if not xml_parameters.initial_momenta is None:
        momenta = read_momenta(xml_parameters.initial_momenta)
    else:
        raise ArgumentError('Please specify a path to momenta to perform a shooting')
    
    template_data_numpy = template.get_data()
    template_data_torch = Variable(torch.from_numpy(template_data_numpy))
    
    momenta_torch = Variable(torch.from_numpy(momenta))
    control_points_torch = Variable(torch.from_numpy(control_points))
    
    exp = Exponential()
    exp.set_initial_control_points(control_points_torch)
    exp.set_initial_template_data(template_data_torch)
    exp.number_of_time_points = 10
    exp.kernel = create_kernel(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width)
    
    
    for i in range(len(momenta_torch)):
        exp.set_initial_momenta(momenta_torch[i])
        exp.update()
        deformedPoints = exp.get_template_data()
        names = [elt + "_"+ str(i) for elt in t_name]
        exp.write_flow(names, t_name_extension, template)
        exp.write_control_points_and_momenta_flow("Shooting_"+str(i))
    
    
    











