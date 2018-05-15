from torch.autograd import Variable

from core.model_tools.deformations.exponential import Exponential
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata
import support.kernel as kernel_factory
from support.utilities.general_settings import *


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
        momenta = read_3D_array(xml_parameters.initial_momenta)
    else:
        raise ArgumentError('Please specify a path to momenta to perform a shooting')
    
    template_data_numpy = template.get_points()
    template_data_torch = Variable(torch.from_numpy(template_data_numpy))
    
    momenta_torch = Variable(torch.from_numpy(momenta))
    control_points_torch = Variable(torch.from_numpy(control_points))
    
    exp = Exponential()
    exp.set_initial_control_points(control_points_torch)
    exp.set_initial_template_data(template_data_torch)
    exp.number_of_time_points = 10
    exp.kernel = kernel_factory.factory(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width)
    exp.set_use_rk2(xml_parameters.use_rk2)
    
    for i in range(len(momenta_torch)):
        exp.set_initial_momenta(momenta_torch[i])
        exp.update()
        deformedPoints = exp.get_template_data()
        names = [elt + "_"+ str(i) for elt in t_name]
        exp.write_flow(names, t_name_extension, template)
        exp.write_control_points_and_momenta_flow("Shooting_"+str(i))
    
    
    











