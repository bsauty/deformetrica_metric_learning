import support.kernels as kernel_factory
from core.model_tools.deformations.geodesic import Geodesic
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata
from support.utilities.general_settings import *
import torch



def run_shooting(xml_parameters):

    import logging
    logger = logging.getLogger(__name__)

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
    
    if not xml_parameters.initial_control_points is None:
        control_points = read_2D_array(xml_parameters.initial_control_points)
    else:
        raise ArgumentError('Please specify a path to control points to perform a shooting')
    
    if not xml_parameters.initial_momenta is None:
        momenta = read_3D_array(xml_parameters.initial_momenta)
    else:
        raise ArgumentError('Please specify a path to momenta to perform a shooting')

    _, b = control_points.shape
    assert Settings().dimension == b, 'Please set the correct dimension in the model.xml file.'
    
    momenta_torch = torch.from_numpy(momenta)
    control_points_torch = torch.from_numpy(control_points)

    template_points = {key: torch.from_numpy(value).type(Settings().tensor_scalar_type) for key, value in template.get_points().items()}

    geodesic = Geodesic()

    if xml_parameters.t0 is None:
        logger.warning('Defaulting geodesic t0 to 1.')
        geodesic.t0 = 0.
    else:
        geodesic.t0 = xml_parameters.t0

    if xml_parameters.tmax == - float('inf'):
        logger.warning('Defaulting geodesic tmax to 1.')
        geodesic.tmax = 1.
    else:
        geodesic.tmax = xml_parameters.tmax

    if xml_parameters.tmin == float('inf'):
        logger.warning('Defaulting geodesic tmin to 0.')
        geodesic.tmin = 0.
    else:
        geodesic.tmin = xml_parameters.tmin

    assert geodesic.tmax >= geodesic.t0, 'The max time {} for the shooting should be larger than t0 {}'\
        .format(geodesic.tmax, geodesic.t0)
    assert geodesic.tmin <= geodesic.t0, 'The min time for the shooting should be lower than t0.'\
        .format(geodesic.tmin, geodesic.t0)

    geodesic.set_control_points_t0(control_points_torch)
    geodesic.concentration_of_time_points = xml_parameters.concentration_of_time_points
    geodesic.set_kernel(kernel_factory.factory(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))
    geodesic.set_use_rk2(xml_parameters.use_rk2)
    geodesic.set_template_points_t0(template_points)

    # Single momenta: single shooting
    if len(momenta.shape) == 2:
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.update()
        names = [elt for elt in t_name]
        geodesic.write('Shooting', names, t_name_extension, template, template.get_data())

    # Several shootings to compute
    else:
        for i in range(len(momenta_torch)):
            geodesic.set_momenta_t0(momenta_torch[i])
            geodesic.update()
            names = [elt for elt in t_name]
            geodesic.write('Shooting' + "_" + str(i), names, t_name_extension, template, template.get_data(), write_adjoint_parameters=True)



    











