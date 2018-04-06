import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from torch.autograd import Variable
import torch

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.core.model_tools.deformations.spatiotemporal_reference_frame import SpatiotemporalReferenceFrame
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from src.in_out.array_readers_and_writers import *

if __name__ == '__main__':

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')
    print('')

    """
    Read command line, prepare output folder, read model xml file.
    """

    assert len(sys.argv) >= 2, 'Usage: ' + sys.argv[0] + " <model.xml> <optional --output-dir=path_to_output>"

    model_xml_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_dir = sys.argv[2][len("--output-dir="):]
        print(">> Setting output directory to:", output_dir)
    else:
        output_dir = 'output'

    if not os.path.exists(output_dir):
        print('>> Creating the output directory: "' + output_dir + '"')
        os.makedirs(output_dir)

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._further_initialization()

    """
    Load the template, control points, momenta, modulation matrix.
    """

    # Template.
    t_list, objects_name, objects_name_extension, _, _ = create_template_metadata(
        xml_parameters.template_specifications)

    template = DeformableMultiObject()
    template.object_list = t_list
    template.update()
    template_data_torch = Variable(torch.from_numpy(template.get_intensities()).type(Settings().tensor_scalar_type),
                                   requires_grad=False)

    # Control points.
    control_points = read_2D_array(xml_parameters.initial_control_points)
    print('>> Reading ' + str(len(control_points)) + ' initial control points from file: '
          + xml_parameters.initial_control_points)
    control_points_torch = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                    requires_grad=False)

    # Momenta.
    momenta = read_3D_array(xml_parameters.initial_momenta)
    print('>> Reading initial momenta from file: ' + xml_parameters.initial_momenta)
    momenta_torch = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=False)

    # Modulation matrix.
    modulation_matrix = read_2D_array(xml_parameters.initial_modulation_matrix)
    print('>> Reading ' + str(modulation_matrix.shape[1]) + '-source initial modulation matrix from file: '
          + xml_parameters.initial_modulation_matrix)
    modulation_matrix_torch = Variable(torch.from_numpy(modulation_matrix).type(Settings().tensor_scalar_type),
                                       requires_grad=False)

    """
    Instantiate the spatiotemporal reference frame, update and write.
    """

    spatiotemporal_reference_frame = SpatiotemporalReferenceFrame()

    spatiotemporal_reference_frame.set_kernel(create_kernel(xml_parameters.deformation_kernel_type,
                                                            xml_parameters.deformation_kernel_width))
    spatiotemporal_reference_frame.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)
    spatiotemporal_reference_frame.set_number_of_time_points(xml_parameters.number_of_time_points)
    spatiotemporal_reference_frame.set_use_rk2(xml_parameters.use_rk2)

    spatiotemporal_reference_frame.set_template_data_t0(template_data_torch)
    spatiotemporal_reference_frame.set_control_points_t0(control_points_torch)
    spatiotemporal_reference_frame.set_momenta_t0(momenta_torch)
    spatiotemporal_reference_frame.set_modulation_matrix_t0(modulation_matrix_torch)
    spatiotemporal_reference_frame.set_t0(xml_parameters.t0)
    spatiotemporal_reference_frame.set_tmin(xml_parameters.tmin)
    spatiotemporal_reference_frame.set_tmax(xml_parameters.tmax)
    spatiotemporal_reference_frame.update()

    spatiotemporal_reference_frame.write('SpatioTemporalReferenceFrame',
                                         objects_name, objects_name_extension, template,
                                         write_adjoint_parameters=True, write_exponential_flow=True)
