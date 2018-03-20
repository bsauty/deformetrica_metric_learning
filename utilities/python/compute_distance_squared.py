import os
import sys

import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.support.utilities.general_settings import Settings


def compute_distance_squared(path_to_mesh_1, path_to_mesh_2, deformable_object_type, attachment_type,
                             kernel_width=None):
    reader = DeformableObjectReader()
    object_1 = reader.create_object(path_to_mesh_1, deformable_object_type.lower())
    object_2 = reader.create_object(path_to_mesh_2, deformable_object_type.lower())

    multi_object_1 = DeformableMultiObject()
    multi_object_1.object_list.append(object_1)
    multi_object_1.update()

    multi_object_2 = DeformableMultiObject()
    multi_object_2.object_list.append(object_2)
    multi_object_2.update()

    multi_object_attachment = MultiObjectAttachment()
    multi_object_attachment.attachment_types.append(attachment_type.lower())
    multi_object_attachment.kernels.append(create_kernel('exact', kernel_width))

    return multi_object_attachment.compute_distances(
        Variable(torch.from_numpy(multi_object_1.get_points()).type(Settings().tensor_scalar_type)),
        multi_object_1, multi_object_2).data.cpu().numpy()[0]


if __name__ == '__main__':

    """
    Basic info printing.
    """

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')
    print('')

    """
    Read command line.
    """

    assert len(sys.argv) in [5, 6], \
        'Usage: ' + sys.argv[0] \
        + " <path_to_mesh_1.vtk> <path_to_mesh_2.vtk> <deformable_object_type> <attachment_type> " \
          "[optional kernel_width]"

    path_to_mesh_1 = sys.argv[1]
    path_to_mesh_2 = sys.argv[2]
    deformable_object_type = sys.argv[3]
    attachment_type = sys.argv[4]

    kernel_width = None
    if len(sys.argv) == 6:
        kernel_width = float(sys.argv[5])

    if not os.path.isfile(path_to_mesh_1):
        raise RuntimeError('The specified source file ' + path_to_mesh_1 + ' does not exist.')
    if not os.path.isfile(path_to_mesh_2):
        raise RuntimeError('The specified source file ' + path_to_mesh_2 + ' does not exist.')

    """
    Core part.
    """

    print(compute_distance_squared(
        path_to_mesh_1, path_to_mesh_2, deformable_object_type, attachment_type, kernel_width))
