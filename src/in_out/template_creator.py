import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel

from os.path import splitext



# Creates a longitudinal dataset object from xml parameters.
def create_template_metadata(templateSpecifications):

    objects_list = []
    objects_name = []
    objects_noise_variance = []
    objects_name_extension = []
    objects_norm = []
    objects_norm_kernel_type = []
    objects_norm_kernel_width = []

    for object_id, object in templateSpecifications.items():
        filename = object['Filename']
        objectType = object['DeformableObjectType'].lower()

        root, extension = splitext(filename)
        reader = DeformableObjectReader()

        objects_list.append(reader.CreateObject(filename, objectType))
        objects_name.append(object_id)
        objects_name_extension.append(extension)
        objects_noise_variance.append(object['NoiseStd']**2)

        if objectType == 'OrientedSurfaceMesh'.lower():
            objects_norm.append('Current')
            objects_norm_kernel_type.append(object['KernelType'])
            objects_norm_kernel_width.append(float(object['KernelWidth']))
        elif objectType == 'NonOrientedSurfaceMesh'.lower():
            objects_norm.append('Varifold')
            objects_norm_kernel_type.append(object['KernelType'])
            objects_norm_kernel_width.append(float(object['KernelWidth']))
        else:
            raise RuntimeError('In DeterminiticAtlas.InitializeTemplateAttributes: '
                               'unknown object type: ' + objectType)

    multi_object_attachment = MultiObjectAttachment()
    multi_object_attachment.attachment_types = objects_norm
    for k in range(len(objects_norm)):
        multi_object_attachment.kernels.append(
            create_kernel(objects_norm_kernel_type[k], objects_norm_kernel_width[k]))

    return objects_list, objects_name, objects_name_extension, objects_noise_variance, objects_norm, multi_object_attachment