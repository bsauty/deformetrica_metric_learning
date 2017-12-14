import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from os.path import splitext
import warnings
from pydeformetrica.src.core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel


def create_dataset(dataset_filenames, visit_ages, subject_ids, template_specifications):
    """
    Creates a longitudinal dataset object from xml parameters. 
    """

    deformable_objects_dataset = []
    for i in range(len(dataset_filenames)):
        deformable_objects_subject = []
        for j in range(len(dataset_filenames[i])):
            deformable_objects_visit = DeformableMultiObject()
            for object_id in template_specifications.keys():
                if object_id not in dataset_filenames[i][j]:
                    raise RuntimeError('The template object with id ' + object_id + ' is not found for the visit'
                                       + str(j) + ' of subject ' + str(i) + '. Check the dataset xml.')
                else:
                    objectType = template_specifications[object_id]['DeformableObjectType']
                    reader = DeformableObjectReader()
                    deformable_objects_visit.object_list.append(
                        reader.CreateObject(dataset_filenames[i][j][object_id], objectType))
            deformable_objects_visit.update()
            deformable_objects_subject.append(deformable_objects_visit)
        deformable_objects_dataset.append(deformable_objects_subject)
    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = visit_ages
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = deformable_objects_dataset
    longitudinal_dataset.update()

    return longitudinal_dataset


def create_template_metadata(template_specifications):
    """
    Creates a longitudinal dataset object from xml parameters.
    """

    objects_list = []
    objects_name = []
    objects_noise_variance = []
    objects_name_extension = []
    objects_norm = []
    objects_norm_kernel_type = []
    objects_norm_kernel_width = []

    for object_id, object in template_specifications.items():
        filename = object['Filename']
        objectType = object['DeformableObjectType'].lower()

        assert objectType in ['SurfaceMesh'.lower(), 'PolyLine'.lower(), 'PointCloud'.lower()], "Unknown object type"

        root, extension = splitext(filename)
        reader = DeformableObjectReader()

        objects_list.append(reader.CreateObject(filename, objectType))
        objects_name.append(object_id)
        objects_name_extension.append(extension)
        objects_noise_variance.append(object['NoiseStd'] ** 2)

        object_norm = _get_norm_for_object(object, object_id)

        objects_norm.append(object_norm)

        if object_norm in ['current', 'varifold']:
            objects_norm_kernel_type.append(object['KernelType'])
            objects_norm_kernel_width.append(float(object['KernelWidth']))

        else:
            objects_norm_kernel_type.append("no_kernel_needed")
            objects_norm_kernel_width.append(0.)

    multi_object_attachment = MultiObjectAttachment()
    multi_object_attachment.attachment_types = objects_norm
    for k in range(len(objects_norm)):
        multi_object_attachment.kernels.append(
            create_kernel(objects_norm_kernel_type[k], objects_norm_kernel_width[k]))

    return objects_list, objects_name, objects_name_extension, objects_noise_variance, objects_norm, multi_object_attachment


def _get_norm_for_object(object, object_id):
    """
    object is a dictionary containing the deformable object properties.
    Here we make sure it is properly set, and deduce the right norm to use.
    """
    object_type = object['DeformableObjectType'].lower()

    if object_type == 'SurfaceMesh'.lower() or object_type == 'PolyLine':
        try:
            object_norm = object['AttachmentType'].lower()
            assert object_norm in ['Varifold'.lower(), 'Current'.lower()]

        except KeyError as e:
            msg = "Watch out, I did not get a distance type for the object {e}, Please make sure you are running shooting, otherwise distances are required.".format(e=object_id)
            warnings.warn(msg)
            object_norm = 'none'

    elif object_type == 'PointCloud'.lower():
        object_norm = 'Current'.lower() #it's automatic for point cloud

    else:
        assert False, "Unknown object type {e}".format(e=object_type)

    return object_norm




