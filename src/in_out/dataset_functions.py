import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from os.path import splitext
import warnings
import math
import numpy as np
import torch
from torch.autograd import Variable

from pydeformetrica.src.core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.support.utilities.general_settings import Settings


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
                    raise RuntimeError('The template object with id ' + object_id + ' is not found for the visit '
                                       + str(j) + ' of subject ' + str(i) + '. Check the dataset xml.')
                else:
                    objectType = template_specifications[object_id]['deformable_object_type']
                    reader = DeformableObjectReader()
                    deformable_objects_visit.object_list.append(
                        reader.create_object(dataset_filenames[i][j][object_id], objectType))
            deformable_objects_visit.update()
            deformable_objects_subject.append(deformable_objects_visit)
        deformable_objects_dataset.append(deformable_objects_subject)
    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = visit_ages
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = deformable_objects_dataset
    longitudinal_dataset.update()

    return longitudinal_dataset


def create_scalar_dataset(group, observations, timepoints):
    """
    Builds a dataset from the given data.
    """

    times = []
    subject_ids = []
    scalars = []

    for subject_id in group:
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
            times_subject = []
            scalars_subject = []
            for i in range(len(observations)):
                if group[i] == subject_id:
                    times_subject.append(timepoints[i])
                    scalars_subject.append(observations[i])
            assert len(times_subject) > 0, subject_id
            assert len(times_subject) == len(scalars_subject)
            times.append(np.array(times_subject))
            scalars.append(Variable(torch.from_numpy(np.array(scalars_subject)).type(Settings().tensor_scalar_type)))

    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = times
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = scalars
    longitudinal_dataset.number_of_subjects = len(subject_ids)
    longitudinal_dataset.total_number_of_observations = len(timepoints)

    return longitudinal_dataset


def read_and_create_scalar_dataset(xml_parameters):
    """
    Read scalar observations e.g. from cognitive scores, and builds a dataset.
    """
    group = np.loadtxt(xml_parameters.group_file, delimiter=',', dtype=str)
    observations = np.loadtxt(xml_parameters.observations_file, delimiter=',')
    timepoints = np.loadtxt(xml_parameters.timepoints_file, delimiter=',')
    return create_scalar_dataset(group, observations, timepoints)


def read_and_create_image_dataset(dataset_filenames, visit_ages, subject_ids, template_specifications):
    """
    Builds a longitudinal dataset of images (non deformable images). Loads everything into memory. #TODO assert on the format of the images !
    """
    deformable_objects_dataset = []

    for i in range(len(dataset_filenames)):
        deformable_objects_subject = []
        for j in range(len(dataset_filenames[i])):
            for object_id in template_specifications.keys():
                if object_id not in dataset_filenames[i][j]:
                    raise RuntimeError('The template object with id ' + object_id + ' is not found for the visit '
                                       + str(j) + ' of subject ' + str(i) + '. Check the dataset xml.')
                else:
                    objectType = template_specifications[object_id]['deformable_object_type']
                    reader = DeformableObjectReader()
                    deformable_object_visit = reader.create_object(dataset_filenames[i][j][object_id], objectType)
                    deformable_object_visit.update()
            deformable_objects_subject.append(deformable_object_visit)
        deformable_objects_dataset.append(deformable_objects_subject)

    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = [np.array(elt) for elt in visit_ages]
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = deformable_objects_dataset
    longitudinal_dataset.update()
    longitudinal_dataset.check_image_shapes()

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
        filename = object['filename']
        object_type = object['deformable_object_type'].lower()

        assert object_type in ['SurfaceMesh'.lower(), 'PolyLine'.lower(), 'PointCloud'.lower(), 'Landmark'.lower(),
                               'Image'.lower()], "Unknown object type"

        root, extension = splitext(filename)
        reader = DeformableObjectReader()

        objects_list.append(reader.create_object(filename, object_type))
        objects_name.append(object_id)
        objects_name_extension.append(extension)

        if object['noise_std'] < 0:
            objects_noise_variance.append(-1.0)
        else:
            objects_noise_variance.append(object['noise_std'] ** 2)

        object_norm = _get_norm_for_object(object, object_id)

        objects_norm.append(object_norm)

        if object_norm in ['current', 'varifold']:
            objects_norm_kernel_type.append(object['kernel_type'])
            objects_norm_kernel_width.append(float(object['kernel_width']))

        else:
            objects_norm_kernel_type.append("no_kernel_needed")
            objects_norm_kernel_width.append(0.)

    multi_object_attachment = MultiObjectAttachment()
    multi_object_attachment.attachment_types = objects_norm
    for k in range(len(objects_norm)):
        multi_object_attachment.kernels.append(
            create_kernel(objects_norm_kernel_type[k], objects_norm_kernel_width[k]))

    return objects_list, objects_name, objects_name_extension, objects_noise_variance, multi_object_attachment


def compute_noise_dimension(template, multi_object_attachment):
    """
    Compute the dimension of the spaces where the norm are computed, for each object.
    """
    assert len(template.object_list) == len(multi_object_attachment.attachment_types)
    assert len(template.object_list) == len(multi_object_attachment.kernels)

    objects_noise_dimension = []
    for k in range(len(template.object_list)):

        if multi_object_attachment.attachment_types[k] in ['current', 'varifold', 'pointcloud']:
            noise_dimension = 1
            for d in range(Settings().dimension):
                length = template.bounding_box[d, 1] - template.bounding_box[d, 0]
                assert length >= 0
                noise_dimension *= math.floor(length / multi_object_attachment.kernels[k].kernel_width + 1.0)
            noise_dimension *= Settings().dimension

        elif multi_object_attachment.attachment_types[k] in ['landmark']:
            noise_dimension = Settings().dimension * template.object_list[k].points.shape[0]

        else:
            raise RuntimeError('Unknown noise dimension for the attachment type: '
                               + multi_object_attachment.attachment_types[k])

        objects_noise_dimension.append(noise_dimension)

    return objects_noise_dimension


def _get_norm_for_object(object, object_id):
    """
    object is a dictionary containing the deformable object properties.
    Here we make sure it is properly set, and deduce the right norm to use.
    """
    object_type = object['deformable_object_type'].lower()

    if object_type == 'SurfaceMesh'.lower() or object_type == 'PolyLine'.lower():
        try:
            object_norm = object['attachment_type'].lower()
            assert object_norm in ['Varifold'.lower(), 'Current'.lower(), 'Landmark'.lower()]

        except KeyError as e:
            msg = "Watch out, I did not get a distance type for the object {e}, Please make sure you are running " \
                  "shooting or a parallel transport, otherwise distances are required.".format(e=object_id)
            warnings.warn(msg)
            object_norm = 'none'

    elif object_type == 'PointCloud'.lower():
        object_norm = 'Current'.lower()  # it's automatic for point cloud

    elif object_type == 'Landmark'.lower():
        object_norm = 'Landmark'.lower()

    elif object_type == 'Image'.lower():
        object_norm = 'L2'
        if 'attachment_type' in object.keys() and not object['attachment_type'].lower() == 'L2'.lower():
            msg = 'Only the "L2" attachment is available for image objects so far. ' \
                  'Overwriting the user-specified invalid attachment: "%s"' % object['attachment_type']
            warnings.warn(msg)

    else:
        assert False, "Unknown object type {e}".format(e=object_type)

    return object_norm
