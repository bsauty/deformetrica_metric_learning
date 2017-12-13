import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject


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
                    raise RuntimeError('The template object with id '+object_id+' is not found for the visit'
                                       +str(j)+' of subject '+str(i)+'. Check the dataset xml.')
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
