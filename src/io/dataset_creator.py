import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from pydeformetrica.src.io.deformable_object_reader import DeformableObjectReader


class DatasetCreator:

    """
    Creates a dataset object.

    """

    def CreateDataset(self, datasetFilenames, visitAges, subjectIds, templateObjects):

        deformableObjects_dataset = []
        for i in range(len(datasetFilenames)):
            deformableObjects_subject = []
            for j in range(len(datasetFilenames[i])):
                deformableObjects_visit = []
                for object_id in templateObjects.keys():
                    if object_id not in datasetFilenames[i][j]:
                        raise RuntimeError('The template object with id '+object_id+' is not found for the visit'
                                           +str(j)+' of subject '+str(i)+'. Check the dataset xml.')
                    else:
                        objectType = templateObjects[object_id]['DeformableObjectType']
                        reader = DeformableObjectReader()
                        deformableObjects_visit.append(reader.CreateObject(datasetFilenames[i][j][object_id],
                                                                           objectType))
                deformableObjects_subject.append(deformableObjects_visit)
            deformableObjects_dataset.append(deformableObjects_subject)

        longitudinalDataset = LongitudinalDataset()
        longitudinalDataset.SetDeformableObjects(deformableObjects_dataset)
        longitudinalDataset.SetTimes(visitAges)
        longitudinalDataset.SetSubjectIds(subjectIds)

        return longitudinalDataset



