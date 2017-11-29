class LongitudinalDataset:

    """
    A longitudinal data set is a collection of sets of deformable objects
    for a series of subjects at multiple time-points.

    """

    # Set the times corresponding to each observation of each subject.
    def SetTimes(self, times):
        self.Times = times

    # Set the subject ids list.
    def SetSubjectIds(self, subjectIds):
        self.SubjectIds = subjectIds

    # Set the deformable objects.
    def SetDeformableObjects(self, deformableObjects):
        self.DeformableObjects = deformableObjects

