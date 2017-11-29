class LongitudinalDataset:

    """
    A longitudinal data set is a collection of sets of deformable objects
    for a series of subjects at multiple time-points.

    """

    def SetTimes(self, times):
        self.Times = times

    def SetSubjectIds(self, subjectIds):
        self.SubjectIds = subjectIds

    def SetDeformableObjects(self, deformableObjects):
        self.DeformableObjects = deformableObjects

