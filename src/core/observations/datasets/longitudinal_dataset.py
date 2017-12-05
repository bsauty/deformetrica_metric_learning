class LongitudinalDataset:

    """
    A longitudinal data set is a collection of sets of deformable objects
    for a series of subjects at multiple time-points.

    """

    # Constructor.
    def __init__(self):
        self.Times = []
        self.SubjectIds = []
        self.DeformableObjects = []

        self.NumberOfSubjects = None

    # Further initialization.
    def Update(self):
        self.NumberOfSubjects = len(self.SubjectIds)

