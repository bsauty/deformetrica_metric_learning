class LongitudinalDataset:

    """
    A longitudinal data set is a collection of sets of deformable objects
    for a series of subjects at multiple time-points.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.times = []
        self.subject_ids = []
        self.deformable_objects = []
        self.number_of_subjects = None


    ################################################################################
    ### Public methods:
    ################################################################################

    def update(self):
        self.number_of_subjects = len(self.deformable_objects)
        assert(self.number_of_subjects == len(self.subject_ids))

    def is_cross_sectional(self):
        """
        Checks whether there is a single visit per subject
        """
        b = True
        for elt in self.deformable_objects: b = (b and len(elt) == 1)
        return b

    def is_time_series(self):
        """
        Checks whether there is a single visit per subject
        """
        return len(self.deformable_objects) == 1 and len(self.deformable_objects[0]) > 1 and \
               len(self.times) == 1 and len(self.deformable_objects[0]) == len(self.times[0])

