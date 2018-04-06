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
        self.total_number_of_observations = None


    ################################################################################
    ### Public methods:
    ################################################################################

    def update(self):
        self.number_of_subjects = len(self.deformable_objects)
        assert(self.number_of_subjects == len(self.subject_ids))
        self.total_number_of_observations = 0
        for i in range(self.number_of_subjects): self.total_number_of_observations += len(self.deformable_objects[i])

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

    def check_image_shapes(self):
        """
        In the case of non deformable objects, checks the dimension of the images are the same.
        """
        shape = None
        for subj in self.deformable_objects:
            for img in subj:
                if shape is None:
                    shape = img.get_intensities().shape
                else:
                    assert img.get_intensities().shape == shape,\
                        "Different images dimensions were detected... please code the projection or normalize before"




