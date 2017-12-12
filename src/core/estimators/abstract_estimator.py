class AbstractEstimator:

    """
    AbstractEstimator object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.CurrentIteration = 0
        self.MaxIterations = None
        self.ConvergenceTolerance = None

        self.PrintEveryNIters = None
        self.SaveEveryNIters = None

        self.Dataset = None
        self.StatisticalModel = None

        # RER = random effects realization.
        self.PopulationRER = None
        self.IndividualRER = None

