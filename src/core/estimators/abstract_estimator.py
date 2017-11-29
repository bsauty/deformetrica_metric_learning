class AbstractEstimator:

    """
    AbstractEstimator object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    # Constructor.
    def __init__(self):
        self.MaxIterations = 100
        self.CurrentIteration = 0
        self.PrintEveryNIters = 1
        self.SaveEveryNIters = 100

