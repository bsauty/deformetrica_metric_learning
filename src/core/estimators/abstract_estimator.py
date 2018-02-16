class AbstractEstimator:

    """
    AbstractEstimator object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.name = 'undefined'
        self.verbose = 1  # If 0, don't print nothing.

        self.optimized_log_likelihood = 'complete'

        self.current_iteration = 0
        self.max_iterations = None
        self.convergence_tolerance = None

        self.print_every_n_iters = None
        self.save_every_n_iters = None

        self.dataset = None
        self.statistical_model = None

        # RER = random effects realization.
        self.population_RER = {}
        self.individual_RER = {}

