from abc import ABC, abstractmethod

from core import default


class AbstractEstimator(ABC):

    """
    AbstractEstimator object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self, statistical_model=None, name='undefined',
                 optimized_log_likelihood=default.optimized_log_likelihood,
                 max_iterations=default.max_iterations, convergence_tolerance=default.convergence_tolerance,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters):

        self.statistical_model = statistical_model
        self.name = name
        self.verbose = 1  # If 0, don't print nothing.
        self.optimized_log_likelihood = optimized_log_likelihood
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.print_every_n_iters = print_every_n_iters
        self.save_every_n_iters = save_every_n_iters

        # RER = random effects realization.
        self.population_RER = {}
        self.individual_RER = {}

    @abstractmethod
    def update(self):
        if self.statistical_model is None:
            raise RuntimeError('statistical_model has not been set')

    @abstractmethod
    def write(self, output_dir):
        pass
