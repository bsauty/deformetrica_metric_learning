import os
import time

from launch.estimate_deterministic_atlas import instantiate_deterministic_atlas_model


class Deformetrica:
    def __init__(self, output_dir):
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def estimate_deterministic_atlas(self, template_specifications, dataset, estimator, **kwargs):

        statistical_model = instantiate_deterministic_atlas_model(dataset, template_specifications, **kwargs)

        # set estimator dataset and model to use
        # estimator.dataset = dataset
        estimator.statistical_model = statistical_model

        # estimator.initial_step_size = xml_parameters.initial_step_size
        # estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
        # estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        # estimator.line_search_shrink = xml_parameters.line_search_shrink
        # estimator.line_search_expand = xml_parameters.line_search_expand
        # estimator.max_iterations = xml_parameters.max_iterations
        # estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        # estimator.convergence_tolerance = xml_parameters.convergence_tolerance
        # estimator.print_every_n_iters = xml_parameters.print_every_n_iters
        # estimator.save_every_n_iters = xml_parameters.save_every_n_iters

        """
        Launch
        """
        print('[ update method of the ' + estimator.name + ' optimizer ]')
        start_time = time.time()
        estimator.update()
        estimator.write()
        end_time = time.time()
        print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

        return statistical_model


