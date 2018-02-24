
from pydeformetrica.src.core.model_tools.manifolds.one_dimensional_exponential import OneDimensionalExponential
from pydeformetrica.src.core.model_tools.manifolds.logistic_exponential import LogisticExponential


class ExponentialFactory:
    def __init__(self):
        self.manifold_type = None
        self.manifold_parameters = None

    def set_manifold_type(self, manifold_type):
        self.manifold_type = manifold_type

    def set_parameters(self, manifold_parameters):
        self.manifold_parameters = manifold_parameters

    def create(self):
        """
        returns an exponential for a manifold of a given type, using the parameters
        """
        if self.manifold_type == 'one_dimensional':
            out = OneDimensionalExponential()
            out.width = self.manifold_parameters['width']
            out.number_of_interpolation_points = self.manifold_parameters['number_of_interpolation_points']
            out.interpolation_points_torch = self.manifold_parameters['interpolation_points_torch']
            out.interpolation_values_torch = self.manifold_parameters['interpolation_values_torch']
            return out

        if self.manifold_type == 'logistic':
            out = LogisticExponential()
            return out

        raise ValueError("Unrecognized manifold type ine exponential factory")