import os.path
import sys
from torch import nn
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

class ScalarNet(nn.Module):
    def __init__(self, in_dimension=2, out_dimension=4):
        super(ScalarNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dimension, in_dimension),
            nn.Tanh(),
            nn.Linear(in_dimension, out_dimension, bias=True),
            nn.Tanh(),
            nn.Linear(out_dimension, out_dimension, bias=True),
            nn.Tanh(),
            nn.Linear(out_dimension, out_dimension, bias=True),
            nn.Tanh(),
            nn.Linear(out_dimension, out_dimension, bias=True),
            nn.ELU(),
            nn.Linear(out_dimension, out_dimension, bias=True),
            nn.ELU(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.ELU(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension, bias=True),
            ])
        self.number_of_parameters = 0
        for elt in self.parameters():
            self.number_of_parameters += len(elt.view(-1))

    def forward(self, x):
        # print(x.data.numpy())
        for layer in self.layers:
            x = layer(x)
        return x

    def get_gradient(self):
        out = np.zeros(self.number_of_parameters)
        pos = 0
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    out[pos:pos+len(layer.weight.view(-1))] = layer.weight.grad.view(-1).data.numpy()
                    pos += len(layer.weight.view(-1))
            except AttributeError:
                pass
            try:
                if layer.bias is not None:
                    out[pos:pos+len(layer.bias.view(-1))] = layer.bias.grad.view(-1).data.numpy()
                    pos += len(layer.bias.view(-1))
            except AttributeError:
                pass
        return out

    def set_parameters(self, nn_parameters):
        """
        sets parameters from the given (flat) variable (should use state_dict)
        """
        pos = 0
        # print("Setting net param", nn_parameters.data.numpy()[0])
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    layer.weight.data = nn_parameters[pos:pos+len(layer.weight.view(-1))].view(layer.weight.size()).data
                    pos += len(layer.weight.view(-1))
            except AttributeError:
                pass
            try:
                if layer.bias is not None:
                    layer.bias.data = nn_parameters[pos:pos+len(layer.bias.view(-1))].view(layer.bias.size()).data
                    pos += len(layer.bias.view(-1))
            except AttributeError:
                pass
        self.assert_rank_condition()

    def get_parameters(self):
        """"
        returns a numpy array with the flattened weights
        """
        out = np.zeros(self.number_of_parameters)
        pos = 0
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    out[pos:pos+len(layer.weight.view(-1))] = layer.weight.view(-1).data.numpy()
                    pos += len(layer.weight.view(-1))
            except AttributeError:
                pass
            try:
                if layer.bias is not None:
                    out[pos:pos+len(layer.bias.view(-1))] = layer.bias.view(-1).data.numpy()
                    pos += len(layer.bias.view(-1))
            except AttributeError:
                pass
        return out

    def assert_rank_condition(self):
        """
        Fletcher condition on generative networks,
        so that the image is (locally) a submanifold of the space of observations
        """
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    np_weight = layer.weight.data.numpy()
                    a, b = np_weight.shape
                    rank = np.linalg.matrix_rank(layer.weight.data.numpy())
                    assert rank == min(a,b), "Weight of layer does not have full rank {}".format(layer)
            except AttributeError:
                pass
