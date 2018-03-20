import os.path
import sys
from torch import nn
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

class ScalarNet(nn.Module):
    def __init__(self, in_dimension=2, out_dimension=4):
        super(ScalarNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dimension, in_dimension),
            nn.ELU(),
            nn.Linear(in_dimension, out_dimension, bias=True),
            nn.ELU(),
            nn.Linear(out_dimension, out_dimension, bias=True),
            nn.ELU(),
            nn.Linear(out_dimension, out_dimension, bias=True),
            nn.ELU()])
        self.number_of_parameters = 0
        for elt in self.parameters():
            self.number_of_parameters += len(elt.view(-1))

    def forward(self, x):
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
                elif layer.bias is not None:
                    out[pos:pos+len(layer.bias.weight.view(-1))] = layer.bias.weight.grad.view(-1).data.numpy()
                    pos += len(layer.bias.weight.view(-1))
            except AttributeError:
                pass
        print("First metric param grad", out[0])
        return out


    def set_parameters(self, nn_parameters):
        """
        sets parameters from the given (flat) variable (should use state_dict)
        """
        pos = 0
        d = {}
        pos = 0
        print("Setting metric param to:", nn_parameters[0].data.numpy())
        for key, val in self.state_dict().items():
            d[key] = nn_parameters[pos:pos+len(val.view(-1))].view(val.size()).data
            pos += len(val.view(-1))
        self.load_state_dict(d)

    def get_parameters(self):
        """"
        returns a numpy array with the flattened weights
        """
        out = np.zeros(self.number_of_parameters)
        pos = 0
        for key, val in self.state_dict().items():
            out[pos:pos+len(val.view(-1))] = val.view(-1).numpy()
            pos += len(val.view(-1))
        # print("First metric param", out[0])
        return out
