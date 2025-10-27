
import math as m
import numpy as np

import torch
from torch import nn

_get_even = lambda xs: xs[:,0::2]
_get_odd = lambda xs: xs[:,1::2]

def _interleave(first, second, order):
    """
    Given 2 rank-2 tensors with same batch dimension, interleave their columns.
    
    The tensors "first" and "second" are assumed to be of shape (B,M) and (B,N)
    where M = N or N+1, repsectively.
    """
    cols = []
    if order == 'even':
        for k in range(second.shape[1]):
            cols.append(first[:,k])
            cols.append(second[:,k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:,-1])
    else:
        for k in range(first.shape[1]):
            cols.append(second[:,k])
            cols.append(first[:,k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:,-1])
    return torch.stack(cols, dim=1)


class _BaseCouplingLayer(nn.Module):
    def __init__(self, dim, partition, nonlinearity):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module.

        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))
        
        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module.
        * nonlinearity: an instance of torch.nn.Module.
        """
        super(_BaseCouplingLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in ['even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"

        self.partition = partition
        
        if (partition == 'even'):
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        
        # store nonlinear function module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        return _interleave(
            self._first(x),
            self.coupling_law(self._second(x), self.nonlinearity(self._first(x))),
            self.partition
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        return _interleave(
            self._first(y),
            self.anticoupling_law(self._second(y), self.nonlinearity(self._first(y))),
            self.partition
        )

    def coupling_law(self, a, b):
        # (a,b) --> g(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")

    def anticoupling_law(self, a, b):
        # (a,b) --> g^{-1}(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")


class AdditiveCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a + b."""
    def coupling_law(self, a, b):
        return (a + b)
    
    def anticoupling_law(self, a, b):
        return (a - b)


class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b."""
    def coupling_law(self, a, b):
        return torch.mul(a,b)
    
    def anticoupling_law(self, a, b):
        return torch.mul(a, torch.reciprocal(b))
    
def NICE_gaussian_loss(h, diag, keepdim=False):
    return torch.sum(diag) - (0.5*torch.sum(torch.pow(h,2),dim=1,keepdim=keepdim) + h.size(1)*0.5*torch.log(torch.tensor(2 * np.pi)))

# wrap above loss functions in Modules:
class GaussianPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(GaussianPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        if self.size_average:
            return torch.mean(-NICE_gaussian_loss(fx, diag))
        else:
            return torch.sum(-NICE_gaussian_loss(fx, diag))
        
class NICEModel(nn.Module):
    def __init__(self, n_features:int, n_flows:int, n_hidden:list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gauss_loss = GaussianPriorNICELoss()

        coupling_layers = []
        for i_flow in range(n_flows):

            if i_flow %2 == 0:
                part = "even"
            else:
                part = "odd"

            half_features = -999
            if part == "even":
                half_features = m.ceil(n_features / 2)
            else:
                half_features = m.floor(n_features / 2)

            # build the list of shift layers 
            shift_layers = []

            shift_layers.append(nn.Linear(half_features, n_hidden[0]))
            shift_layers.append(nn.ReLU(True))

            for i_hidden in range(1, len(n_hidden)):
                shift_layers.append(nn.Linear(n_hidden[i_hidden-1], n_hidden[i_hidden]))
                shift_layers.append(nn.ReLU(True))

            shift_layers.append( nn.Linear(n_hidden[-1], half_features))


            # add the affine layer, made from those shift layers
            coupling_layers.append(
                AdditiveCouplingLayer(
                    dim = n_features,
                    partition = part,
                    nonlinearity = nn.Sequential(*shift_layers)
                )
            )


        self.flows = nn.Sequential(*coupling_layers)
    
        self.scaling_diag = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        
        y = self.flows(x)

        return torch.matmul(y, torch.diag(torch.exp(self.scaling_diag)))
    
    def inverse(self, y):
        """Invert a set of draws from gaussians"""
        with torch.no_grad():
            x = torch.matmul(y, torch.diag(torch.reciprocal(torch.exp(self.scaling_diag))))
            
            for flow in self.flows:
                x = flow.inverse(x)
            
        return x
    
    def loss_fn(self, model_pred):
        return self.gauss_loss(model_pred, self.scaling_diag)
    
    def train_batch(self, data):

        enc = self.forward(data)
        loss = self.loss_fn(enc)
        loss.backward()

        return loss

