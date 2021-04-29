import torch
from torch.nn.utils.convert_parameters import _check_param_device


class AttributeDict(dict):
    """
    Utility to allow ad.foo to access values inside dictionary
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


class PositiveInt(int):
    def __init__(self, value):
        value = int(value)
        if value < 1:
            raise ValueError("You have to provide a positive integer")
        self(PositiveInt, self).__init__(value)


class WelfordMeanM2:
    # For a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def __init__(self, count=0, mean=0, m2=0):
        self._count = count
        self._mean = mean
        self._M2 = m2

    def update(self, newValue):
        self._count += 1
        delta = newValue - self._mean
        self._mean += delta / self._count
        delta2 = newValue - self._mean
        self._M2 += delta * delta2

    # Retrieve the mean, variance and sample variance from an aggregate
    def estimate(self):
        if self._count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (self._mean,
                                                self._M2 / self._count, self._M2 / (self._count - 1))
            return (mean, variance, sampleVariance)


def parameters_gradients_to_vector(parameters):
    r"""Convert parameters and parameter gradienst to  vectors
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
        The gradients as a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None
    vec = []
    grd = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        vec.append(param.view(-1))
        grd.append(param.grad.view(-1))
    return torch.cat(vec), torch.cat(grd)


def zero_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
