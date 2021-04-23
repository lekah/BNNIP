import torch
from torch.nn.utils.convert_parameters import _check_param_device



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
