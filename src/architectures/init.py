import torch
from torch import nn


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(module, activation="relu"):
    """Initialize a linear layer.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                # don't reset if resetted already (might want special)
                continue
        except AttributeError:
            pass

        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
