import math
import functools
import hydra
from omegaconf import OmegaConf
import numpy as np


# Define useful resolver for hydra config
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("str", lambda x: str(x))
OmegaConf.register_new_resolver("prod", lambda x: np.prod(x))
OmegaConf.register_new_resolver("where", lambda condition, x, y: x if condition else y)
OmegaConf.register_new_resolver("isequal", lambda x, y: x == y)
OmegaConf.register_new_resolver("pi", lambda x: x * math.pi)


def partialclass(cls, *args, **kwds):
    """Return a class instance with partial __init__

    Input:
        cls [str]: class to instantiate
    """
    cls = hydra.utils.get_class(cls)

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def partialfunction(func, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)
