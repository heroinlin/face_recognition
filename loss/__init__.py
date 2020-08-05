from .focal import FocalLoss
import torch.nn as nn

__all_ = ["FocalLoss"]

__loss_factory = {
    'FocalLoss': FocalLoss,
    'Softmax': nn.CrossEntropyLoss
}


def get_names():
    return list(__loss_factory.keys())


def init_loss(name, *args, **kwargs):
    if name not in list(__loss_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __loss_factory[name](*args, **kwargs)