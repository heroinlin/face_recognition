from .resnet import *

__model_factory = {
    # image classification models
    'resnet_50': ResNet_50,
    'resnet_101': ResNet_101,
    'resnet_152': ResNet_152,
}


def get_names():
    return list(__model_factory.keys())


def init_backbone(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
