from .resnet import ResNet_50, ResNet_101, ResNet_152
from .irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from .mobile_facenet import MobileFaceNet

__backbone_factory = {
    # image classification models
    'ResNet_50': ResNet_50,
    'ResNet_101': ResNet_101,
    'ResNet_152': ResNet_152,
    'IR_50': IR_50,
    'IR_101': IR_101,
    'IR_152': IR_152,
    'IR_SE_50': IR_SE_50,
    'IR_SE_101': IR_SE_101,
    'IR_SE_152': IR_SE_152,
    'MobileFaceNet': MobileFaceNet,
}


def get_names():
    return list(__backbone_factory.keys())


def init_backbone(name, *args, **kwargs):
    if name not in list(__backbone_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __backbone_factory[name](*args, **kwargs)
