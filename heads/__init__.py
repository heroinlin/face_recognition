from .metrics import ArcFace, CosFace, SphereFace, Softmax, Am_softmax

__head_factory = {
    'ArcFace': ArcFace,
    'CosFace': CosFace,
    'SphereFace': SphereFace,
    'Am_softmax': Am_softmax
}


def get_names():
    return list(__head_factory.keys())


def init_head(name, *args, **kwargs):
    if name not in list(__head_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __head_factory[name](*args, **kwargs)