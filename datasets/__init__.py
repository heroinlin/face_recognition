from .MS_Celeb_1M import MSCeleb
from .glintasia import GlintAsia

__data_factory = {
    # image classification models
    'Ms_celeb': MSCeleb,
    'GlintAsia': GlintAsia,
    'default': MSCeleb,
}


def get_names():
    return list(__data_factory.keys())


def init_database(name, *args, **kwargs):
    if name not in get_names():
        raise KeyError('Unknown model: {}'.format(name))
    return __data_factory[name](*args, **kwargs)
