from .resnet import resnet50, resnet152
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, se_resnet101_ibn_a
from .resnext_ibn_a import resnext50_ibn_a, resnext101_ibn_a
from .resnext_ibn_a_2_head import resnext101_ibn_a_2_head
from .resnest import resnest50
from .regnet.regnet import regnety_800mf, regnety_1600mf, regnety_3200mf
from .mixstyle import MixStyle, MixStyle2
from .STNModule import SpatialTransformer
from .resnext_ibn_a_attention import resnext101_ibn_a_attention
# from .nfnet import dm_nfnet_f0

factory = {
    'resnet50': resnet50,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnext101_ibn_a': resnext101_ibn_a,
    'resnext101_ibn_a_2_head': resnext101_ibn_a_2_head,
    'resnext101_ibn_a_attention': resnext101_ibn_a_attention,
    'resnest50': resnest50,
    'regnety_800mf': regnety_800mf,
    'regnety_1600mf': regnety_1600mf,
    'regnety_3200mf': regnety_3200mf,
    'resnet152': resnet152,
}
def build_backbone(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)