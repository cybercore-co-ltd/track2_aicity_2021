# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline_reduce, Baseline, Baseline_2_Head
from .multiheads_baseline import Baseline as MultiHeadsBaseline



def build_model(cfg, num_classes):
    if cfg.MODEL.MODEL_TYPE == 'baseline_reduce':
        print("using global feature baseline reduce")
        model = Baseline_reduce(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'baseline':
        print("using global feature baseline")
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'baseline_2_head':
        print("using low-level feature + high-level feature and GeM Pooling + Adaptive Pooling")
        model = Baseline_2_Head(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'baseline_multiheads':
        print("using global feature baseline")
        model = MultiHeadsBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    
    else:
        print("unsupport model type")
        model = None

    return model
