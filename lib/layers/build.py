# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .metric_learning import ContrastiveLoss
from .metric_learning import ContrastiveLoss, SupConLoss
from torch.utils.tensorboard import SummaryWriter

def make_loss(cfg, num_classes):    # modified by gu
    make_loss.update_iter_interval = 500
    make_loss.id_loss_history = []
    make_loss.metric_loss_history = []
    make_loss.ID_LOSS_WEIGHT = cfg.MODEL.ID_LOSS_WEIGHT
    make_loss.TRIPLET_LOSS_WEIGHT = cfg.MODEL.TRIPLET_LOSS_WEIGHT
    
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_func = TripletLoss(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'contrastive':
        metric_loss_func = ContrastiveLoss(cfg.SOLVER.MARGIN)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'supconloss':
        metric_loss_func = SupConLoss(num_ids=int(cfg.SOLVER.IMS_PER_BATCH/cfg.DATALOADER.NUM_INSTANCE), views=cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'none':
        def metric_loss_func(feat, target):
            return 0
    else:
        print('got unsupported metric loss type {}'.format(
            cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    else:
        id_loss_func = F.cross_entropy

    def loss_func(score, feat, target):
        _id_loss = id_loss_func(score, target)
        _metric_loss = metric_loss_func(feat, target)
        make_loss.id_loss_history.append(_id_loss.item())
        make_loss.metric_loss_history.append(_metric_loss.item())
        if len(make_loss.id_loss_history)==0:
            pass
        elif (len(make_loss.id_loss_history) % make_loss.update_iter_interval == 0):
            
            _id_history = np.array(make_loss.id_loss_history)
            id_mean = _id_history.mean()
            id_std = _id_history.std()
            
            _metric_history = np.array(make_loss.metric_loss_history)
            metric_mean = _metric_history.mean()
            metric_std = _metric_history.std()
            
            id_weighted = id_std
            metric_weighted = metric_std
            if id_weighted > metric_weighted:
                new_weight = 1 - (id_weighted-metric_weighted)/id_weighted
                make_loss.ID_LOSS_WEIGHT = make_loss.ID_LOSS_WEIGHT*0.9+new_weight*0.1

            make_loss.id_loss_history = []
            make_loss.metric_loss_history = []
            print(f"update weighted loss ID_LOSS_WEIGHT={round(make_loss.ID_LOSS_WEIGHT,3)},TRIPLET_LOSS_WEIGHT={make_loss.TRIPLET_LOSS_WEIGHT}")
        else:
            pass
        return make_loss.ID_LOSS_WEIGHT * _id_loss, make_loss.TRIPLET_LOSS_WEIGHT * _metric_loss
    return loss_func