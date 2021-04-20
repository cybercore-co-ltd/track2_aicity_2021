# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import numpy as np
import random
from torch import nn

from .backbones import build_backbone
from lib.layers.pooling import GeM
from lib.layers.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            #nn.init.constant_(m.weight, 1.0)
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def build_embedding_head(option, input_dim, output_dim, dropout_prob):
    reduce = None
    if option == 'fc':
        reduce = nn.Linear(input_dim, output_dim)
    elif option == 'dropout_fc':
        reduce = [nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                 ]
        reduce = nn.Sequential(*reduce)
    elif option == 'bn_dropout_fc':
        reduce = [nn.BatchNorm1d(input_dim),
                  nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                  ]
        reduce = nn.Sequential(*reduce)
    elif option == 'mlp':
        reduce = [nn.Linear(input_dim, output_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(output_dim, output_dim),
                 ]
        reduce = nn.Sequential(*reduce)
    else:
        print('unsupported embedding head options {}'.format(option))
    return reduce


class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)


class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return intermediate, torch.softmax(out, dim=1)


class MultiHeads(nn.Module):
    def __init__(self, feature_dim=256, groups=4, mode='S', backbone_fc_dim=1024):
        super(MultiHeads, self).__init__()
        self.mode = mode
        self.groups = groups
        # self.Backbone = backbone[resnet]
        self.instance_fc = FC(backbone_fc_dim, feature_dim)
        self.GDN = GDN(feature_dim, groups)
        self.group_fc = nn.ModuleList([FC(backbone_fc_dim, feature_dim) for i in range(groups)])
        self.feature_dim = feature_dim

    def forward(self, x):
        B = x.shape[0]
        # x = self.Backbone(x)  # (B,4096)
        instacne_representation = self.instance_fc(x)

        # GDN
        group_inter, group_prob = self.GDN(instacne_representation)
        # print(group_prob)
        # group aware repr
        v_G = [Gk(x) for Gk in self.group_fc]  # (B,512)

        # self distributed labeling
        group_label_p = group_prob.data
        group_label_E = group_label_p.mean(dim=0)
        group_label_u = (group_label_p - group_label_E.unsqueeze(dim=-1).expand(self.groups, B).T) / self.groups + (
                1 / self.groups)
        group_label = torch.argmax(group_label_u, dim=1).data

        # group ensemble
        group_mul_p_vk = list()
        if self.mode == 'S':
            for k in range(self.groups):
                Pk = group_prob[:, k].unsqueeze(dim=-1).expand(B, self.feature_dim)
                group_mul_p_vk.append(torch.mul(v_G[k], Pk))
            group_ensembled = torch.stack(group_mul_p_vk).sum(dim=0)
        # instance , group aggregation
        final = instacne_representation + group_ensembled
        return group_inter, final, group_prob, group_label


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes

        if pretrain_choice == 'imagenet' and model_name not in ['tf_efficientnet_l2_ns', 'seresnext50_32x4d', 'dm_nfnet_f0', 'nf_resnet50', 'dm_nfnet_f1']:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.MultiHeads = MultiHeads(feature_dim=2048, groups=32, mode='S', backbone_fc_dim=2048)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        #self.bottleneck = IBN(self.in_planes)

        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.reduce = nn.Sequential(
            nn.Conv2d(3072, 2048, kernel_size=1, bias=False)
        )


    def forward(self, x, label=None, return_featmap=False):

        featmap = self.base(x)  # (b, 2048, 1, 1)

        if return_featmap:
            return featmap
        global_feat = self.gap(featmap)
        global_feat = global_feat.flatten(1)

        # MultiHeads
        _, global_feat, _, _ = self.MultiHeads(global_feat)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat#global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path, skip_fc=True):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        
        for i in param_dict:
            y = i.replace('module', 'base')
            if skip_fc and 'classifier' in i:
                continue
            
            if self.state_dict()[y].shape != param_dict[i].shape:
                pass
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[y].shape, param_dict[i].shape))
                continue
            self.state_dict()[y].copy_(param_dict[i])
            
class Baseline_2_Head(Baseline):
    in_planes = 2048 + 1024
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline_2_Head, self).__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg)
        
        self.gap_1 = GeM()
        self.gap_2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, label=None, return_featmap=False):
        featmap_low, featmap = self.base(x)  # (b, 2048, 1, 1)
        if return_featmap:
            return featmap_low, featmap
        
        # process low-level feature
        global_feat_low_gem = self.gap_1(featmap_low)
        global_feat_low_ada = self.gap_2(featmap_low)
        
        global_feat_low_gem = global_feat_low_gem.flatten(1)
        global_feat_low_ada = global_feat_low_ada.flatten(1)
        
        featmap_low = global_feat_low_gem + global_feat_low_ada
        
        # process high-level features
        global_feat_gem = self.gap_1(featmap)
        global_feat_ada = self.gap_2(featmap)
        
        global_feat_gem = global_feat_gem.flatten(1)
        global_feat_ada = global_feat_ada.flatten(1)
        
        featmap = global_feat_gem + global_feat_ada
        
        # cat low-level features and high-level feature
        global_feat = torch.cat((featmap, featmap_low), dim=1)
        
        
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat#global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat


if __name__=='__main__':
    x = torch.rand(32, 2048)
    model = MultiHeads(feature_dim=2048, groups=4, mode='S', backbone_fc_dim=2048)
    feat = model(x)
    print()