""" A plug and play Spatial Transformer Module in Pytorch """ 
import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 




class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.
    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """
    def __init__(self, in_channels, spatial_dims, kernel_size,use_dropout=True):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims 
        self._in_ch = in_channels 
        self._ksize = kernel_size
        self.dropout = use_dropout

        # localization net 
        self.conv1_stn = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2_stn = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3_stn = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4_stn = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv5_stn = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        self.fc1_stn = nn.Linear(32*20*20, 512)
        self.fc2_stn = nn.Linear(512, 6)


    def forward(self, x): 
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        batch_images = x
        x = F.relu(self.conv1_stn(x.detach()))
        x = F.relu(self.conv2_stn(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3_stn(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv4_stn(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5_stn(x))
        x = F.max_pool2d(x, 2)
        # print("Pre view size:{}".format(x.size()))
        x = x.view(-1, 32*20*20)
        if self.dropout:
            x = F.dropout(self.fc1_stn(x), p=0.3)
            x = self.fc2_stn(x)
        else:
            x = self.fc1_stn(x)
            x = self.fc2_stn(x) # params [Nx6]
        # import ipdb; ipdb.set_trace()
        x = x.view(-1, 2,3) # change it to the 2x3 matrix 
        # print(x.size())
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        # print("rois found to be of size:{}".format(rois.size()))
        return rois, affine_grid_points