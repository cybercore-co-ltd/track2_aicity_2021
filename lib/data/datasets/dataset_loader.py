# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import cv2
from glob import glob
from random import randint
import numpy as np
import random
from PIL import Image

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            # img = cv2.imread(img_path, 1) #BGR
            # img = Image.fromarray(img)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, change_background=False):
        self.dataset = dataset
        self.transform = transform
        self.change_background = change_background
        if self.change_background:
            self.mask_list = glob('/media/data/ai-city/Track2/AIC21_Track2_ReID/AIC21_Track2_ReID/track2_segmented/mask/*.npy')

        self.path = '/media/data/ai-city/Track2/AIC21_Track2_ReID/AIC21_Track2_ReID/track2_segmented/'
        self._ori_len = len(self.dataset)
        self.times = 1
        # print("I'm ngocnt")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, domain = self.dataset[index]
        if self.change_background==True:
            prob = randint(1, 10)/10
            img_name = img_path.split('/')[-1]
            mask_path = self.path+'mask/'+img_name.split('.')[0]+'.npy'
            if (prob>=0.5) and (mask_path in self.mask_list):
                # import ipdb; ipdb.set_trace()
                foreground=read_image(self.path+'foreground/'+img_name.split('.')[0]+'.jpg')
                mask = np.load(mask_path)
                width, height = foreground.size
                
                # select background
                background_path = random.choice(glob(self.path+'background_painted/*.jpg'))
                background = read_image(background_path)
                background = background.resize((width, height))
                # merge
                merge = background * (1 - np.stack([mask, mask, mask], axis=2)) + foreground
                img = Image.fromarray(merge)
                
            else:
                img = read_image(img_path)
        else:
            img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, domain, img_path
