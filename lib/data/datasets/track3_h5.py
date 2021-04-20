# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET
import h5py

from .bases import BaseImageDataset

class Track3(BaseImageDataset):
    """
      ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   333 |    36935 |        36
  query    |   333 |     1052 |        ?
  gallery  |   333 |    18290 |        ?
  ----------------------------------------

    """
    dataset_dir = '/media/data/ai-city/Aic_track3/train'
    h5_files = ['S01_data.h5', 'S03_data.h5', 'S04_data.h5']
    def __init__(self, root='', verbose=True, **kwargs):
        super(Track3, self).__init__()
        # import ipdb; ipdb.set_trace()
        self.img_dir = osp.join(self.dataset_dir, 'JPEGImages')
        train = []
        for h5_file_name in h5_files:
            self.data_file = osp.join(self.dataset_dir, h5_file_name)
            data_ = self._process_dir(self.img_dir, self.data_file)
            train.append(data_)
    
        train = self.relabel(train)
        if verbose:
            print("=> AI CITY 2020 data loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train = train
        # self.query = query
        # self.gallery = gallery

        self.train_tracks = self._read_tracks(os.path.join(self.dataset_dir, 'train_track.txt'))
        self.test_tracks = self._read_tracks(os.path.join(self.dataset_dir, 'test_track.txt'))

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, img_dir, data_file, relabel=False):
        dataset = []
        
        
        
        
        if label_path:
            tree = ET.parse(label_path, parser=ET.XMLParser(encoding='utf-8'))
            objs = tree.find('Items')
            for obj in objs:
                image_name = obj.attrib['imageName']
                img_path = osp.join(img_dir, image_name)
                pid = int(obj.attrib['vehicleID'])
                camid = int(obj.attrib['cameraID'][1:])
                dataset.append((img_path, pid, camid))
                #dataset.append((img_path, camid, pid))
            if relabel: dataset = self.relabel(dataset)
        else:
            with open(list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    img_path = osp.join(img_dir, line)
                    pid = 0
                    camid = 0
                    dataset.append((img_path, pid, camid))
        return dataset

if __name__ == '__main__':
    dataset = AICity20(root='/media/data/ai-city/Track2/AIC21_Track2_ReID')
