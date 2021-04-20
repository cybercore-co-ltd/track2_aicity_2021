# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET


from .bases import BaseImageDataset


class AICity20ReCam(BaseImageDataset):
    """
    将AI City train 中333个ID， 1-95为测试集, 241-478为训练集
    测试集中随机取500张作为query
    """
    dataset_dir = 'AIC21_Track2_ReID/AIC21_Track2_ReID'
    dataset_aug_dir = 'AIC20_ReID_Cropped/'
    dataset_blend_dir = 'AIC20_ReID_blend/'

    def __init__(self, root='', verbose=True, **kwargs):
        super(AICity20ReCam, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        train_list_path = osp.join(self.dataset_dir, 'name_train.txt')
        query_list_path = osp.join(self.dataset_dir, 'name_query.txt')
        gallery_list_path = osp.join(self.dataset_dir, 'name_test.txt')
        
        self.train_label_path = osp.join(self.dataset_dir, 'train_label.xml')
        self.query_label_path = osp.join(self.dataset_dir, 'query_label.xml')
        self.gallery_label_path = osp.join(self.dataset_dir, 'test_label.xml')
    
        self._check_before_run()

        train = self._process_dir(self.train_dir, train_list_path, self.train_label_path, relabel=False)
        query = self._process_dir(self.query_dir, query_list_path, None)
        gallery = self._process_dir(self.gallery_dir, gallery_list_path, None)
        # train += self._process_dir(self.train_aug_dir, train_list_path, relabel=False)
        # train += self._process_dir(os.path.join(root, self.dataset_blend_dir, 'image_train')
        #                            , train_list_path, relabel=False)

        train = train+query+gallery

        train = self.relabel(train)
        if verbose:
            print("=> aicity trainval for ReCamID loaded")
            # self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        self.train_tracks = self._read_tracks(osp.join(self.dataset_dir, 'train_track.txt'))
        self.test_tracks = self._read_tracks(osp.join(self.dataset_dir, 'test_track.txt'))
        # import ipdb; ipdb.set_trace()
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, img_dir, list_path, label_path, relabel=False):
        dataset = []
        if label_path:
            tree = ET.parse(label_path, parser=ET.XMLParser(encoding='utf-8'))
            objs = tree.find('Items')
            for obj in objs:
                image_name = obj.attrib['imageName']
                img_path = osp.join(img_dir, image_name)
                pid = int(obj.attrib['cameraID'][1:])
                camid = int(obj.attrib['cameraID'][1:])
                domain=0
                dataset.append((img_path, pid, camid, domain))
            if relabel: dataset = self.relabel(dataset)
        else:
            with open(list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    img_path = osp.join(img_dir, line)
                    pid = 0
                    camid = 0
                    domain=0
                    dataset.append((img_path, pid, camid, domain))
        return dataset

if __name__ == '__main__':
    dataset = AICity20ReCam(root='/home/zxy/data/ReID/vehicle')
