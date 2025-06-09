from torch.utils.data import Dataset
import pickle
import os

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import random

from .utils import pre_text, type_transform_helper, load_anno, open_img

class CapDataset(Dataset):
    def __init__(self, config, medium, vis_processor, text_processor, split):
        super(CapDataset, self).__init__()
        self.config = config
        self.batch_size = config['batch_size_{}'.format(medium)]
        self.medium = medium  # "webvid / cc3m / msrvtt" 
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.split = split  # train / val / test     
        
        self.root_vis = config['root_raw_vis_{}_{}'.format(medium, split)]

        # get the mapping between caption and image/video
        mapping_path = config.get('mapping_path_{}_{}'.format(medium, split), None) 
        with open(mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)
        
        # These are the main ids of the dataset (typically one pro image/vid)
        self.ids = list(self.mapping.keys())
        num_samples = config['num_samples_{}'.format(self.medium)]
        if num_samples > 0:
            self.ids = self.ids[:num_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item  = self.mapping[self.ids[index]]
        # _id = self.ids[index]
        #############################  Textal features  #############################
        caption = item['caption']
        # caption_ = pre_text(caption)
        caption = self.text_processor(caption)        
        # add [CLS] token
        caption = '[CLS] ' + caption

        if self.medium == 'cc3m':
            pth = os.path.join(self.root_vis, item['file'])
            vis = open_img(pth)
            vis = self.vis_processor(vis).unsqueeze(0)
        else:
            pth = os.path.join(self.root_vis, item['file'])
            f_names = os.listdir(pth)
            f_names.sort(key=lambda f_n: int(f_n.split('.')[0]))
            pth = [os.path.join(pth, f_name) for f_name in f_names]
            vis = [Image.open(p).convert('RGB') for p in pth]
            vis = [self.vis_processor(v).unsqueeze(0) for v in vis]
            vis = torch.cat(vis, dim=0)

        # Get negative vis
        neg_index = random.randint(0, len(self) - 1)
        while neg_index == index:
            neg_index = random.randint(0, len(self) - 1)

        neg_item = self.mapping[self.ids[neg_index]]

        if self.medium == 'cc3m':
            neg_pth = os.path.join(self.root_vis, neg_item['file'])
            neg_vis = open_img(neg_pth)
            neg_vis = self.vis_processor(neg_vis).unsqueeze(0)
        else:
            neg_pth = os.path.join(self.root_vis, neg_item['file'])
            neg_f_names = os.listdir(neg_pth)
            neg_f_names.sort(key=lambda f_n: int(f_n.split('.')[0]))
            neg_pth = [os.path.join(neg_pth, neg_f_name) for neg_f_name in neg_f_names]
            neg_vis = [Image.open(p).convert('RGB') for p in neg_pth]
            neg_vis = [self.vis_processor(v).unsqueeze(0) for v in neg_vis]
            neg_vis = torch.cat(neg_vis, dim=0)

        # return caption, vis
        return vis, caption, neg_vis


class VideoTextRetDataset(Dataset):
    def __init__(self, config, vis_processor, text_processor, medium, split):
        super(VideoTextRetDataset, self).__init__()

        self.config = config
        self.batch_size = config['batch_size_{}'.format(medium)]
        self.medium = medium  # "webvid / cc3m / msrvtt" 
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.split = split  # train / val / test

       
        self.root_vis = config['root_raw_vis_{}_{}'.format(medium, split)]

        anno_path = config['annotation_{}_{}'.format(medium, split)]
        self.raw_anno_list = load_anno(anno_path)
        self.text    = []
        self.vis     = []
        self.txt2vis = {}
        self.vis2txt = {}
        self.build_data()
        self.anno_list = [dict(vis=v) for v in self.vis]
        # print('bla')

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        pth = self.anno_list[index]['vis']
        f_names = os.listdir(pth)
        f_names.sort(key=lambda f_n: int(f_n.split('.')[0]))
        pth = [os.path.join(pth, f_name) for f_name in f_names]
        vis = [Image.open(p).convert('RGB') for p in pth]
        vis = [self.vis_processor(v) for v in vis]
        # vis = [transforms.PILToTensor()(v).unsqueeze(0) for v in vis]
        vis = torch.cat(vis, dim=0)
        # vis = self.trans(vis)

        return vis, index

    def build_data(self):
        """each image may have multiple ground_truth text, e.g., COCO and Flickr30K"""
        txt_id = 0
        for vis_id, ann in enumerate(self.raw_anno_list):
            self.vis.append(ann["vis"])
            self.vis2txt[vis_id] = []
            _captions = ann["caption"] \
                if isinstance(ann["caption"], list) else [ann["caption"], ]
            for i, caption in enumerate(_captions):
                # self.text.append(pre_text(caption))
                self.text.append(self.text_processor(caption))
                self.vis2txt[vis_id].append(txt_id)
                self.txt2vis[txt_id] = vis_id
                txt_id += 1


def load_datasets(config, vis_processor, text_processor, split):
    if config['stage'] == 'stage_1':
        if split != 'test':
            cc3m_dataset   = CapDataset(config, 'cc3m', vis_processor, text_processor, split)
            webvid_dataset = CapDataset(config, 'webvid', vis_processor, text_processor, split)
            datasets = {
                'cc3m': cc3m_dataset,
                'webvid': webvid_dataset
            }
        else:  # Test with msrvtt_1k --> video retieval
            msrvtt_dataset = VideoTextRetDataset(config, vis_processor, text_processor, 'msrvtt', split)
            datasets = {
                'msrvtt': msrvtt_dataset
            }
    return datasets