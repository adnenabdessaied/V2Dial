import os
import pandas as pd
# import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .video_utils import read_frames_decord


def load_file(file_name):
    annos = None
    if os.path.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if os.path.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if os.path.splitext(file_name)[1] == '.json':
            annos = json.load(fp)
    return annos


class NextQADataset(Dataset):
    def __init__(self, config, medium, vis_processor, text_processor, split):

        super().__init__()
        self.config = config
        self.medium = medium
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.split = split

        self.batch_size = config['batch_size_test_{}'.format(medium)] if split == 'test' else config['batch_size_{}'.format(medium)] 
        self.root_vis = config['root_raw_vis_{}_{}'.format(medium, split)]
        with open(config['vid_mapping_nextqa'], 'r') as f:
            self.video_mapping = json.load(f)
        
        self.sample_list = load_file(self.config['anno_nextqa_{}'.format(split)])

        if split == 'test':
            self.sample_list = self.sample_list[config['start_idx_gen']: config['end_idx_gen']]
            self.captions = load_file(self.config['next_qa_captions_{}'.format(split)])
        else:
            self.captions = None

        num_samples = config['num_samples_{}'.format(self.medium)]
        if num_samples > 0:
            self.sample_list = self.sample_list[:num_samples]
        
    def __len__(self):
        return len(self.sample_list)


    def load_vid(self, vid_id):
        vid_dir_path = os.path.join(self.root_vis, self.video_mapping[vid_id] + '.mp4')

        frames, _, _ = read_frames_decord(vid_dir_path, self.config.num_frames)
        frames = [self.vis_processor(f).unsqueeze(0) for f in frames]

        vis = torch.cat(frames, dim=0)
        return vis

    def __getitem__(self, idx):
        if self.split == 'test':
            idx += self.config['start_idx_gen']
            
        cur_sample = self.sample_list.loc[idx]
        video_id, ques, ans, qid = str(cur_sample['video']), str(cur_sample['question']),\
                                    str(cur_sample['answer']), str(cur_sample['qid'])
        
        history = self.text_processor(ques)
        answer = self.text_processor(ans)
        if self.split == 'test':
            caption = self.text_processor(self.captions[video_id])
        else:
            caption = self.text_processor('please answer the following question based on the video')
        vis = self.load_vid(video_id)

        return vis, caption, history, answer, video_id, qid

def load_nextqa_dataset(config, vis_processor, text_processor, split):
    # data_file = config['anno_avsd_{}'.format(split)]
    # dataset_list = get_dataset(config, split, tokenizer_enc_dec)
    dataset = NextQADataset(config, 'nextqa', vis_processor, text_processor, split)
    return dataset
