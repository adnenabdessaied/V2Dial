# coding: utf-8
# author: noctli
import json
import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from itertools import chain
from torchvision import transforms
from .utils import type_transform_helper
from itertools import chain
from .video_utils import read_frames_decord


def tokenize(text, tokenizer, return_tensor=False):
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if return_tensor:
        return torch.tensor(tokenized_text).long()
    return tokenized_text


def get_dataset(config, split):
    if split != 'test':
        dialog_pth = config[f'anno_avsd_{split}']
    else:
        dialog_pth = config['anno_avsd_test_dstc_{}'.format(config['dstc'])]
    n_history = config['num_hist_turns_avsd']
    undisclosed_only = split == 'test'
    dialog_data = json.load(open(dialog_pth, 'r'))
    dialog_list = []
    vid_set = set()
    pbar = tqdm(dialog_data['dialogs'])
    pbar.set_description('[INFO] Loading AVSD - {}'.format(split))
    for dialog in pbar:
        # if config['dstc'] != 10:
        caption = dialog['caption']
        summary = dialog['summary']
        # else:
        #     caption = 'no'
        #     summary = 'no'

        questions = [d['question'] for d in dialog['dialog']]
        answers = [d['answer'] for d in dialog['dialog']]
        vid = dialog["image_id"]
        vid_set.add(vid)
        if undisclosed_only:
            it = range(len(questions) - 1, len(questions))
        else:
            it = range(len(questions))
        qalist=[]
        history = []
        if undisclosed_only:
            for n in range(len(questions)-1):
                qalist.append(questions[n])
                qalist.append(answers[n])
            history=qalist[max(-len(qalist),-n_history*2):]
        for n in it:
            if undisclosed_only:
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            answer = answers[n]
            history.append(question)
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption, 'summary': summary}
            else:
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption, 'summary': summary}

            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]

    return dialog_list


def build_input_from_segments(caption, history, reply, tokenizer, drop_caption=False):
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """

    bos, eos = tokenizer.convert_tokens_to_ids(['<s>', '</s>'])
    sep = eos
    
    instance = {}
    instance["lm_labels"] = reply + [eos]
    caption = list(chain(*caption))

    if not drop_caption:
        # sequence = [[bos] + list(chain(*caption))] + history + [reply + ([eos] if with_eos else [])]

        # NOTE It is important not to include the reply in the input of the encoder -- > the decoder will just
        # learn to copy it --> low train/val loss but no learning is happening
        sequence = [[bos] + caption + [eos]] + [[sep] + s for s in history] + [[eos]]
    else:
        sequence = [[bos]] + [[sep] + s for s in history] + [[eos]]

    instance["input_ids"] = list(chain(*sequence))
    return instance


class AVSDDataSet(Dataset):
    def __init__(self, config, medium, vis_processor, text_processor, split
        # tokenizer, features=None, drop_rate=0.0, train=True
        ):
        self.config = config
        self.medium = medium
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.split = split
        self.batch_size = config['batch_size_test_{}'.format(medium)] if split == 'test' else config['batch_size_{}'.format(medium)] 
        self.root_vis = config['root_raw_vis_{}_{}'.format(medium, split)]
        self.dialogs = get_dataset(config, split) 

        if split == 'test':
            self.dialogs = self.dialogs[config['start_idx_gen']: config['end_idx_gen']]

        num_samples = config['num_samples_{}'.format(self.medium)]
        if num_samples > 0:
            self.dialogs = self.dialogs[:num_samples]

    def __len__(self):
        return len(self.dialogs)
    
    def load_vid(self, vid_id):
        vid_dir_path = os.path.join(self.root_vis, vid_id + '.mp4')

        frames, _, _ = read_frames_decord(vid_dir_path, self.config.num_frames)
        frames = [self.vis_processor(f).unsqueeze(0) for f in frames]

        vis = torch.cat(frames, dim=0)
        return vis

    def load_vid_old(self, vid_id):
        # if vid_id == 'QQM8M':
        #     print('bla')
        vid_dir_path = os.path.join(self.root_vis, vid_id)
        frame_paths = [os.path.join(vid_dir_path, f) for f in  os.listdir(vid_dir_path)]
        frame_paths.sort()
        num_avail_frames = len(frame_paths)
        delta =  int(num_avail_frames / (self.config['num_frames'] - 1))
        ran = list(range(0, num_avail_frames, delta))
        if len(ran) < self.config['num_frames']:
            ran.extend([num_avail_frames - 1 for _ in range(self.config['num_frames'] - len(ran))])
        if len(ran) > self.config['num_frames']:
            ran = ran[:self.config['num_frames']]
        assert len(ran) == self.config['num_frames'], f"vid {vid_id} - loaded {len(ran)}/{len(frame_paths)} frames"
        frame_paths = [frame_paths[i] for i in ran]
        vis = [Image.open(p).convert('RGB') for p in frame_paths]
        vis = [transforms.PILToTensor()(v).unsqueeze(0) for v in vis]
        vis = torch.cat(vis, dim=0)
        vis = self.trans(vis)
        return vis

    def __getitem__(self, index):
        dialog  = self.dialogs[index]
        vid_id  = dialog['vid']
        
        caption = dialog['caption']
        summary = dialog['summary'] 
        history = dialog['history']
        answer  = dialog['answer']
        
        caption = self.text_processor(caption)
        summary = self.text_processor(summary)
        if self.config.dstc != 10:
            caption = caption + ' ' + summary

        history = [self.text_processor(h) for h in history]
        answer  = self.text_processor(answer, remove_period=True)
        
        if self.config.embed_from_llm:
            if self.config.llm_family in ['llama', 'mistral']:
                cls_tok = ''
                sep_tok = ' '
                bos_tok = '<s>'
                eos_tok = '</s>'
            else:
                cls_tok = '<s>'
                sep_tok = '</s>'
                bos_tok = '<pad>'
                eos_tok = '</s>'
        else:
            cls_tok = '[CLS]' 
            sep_tok = '[SEP]'
            bos_tok = '[SEP]'
            eos_tok = '[SEP]'
        
        caption = cls_tok + caption + sep_tok
        history = sep_tok.join(history)
        history = history + sep_tok

        # load the video frames
        vis = self.load_vid(vid_id)

        return vis, caption, history, answer, vid_id


def load_avsd_dataset(config, vis_processor, text_processor, split):
    # data_file = config['anno_avsd_{}'.format(split)]
    # dataset_list = get_dataset(config, split, tokenizer_enc_dec)
    dataset = AVSDDataSet(config, 'avsd', vis_processor, text_processor, split)
    return dataset
