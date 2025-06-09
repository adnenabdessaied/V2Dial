# coding: utf-8
# author: noctli
import json
import os
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from itertools import chain
from torchvision import transforms
from .utils import type_transform_helper
from .utils import open_img

def tokenize(text, tokenizer, return_tensor=False):
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if return_tensor:
        return torch.tensor(tokenized_text).long()
    return tokenized_text


def get_dataset(config, split):

    dialog_pth    = config['anno_visdial_{}'.format(split)]
    dialog_data   = json.load(open(dialog_pth, 'r'))['data']

    all_answers   = dialog_data['answers']
    all_questions = dialog_data['questions']
    dialog_list   = []
    n_history     = config['num_hist_turns_visdial']
    vid_set       = set()
    undisclosed_only = False

    pbar = tqdm(dialog_data['dialogs'])
    pbar.set_description('[INFO] Loading VisDial - {}'.format(split))
    for dialog in pbar:
        caption   = dialog['caption'] + ' .'
        questions = [all_questions[d['question']] + ' ?' for d in dialog['dialog']]
        answers   = [all_answers[d['answer']] + ' .' for d in dialog['dialog']]

        # answer_opts = [[all_answers[key] for key in d['answer_options']] for d in dialog['dialog']]
        # if 'test' in config['anno_visdial_{}'.format(split)]:
        #     gt_indices = [-1 for _ in range(len(questions))]
        # else:
        #     gt_indices = [d['gt_index'] for d in dialog['dialog']]

        vid = dialog['image_id']
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
            # answer_opt = answer_opts[n]
            # gt_index = gt_indices[n]
            history.append(question)
            # if n_history == 0:
            #     item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption, 'round': n+1, 'answer_opts': answer_opt, 'gt_index': gt_index}
            # else:
            #     item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption, 'round': n+1, 'answer_opts': answer_opt, 'gt_index': gt_index}
            
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption, 'round': n+1}
            else:
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption, 'round': n+1}

            
            
            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]

    return dialog_list


class VisDial(Dataset):
    def __init__(self, config, medium, vis_processor, text_processor, split
        # tokenizer, features=None, drop_rate=0.0, train=True
        ):
        self.config = config
        self.medium = medium
        self.split = split
        self.vis_processor = vis_processor
        self.text_processor = text_processor
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
    
    def load_img(self, vid_id):
        file_pth = os.path.join(self.root_vis, f'{vid_id}.jpg')
        vis = open_img(file_pth)
        vis = self.vis_processor(vis).unsqueeze(0)
        return vis

    def __getitem__(self, index):
        dialog  = self.dialogs[index]

        vid_id  = dialog['vid']
        caption = dialog['caption']
        history = dialog['history']
        answer  = dialog['answer']
        d_round = dialog['round']

        caption = self.text_processor(caption)
        history = [self.text_processor(h) for h in history]
        answer  = self.text_processor(answer, remove_period=True)

        
        # if self.split == 'test':
        #     answer_opts  = dialog['answer_opts'] 
        #     answer_opts  = [self.text_processor(a) for a in answer_opts]

        #     gt_index     = dialog['gt_index']
        #     dialog_round = dialog['round']

        #     dense_key    = str(vid_id) + '_' + str(dialog_round)
        #     gt_relevance = self.dense_annos.get(dense_key, -1)
        #     # eval_data = (answer_opts, gt_index, gt_relevance)

        
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
        vis = self.load_img(vid_id)

        # if self.split == 'test':
        #     return vis, caption, history, answer, vid_id, answer_opts, gt_relevance, gt_index

        # else:
        return vis, caption, history, answer, vid_id,  d_round
    


def load_visdial_dataset(config, vis_processor, text_processor, split):
    dataset = VisDial(config, 'visdial', vis_processor, text_processor, split)
    return dataset
