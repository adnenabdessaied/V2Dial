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
    n_history     = config['num_hist_turns']
    vid_set       = set()
    
    pbar = tqdm(dialog_data['dialogs'])
    pbar.set_description('[INFO] Loading VisDial - {}'.format(split))
    for dialog in pbar:
        caption   = dialog['caption']
        questions = [all_questions[d['question']] for d in dialog['dialog']]
        answers   = [all_answers[d['answer']] for d in dialog['dialog']]

        vid = dialog["image_id"]
        vid_set.add(vid)
        # if undisclosed_only:
        #     it = range(len(questions) - 1, len(questions))
        # else:
        it = range(len(questions))
        qalist=[]
        history = []
        # if undisclosed_only:
        #     for n in range(len(questions)-1):
        #         qalist.append(questions[n])
        #         qalist.append(answers[n])
        #     history=qalist[max(-len(qalist),-n_history*2):]

        for n in it:
            # if undisclosed_only:
            #     assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            answer = answers[n]
            history.append(question)
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption}
            else:
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption}
            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]
    return dialog_list


class Champagne(Dataset):
    def __init__(self, config, medium, vis_processor, text_processor, split):

        self.config = config
        self.medium = medium
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.split = split
        self.batch_size = config['batch_size_{}'.format(medium)]
        
        self.root_vis = config['root_raw_vis_{}_{}'.format(medium, split)]

        # get the mapping between caption and image/video
        mapping_path = config.get('mapping_path_{}_{}'.format(medium, split), None) 
        with open(mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)

        ids = list(self.mapping.keys())
        ids.sort()

        # reserve some samples for validation
        if split == 'train':
            self.ids = ids[config.num_val_samples:]
        elif split == 'val':
            self.ids = ids[:config.num_val_samples]

        num_samples = config['num_samples_{}'.format(self.medium)]
        if num_samples > 0:
            self.ids = self.ids[:num_samples]

    def __len__(self):
        return len(self.ids)
    
    
    def padding(self, seq, pad_token, max_len=None): 
        if max_len is None:
            max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        orig_len = [s.size(0) for s in seq]
        return result, orig_len

    def __getitem__(self, index):
        item = self.mapping[self.ids[index]]
        # load the videos
        pth = os.path.join(self.root_vis, item['path'])
        f_names = os.listdir(pth)
        if len(f_names) == 0:
            with open('/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new/emergency/item.pkl', 'rb') as f:
                item = pickle.load(f)
        
        # load the videos
        pth = os.path.join(self.root_vis, item['path'])
        f_names = os.listdir(pth)
        f_names.sort()

        if len(f_names) < self.config['num_frames']:
            f_names += [f_names[-1]] * (self.config['num_frames'] - len(f_names))
        elif len(f_names) > self.config['num_frames']:
            f_names = f_names[:self.config['num_frames']]

        pth = [os.path.join(pth, f_name) for f_name in f_names]
        try:
            vis = [Image.open(p).convert('RGB') for p in pth]
        except:
            with open('/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new/emergency/item.pkl', 'rb') as f:
                item = pickle.load(f)

            # load the videos
            pth = os.path.join(self.root_vis, item['path'])
            f_names = os.listdir(pth)
            f_names.sort()
            
            pth = [os.path.join(pth, f_name) for f_name in f_names]
            vis = [Image.open(p).convert('RGB') for p in pth]
    
        vis = [self.vis_processor(v).unsqueeze(0) for v in vis]
        vis = torch.cat(vis, dim=0)

        dialog = item['dialog']

        caption = dialog['caption']
        history = dialog['history']
        answer  = dialog['answer']

        caption = self.text_processor(caption)
        history = [self.text_processor(h) for h in history]
        answer  = self.text_processor(answer, remove_period=True)

        if self.config.embed_from_llm:
            if self.config.llm_family in ['llama', 'mistral']:
                cls_tok = ''
                sep_tok = ''
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

        # preprocess the textual data
        caption = cls_tok + caption + sep_tok
        history = sep_tok.join(history)
        history = history + sep_tok
        # if self.config.llm_family == 'flan_t5':
            # answer = '<s> ' + self.text_processor(answer) + ' </s>'
        # else:
        # answer = self.text_processor(answer) + eos_tok

        return vis, caption, history, answer


    # def collate_fn(self, batch):
        
    #     BOS, EOS, SEP = self.tokenizer_enc_dec.convert_tokens_to_ids(['<s>', '</s>', '</s>'])

    #     vis_list, cap_list, hist_list, ques_list, ans_list, index_list, vid_id_list = [], [], [], [], [], [], []
    #     batch_size = len(batch)
    #     for b in batch:
    #         vis_list.append(b[0])
    #         cap = [BOS] + tokenize(b[1], self.tokenizer_enc_dec) + [EOS]
    #         cap_list.append(torch.tensor(cap))
    #         if len(b[2])!=0:
    #             hist = [[SEP] + tokenize(s, self.tokenizer_enc_dec) for s in b[2]] + [[EOS]]
    #             hist_list.append(torch.tensor(list(chain(*hist))))
    #         else:
    #             hist = [SEP] + tokenize(b[3], self.tokenizer_enc_dec) + [EOS]
    #             hist_list.append(torch.tensor(hist))

    #         ques = tokenize(b[3], self.tokenizer_enc_dec) + [EOS]
    #         ques_list.append(torch.tensor(ques))
    #         ans = tokenize(b[4], self.tokenizer_enc_dec) + [EOS]
    #         ans_list.append(torch.tensor(ans))
    #         index_list.append(b[5])
    #         vid_id_list.append(b[6])

    #     # pad and keep track of the original lengths
    #     cap_input_ids,  cap_orig_lens  = self.padding(cap_list,  self.tokenizer_experts.pad_token_id) 
    #     hist_input_ids, hist_orig_lens = self.padding(hist_list, self.tokenizer_experts.pad_token_id)
    #     ques_input_ids, ques_orig_lens = self.padding(ques_list, self.tokenizer_experts.pad_token_id)
    #     ans_input_ids,  _              = self.padding(ans_list, -100)

    #     cap_attention_mask  = cap_input_ids  != self.tokenizer_experts.pad_token_id
    #     hist_attention_mask = hist_input_ids != self.tokenizer_experts.pad_token_id
    #     ques_attention_mask = ques_input_ids != self.tokenizer_experts.pad_token_id

    #     total_orig_lens = [sum(l) for l in zip(cap_orig_lens, hist_orig_lens, ques_orig_lens)]
    #     max_len = max(total_orig_lens)

    #     dummy_input_ids_enc_dec = torch.full((batch_size, max_len), self.tokenizer_experts.pad_token_id)
    #     enc_dec_attention_mask = torch.zeros_like(dummy_input_ids_enc_dec, dtype=torch.bool)
    #     for i, l in enumerate(total_orig_lens):
    #         enc_dec_attention_mask[i][:l] = True
    #     # add the masking of the visual input
    #     num_query_tok = self.config['num_temporal_query_tokens_{}'.format(self.config['bert_size'])]
    #     if self.medium in ['avsd', 'msrvtt', 'webvid', 'champagne']:
    #         vis_attention_mask = torch.ones((batch_size, 2 * num_query_tok), dtype=torch.bool)  # *2 for spatial and temporal queries
    #     else:
    #         vis_attention_mask = torch.ones((batch_size, num_query_tok), dtype=torch.bool)  # only spatial queries

    #     enc_dec_attention_mask = torch.concat((vis_attention_mask, enc_dec_attention_mask), dim=1)
    #     # Now prepare the data
    #     vis = torch.stack(vis_list, dim=0)
    #     cap = {
    #         'input_ids': cap_input_ids,
    #         'attention_mask': cap_attention_mask,
    #         'orig_lens': cap_orig_lens
    #     }

    #     hist = {
    #         'input_ids': hist_input_ids,
    #         'attention_mask': hist_attention_mask,
    #         'orig_lens': hist_orig_lens
    #     }

    #     ques = {
    #         'input_ids': ques_input_ids,
    #         'attention_mask': ques_attention_mask,
    #         'orig_lens': ques_orig_lens
    #     }

    #     ans = {
    #         'input_ids': ans_input_ids,
    #     }

    #     enc_dec_input = {
    #         'input_ids': dummy_input_ids_enc_dec,
    #         'attention_mask': enc_dec_attention_mask,
    #     }

    #     index = torch.tensor(index_list)
    #     return vis, cap, hist, ques, ans, enc_dec_input, index, vid_id_list


def load_champagne_dataset(config, vis_processor, text_processor, split):
    dataset = Champagne(config, 'champagne', vis_processor, text_processor, split)
    return dataset