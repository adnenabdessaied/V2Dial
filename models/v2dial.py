import json
import re
import glog as logging
import random
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# from minigpt4.common.registry import registry
from .backbones.blip2 import Blip2Base, disabled_train
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
# from .backbones.encoder_decoder.xflan_t5 import T5ForConditionalGeneration
from .backbones.modeling_mistral import MistralForCausalLM
from .backbones.modeling_llama_v2 import LlamaForCausalLM
from .backbones.moes import MoELayer, Pooler
# from .backbones.moes_huggingface import MoEPooler
# from .backbones.moes_huggingface import MoELayer, MoEPooler
from .modules.temporal_modelling import SpatialAttention, TemporalAttention 
from .common.dist_utils import concat_all_gather, all_gather_with_grad
from .utils import MLM
from utils.dist import is_main_process

# from minigpt4.models.modeling_llama_v2 import LlamaForCausalLM as llm_model
# minigpt4.models.modeling_mistral import MistralForCausalLM as llm_model
# from minigpt4.conversation.conversation import Conversation, SeparatorStyle, StoppingCriteriaList, StoppingCriteriaSub

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import time
import numpy as np

# from minigpt4.models import policies

class V2DialAbstract(Blip2Base):
    def __init__(self):
        super(V2DialAbstract, self).__init__()

    def shift_right(self, input_ids):
        decoder_start_token_id = self.llm.config.decoder_start_token_id
        pad_token_id = self.llm.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )


        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def encode_vis(self, image, device, is_vid=True):
        num_frames = image.size(1)
        bs_pre_reshape = image.size(0)
        if len(image.shape) > 4: 
            image = image.view(-1, *image.shape[-3:]) # for video input flatten the batch and time dimension (4,50,3,224,224) -> (200,3,224,224)
        # with self.maybe_autocast():  # inherited from Blip2Base
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # (200,3,224,224) -> (200,257,1408)
        image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)

        bs, pn, hs = image_embeds.shape
        if self.vit_token_pooling: # concat the each 4 tokens into one token (200,64,5632)
            image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) # (200,64,5632)

        vis_embed = self.vit_proj(image_embeds) # project to LLM input size (200,64,5632) -> (200,64, d_hidden)

        # reshape the video features
        vis_embed = vis_embed.view(bs_pre_reshape, num_frames, -1, vis_embed.size(-1))

        # Perfrom spatial temporal attention
        vis_embed_spatial = self.spatial_att(vis_embed) 
        vis_feat_len = vis_embed_spatial.size(1)
        
        if not self.config.embed_from_llm:
            vis_embed_spatial = vis_embed_spatial + self.token_type_embedding(torch.zeros(bs_pre_reshape, vis_feat_len).long().to(device))
        vis_spatial_mask  = torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

        vis_embed_temporal, vis_temporal_mask = None, None

        if is_vid:
            vis_embed_temporal = self.temporal_att(vis_embed) 
            if not self.config.embed_from_llm:
                vis_embed_temporal = vis_embed_temporal + self.token_type_embedding(torch.ones(bs_pre_reshape, vis_feat_len).long().to(device))
            vis_temporal_mask  =  torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

        return vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask
    
    def tokenize_text(self, text, device, add_bos=False, add_eos=False, max_len=None):
        if max_len:
            text_tokenized = self.tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=max_len,
                truncation=True,
                add_special_tokens=False,
                return_special_tokens_mask=True
            ).to(device)
        else:
            text_tokenized = self.tokenizer(
                text,
                return_tensors='pt',
                padding='longest',
                add_special_tokens=False,
                return_special_tokens_mask=True
            ).to(device)

        text_ids = text_tokenized.input_ids
        text_attention_mask = text_tokenized.attention_mask

        if add_bos:
            bos_ids = torch.LongTensor(text_ids.size(0), 1).fill_(self.tokenizer.bos_token_id).to(device)
            bos_att = torch.LongTensor(text_ids.size(0), 1).fill_(1).to(device)

            text_ids = torch.cat([bos_ids, text_ids], dim=1)
            text_attention_mask = torch.cat([bos_att, text_attention_mask], dim=1)
        
        if add_eos:
            eos_ids = torch.LongTensor(text_ids.size(0), 1).fill_(self.tokenizer.eos_token_id).to(device)
            eos_att = torch.LongTensor(text_ids.size(0), 1).fill_(1).to(device)
            
            text_ids = torch.cat([text_ids, eos_ids], dim=1)
            text_attention_mask = torch.cat([text_attention_mask, eos_att], dim=1)


        return text_ids, text_attention_mask

    def get_extended_attention_mask(self, attention_mask=None):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        return extended_attention_mask

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class V2DialBase(V2DialAbstract):
    def __init__(self, config):
        super(V2DialBase, self).__init__()
        self.config = config

        ################## 1. Select Tokenizer -- We use BERT tokenizer ##################
        bert_config = BertConfig.from_pretrained('bert-{}-uncased'.format(config.expert_size))

        tokenizer = AutoTokenizer.from_pretrained('bert-{}-uncased'.format(config.expert_size))

        text_embedding = BertEmbeddings(bert_config)
        text_embedding.apply(self.init_weights)

        token_type_embedding = nn.Embedding(3, bert_config.hidden_size)  # Number of modality types (temp/spa/text)
        token_type_embedding.apply(self.init_weights)

        # Define the masking strategy
        mlm_collactor = DataCollatorForLanguageModeling(
            tokenizer, mlm=True, mlm_probability=config.masking_prob, return_tensors='pt')
        
        ################## 2. Select the backbone ViT ##################
        logging.info('[INFO] Loading ViT in progress')
        if config.freeze_vit:
            # vit_precision = 'fp16' if config.fp16 else 'fp32'
            logging.info(f'[INFO] ViT precision: {config.vit_precision}')
            visual_encoder, ln_vision = self.init_vision_encoder(
                config.vit_model, config.image_res, drop_path_rate=0, use_grad_checkpoint=False, precision=config.vit_precision
            )
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info('[INFO] ViT frozen')
      
        else:
            vit_precision = 'fp32'
            visual_encoder, ln_vision = self.init_vision_encoder(
                config.vit_model, config.image_res, drop_path_rate=0, use_grad_checkpoint=False, vit_precision=vit_precision
            )
            logging.info('[INFO] ViT hot')
        logging.info('[INFO] ViT successfully loaded')

        ################## 3. Define the ViT-Expert communication Interface ##################
        self.system_prompt = False
        self.vit_token_pooling = config.vit_token_pooling
        if self.vit_token_pooling:
            vit_proj = nn.Linear(
                1408*4, bert_config.hidden_size
            )
        else:
            vit_proj = nn.Linear(
                1408, bert_config.hidden_size
            )
        vit_proj.apply(self.init_weights)
        
        spatial_att  = SpatialAttention(input_dim=bert_config.hidden_size)
        temporal_att = TemporalAttention(input_dim=bert_config.hidden_size)
    
        spatial_att.apply(self.init_weights)
        temporal_att.apply(self.init_weights)

        ################## 4. Define the Expert layers  ##################
        moe_layers = []

        for moe_layer_idx in range(config.num_moe_layers):
            if moe_layer_idx < self.config.num_moe_modality_layers:
                expert_flag = 'modalities'
            else: 
                expert_flag = 'fusion'
            moe_layer = MoELayer(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                expert_flag,
                use_sep_spatial_temp_experts=config.use_sep_spatial_temp_experts
            )
            moe_layer.apply(self.init_weights)
            moe_layers.append(moe_layer)

            logging.info(f'[INFO] {moe_layer_idx+1}/{config.num_moe_layers} MoE layers successfully loaded')

        moe_layers = nn.ModuleList(moe_layers)
        moe_norm   = nn.LayerNorm(bert_config.hidden_size)

        ################## 5. Define the projection layers for contrastive learning  ##################
        temp_proj    = nn.Linear(bert_config.hidden_size, config.joint_dim)
        spatial_proj = nn.Linear(bert_config.hidden_size, config.joint_dim)
        vision_proj  = nn.Linear(bert_config.hidden_size, config.joint_dim)
        cap_proj     = nn.Linear(bert_config.hidden_size, config.joint_dim)

        temp_proj.apply(self.init_weights)
        spatial_proj.apply(self.init_weights)
        vision_proj.apply(self.init_weights)
        cap_proj.apply(self.init_weights)

        ################## 6. Define the pooler for matching loss  ##################
        pooler = Pooler(bert_config.hidden_size)
        pooler.apply(self.init_weights)

        ################## 5. Attach the matching heads  ##################
        stm_head = nn.Linear(bert_config.hidden_size, 2)
        vcm_head = nn.Linear(bert_config.hidden_size, 2)
        lm_head  = nn.Linear(bert_config.hidden_size, len(tokenizer))

        stm_head.apply(self.init_weights)
        vcm_head.apply(self.init_weights)
        lm_head.apply(self.init_weights)

        temp = nn.Parameter(0.07 * torch.ones([]))
        # temp = 0.07

        # Attach the components to self
        self.tokenizer            = tokenizer
        self.mlm_collactor        = mlm_collactor
        self.text_embedding       = text_embedding
        self.token_type_embedding = token_type_embedding
        self.visual_encoder       = visual_encoder
        self.ln_vision            = ln_vision
        self.vit_proj             = vit_proj
        self.moe_layers           = moe_layers
        self.moe_norm             = moe_norm   
        self.spatial_att          = spatial_att
        self.temporal_att         = temporal_att
        self.temp_proj            = temp_proj
        self.spatial_proj         = spatial_proj
        self.vision_proj          = vision_proj
        self.cap_proj             = cap_proj
        self.pooler               = pooler
        self.stm_head             = stm_head
        self.vcm_head             = vcm_head
        self.lm_head              = lm_head
        self.temp                 = temp

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def build_query_embeds(self, num_query_tokens, dim_query_tokens):
        query_embeds = nn.Parameter(
            torch.zeros(1, num_query_tokens, dim_query_tokens)
        )
        query_embeds.data.normal_(mean=0.0, std=0.02)
        return query_embeds

    def encode_caption(self, cap):
        cap_output = self.cap_expert(
            input_ids=cap.input_ids,
            attention_mask=cap.attention_mask,
            return_dict=True,
        )
        cap_embeds = cap_output.last_hidden_state
        pooled_cap_embeds = cap_embeds[:, 0]
        return cap_embeds, pooled_cap_embeds

    def encode_vis_old(self, vis, media_type):
        # if media_type == 'webvid':
        #     bs, num_frames, c, h, w = vis.size()
        #     # reshape
        #     vis = vis.view(bs * num_frames, c, h, w)
        vis_embed = self.beit(vis).last_hidden_state
        # vis_embed = self.beit_layernorm(vis_output.last_hidden_state)
        # remove cls token embedding
        vis_embed = vis_embed[:, :, 1:, :]
        vis_embed = self.beit_lin(vis_embed)
        # perform spatial attention
        vis_spatial_embed = self.spatial_att(vis_embed)
        vis_temp_embed = self.tempotal_att(vis_embed) if media_type in ['webvid', 'msrvtt', 'champagne', 'avsd'] else None

        return vis_spatial_embed, vis_temp_embed

    def encode_queries(self, query_embeds, vis_embeds, vis_mode):
        if vis_mode == 'spatial':
            expert     = self.spatial_expert
            layer_norm = self.spatial_layernorm
        elif vis_mode == 'temporal':
            expert     = self.temporal_expert
            layer_norm = self.temporal_layernorm
        else:
            raise ValueError(f'[ERROR] {vis_mode} not implemented!')

        attention_mask = torch.ones(
            query_embeds.size()[:-1], dtype=torch.long).to(vis_embeds.device)

        vis_attention_mask = torch.ones(
            vis_embeds.size()[:-1], dtype=torch.long).to(vis_embeds.device)

        if self.config['expert_layer_type'] == 'bert':

            output_dict = expert(
                encoder_embeds=query_embeds,
                encoder_hidden_states=vis_embeds,
                encoder_attention_mask=vis_attention_mask,
            )
            query_embeds = layer_norm(output_dict.last_hidden_state)
            pooled_query_embeds = output_dict.pooler_output

        elif self.config['expert_layer_type'] == 'bart':
            output_dict = expert(
                inputs_embeds=query_embeds,
                attention_mask=attention_mask,
                cross_embeds=vis_embeds,
                cross_attention_mask=vis_attention_mask,
            )

            query_embeds = layer_norm(output_dict.last_hidden_state)
            pooled_query_embeds = query_embeds[:, 0]

        return query_embeds, pooled_query_embeds

    # def encode_vis(self, image, device, is_vid=True):
    #     num_frames = image.size(1)
    #     bs_pre_reshape = image.size(0)
    #     if len(image.shape) > 4: 
    #         image = image.view(-1, *image.shape[-3:]) # for video input flatten the batch and time dimension (4,50,3,224,224) -> (200,3,224,224)
    #     # with self.maybe_autocast():  # inherited from Blip2Base
    #     image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # (200,3,224,224) -> (200,257,1408)
    #     image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)

    #     bs, pn, hs = image_embeds.shape
    #     if self.vit_token_pooling: # concat the each 4 tokens into one token (200,64,5632)
    #         image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) # (200,64,5632)

    #     vis_embed = self.vit_proj(image_embeds) # project to llama input size (200,64,5632) -> (200,64,4096)

    #     # reshape the video features
    #     vis_embed = vis_embed.view(bs_pre_reshape, num_frames, -1, vis_embed.size(-1))


    #     # Perfrom spatial temporal attention
    #     vis_embed_spatial = self.spatial_att(vis_embed) 
    #     vis_feat_len = vis_embed_spatial.size(1)
        
    #     vis_embed_spatial = vis_embed_spatial + self.token_type_embedding(torch.zeros(bs_pre_reshape, vis_feat_len).long().to(device))
    #     vis_spatial_mask  = torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

    #     vis_embed_temporal, vis_temporal_mask = None, None

    #     if is_vid:
    #         vis_embed_temporal = self.temporal_att(vis_embed) + self.token_type_embedding(torch.ones(bs_pre_reshape, vis_feat_len).long().to(device))
    #         vis_temporal_mask  =  torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

    #     return vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask
    
    def encode_vis_with_seq_spa_temp_att(self, image, device, is_vid=True):
        num_frames = image.size(1)
        bs_pre_reshape = image.size(0)
        if len(image.shape) > 4: 
            image = image.view(-1, *image.shape[-3:]) # for video input flatten the batch and time dimension (4,50,3,224,224) -> (200,3,224,224)
        # with self.maybe_autocast():  # inherited from Blip2Base
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # (200,3,224,224) -> (200,257,1408)
        image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)

        bs, pn, hs = image_embeds.shape
        if self.vit_token_pooling: # concat the each 4 tokens into one token (200,64,5632)
            image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) # (200,64,5632)

        vis_embed = self.vit_proj(image_embeds) # project to llama input size (200,64,5632) -> (200,64,4096)

        # reshape the video features
        vis_embed = vis_embed.view(bs_pre_reshape, num_frames, -1, vis_embed.size(-1))
        size_orig = vis_embed.size()

        # Perfrom spatial temporal attention
        vis_embed = self.spatial_att(vis_embed) 
        if is_vid:
            vis_embed = vis_embed.view(size_orig)
            vis_embed = self.temporal_att(vis_embed)
        
        vis_feat_len = vis_embed.size(1)

        vis_embed = vis_embed + self.token_type_embedding(torch.zeros(bs_pre_reshape, vis_feat_len).long().to(device))
        vis_mask  =  torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

        return vis_embed, vis_mask

    def tokenize_text(self, text, device, add_bos=False, add_eos=False, max_len=None):
        if max_len:
            text_tokenized = self.tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=max_len,
                truncation=True,
                add_special_tokens=False,
                return_special_tokens_mask=True
            ).to(device)
        else:
            text_tokenized = self.tokenizer(
                text,
                return_tensors='pt',
                padding='longest',
                add_special_tokens=False,
                return_special_tokens_mask=True
            ).to(device)

        text_ids = text_tokenized.input_ids
        text_attention_mask = text_tokenized.attention_mask

        if add_bos:
            bos_ids = torch.LongTensor(text_ids.size(0), 1).fill_(self.tokenizer.bos_token_id).to(device)
            bos_att = torch.LongTensor(text_ids.size(0), 1).fill_(1).to(device)

            text_ids = torch.cat([bos_ids, text_ids], dim=1)
            text_attention_mask = torch.cat([bos_att, text_attention_mask], dim=1)
        
        if add_eos:
            eos_ids = torch.LongTensor(text_ids.size(0), 1).fill_(self.tokenizer.eos_token_id).to(device)
            eos_att = torch.LongTensor(text_ids.size(0), 1).fill_(1).to(device)
            
            text_ids = torch.cat([text_ids, eos_ids], dim=1)
            text_attention_mask = torch.cat([text_attention_mask, eos_att], dim=1)


        return text_ids, text_attention_mask

    def encode_text(self, text, max_len, device):
        text_tokenized = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            max_length=max_len,
            truncation=True,
            add_special_tokens=False
        ).to(device)
        text_ids = text_tokenized.input_ids
        text_embeds = self.embed(text_ids)
        text_attention_mask = text_tokenized.attention_mask
        return text_embeds, text_ids, text_attention_mask
        
    def encode_spatial_toks(self, batch_size, device):
        # ['<vis>', '<spatial>', '<temporal>', '<caption>', '<history>']

        special_toks_ids = self.tokenizer(
            '<s><vis><spatial><temporal><caption><history></s>',
            return_tensors='pt',
            padding='longest',
            truncation=True,
            add_special_tokens=False
        ).to(device)

        special_toks_embeds = self.embed(special_toks_ids.input_ids)
        special_toks_embeds = special_toks_embeds.repeat(batch_size, 1, 1)
        return special_toks_embeds

    def construt_input_embeds_stage_1(self, vis_embed, cap_embed, special_toks_embeds, cap_attention_mask, media_type, device):
        batch_size = vis_embed.size(0)
        embed_dim = vis_embed.size(-1)
        vis_embed = vis_embed.view(batch_size, -1, embed_dim)

        input_embeds = []
        input_attention_mask = []
        special_toks_indices = {
            '<s>': 0,
            '<vis>': 1,
            '<spatial>': 2,
        }
        # special_toks_embeds = <s> <vis> <spatial> <temporal> <caption> <history> </s>
        # for video: <s><vis><spatial>[spatial_featurres]<temporal>[temporal_featurres]<caption>[caption_features]</s>
        # for image: <s><vis><spatial>[spatial_featurres]<caption>[caption_features]</s>

        input_embeds.append(special_toks_embeds[:, 0:3, :])  # <s> <vis> <spatial>
        input_attention_mask.append(torch.ones(input_embeds[-1].size()[:-1], dtype=torch.long).to(device))
        
        input_embeds.append(vis_embed.clone())  # [spatial_features]
        input_attention_mask.append(torch.ones(input_embeds[-1].size()[:-1], dtype=torch.long).to(device))
        
        if media_type == 'webvid':
            # here we copy the original vis_embeds twice and will apply spatial and temporal attention later
            input_embeds.append(special_toks_embeds[:, 3:4, :])  # <temporal>
            input_attention_mask.append(torch.ones(input_embeds[-1].size()[:-1], dtype=torch.long).to(device))
            special_toks_indices['<temporal>'] = special_toks_indices['<spatial>'] + input_embeds[-2].size(1) + 1

            input_embeds.append(vis_embed.clone())  # [temporal_features]
            input_attention_mask.append(torch.ones(input_embeds[-1].size()[:-1], dtype=torch.long).to(device))


        input_embeds.append(special_toks_embeds[:, 4:5, :])  # <caption>
        input_attention_mask.append(torch.ones(input_embeds[-1].size()[:-1], dtype=torch.long).to(device))

        if media_type == 'webvid':
            special_toks_indices['<caption>'] = special_toks_indices['<temporal>'] + input_embeds[-2].size(1) + 1
        elif media_type == 'cc3m':
            special_toks_indices['<caption>'] = special_toks_indices['<spatial>'] + input_embeds[-2].size(1) + 1

        input_embeds.append(cap_embed)  # [caption_features]
        input_attention_mask.append(cap_attention_mask)

        input_embeds.append(special_toks_embeds[:, 6:7, :])  # </s>
        input_attention_mask.append(torch.ones(input_embeds[-1].size()[:-1], dtype=torch.long).to(device))
        special_toks_indices['</s>'] = special_toks_indices['<caption>'] + input_embeds[-2].size(1) + 1

        input_embeds = torch.cat(input_embeds, dim=1)
        input_attention_mask = torch.cat(input_attention_mask, dim=1)
        assert input_embeds.size()[:-1] == input_attention_mask.size()

        return input_embeds, input_attention_mask, special_toks_indices

    def construct_global_input(self, cap_ids, cap_attention_mask, vid_feat_len, media_type, device):
        # for video: <s><vis><spatial>[spatial_featurres]<temporal>[temporal_features]<caption>[caption_features]</s>
        # for image: <s><vis><spatial>[spatial_featurres]<caption>[caption_features]</s>
        batch_size = cap_ids.size(0)
        special_toks_indices = {
            '<s>': 0,
            '<vis>': 1,
            '<spatial>': 2,
        }
        
        ids = [self.added_vocab['<s>']] + [self.added_vocab['<vis>']] + [self.added_vocab['<spatial>']]
        ids += vid_feat_len * [self.added_vocab['<pad>']]
        if media_type == 'webvid':
            ids += [self.added_vocab['<temporal>']]
            special_toks_indices['<temporal>'] = len(ids) - 1
            ids += vid_feat_len * [self.added_vocab['<pad>']]

        ids += [self.added_vocab['<caption>']]
        special_toks_indices['<caption>'] = len(ids) - 1
        ids += cap_ids.size(1) * [self.added_vocab['<pad>']]

        ids += [self.added_vocab['</s>']]
        special_toks_indices['</s>'] = len(ids) - 1
        total_len = len(ids)

        ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        ids[:, special_toks_indices['<caption>'] + 1: special_toks_indices['</s>']] = cap_ids
        
        mask = torch.ones((batch_size, total_len), device=device)
        mask[:, special_toks_indices['<caption>'] + 1: special_toks_indices['</s>']] = cap_attention_mask 

        return ids, mask, special_toks_indices

    def compute_contrastive_loss(self, x, y_all, y, x_all):
        sim_x2y = torch.mm(x, y_all.t())  # (bs, bs*ngpus)
        sim_x2y = sim_x2y / self.temp

        sim_y2x = torch.mm(y, x_all.t())  # (bs, bs*ngpus)
        sim_y2x = sim_y2x / self.temp
        
        rank = dist.get_rank() if self.config['distributed'] else 0
        
        bs = x.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            x.device
        )
        loss_contrastive = (
            F.cross_entropy(sim_x2y, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_y2x, targets, label_smoothing=0.1)
            ) / 2

        return loss_contrastive, sim_x2y, sim_y2x
    
    def get_extended_attention_mask(self, attention_mask=None):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        return extended_attention_mask

    def shared_forward(
            self,
            vis_spatial, vis_spatial_mask, vis_temporal, vis_temporal_mask,
            cap_ids, cap_mask, is_vid, device):
    
        # is_vid = media_type == 'webvid'
        # batch_size = len(cap)
        vis_feat_len = vis_spatial.size(1)
        input_embeds = []
        input_masks  = []

        input_embeds.append(vis_spatial)
        input_masks.append(vis_spatial_mask)

        if is_vid:
            input_embeds.append(vis_temporal)
            input_masks.append(vis_temporal_mask)
       
        cap_embeds = self.text_embedding(cap_ids) + self.token_type_embedding(torch.ones_like(cap_ids).long().fill_(2))
        cap_feat_len = cap_embeds.size(1)

        input_embeds.append(cap_embeds)
        input_masks.append(cap_mask)

        input_embeds = torch.cat(input_embeds, dim=1)
        input_masks  = torch.cat(input_masks, dim=1)

        # expand the mask
        input_masks = self.get_extended_attention_mask(attention_mask=input_masks)

        # MoEs feed-forward
        for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
            if moe_layer_idx < self.config.num_moe_modality_layers:
                expert_flag = 'modalities'
            else: 
                expert_flag = 'fusion'
            
            input_embeds = moe_layer(input_embeds, vis_feat_len, cap_feat_len, expert_flag, is_vid=is_vid, mask=input_masks)

        #TODO normalize the output () !!!!!!
        input_embeds = self.moe_norm(input_embeds)

        # return the features
        spatial_feats  = input_embeds[:, :vis_feat_len]
        temporal_feats = input_embeds[:, vis_feat_len:2*vis_feat_len] if is_vid else None
        cap_feats      = input_embeds[:, -cap_feat_len:]
        cls_feats      = self.pooler(cap_feats)

        moe_outputs = {
            'spatial_feats': spatial_feats,
            'temporal_feats': temporal_feats,
            'cap_feats': cap_feats,
            'cls_feats': cls_feats,
        }

        return moe_outputs

    def shared_forward_no_sep_spatial_temporal_experts(
            self,
            vis, vis_mask,
            cap_ids, cap_mask, is_vid, device):
    
        # is_vid = media_type == 'webvid'
        # batch_size = len(cap)
        vis_feat_len = vis.size(1)
        input_embeds = []
        input_masks  = []

        input_embeds.append(vis)
        input_masks.append(vis_mask)

        # if is_vid:
        #     input_embeds.append(vis_temporal)
        #     input_masks.append(vis_temporal_mask)
       
        cap_embeds = self.text_embedding(cap_ids) + self.token_type_embedding(torch.ones_like(cap_ids).long().fill_(2))
        cap_feat_len = cap_embeds.size(1)

        input_embeds.append(cap_embeds)
        input_masks.append(cap_mask)

        input_embeds = torch.cat(input_embeds, dim=1)
        input_masks  = torch.cat(input_masks, dim=1)

        # expand the mask
        input_masks = self.get_extended_attention_mask(attention_mask=input_masks)

        # MoEs feed-forward
        for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
            if moe_layer_idx < self.config.num_moe_modality_layers:
                expert_flag = 'modalities'
            else: 
                expert_flag = 'fusion'
            
            input_embeds = moe_layer(input_embeds, vis_feat_len, cap_feat_len, expert_flag, is_vid=is_vid, mask=input_masks)

        #TODO normalize the output () !!!!!!
        input_embeds = self.moe_norm(input_embeds)

        # return the features
        vis_feats = input_embeds[:, :vis_feat_len]
        cap_feats = input_embeds[:, -cap_feat_len:]
        cls_feats = self.pooler(cap_feats)

        moe_outputs = {
            'vis_feats': vis_feats,
            'cap_feats': cap_feats,
            'cls_feats': cls_feats,
        }

        return moe_outputs

    def vcm_iteration(self, vis, cap, neg_vis, is_vid, device):
        # Prepare the vis data
        # is_vid = media_type == 'webvid'
        num_positive_samples = len(cap) // 2
        num_negative_samples = len(cap) - num_positive_samples

        vcm_labels = torch.cat([torch.ones(num_positive_samples), torch.zeros(num_negative_samples)]).to(device)
        vcm_labels = vcm_labels[torch.randperm(vcm_labels.size(0))].long()

        # now get the mixed vis data

        vis_mixed = [p if vcm_labels[i] == 1 else n for i, (p, n) in enumerate(zip(vis, neg_vis))]
        vis_mixed = torch.stack(vis_mixed, dim=0)

        cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=self.config.max_cap_len)

        if self.config.use_sep_spatial_temp_experts: 
            vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = self.encode_vis(vis_mixed, device, is_vid=is_vid)
            moe_outputs = self.shared_forward(
                vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask, cap_ids, cap_mask, is_vid, device)
        else:
            vis_embed, vis_mask = self.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)
            moe_outputs = self.shared_forward_no_sep_spatial_temporal_experts(
                vis_embed, vis_mask, cap_ids, cap_mask, is_vid, device)

        vcm_logits = self.vcm_head(moe_outputs['cls_feats'])
        loss_vcm = F.cross_entropy(vcm_logits, vcm_labels)
        return loss_vcm

    def stm_iteration(self, vis, cap, neg_vis, is_vid, device):
        num_positive_samples = len(cap) // 2
        num_negative_samples = len(cap) - num_positive_samples

        stm_labels = torch.cat([torch.ones(num_positive_samples), torch.zeros(num_negative_samples)]).to(device)
        stm_labels = stm_labels[torch.randperm(stm_labels.size(0))].long()

        vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = self.encode_vis(vis, device, is_vid=is_vid)
        neg_vis_embed_spatial, _           , neg_vis_embed_temporal, _             = self.encode_vis(neg_vis, device, is_vid=is_vid)

        # now get the mixed vis data
        vis_embed_spatial_mixed  = []
        vis_embed_temporal_mixed = []

        for i, (pos_spatial, pos_temporal, neg_spatial, neg_temporal) in enumerate(
            zip(vis_embed_spatial, vis_embed_temporal, S, neg_vis_embed_temporal)):
            if stm_labels[i] == 1:
                vis_embed_spatial_mixed.append(pos_spatial)
                vis_embed_temporal_mixed.append(pos_temporal)
            else:
                # 50% negative spatial / 50% negative temporal
                if torch.rand(1).item() < 0.5:
                    vis_embed_spatial_mixed.append(pos_spatial)
                    vis_embed_temporal_mixed.append(neg_temporal)
                else:
                    vis_embed_spatial_mixed.append(neg_spatial)
                    vis_embed_temporal_mixed.append(pos_temporal)

        vis_embed_spatial_mixed = torch.stack(vis_embed_spatial_mixed, dim=0)
        vis_embed_temporal_mixed = torch.stack(vis_embed_temporal_mixed, dim=0)

        cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=self.config.max_cap_len)

        moe_outputs = self.shared_forward(
            vis_embed_spatial_mixed, vis_spatial_mask, vis_embed_temporal_mixed, vis_temporal_mask, cap_ids, cap_mask, is_vid, device)

        stm_logits = self.vcm_head(moe_outputs['cls_feats'])
        loss_stm = F.cross_entropy(stm_logits, stm_labels)
        return loss_stm

    def mlm_iteration(self, vis, cap, is_vid, device):
        if self.config.use_sep_spatial_temp_experts: 
            vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = self.encode_vis(vis, device, is_vid=is_vid)
        else:
            vis_embed, vis_mask = self.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)

        cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=self.config.max_cap_len)
        cap_ids = cap_ids.tolist()

        # NOTE We make sure to mask some tokens here to avoid nan loss later
        mlm_output = self.mlm_collactor(cap_ids)
        cap_ids = mlm_output['input_ids'].to(device)
        labels_cap = mlm_output['labels'].to(device)

        if self.config.use_sep_spatial_temp_experts: 
            moe_outputs = self.shared_forward(
                vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask, cap_ids, cap_mask, is_vid, device)
        else:
            moe_outputs = self.shared_forward_no_sep_spatial_temporal_experts(
                vis_embed, vis_mask, cap_ids, cap_mask, is_vid, device)

        mlm_logits = self.lm_head(moe_outputs['cap_feats'])
        loss_mlm   = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), labels_cap.view(-1))   
        return loss_mlm

    def vcc_iteration(self, vis, cap, is_vid, device):
        vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = self.encode_vis(vis, device, is_vid=is_vid)
        cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=self.config.max_cap_len)
        
        if self.config.use_sep_spatial_temp_experts: 
            moe_outputs = self.shared_forward(
                vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask, cap_ids, cap_mask, is_vid, device)
            vis_feats = moe_outputs['spatial_feats']
            if is_vid:
                vis_feats = torch.cat([moe_outputs['spatial_feats'], moe_outputs['temporal_feats']], dim=1)
        else:
            vis_embed, vis_mask = self.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)
            moe_outputs = self.shared_forward_no_sep_spatial_temporal_experts(
                vis_embed, vis_mask, cap_ids, cap_mask, is_vid, device)
            vis_feats = moe_outputs['vis_feats']

        cap_feats = F.normalize(self.cap_proj(moe_outputs['cls_feats']), dim=-1)
        vis_feats = F.normalize(self.vision_proj(vis_feats), dim=-1)

        vis_feats_all = concat_all_gather(vis_feats)
        cap_feats_all = concat_all_gather(cap_feats)

        sim_v2c = torch.matmul(
            vis_feats.unsqueeze(1), cap_feats_all.unsqueeze(-1)
        ).squeeze()

        sim_v2c, _ = sim_v2c.max(-1)
        sim_v2c    = sim_v2c / self.temp

        sim_c2v = torch.matmul(
            cap_feats.unsqueeze(1).unsqueeze(1), vis_feats_all.permute(0, 2, 1)
        ).squeeze()

        sim_c2v, _ = sim_c2v.max(-1)
        sim_c2v    = sim_c2v / self.temp

        rank = dist.get_rank() if self.config['distributed'] else 0

        bs = vis_feats.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )
        loss_vcc = (
            F.cross_entropy(sim_v2c, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_c2v, targets, label_smoothing=0.1)
        ) / 2
        return loss_vcc

    def stc_iteration(self, vis, cap, is_vid, device):
        vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = self.encode_vis(vis, device, is_vid=is_vid)
        cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=self.config.max_cap_len)
        moe_outputs = self.shared_forward(
            vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask, cap_ids, cap_mask, is_vid, device)

        spatial_feats  = F.normalize(self.spatial_proj(moe_outputs['spatial_feats']), dim=-1)
        temporal_feats = F.normalize(self.temp_proj(moe_outputs['temporal_feats']), dim=-1)

        spatial_feats_all  = concat_all_gather(spatial_feats)
        temporal_feats_all = concat_all_gather(temporal_feats)

        sim_s2t = torch.matmul(
            spatial_feats.unsqueeze(1), temporal_feats_all
        )

        sim_s2t, _ = sim_s2t.max(-1)
        sim_s2t, _ = sim_s2t.max(-1)
        sim_s2t    = sim_s2t / self.temp

        sim_t2s = torch.matmul(
            temporal_feats.unsqueeze(1), spatial_feats_all
        )

        sim_t2s, _ = sim_t2s.max(-1)
        sim_t2s, _ = sim_t2s.max(-1)
        sim_t2s    = sim_t2s / self.temp

        rank = dist.get_rank() if self.config['distributed'] else 0
        bs = vis.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )
        loss_stc = (
            F.cross_entropy(sim_s2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2s, targets, label_smoothing=0.1)
        ) / 2
        return loss_stc


    def forward(self, vis, cap, neg_vis, media_type):
        device = vis.device        
        is_vid = media_type == 'webvid'
        loss_stc = torch.tensor(0).to(device)
        loss_stm = torch.tensor(0).to(device)
        loss_vcc = torch.tensor(0).to(device)
        loss_vcm = torch.tensor(0).to(device)
        loss_mlm = torch.tensor(0).to(device)

        if self.config.loss_dict['vcm'] != 0:
            loss_vcm = self.vcm_iteration(vis, cap, neg_vis, is_vid, device)

        if self.config.loss_dict['vcc'] != 0:
            loss_vcc = self.vcc_iteration(vis, cap, is_vid, device)

        if self.config.loss_dict['stm'] != 0 and is_vid:
            loss_stm = self.stm_iteration(vis, cap, neg_vis, is_vid, device)

        if self.config.loss_dict['stc'] != 0 and is_vid:
            loss_stc = self.stc_iteration(vis, cap, is_vid, device)

        if self.config.loss_dict['mlm'] != 0:
            loss_mlm = self.mlm_iteration(vis, cap, is_vid, device)

        return dict(
            loss_stc = loss_stc * self.config.loss_dict['stc'],
            loss_stm = loss_stm * self.config.loss_dict['stm'],
            loss_vcc = loss_vcc * self.config.loss_dict['vcc'],
            loss_vcm = loss_vcm * self.config.loss_dict['vcm'],
            loss_mlm = loss_mlm * self.config.loss_dict['mlm'],
        )

    def forward__(self, vis, cap, neg_vis, media_type):

        device = vis.device        
        self.vcm_matching(vis, cap, neg_vis, media_type, device)
        self.shared_forward(vis, cap, media_type, device)
        

        # First init all losses to zeros
        loss_stc = torch.tensor(0).to(device)
        loss_stm = torch.tensor(0).to(device)
        loss_vcc = torch.tensor(0).to(device)
        loss_vcm = torch.tensor(0).to(device)
        loss_mlm = torch.tensor(0).to(device)

        batch_size = len(cap)
        # First get the visual features depending on the media type
        vis_embed = self.encode_vis(vis)
        neg_vis_embed = self.encode_vis(neg_vis)

        embed_dim = vis_embed.size(-1) 
        num_frames = vis.size(1)
        # reshape the video features
        vis_embed     = vis_embed.view(batch_size, num_frames, -1, embed_dim)
        neg_vis_embed = neg_vis_embed.view(batch_size, num_frames, -1, embed_dim)

        # Perfrom spatial temporal attention and reshape
        vis_embed_spatial = self.spatial_att(vis_embed)
        # vis_embed_spatial = vis_embed_spatial.view(batch_size, -1, embed_dim)
        
        neg_vis_embed_spatial = self.spatial_att(neg_vis_embed)
        # neg_vis_embed_spatial = neg_vis_embed_spatial.view(batch_size, -1, embed_dim)

        if media_type == 'webvid':
            vis_embed_temporal = self.temporal_att(vis_embed)
            # vis_embed_temporal = vis_embed_temporal.view(batch_size, -1, embed_dim)

            neg_vis_embed_temporal = self.temporal_att(neg_vis_embed)
            # neg_vis_embed_temporal = neg_vis_embed_temporal.view(batch_size, -1, embed_dim)

        spatial_feat_len = vis_embed_spatial.size(1)

        # construct the global input tensor --> use place holder for vis features
        cap_ids, cap_attention_mask = self.tokenize_text(cap, device, max_len=self.config.max_cap_len)
        input_ids, input_mask, special_toks_indices = self.construct_global_input(cap_ids, cap_attention_mask, spatial_feat_len, media_type, device)

        input_embeds = self.embed(input_ids)
      
        if media_type == 'webvid':
            input_embeds[:, special_toks_indices['<spatial>'] + 1: special_toks_indices['<temporal>'], :] = vis_embed_spatial
            input_embeds[:, special_toks_indices['<temporal>'] + 1: special_toks_indices['<caption>'], :] = vis_embed_temporal

        elif media_type == 'cc3m':
            input_embeds[:, special_toks_indices['<spatial>'] + 1: special_toks_indices['<caption>'], :] = vis_embed_spatial

        # LLM --> MoEs
        input_embeds = self.moe_llm_bottleneck(input_embeds)
        input_embeds_orig = input_embeds.clone()

        neg_vis_embed_spatial = self.moe_llm_bottleneck(neg_vis_embed_spatial)

        if media_type == 'webvid':
            neg_vis_embed_temporal = self.moe_llm_bottleneck(neg_vis_embed_temporal)

        for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
            if moe_layer_idx < self.config.num_moe_modality_layers:
                expert_flag = 'modalities'
            else: 
                expert_flag = 'fusion'

            input_embeds = moe_layer(input_embeds, special_toks_indices, expert_flag, mask=input_mask)

        #TODO normalize the output () !!!!!!

        #-------------------- Contrastive losses  --------------------#
        cap_proj_feats      = F.normalize(self.cap_proj(input_embeds[:, special_toks_indices['<caption>'], :]), dim=-1)  # (bs*gpus, H)
        vis_proj_feats      = F.normalize(self.vision_proj(input_embeds[:, special_toks_indices['<vis>'], :]), dim=-1)  # (bs*gpus, H)
        if media_type == 'webvid':
            spatial_proj_feats  = F.normalize(self.spatial_proj(input_embeds[:, special_toks_indices['<spatial>'], :]), dim=-1)  # (bs*gpus, H)
            temp_proj_feats = F.normalize(self.temp_proj(input_embeds[:, special_toks_indices['<temporal>'], :]), dim=-1)  # (bs*gpus, H)

        if self.config.loss_dict['vcc'] != 0:
            vis_proj_feats_all = concat_all_gather(vis_proj_feats)  # (bs*gpus, H)
            cap_proj_feats_all = concat_all_gather(cap_proj_feats)  # (bs*gpus, H)

            loss_vcc, _, _ = self.compute_contrastive_loss(vis_proj_feats, cap_proj_feats_all, cap_proj_feats, vis_proj_feats_all)

        # 1- Spatial-Temporal
        if media_type == 'webvid':
            if self.config.loss_dict['stc'] != 0:
                spatial_proj_feats_all = concat_all_gather(spatial_proj_feats)  # (bs*gpus, H)
                temp_proj_feats_all = concat_all_gather(temp_proj_feats)  # (bs*gpus, H)
                loss_stc, _, _ = self.compute_contrastive_loss(temp_proj_feats, spatial_proj_feats_all, spatial_proj_feats, temp_proj_feats_all)


        #--------------------  Matching losses  --------------------#
        if self.config.loss_dict['vcm'] != 0:
            # Negative caption with positive visual
            neg_cap_ids, neg_cap_attention_mask, = self.tokenize_text(neg_cap, device, max_len=self.config.max_cap_len)
            neg_cap_embed = self.moe_llm_bottleneck(self.embed(neg_cap_ids))
            input_embeds_neg_cap = input_embeds_orig.clone().detach()
            input_embeds_neg_cap[:, special_toks_indices['<caption>'] + 1:special_toks_indices['</s>']] = neg_cap_embed
            input_mask_neg_cap = input_mask.clone().detach()
            input_mask_neg_cap[:, special_toks_indices['<caption>'] + 1:special_toks_indices['</s>']] = neg_cap_attention_mask

            # Negative visual with positive caption
            input_embeds_neg_vis = input_embeds_orig.clone().detach()
            input_mask_neg_vis = input_mask.clone().detach()

            # neg_vis_embed = self.encode_vis(neg_vis)
            
            # # reshape video features
            # neg_vis_embed = neg_vis_embed.reshape(batch_size, num_frames, -1, embed_dim)
            
            # # Perfrom spatial temporal attention and reshape
            # neg_vis_embed_spatial = self.spatial_att(neg_vis_embed)
            # neg_vis_embed_spatial = neg_vis_embed_spatial.reshape(batch_size, -1, embed_dim)
            if media_type == 'webvid':
                # neg_vis_embed_temporal = self.temporal_att(neg_vis_embed)
                # neg_vis_embed_temporal = neg_vis_embed_temporal.reshape(batch_size, -1, embed_dim)

                input_embeds_neg_vis[:, special_toks_indices['<spatial>']  + 1: special_toks_indices['<temporal>'], :] = neg_vis_embed_spatial
                input_embeds_neg_vis[:, special_toks_indices['<temporal>'] + 1: special_toks_indices['<caption>'], :] = neg_vis_embed_temporal

            elif media_type == 'cc3m':
                # neg_vis_embed_spatial = self.moe_llm_bottleneck(neg_vis_embed_spatial)
                input_embeds_neg_vis[:, special_toks_indices['<spatial>'] + 1: special_toks_indices['<caption>'], :] = neg_vis_embed_spatial

            # Construct the input of VCM
            final_input_embeds_vcm = torch.cat([input_embeds_orig, input_embeds_neg_cap, input_embeds_neg_vis], dim=0)
            final_input_mask_vcm = torch.cat([input_mask, input_mask_neg_cap, input_mask_neg_vis], dim=0)

            for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
                if moe_layer_idx < self.config.num_moe_modality_layers:
                    expert_flag = 'modalities'
                else: 
                    expert_flag = 'fusion'
                final_input_embeds_vcm = moe_layer(final_input_embeds_vcm, special_toks_indices, expert_flag, mask=final_input_mask_vcm)
            
            pooled_caption = self.caption_pooler(final_input_embeds_vcm, special_toks_indices['<caption>'])
            pooled_vis     = self.vis_pooler(final_input_embeds_vcm, special_toks_indices['<vis>'])

            vcm_feats = torch.mul(pooled_caption, pooled_vis)
            vcm_logits = self.vcm_head(vcm_feats)
            vcm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(device)

            # random permutation of the logits and labels --> make the task not trivial to learn
            # perm_idx = torch.randperm(vcm_logits.size(0), device=device)
            # perm_idx_extended = perm_idx.unsqueeze(-1).repeat(1, vcm_logits.size(-1))

            # # Shuffle
            # vcm_logits = vcm_logits.scatter(0, perm_idx_extended, vcm_logits)
            # vcm_labels = vcm_labels.scatter(0, perm_idx, vcm_labels)
            
            # class_weight = torch.FloatTensor([1.0, 1.0/3]).to(device)

            loss_vcm = F.cross_entropy(vcm_logits, vcm_labels)  # , weight=class_weight)

        if media_type == 'webvid':
            if self.config.loss_dict['stm'] != 0:
                # Negative spatial with positive temporal
                input_embeds_neg_spatial = input_embeds_orig.clone().detach()
                input_mask_neg_spatial   = input_mask.clone().detach()
                input_embeds_neg_spatial[:, special_toks_indices['<spatial>']  + 1: special_toks_indices['<temporal>'], :] = neg_vis_embed_spatial

                # Positive spatial with negative temporal
                input_embeds_neg_temporal = input_embeds_orig.clone().detach()
                input_mask_neg_temporal = input_mask.clone().detach()
                input_embeds_neg_temporal[:, special_toks_indices['<temporal>']  + 1: special_toks_indices['<caption>'], :] = neg_vis_embed_temporal

                # Construct the input of STM
                final_input_embeds_stm = torch.cat([input_embeds_orig, input_embeds_neg_spatial, input_embeds_neg_temporal], dim=0)
                final_input_mask_stm   = torch.cat([input_mask, input_mask_neg_spatial, input_mask_neg_temporal], dim=0)

                for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
                    if moe_layer_idx < self.config.num_moe_modality_layers:
                        expert_flag = 'modalities'
                    else: 
                        expert_flag = 'fusion'
                    final_input_embeds_stm = moe_layer(final_input_embeds_stm, special_toks_indices, expert_flag, mask=final_input_mask_stm)
                
                pooled_spatial = self.spatial_pooler(final_input_embeds_stm, special_toks_indices['<spatial>'])
                pooled_temporal = self.temporal_pooler(final_input_embeds_stm, special_toks_indices['<temporal>'])

                stm_feats = torch.mul(pooled_spatial, pooled_temporal)
                stm_logits = self.stm_head(stm_feats)
                stm_labels = torch.cat(
                    [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                    dim=0,
                ).to(device)

                # random permutation of the logits and labels --> make the task not trivial to learn
                # perm_idx = torch.randperm(stm_logits.size(0), device=device)
                # perm_idx_extended = perm_idx.unsqueeze(-1).repeat(1, stm_logits.size(-1))

                # # Shuffle
                # stm_logits = stm_logits.scatter(0, perm_idx_extended, stm_logits)
                # stm_labels = stm_labels.scatter(0, perm_idx, stm_labels)

                # class_weight = torch.FloatTensor([1.0, 1.0/3]).to(device)
                loss_stm = F.cross_entropy(stm_logits, stm_labels) # , weight=class_weight)

        if self.config.loss_dict['mlm'] != 0:
            masked_cap_ids, labels = self.mlm(cap_ids.clone())
            masked_cap_embeds = self.moe_llm_bottleneck(self.embed(masked_cap_ids))
            # inject the masked embeddings instead of the original ones
            # input_embeds_mlm[:, special_toks_indices['<caption>']+1 : special_toks_indices['</s>'], :] = masked_cap_embeds
            
            for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
                if moe_layer_idx < self.config.num_moe_modality_layers:
                    expert_flag = 'modalities'
                else: 
                    expert_flag = 'fusion'
                masked_cap_embeds = moe_layer(masked_cap_embeds, special_toks_indices, expert_flag, mask=cap_attention_mask, only_text=True)

            # extract the caption last hidden states 
            # masked_cap_embeds_last = input_embeds_mlm[:, special_toks_indices['<caption>']+1 : special_toks_indices['</s>'], :]
            lm_logits = self.lm_head(masked_cap_embeds)
            loss_mlm = F.cross_entropy(
                lm_logits.view(-1, len(self.tokenizer)),
                labels.view(-1),
                ignore_index=self.mlm.padding_token
            )

        return dict(
            loss_stc = loss_stc * self.config.loss_dict['stc'],
            loss_stm = loss_stm * self.config.loss_dict['stm'],
            loss_vcc = loss_vcc * self.config.loss_dict['vcc'],
            loss_vcm = loss_vcm * self.config.loss_dict['vcm'],
            loss_mlm = loss_mlm * self.config.loss_dict['mlm'],
        )


    def get_vis_enc_for_eval(self, vis, media_type):
        # First get the visual features depending on the media type
        vis_spatial_embed, vis_temporal_embed = self.encode_vis(vis, media_type)

        # Expand the query tokens 
        spatial_query_embeds = self.spatial_query_embeds.expand(vis_spatial_embed.size(0), -1, -1)
        
        # Run the spatial expert
        spatial_query_embeds, pooled_spatial_query_embeds = self.encode_queries(
            spatial_query_embeds, vis_spatial_embed, vis_mode='spatial')
        
        temporal_query_embeds = self.spatial_query_embeds.expand(vis_temporal_embed.size(0), -1, -1)
        temporal_query_embeds, pooled_temporal_query_embeds = self.encode_queries(
            temporal_query_embeds, vis_temporal_embed, vis_mode='temporal')

        vis_pooled = torch.cat((pooled_spatial_query_embeds, pooled_temporal_query_embeds), dim=1)
        vis_embeds = torch.cat((spatial_query_embeds, temporal_query_embeds), dim=1)

        return vis_embeds, vis_pooled

    def get_expert_encoder(self, expert):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = None
        if expert == 'cap':
            encoder = self.cap_expert
        if expert == 'spatial':
            encoder = self.spatial_expert
        if expert == 'temporal':
            encoder = self.temporal_expert
        if expert == 'sap_att_grounding':
            encoder = self.spa_temp_grounding_expert
        if expert == 'vis_cap_grounding':
            encoder = self.vis_cap_grounding_expert
        assert encoder is not None
        return encoder.bert if hasattr(encoder, "bert") else encoder



class V2Dial(V2DialAbstract):
    def __init__(self, config):
        super(V2Dial, self).__init__()
        self.config = config

        ################## 1. Select Tokenizer -- We use BERT tokenizer ##################
        bert_config = BertConfig.from_pretrained('bert-{}-uncased'.format(config.expert_size))
        tokenizer = AutoTokenizer.from_pretrained('bert-{}-uncased'.format(config.expert_size))

        text_embedding = BertEmbeddings(bert_config)
        text_embedding.apply(self.init_weights)

        token_type_embedding = nn.Embedding(3, bert_config.hidden_size)  # Number of modalities (temp/spa/cap/hist-ques-ans)
        token_type_embedding.apply(self.init_weights)

        ################## 1. Select LLM -- We use BERT tokenizer ##################
        if config.llm_family == 'llama':
            logging.info('[INFO] LLM: LLAMA v2')
            llm_model = LlamaForCausalLM

        elif config.llm_family == 'mistral':
            logging.info('[INFO] LLM: Mistral')
            llm_model = MistralForCausalLM

        elif config.llm_family == 'flan_t5':
            logging.info('[INFO] LLM: Flan T5')
            llm_model = T5ForConditionalGeneration

        elif config.llm_family == 'bart':
            logging.info('[INFO] LLM: BART')
            llm_model = BartForConditionalGeneration
        else:
            raise ValueError


        llm_tokenizer = AutoTokenizer.from_pretrained(
                config.llm_name,
                use_fast=False,
                token='your_token'
            ) 
        # set the padding token to eos token for llama
        if config.llm_family == 'llama':
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        #________________________________ LLM Quantization ________________________________#
        if config.llm_family in ['mistral', 'llama']:
            dtype=None
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            if config.fp16:
                dtype = torch.float16
                if config.llm_family == 'flan_t5':
                    dtype = torch.bfloat16
            else:
                dtype = torch.float32
            quantization_config = None

        # llm_model.generate()    
        llm = llm_model.from_pretrained(
            config.llm_name,
            token='your_token',
            torch_dtype=dtype,
            quantization_config=quantization_config
        )        
        
        if config.llm_family == 'llama':
            llm_embed = llm.model.embed_tokens
        elif config.llm_family == 'flan_t5':
            llm_embed = llm.shared
        elif config.llm_family == 'mistral':
            llm_embed = llm.model.embed_tokens
        elif config.llm_family == 'bart':
            llm_embed = llm.model.shared
        else:
            raise ValueError

        # llm.resize_token_embeddings(len(self.tokenizer))
        if quantization_config is not None:
        # Gradient checkpointing is not compatible with DDP!! 
            llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
           

        if config.freeze_llm:
            for _, param in llm.named_parameters():
                param.requires_grad = False
            logging.info('[INFO] LLM frozen')
        else:
            if config.use_lora_llm:
            # load the lora config
                with open(config.lora_config, 'r') as f: 
                    lora_config = json.load(f)

                if config.llm_family in ['llama', 'mistral']:
                    lora_config['target_modules'] = ['q_proj', 'v_proj']

                elif config.llm_family in ['flan_t5']:
                    lora_config['target_modules'] = ['q', 'v']

                lora_config = LoraConfig(**lora_config)
                llm = get_peft_model(llm, lora_config)

                logging.info('[INFO] LLM hot with lora')
            else:
                logging.info('[INFO] LLM hot')
                
            logging.info('[INFO] LLM successfully loaded')
        
        for _, param in llm_embed.named_parameters():
            param.data = param.data.float()
            param.requires_grad = True

        llm_to_moe = nn.Linear(llm.config.hidden_size, bert_config.hidden_size)
        llm_to_moe.apply(self.init_weights)

        moe_to_llm = nn.Linear(bert_config.hidden_size, llm.config.hidden_size)
        moe_to_llm.apply(self.init_weights)

        ################## 2. Select the backbone ViT ##################
        logging.info('[INFO] Loading ViT in progress')
        if config.freeze_vit:
            # vit_precision = 'fp16' if config.fp16 else 'fp32'
            logging.info(f'[INFO] ViT precision: {config.vit_precision}')
            visual_encoder, ln_vision = self.init_vision_encoder(
                config.vit_model, config.image_res, drop_path_rate=0, use_grad_checkpoint=False, precision=config.vit_precision
            )
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info('[INFO] ViT frozen')
      
        else:
            vit_precision = 'fp32'
            visual_encoder, ln_vision = self.init_vision_encoder(
                config.vit_model, config.image_res, drop_path_rate=0, use_grad_checkpoint=False, vit_precision=vit_precision
            )
            logging.info('[INFO] ViT hot')
        logging.info('[INFO] ViT successfully loaded')

        ################## 3. Define the ViT-Expert communication Interface ##################
        self.system_prompt = False
        self.vit_token_pooling = config.vit_token_pooling
        if self.vit_token_pooling:
            vit_proj = nn.Linear(
                1408*4, bert_config.hidden_size
            )
        else:
            vit_proj = nn.Linear(
                1408, bert_config.hidden_size
            )
        vit_proj.apply(self.init_weights)

        spatial_att  = SpatialAttention(input_dim=bert_config.hidden_size)
        temporal_att = TemporalAttention(input_dim=bert_config.hidden_size)
        
        spatial_att.apply(self.init_weights)
        temporal_att.apply(self.init_weights)

        ################## 4. Define the Expert layers  ##################
        moe_layers = None
        moe_norm = None
        if config.use_moes:
            moe_layers = []

            for moe_layer_idx in range(config.num_moe_layers):
                if moe_layer_idx < self.config.num_moe_modality_layers:
                    expert_flag = 'modalities'
                else: 
                    expert_flag = 'fusion'
                moe_layer = MoELayer(
                    bert_config.hidden_size,
                    bert_config.num_attention_heads,
                    expert_flag,
                    has_hist=True,
                    use_sep_spatial_temp_experts=config.use_sep_spatial_temp_experts
                )
                
                moe_layer.apply(self.init_weights)
                moe_layers.append(moe_layer)

                logging.info(f'[INFO] {moe_layer_idx+1}/{config.num_moe_layers} MoE layers successfully loaded')

            moe_layers = nn.ModuleList(moe_layers)
            moe_norm   = nn.LayerNorm(bert_config.hidden_size)
            
        ################## 5. Define the projection layers for contrastive learning  ##################
        # temp_proj    = nn.Linear(bert_config.hidden_size, config.joint_dim)
        # spatial_proj = nn.Linear(bert_config.hidden_size, config.joint_dim)
        # vision_proj  = nn.Linear(bert_config.hidden_size, config.joint_dim)
        # cap_proj     = nn.Linear(bert_config.hidden_size, config.joint_dim)

        # temp_proj.apply(self.init_weights)
        # spatial_proj.apply(self.init_weights)
        # vision_proj.apply(self.init_weights)
        # cap_proj.apply(self.init_weights)

        ################## 6. Define the pooler for matching loss  ##################
        # pooler = Pooler(bert_config.hidden_size)
        # pooler.apply(self.init_weights)

        ################## 5. Attach the matching heads  ##################
        # stm_head = nn.Linear(bert_config.hidden_size, 2)
        # vcm_head = nn.Linear(bert_config.hidden_size, 2)
        # lm_head  = nn.Linear(bert_config.hidden_size, len(tokenizer))

        # stm_head.apply(self.init_weights)
        # vcm_head.apply(self.init_weights)
        # lm_head.apply(self.init_weights)

        temp = nn.Parameter(0.07 * torch.ones([]))
        # temp = 0.07

        # Attach the components to self
        if self.config.embed_from_llm:
            self.tokenizer            = llm_tokenizer
            self.text_embedding       = llm_embed
        else:
            self.tokenizer            = tokenizer
            self.text_embedding       = text_embedding
            self.token_type_embedding = token_type_embedding
        
        self.llm                      = llm
        self.llm_to_moe               = llm_to_moe
        self.moe_to_llm               = moe_to_llm
        self.visual_encoder           = visual_encoder
        self.ln_vision                = ln_vision
        self.vit_proj                 = vit_proj
        self.moe_layers               = moe_layers
        self.moe_norm                 = moe_norm   
        self.spatial_att              = spatial_att
        self.temporal_att             = temporal_att
        # self.temp_proj            = temp_proj
        # self.spatial_proj         = spatial_proj
        # self.vision_proj          = vision_proj
        # self.cap_proj             = cap_proj
        # self.pooler               = pooler
        # self.stm_head             = stm_head
        # self.vcm_head             = vcm_head
        # self.lm_head              = lm_head
        self.temp                 = temp

    def construct_global_input(self, cap_ids, cap_attention_mask, hist_ids, hist_attention_mask, vid_feat_len, device):
        # for video: <s><vis><spatial>[spatial_feats]<temporal>[temp_feats]<caption>[cap_feats]<history>[hist_feats]</s>

        batch_size = cap_ids.size(0)
        special_toks_indices = {
            '<s>': 0,
            '<vis>': 1,
            '<spatial>': 2,
        }
        
        ids = [self.added_vocab['<s>']] + [self.added_vocab['<vis>']] + [self.added_vocab['<spatial>']]
        ids += vid_feat_len * [self.added_vocab['<pad>']]

        ids += [self.added_vocab['<temporal>']]
        special_toks_indices['<temporal>'] = len(ids) - 1
        ids += vid_feat_len * [self.added_vocab['<pad>']]

        ids += [self.added_vocab['<caption>']]
        special_toks_indices['<caption>'] = len(ids) - 1
        ids += cap_ids.size(1) * [self.added_vocab['<pad>']]

        ids += [self.added_vocab['<history>']]
        special_toks_indices['<history>'] = len(ids) - 1
        ids += hist_ids.size(1) * [self.added_vocab['<pad>']]

        ids += [self.added_vocab['</s>']]
        special_toks_indices['</s>'] = len(ids) - 1
        total_len = len(ids)

        ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        ids[:, special_toks_indices['<caption>'] + 1: special_toks_indices['<history>']] = cap_ids
        ids[:, special_toks_indices['<history>'] + 1: special_toks_indices['</s>']] = hist_ids


        mask = torch.ones((batch_size, total_len), device=device)
        mask[:, special_toks_indices['<caption>'] + 1: special_toks_indices['<history>']] = cap_attention_mask 
        mask[:, special_toks_indices['<history>'] + 1: special_toks_indices['</s>']] = hist_attention_mask 

        return ids, mask, special_toks_indices

    def construct_reg_labels(self, regress_ids, start_regress_idx, full_embeds, device):

        full_labels = torch.LongTensor(full_embeds.size(0), full_embeds.size(1)).fill_(-100).to(device)
    
        for i in range(regress_ids.size(0)):

            full_labels[i, start_regress_idx[i]: start_regress_idx[i] + regress_ids[i].size(-1)] = regress_ids[i]
            # Add </s> to the labels -- just before the response starts
            full_labels[i, start_regress_idx[i] - 1] = self.tokenizer.eos_token_id

        # labels = regress_ids.masked_fill(
        #     regress_ids == self.tokenizer.pad_token_id, -100
        # ).to(device)

        # eos_from_cond = torch.LongTensor(labels.size(0), 1).fill_(self.tokenizer.eos_token_id).to(device)
        # labels = torch.concat([eos_from_cond, labels], dim=1)

        # full_labels = torch.LongTensor(labels.size(0), full_len).fill_(-100).to(device)

        # full_labels[:, len_cond-1:] = labels

        return full_labels

    def rearrange_llm_input_decoder_only(self, input_embeds, output_emebds, input_mask, cap_mask, hist_mask, output_mask, spatial_feat_len):
        '''
            Push all pads to the right
        '''
        # full_embeds = <s><vis><spa>[...]<temp>[...]<cap>[...][pad]<hist>[...][pad]</s>[ans ...][pad]
        # ------------> <s><vis><spa>[...]<temp>[...]<cap>[...]<hist>[...]</s>[ans ...][-----pad-----]

        init_len = input_embeds.size(1) + output_emebds.size(1)
        
        # First, we compute the initial offset of the visual features
        offset = 3 + spatial_feat_len + 1 + spatial_feat_len  #  --> input_embeds[offset] = h_<cap>
        
        offset_embeds = input_embeds[:, :offset, :]
        offset_mask   = input_mask[:, :offset]

        rest_input_embdes = input_embeds[:, offset:, :]
        rest_input_mask   = input_mask[:, offset:]
        
        start_output_idx = []
        full_embeds = []
        full_masks  = []

        for i in range(input_embeds.size(0)):
            output_emebd_i = output_emebds[i] 
            output_mask_i  = output_mask[i]

            cap_mask_i = cap_mask[i]
            len_cap_i  = cap_mask_i.sum()
            end_cap_i  = len_cap_i + 1  # +1 for the <caption> token

            cap_embdes_i_to_keep = rest_input_embdes[i, :end_cap_i, :]
            cap_mask_i_to_keep   = rest_input_mask[i, :end_cap_i,]
            cap_embeds_i_to_push = rest_input_embdes[i, end_cap_i:cap_mask_i.size(-1) + 1, :]  # +1 for the <caption> token
            cap_mask_i_to_push   = rest_input_mask[i, end_cap_i: cap_mask_i.size(-1) + 1]  # +1 for the <caption> token

            hist_mask_i  = hist_mask[i]
            len_hist_i   = hist_mask_i.sum()
            start_hist_i = cap_mask_i.size(-1) + 1
            end_hist_i   = start_hist_i + len_hist_i + 1  # +1 for <history> token

            # fianl token to keep is </s> which is the last in input_embdes/rest_input_embdes
            final_tok_embedding_i  = rest_input_embdes[i, -1, :].unsqueeze(0)
            final_tok_mask_i       = rest_input_mask[i, -1].unsqueeze(0)

            hist_embdes_i_to_keep = rest_input_embdes[i, start_hist_i:end_hist_i, :]
            hist_mask_i_to_keep   = rest_input_mask[i, start_hist_i:end_hist_i]

            # these two do not consider the last </s> token --> we don't need to extra remove it from them
            hist_embdes_i_to_push = rest_input_embdes[i, end_hist_i: cap_mask_i.size(-1) + 1 + hist_mask_i.size(-1) + 1, :]
            hist_mask_i_to_push   = rest_input_mask[i, end_hist_i: cap_mask_i.size(-1) + 1 + hist_mask_i.size(-1) + 1]

            full_embed_i = torch.cat(
                [cap_embdes_i_to_keep, hist_embdes_i_to_keep, final_tok_embedding_i, output_emebd_i, cap_embeds_i_to_push, hist_embdes_i_to_push],
                dim=0
            )

            full_mask_i = torch.cat(
                [cap_mask_i_to_keep, hist_mask_i_to_keep, final_tok_mask_i, output_mask_i, cap_mask_i_to_push, hist_mask_i_to_push],
                dim=0
            )

            start_output_idx.append(offset + cap_embdes_i_to_keep.size(0) + hist_embdes_i_to_keep.size(0) + 1 - 1)

            full_embeds.append(full_embed_i)
            full_masks.append(full_mask_i)

        # Now stack to get the batch
        full_embeds = torch.stack(full_embeds, dim=0)
        full_masks = torch.stack(full_masks, dim=0)

        # Add the offset visual features
        full_embeds = torch.cat([offset_embeds, full_embeds], dim=1)
        full_masks = torch.cat([offset_mask, full_masks], dim=1)

        final_len = full_embeds.size(1)

        # Sanity check
        assert init_len == final_len, 'The reconstructed embeds have length ({}) which is not the same as the length of initial embeds ({})'.format(
            final_len, init_len
        )

        return full_embeds, full_masks, start_output_idx

    def pad_to_right_enc_dec(self, cap_embeds, cap_masks, hist_embeds, hist_masks, device):
        """
        pushes all in-between pad tokens to the right 
        """
        res_embeds = []
        res_mask = []
        for cap_embed, cap_mask, hist_embed, hist_mask in zip(cap_embeds, cap_masks, hist_embeds, hist_masks):
            len_cap = sum(cap_mask)
            len_hist = sum(hist_mask)

            batch_embed = torch.cat([cap_embed[:len_cap], hist_embed[:len_hist], cap_embed[len_cap:], hist_embed[len_hist:]], dim=0)
            batch_mask  = torch.zeros(batch_embed.size(0)).long().to(device)
            batch_mask[:len_cap+len_hist] = 1

            res_embeds.append(batch_embed)
            res_mask.append(batch_mask)

        res_embeds = torch.stack(res_embeds, dim=0)
        res_mask   = torch.stack(res_mask, dim=0)

        return res_embeds, res_mask

    def pad_to_right_dec_only(self, cap_embeds, cap_masks, hist_embeds, hist_masks, regress_embeds, regress_masks, device):
        """
        pushes all in-between pad tokens to the right 
        """
        res_embeds = []
        res_mask = []
        regress_limits_txt_input = []
        for cap_embed, cap_mask, hist_embed, hist_mask, regress_emebd, regress_mask in zip(
            cap_embeds, cap_masks, hist_embeds, hist_masks, regress_embeds, regress_masks):

            len_cap  = sum(cap_mask)
            len_hist = sum(hist_mask)
            len_ans  = sum(regress_mask)
            regress_limits_txt_input.append((len_cap+len_hist, len_cap+len_hist+len_ans))

            batch_embed = torch.cat([cap_embed[:len_cap], hist_embed[:len_hist], regress_emebd, cap_embed[len_cap:], hist_embed[len_hist:]], dim=0)
            batch_mask  = torch.zeros(batch_embed.size(0)).long().to(device)
            batch_mask[:len_cap+len_hist+len_ans] = 1

            res_embeds.append(batch_embed)
            res_mask.append(batch_mask)

        res_embeds = torch.stack(res_embeds, dim=0)
        res_mask   = torch.stack(res_mask, dim=0)

        return res_embeds, res_mask, regress_limits_txt_input

    def pad_to_right_dec_only_gen_mode(self, cap_embeds, cap_masks, hist_embeds, hist_masks, device):
        """
        pushes all in-between pad tokens to the right 
        """
        res_embeds = []
        res_mask = []
        for cap_embed, cap_mask, hist_embed, hist_mask in zip(cap_embeds, cap_masks, hist_embeds, hist_masks):

            len_cap  = sum(cap_mask)
            len_hist = sum(hist_mask)

            batch_embed = torch.cat([cap_embed[:len_cap], hist_embed[:len_hist], cap_embed[len_cap:], hist_embed[len_hist:]], dim=0)
            batch_mask  = torch.zeros(batch_embed.size(0)).long().to(device)
            batch_mask[:len_cap+len_hist] = 1

            res_embeds.append(batch_embed)
            res_mask.append(batch_mask)

        res_embeds = torch.stack(res_embeds, dim=0)
        res_mask   = torch.stack(res_mask, dim=0)

        return res_embeds, res_mask

    def encode_vis_with_seq_spa_temp_att(self, image, device, is_vid=True):
        num_frames = image.size(1)
        bs_pre_reshape = image.size(0)
        if len(image.shape) > 4: 
            image = image.view(-1, *image.shape[-3:]) # for video input flatten the batch and time dimension (4,50,3,224,224) -> (200,3,224,224)
        # with self.maybe_autocast():  # inherited from Blip2Base
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # (200,3,224,224) -> (200,257,1408)
        image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)

        bs, pn, hs = image_embeds.shape
        if self.vit_token_pooling: # concat the each 4 tokens into one token (200,64,5632)
            image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) # (200,64,5632)

        vis_embed = self.vit_proj(image_embeds) # project to llama input size (200,64,5632) -> (200,64,4096)

        # reshape the video features
        vis_embed = vis_embed.view(bs_pre_reshape, num_frames, -1, vis_embed.size(-1))
        size_orig = vis_embed.size()

        # Perfrom spatial temporal attention
        vis_embed = self.spatial_att(vis_embed) 
        if is_vid:
            vis_embed = vis_embed.view(size_orig)
            vis_embed = self.temporal_att(vis_embed)
        
        vis_feat_len = vis_embed.size(1)

        # vis_embed = vis_embed + self.token_type_embedding(torch.zeros(bs_pre_reshape, vis_feat_len).long().to(device))
        vis_mask  =  torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

        return vis_embed, vis_mask

    def moe_forward_no_sep_spatial_temporal(
        self,
        vis, vis_mask,
        cap_ids, cap_mask, hist_ids, hist_mask,
        is_vid, device):

        # is_vid = media_type == 'webvid'
        # batch_size = len(cap)
        vis_feat_len = vis.size(1)
        input_embeds = []
        input_masks  = []

        input_embeds.append(vis)
        input_masks.append(vis_mask)

        # if is_vid:
        #     input_embeds.append(vis_temporal)
        #     input_masks.append(vis_temporal_mask)
       
        if self.config.embed_from_llm:
            cap_embeds = self.llm_to_moe(self.text_embedding(cap_ids))
        else:        
            cap_embeds = self.text_embedding(cap_ids) + self.token_type_embedding(torch.ones_like(cap_ids).long().fill_(2))
        
        cap_feat_len = cap_embeds.size(1)

        input_embeds.append(cap_embeds)
        input_masks.append(cap_mask)

        if self.config.embed_from_llm:
            hist_embeds = self.llm_to_moe(self.text_embedding(hist_ids))
        else:
            hist_embeds = self.text_embedding(hist_ids) + self.token_type_embedding(torch.ones_like(hist_ids).long().fill_(2))
        
        hist_feat_len = hist_embeds.size(1)

        input_embeds.append(hist_embeds)
        input_masks.append(hist_mask)

        input_embeds = torch.cat(input_embeds, dim=1)
        input_masks  = torch.cat(input_masks, dim=1)

        # expand the mask
        input_masks = self.get_extended_attention_mask(attention_mask=input_masks)

        # MoEs feed-forward
        for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
            if moe_layer_idx < self.config.num_moe_modality_layers:
                expert_flag = 'modalities'
            else: 
                expert_flag = 'fusion'
            
            input_embeds = moe_layer(input_embeds, vis_feat_len, cap_feat_len, expert_flag, hist_feat_len, is_vid=is_vid, mask=input_masks)

        #TODO normalize the output () !!!!!!
        input_embeds = self.moe_norm(input_embeds)

        # return the features
        vis_embeds  = input_embeds[:, :vis_feat_len]
        # temporal_embeds = input_embeds[:, vis_feat_len:2*vis_feat_len] if is_vid else None
        cap_embeds      = input_embeds[:, -(cap_feat_len + hist_feat_len): -hist_feat_len]
        hist_embeds     = input_embeds[:, -hist_feat_len:]
        # cls_feats      = self.pooler(cap_feats)

        moe_outputs = {
            'vis_embeds': vis_embeds,
            # 'temporal_embeds': temporal_embeds,
            'cap_embeds': cap_embeds,
            'hist_embeds': hist_embeds,
            # 'cls_feats': cls_feats,
            # 'last_hidden': input_embeds
        }

        return moe_outputs

    def moe_forward(
            self,
            vis_spatial, vis_spatial_mask, vis_temporal, vis_temporal_mask,
            cap_ids, cap_mask, hist_ids, hist_mask,
            is_vid, device):
    
        # is_vid = media_type == 'webvid'
        # batch_size = len(cap)
        vis_feat_len = vis_spatial.size(1)
        input_embeds = []
        input_masks  = []

        input_embeds.append(vis_spatial)
        input_masks.append(vis_spatial_mask)

        if is_vid:
            input_embeds.append(vis_temporal)
            input_masks.append(vis_temporal_mask)
       
        if self.config.embed_from_llm:
            cap_embeds = self.llm_to_moe(self.text_embedding(cap_ids))
        else:        
            cap_embeds = self.text_embedding(cap_ids) + self.token_type_embedding(torch.ones_like(cap_ids).long().fill_(2))
        
        cap_feat_len = cap_embeds.size(1)

        input_embeds.append(cap_embeds)
        input_masks.append(cap_mask)

        if self.config.embed_from_llm:
            hist_embeds = self.llm_to_moe(self.text_embedding(hist_ids))
        else:
            hist_embeds = self.text_embedding(hist_ids) + self.token_type_embedding(torch.ones_like(hist_ids).long().fill_(2))
        
        hist_feat_len = hist_embeds.size(1)

        input_embeds.append(hist_embeds)
        input_masks.append(hist_mask)

        input_embeds = torch.cat(input_embeds, dim=1)
        input_masks  = torch.cat(input_masks, dim=1)

        # expand the mask
        input_masks = self.get_extended_attention_mask(attention_mask=input_masks)

        # MoEs feed-forward
        for moe_layer_idx, moe_layer in enumerate(self.moe_layers):
            if moe_layer_idx < self.config.num_moe_modality_layers:
                expert_flag = 'modalities'
            else: 
                expert_flag = 'fusion'
            
            input_embeds = moe_layer(
                input_embeds, vis_feat_len, cap_feat_len, expert_flag, hist_feat_len,
                is_vid=is_vid,
                mask=input_masks,
                expert_permutation=self.config.expert_permutation
                )

        #TODO normalize the output () !!!!!!
        input_embeds = self.moe_norm(input_embeds)

        # return the features
        spatial_embeds  = input_embeds[:, :vis_feat_len]
        temporal_embeds = input_embeds[:, vis_feat_len:2*vis_feat_len] if is_vid else None
        cap_embeds      = input_embeds[:, -(cap_feat_len + hist_feat_len): -hist_feat_len]
        hist_embeds     = input_embeds[:, -hist_feat_len:]
        # cls_feats      = self.pooler(cap_feats)

        moe_outputs = {
            'spatial_embeds': spatial_embeds,
            'temporal_embeds': temporal_embeds,
            'cap_embeds': cap_embeds,
            'hist_embeds': hist_embeds,
            # 'cls_feats': cls_feats,
            # 'last_hidden': input_embeds
        }

        return moe_outputs

    def forward(self, vis, cap, hist, ans, media_type):

        device   = vis.device
        is_vid   = media_type in ['webvid', 'champagne', 'avsd', 'nextqa']
        loss_stc = torch.tensor(0)
        loss_stm = torch.tensor(0)
        loss_vhc = torch.tensor(0)
        loss_vhm = torch.tensor(0)
        loss_gen = torch.tensor(0)
        
        # construct the global input tensor --> use place holder for vis features
        cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=None)
        hist_ids, hist_mask = self.tokenize_text(hist, device, max_len=None)
        if self.config.use_moes:
            # First get the visual features depending on the media type
            if self.config.use_sep_spatial_temp_experts:
                vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = self.encode_vis(vis, device, is_vid=is_vid)
                spatial_feat_len = vis_embed_spatial.size(1)

            else:
                vis_embed, vis_mask = self.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)


            if self.config.use_sep_spatial_temp_experts:
                moe_outputs = self.moe_forward(
                    vis_embed_spatial, vis_spatial_mask,
                    vis_embed_temporal, vis_temporal_mask,
                    cap_ids, cap_mask,
                    hist_ids, hist_mask,
                    is_vid, device
                )
                spatial_embeds  = self.moe_to_llm(moe_outputs['spatial_embeds'])
                temporal_embeds = self.moe_to_llm(moe_outputs['temporal_embeds']) if is_vid else None
                # cap_embeds      = self.moe_to_llm(moe_outputs['cap_embeds'])
                # hist_embeds     = self.moe_to_llm(moe_outputs['hist_embeds'])

            else:
                moe_outputs = self.moe_forward_no_sep_spatial_temporal(
                    vis_embed, vis_mask,
                    cap_ids, cap_mask,
                    hist_ids, hist_mask,
                    is_vid, device
                )
                vis_embeds  = self.moe_to_llm(moe_outputs['vis_embeds'])
                # temporal_embeds = self.moe_to_llm(moe_outputs['temporal_embeds']) if is_vid else None
            cap_embeds      = self.moe_to_llm(moe_outputs['cap_embeds'])
            hist_embeds     = self.moe_to_llm(moe_outputs['hist_embeds'])
        else:
            cap_embeds  = self.llm_to_moe(self.text_embedding(cap_ids))
            hist_embeds = self.llm_to_moe(self.text_embedding(hist_ids))

            vis_embeds, vis_mask = self.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)

        ans = [a + self.tokenizer.eos_token for a in ans]

        if self.config.llm_family in ['llama', 'mistral']:
            bos = torch.ones_like(cap_ids[:, :1]) * self.tokenizer.bos_token_id
            bos_embeds = self.text_embedding(bos)
            bos_mask = cap_mask[:, :1]

            # add corresponding eos

            regress_ids, regress_mask = self.tokenize_text(ans, device, max_len=None)  # pad the longest
            
            regress_embeds = self.text_embedding(regress_ids)

            inputs_embeds, attention_mask, regress_limits_txt_input = self.pad_to_right_dec_only(cap_embeds, cap_mask, hist_embeds, hist_mask, regress_embeds, regress_mask, device)

            if is_vid:
                inputs_embeds = torch.cat([bos_embeds, spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                attention_mask = torch.cat([bos_mask, vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
            else:
                inputs_embeds = torch.cat([bos_embeds, spatial_embeds, inputs_embeds], dim=1)
                attention_mask = torch.cat([bos_mask, vis_spatial_mask, attention_mask], dim=1)

            labels = torch.zeros(inputs_embeds.size()[:-1]).fill_(-100).long().to(device)

            for i in range(labels.size(0)):
                start_regress = regress_limits_txt_input[i][0] + 1 + spatial_feat_len + spatial_feat_len * int(is_vid)  # offset (bos + spatial + temporal)
                end_regress   = regress_limits_txt_input[i][1] + 1 + spatial_feat_len + spatial_feat_len * int(is_vid) # offset (bos + spatial + temporal)

                labels[i, start_regress:end_regress] = regress_ids[i, :regress_mask[i].sum()]


            # get causal attention mask


            # Compute the regression embeds
            
            # Now we need to right-pad the input to LLM (at least for llama) to avoid nan loss values
            # This means, all pad tokens have to be placed to the right
            # full_embeds = <s><vis><spa>[...]<temp>[...]<cap>[...][pad]<hist>[...][pad]</s>[ans ...][pad]
            # ------------> <s><vis><spa>[...]<temp>[...]<cap>[...]<hist>[...]</s>[ans ...][-----pad-----]

            # full_embeds, full_masks, start_output_idx = self.rearrange_llm_input_dec_only(cond_embeds, regress_embeds, cond_mask, cap_mask, hist_mask, regress_mask, spatial_feat_len)

            # labels = self.construct_reg_labels(regress_ids, start_output_idx, full_embeds, device)

            lm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True 
            )
            loss_gen = lm_outputs.loss

        # Encoder Decoder
        else:
            inputs_embeds, attention_mask = self.pad_to_right_enc_dec(cap_embeds, cap_mask, hist_embeds, hist_mask, device)

            # now merge the multi-modal inputs
            if self.config.use_moes:
                if self.config.use_sep_spatial_temp_experts:
                    if is_vid:
                        inputs_embeds = torch.cat([spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([spatial_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([vis_spatial_mask, attention_mask], dim=1)
                else:
                    inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
                    attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            else:
                inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

            decoder_ids, decoder_mask = self.tokenize_text(ans, device, max_len=None)  # pad the longest
            
            labels = decoder_ids.masked_fill(decoder_ids == self.tokenizer.pad_token_id, -100)
            decoder_ids = self.shift_right(labels)
            decoder_inputs_embeds = self.text_embedding(decoder_ids)

            lm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_mask,
                labels=labels,
                return_dict=True
            )

            loss_gen = lm_outputs.loss

        return dict(
            loss_stc = loss_stc * self.config.loss_dict['stc'],
            loss_stm = loss_stm * self.config.loss_dict['stm'],
            loss_vhc = loss_vhc * self.config.loss_dict['vhc'],
            loss_vhm = loss_vhm * self.config.loss_dict['vhm'],
            loss_gen = loss_gen * self.config.loss_dict['gen'],
        )


class V2DialNoMoes(V2Dial):
    def __init__(self, config):
        super(V2DialNoMoes, self).__init__(config)

    def encode_vis(self, image, device, is_vid=True):
        num_frames = image.size(1)
        bs_pre_reshape = image.size(0)
        if len(image.shape) > 4: 
            image = image.view(-1, *image.shape[-3:]) # for video input flatten the batch and time dimension (4,50,3,224,224) -> (200,3,224,224)
        # with self.maybe_autocast():  # inherited from Blip2Base
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # (200,3,224,224) -> (200,257,1408)
        image_embeds = image_embeds[:,1:,:] # remove the first token (CLS) (200,256,1408)

        bs, pn, hs = image_embeds.shape
        if self.vit_token_pooling: # concat the each 4 tokens into one token (200,64,5632)
            image_embeds = image_embeds.view(bs, int(pn/4), int(hs*4)) # (200,64,5632)

        vis_embed = self.vit_proj(image_embeds) # project to LLM input size (200,64,5632) -> (200,64, d_hidden)

        # reshape the video features
        vis_embed = vis_embed.view(bs_pre_reshape, num_frames, -1, vis_embed.size(-1))


        # Perfrom spatial temporal attention
        if is_vid:
            vis_embed = self.temporal_att(vis_embed) 
            if not self.config.embed_from_llm:
                vis_embed_temporal = vis_embed_temporal + self.token_type_embedding(torch.ones(bs_pre_reshape, vis_feat_len).long().to(device))
            # vis_temporal_mask  =  torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

        vis_embed = self.spatial_att(vis_embed) 
        vis_feat_len = vis_embed_spatial.size(1)
        
        if not self.config.embed_from_llm:
            vis_embed_spatial = vis_embed_spatial + self.token_type_embedding(torch.zeros(bs_pre_reshape, vis_feat_len).long().to(device))
        vis_mask  = torch.ones((bs_pre_reshape, vis_feat_len)).to(device)

        return vis_embed, vis_mask
            

    def forward(self, vis, cap, hist, ans, media_type):

        device   = vis.device
        is_vid   = media_type in ['webvid', 'champagne', 'avsd', 'nextqa']
        loss_stc = torch.tensor(0)
        loss_stm = torch.tensor(0)
        loss_vhc = torch.tensor(0)
        loss_vhm = torch.tensor(0)
        loss_gen = torch.tensor(0)
        
        # First get the visual features depending on the media type
        vis_embed, vis_mask = self.encode_vis(vis, device, is_vid=is_vid)

        # spatial_feat_len = vis_embed_spatial.size(1)

        # construct the global input tensor --> use place holder for vis features
        # text = (c + h for c,h in zip(cap, hist))
        # cap_ids, cap_mask = self.tokenize_text(cap, device, max_len=None)
        # hist_ids, hist_mask = self.tokenize_text(hist, device, max_len=None)
        # text_ids, text_mask = self.tokenize_text(text, device, max_len=None)

        text_embeds = self.text_embedding(text_ids)
        # moe_outputs = self.moe_forward(
        #     vis_embed_spatial, vis_spatial_mask,
        #     vis_embed_temporal, vis_temporal_mask,
        #     cap_ids, cap_mask,
        #     hist_ids, hist_mask,
        #     is_vid, device
        # )
        # spatial_embeds  = self.moe_to_llm(moe_outputs['spatial_embeds'])
        # temporal_embeds = self.moe_to_llm(moe_outputs['temporal_embeds']) if is_vid else None
        # cap_embeds      = self.moe_to_llm(moe_outputs['cap_embeds'])
        # hist_embeds     = self.moe_to_llm(moe_outputs['hist_embeds'])

        ans = [a + self.tokenizer.eos_token for a in ans]

        if self.config.llm_family in ['llama', 'mistral']:
            bos = torch.ones_like(cap_ids[:, :1]) * self.tokenizer.bos_token_id
            bos_embeds = self.text_embedding(bos)
            bos_mask = cap_mask[:, :1]

            # add corresponding eos

            regress_ids, regress_mask = self.tokenize_text(ans, device, max_len=None)  # pad the longest
            
            regress_embeds = self.text_embedding(regress_ids)

            inputs_embeds, attention_mask, regress_limits_txt_input = self.pad_to_right_dec_only(cap_embeds, cap_mask, hist_embeds, hist_mask, regress_embeds, regress_mask, device)

            if is_vid:
                inputs_embeds = torch.cat([bos_embeds, spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                attention_mask = torch.cat([bos_mask, vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
            else:
                inputs_embeds = torch.cat([bos_embeds, spatial_embeds, inputs_embeds], dim=1)
                attention_mask = torch.cat([bos_mask, vis_spatial_mask, attention_mask], dim=1)

            labels = torch.zeros(inputs_embeds.size()[:-1]).fill_(-100).long().to(device)

            for i in range(labels.size(0)):
                start_regress = regress_limits_txt_input[i][0] + 1 + spatial_feat_len + spatial_feat_len * int(is_vid)  # offset (bos + spatial + temporal)
                end_regress   = regress_limits_txt_input[i][1] + 1 + spatial_feat_len + spatial_feat_len * int(is_vid) # offset (bos + spatial + temporal)

                labels[i, start_regress:end_regress] = regress_ids[i, :regress_mask[i].sum()]


            # get causal attention mask


            # Compute the regression embeds
            
            # Now we need to right-pad the input to LLM (at least for llama) to avoid nan loss values
            # This means, all pad tokens have to be placed to the right
            # full_embeds = <s><vis><spa>[...]<temp>[...]<cap>[...][pad]<hist>[...][pad]</s>[ans ...][pad]
            # ------------> <s><vis><spa>[...]<temp>[...]<cap>[...]<hist>[...]</s>[ans ...][-----pad-----]

            # full_embeds, full_masks, start_output_idx = self.rearrange_llm_input_dec_only(cond_embeds, regress_embeds, cond_mask, cap_mask, hist_mask, regress_mask, spatial_feat_len)

            # labels = self.construct_reg_labels(regress_ids, start_output_idx, full_embeds, device)

            lm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True 
            )
            loss_gen = lm_outputs.loss

        # Encoder Decoder
        else:
            # inputs_embeds, attention_mask = self.pad_to_right_enc_dec(cap_embeds, cap_mask, hist_embeds, hist_mask, device)

            # now merge the multi-modal inputs
            # if is_vid:
            #     inputs_embeds = torch.cat([spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
            #     attention_mask = torch.cat([vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
            # else:
            inputs_embeds = torch.cat([vis_embed, text_embeds], dim=1)
            attention_mask = torch.cat([vis_mask, text_mask], dim=1)

            decoder_ids, decoder_mask = self.tokenize_text(ans, device, max_len=None)  # pad the longest
            
            labels = decoder_ids.masked_fill(decoder_ids == self.tokenizer.pad_token_id, -100)
            decoder_ids = self.shift_right(labels)
            decoder_inputs_embeds = self.text_embedding(decoder_ids)

            lm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_mask,
                labels=labels,
                return_dict=True
            )

            loss_gen = lm_outputs.loss

        return dict(
            loss_stc = loss_stc * self.config.loss_dict['stc'],
            loss_stm = loss_stm * self.config.loss_dict['stm'],
            loss_vhc = loss_vhc * self.config.loss_dict['vhc'],
            loss_vhm = loss_vhm * self.config.loss_dict['vhm'],
            loss_gen = loss_gen * self.config.loss_dict['gen'],
        )