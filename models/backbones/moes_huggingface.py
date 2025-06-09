import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from timm.models.layers import DropPath
import warnings
from torch import Tensor
from typing import Optional, Tuple


from .bert.xbert import BertLayer, BertAttention, BertIntermediate, BertOutput, BertConfig

class MoELayer(nn.Module):
    def __init__(self, config, expert_type):
        super(MoELayer, self).__init__()
        self.config = config
        self.expert_type = expert_type
        self.bert_config = BertConfig.from_pretrained('bert-large-uncased')
        
        # Shared across all experts
        self.attention = BertAttention(self.bert_config)

        # One for each expert
        if expert_type == 'modalities':
            # Spatial expert
            self.intermediate_spatial  = BertIntermediate(self.bert_config)
            self.output_spatial        = BertOutput(self.bert_config)

            # Temporal expert
            self.intermediate_temporal = BertIntermediate(self.bert_config)
            self.output_temporal       = BertOutput(self.bert_config)

            # Vis Expert
            self.intermediate_vis      = BertIntermediate(self.bert_config)
            self.output_vis            = BertOutput(self.bert_config)

            # Caption Expert
            self.intermediate_caption  = BertIntermediate(self.bert_config)
            self.output_caption        = BertOutput(self.bert_config)

            if config.stage != 'stage_1':
            # History Expert
                self.intermediate_history = BertIntermediate(self.bert_config)
                self.output_history       = BertOutput(self.bert_config)

        # Fusion expert
        elif expert_type == 'fusion':
            self.intermediate_fusion = BertIntermediate(self.bert_config)
            self.output_fusion       = BertOutput(self.bert_config)
        else:
            raise ValueError

        self._init_weights()

    def _init_weights(self):
        for _, m in dict(self.named_modules()).items():
            if isinstance(m, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
                m.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_() 


    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.bert_config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # if self.config.is_decoder:
            #     extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
            #         input_shape, attention_mask, device
            #     )
            # else:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


    def forward(self, hidden_states, special_toks_indices, expert_flag, mask=None, only_text=False, output_attentions=False):

        input_shape = hidden_states.size()[:-1]
        # dtype = mask.dtype
        # device = mask.device
        extended_attention_mask = self.get_extended_attention_mask(mask, input_shape, dtype=torch.float32)
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            head_mask=None
            )
        attention_output = self_attention_outputs[0]
        # outputs = self_attention_outputs[1:]

        len_init = attention_output.size(1)
        # bs, h_dim = x.size(0), x.size(-1)
        # device = x.device


        if expert_flag == 'modalities':
            if only_text:
                intermediate_output = self.intermediate_caption(attention_output)
                layer_output        = self.output_caption(intermediate_output, attention_output)
            else:
                # split the input first into different parts/modalities
                unchanged = attention_output[:, :special_toks_indices['<vis>'], :]
                end_idx_spatial = special_toks_indices.get('<temporal>', special_toks_indices['<caption>'])
                attention_spatial = attention_output[:, special_toks_indices['<vis>']:end_idx_spatial, :]

                end_idx_caption = special_toks_indices.get('<history>', special_toks_indices['</s>'] + 1)
                attention_caption = attention_output[:, special_toks_indices['<caption>']: end_idx_caption, :]

                attention_temporal, attention_history = None, None

                if '<temporal>' in special_toks_indices:
                    end_idx_temporal = special_toks_indices['<caption>']
                    attention_temporal = attention_output[:, special_toks_indices['<temporal>']:end_idx_temporal, :]

                if '<history>' in special_toks_indices:
                    end_idx_history = special_toks_indices['</s>'] + 1
                    attention_history = attention_output[:, special_toks_indices['<history>']:end_idx_history, :]
                
                # Expert activation
                # 1- Spatial
                intermediate_spatial = self.intermediate_spatial(attention_spatial)
                output_sapatial      = self.output_spatial(intermediate_spatial, attention_spatial)
                
                output_vis = output_sapatial
                
                # 2- Temporal
                if attention_temporal is not None:
                    intermediate_temporal = self.intermediate_temporal(attention_temporal)
                    output_temporal       = self.output_temporal(intermediate_temporal, attention_temporal)

                    attention_vis = torch.concat([output_sapatial, output_temporal], dim=1)
                    intermediate_vis = self.intermediate_vis(attention_vis)
                    output_vis = self.output_vis(intermediate_vis, attention_vis)

                # 3- Caption
                intermediate_caption = self.intermediate_caption(attention_caption)
                output_caption       = self.output_caption(intermediate_caption, attention_caption)

                # 4- History
                if attention_history is not None:
                    intermediate_history = self.intermediate_history(attention_history)
                    output_history       = self.output_history(intermediate_history, attention_history)

                output_list = [unchanged, output_vis, output_caption]

                if attention_history is not None:
                    output_list.append(output_history)
                
                # Concat the features back
                layer_output = torch.concat(output_list, dim=1)
                assert layer_output.size(1) == len_init, 'Reconstructed features length is {} != original features len = {}'.format(
                    layer_output.size(1), len_init
                )

        elif expert_flag == 'fusion':
            intermediate_output = self.intermediate_fusion(attention_output)
            layer_output        = self.output_fusion(intermediate_output, attention_output)

        return layer_output


class MoEPooler(nn.Module):
    def __init__(self):
        super(MoEPooler, self).__init__()

        self.bert_config = BertConfig.from_pretrained('bert-large-uncased')
        hidden_size = self.bert_config.hidden_size

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for _, m in dict(self.named_modules()).items():
            if isinstance(m, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
                m.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_() 

    def forward(self, hidden_states, idx):
        pooled_states = hidden_states[:, idx]
        pooled_output = self.dense(pooled_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
