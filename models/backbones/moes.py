import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # if mask.dim() != x.dim():
            #     expanded_mask = mask[:, None, None, :].expand(B, 1, N, N) 
            # else:
            #     expanded_mask = mask
            mask = mask.bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MoELayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        expert_type,
        use_sep_spatial_temp_experts=True,
        has_hist=False,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LlamaRMSNorm,
    ):
        super().__init__()
        self.has_hist = has_hist
        self.use_sep_spatial_temp_experts = use_sep_spatial_temp_experts
        self.norm_att = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        if expert_type == 'modalities':
            # EXPERT CONSTRUCTION
            if use_sep_spatial_temp_experts:
                # Spatial expert          
                self.norm_spatial = norm_layer(dim)
                self.mlp_spatial = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                )

                # Temporal expert
                self.norm_temp = norm_layer(dim)
                self.mlp_temp = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                )

            # Vis expert
            self.norm_vis = norm_layer(dim)
            self.mlp_vis = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

            # caption expert
            self.norm_cap = norm_layer(dim)
            self.mlp_cap = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

            # history expert
            if has_hist:
                self.norm_hist = norm_layer(dim)
                self.mlp_hist = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                )
    
        elif expert_type == 'fusion':
            # Fusion expert
            self.norm_fusion = norm_layer(dim)
            self.mlp_fusion = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            raise ValueError


    def forward(self, x, vis_feat_len, cap_feat_len, expert_flag, hist_feat_len=None, is_vid=False, mask=None, only_text=False, expert_permutation=None):

        if self.has_hist:
            assert hist_feat_len is not None

        x_shortcut, attn = self.attn(self.norm_att(x), mask=mask)
        x = x + self.drop_path(x_shortcut)
        len_init = x.size(1)
        # bs, h_dim = x.size(0), x.size(-1)
        # device = x.device
        # if only_text:
        #     # end_idx_caption = special_toks_indices.get('<history>', special_toks_indices['</s>'] + 1)
        #     # x = x[:, special_toks_indices['<caption>']: end_idx_caption, :]
        #     x = x + self.drop_path(self.mlp_cap(self.norm_cap(x)))

        if expert_flag == 'modalities':
            if self.use_sep_spatial_temp_experts:
                x_spatial = x[:, :vis_feat_len]
                if expert_permutation is not None:
                    if expert_permutation['spatial'] == 'temporal':
                        x_spatial = x_spatial + self.drop_path(self.mlp_temp(self.norm_temp(x_spatial)))
                    elif expert_permutation['spatial'] == 'caption':
                        x_spatial = x_spatial + self.drop_path(self.mlp_cap(self.norm_cap(x_spatial)))
                    elif expert_permutation['spatial'] == 'history':
                        x_spatial = x_spatial + self.drop_path(self.mlp_hist(self.norm_hist(x_spatial)))
                    elif expert_permutation['spatial'] == 'spatial':
                        x_spatial = x_spatial + self.drop_path(self.mlp_spatial(self.norm_spatial(x_spatial)))
                    x_vis = x_spatial

                else:
                    x_spatial = x_spatial + self.drop_path(self.mlp_spatial(self.norm_spatial(x_spatial)))
                    x_vis = x_spatial

                if is_vid:
                    x_temporal = x[:, vis_feat_len:2*vis_feat_len]
                    if expert_permutation is not None:
                        if expert_permutation['temporal'] == 'spatial':
                            x_temporal = x_temporal + self.drop_path(self.mlp_spatial(self.norm_spatial(x_temporal)))
                        elif expert_permutation['temporal'] == 'caption':
                            x_temporal = x_temporal + self.drop_path(self.mlp_cap(self.norm_cap(x_temporal)))
                        elif expert_permutation['temporal'] == 'history':
                            x_temporal = x_temporal + self.drop_path(self.mlp_hist(self.norm_hist(x_temporal)))
                        elif expert_permutation['temporal'] == 'temporal':
                            x_temporal = x_temporal + self.drop_path(self.mlp_temp(self.norm_temp(x_temporal)))
                    else:
                        x_temporal = x_temporal + self.drop_path(self.mlp_temp(self.norm_temp(x_temporal)))
                    x_vis = torch.concat([x_spatial, x_temporal], dim=1)
                    x_vis = x_vis + self.drop_path(self.mlp_vis(self.norm_vis(x_vis)))
            else:
                x_vis = x[:, :vis_feat_len]
                x_vis = x_vis + self.drop_path(self.mlp_vis(self.norm_vis(x_vis)))

            if self.has_hist:
                x_caption = x[:, -(cap_feat_len + hist_feat_len): -hist_feat_len]
                if expert_permutation is not None:
                    if expert_permutation['caption'] == 'spatial':
                        x_caption = x_caption + self.drop_path(self.mlp_spatial(self.norm_spatial(x_caption)))
                    elif expert_permutation['caption'] == 'temporal':
                        x_caption = x_caption + self.drop_path(self.mlp_temp(self.norm_temp(x_caption)))
                    elif expert_permutation['caption'] == 'history':
                        x_caption = x_caption + self.drop_path(self.mlp_hist(self.norm_hist(x_caption)))
                    elif expert_permutation['caption'] == 'caption':
                        x_caption = x_caption + self.drop_path(self.mlp_cap(self.norm_cap(x_caption)))
                else:
                    x_caption = x_caption + self.drop_path(self.mlp_cap(self.norm_cap(x_caption)))


                x_history = x[:, -hist_feat_len:]
                if expert_permutation is not None:
                    if expert_permutation['history'] == 'spatial':
                        x_history = x_history + self.drop_path(self.mlp_spatial(self.norm_spatial(x_history)))
                    elif expert_permutation['history'] == 'temporal':
                        x_history = x_history + self.drop_path(self.mlp_temp(self.norm_temp(x_history)))
                    elif expert_permutation['history'] == 'caption':
                        x_history = x_history + self.drop_path(self.mlp_cap(self.norm_cap(x_history)))
                    elif expert_permutation['history'] == 'history':
                        x_history = x_history + self.drop_path(self.mlp_hist(self.norm_hist(x_history)))
                else:
                    x_history = x_history + self.drop_path(self.mlp_hist(self.norm_hist(x_history)))
                # concat the features back
                x = torch.cat([x_vis, x_caption, x_history], dim=1)
            else:
                x_caption = x[:, -cap_feat_len:]
                x_caption = x_caption + self.drop_path(self.mlp_cap(self.norm_cap(x_caption)))
                x = torch.cat([x_vis, x_caption], dim=1)

            assert x.size(1) == len_init, 'Reconstructed features length is {} != original features len = {}'.format(
                x.size(1), len_init
            )

        elif expert_flag == 'fusion':
            x = x + self.drop_path(self.mlp_fusion(self.norm_fusion(x)))

        return x


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_states = hidden_states[:, 0]
        pooled_output = self.dense(pooled_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output