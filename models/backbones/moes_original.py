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
            if mask.dim() != x.dim():
                expanded_mask = mask[:, None, None, :].expand(B, 1, N, N) 
            else:
                expanded_mask = mask
            expanded_mask = expanded_mask.bool()
            attn = attn.masked_fill(~expanded_mask, float("-inf"))

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
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        norm_layer=LlamaRMSNorm,
    ):
        super().__init__()
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

        # EXPERT CONSTRUCTION
        mlp_hidden_dim = int(dim * mlp_ratio)


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
        self.norm_hist = norm_layer(dim)
        self.mlp_hist = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
   
        # Fusion expert
        self.norm_fusion = norm_layer(dim)
        self.mlp_fusion = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    # expert_flag:{Only Text : 00 , Only Image : 01, Fusion : 10, Text & Image : 11} (BINARY)
    
    # expert_flag:
    # 0: 

    def forward(self, x, special_toks_indices, expert_flag, mask=None):
        x_shortcut, attn = self.attn(self.norm_att(x), mask=mask)
        x = x + self.drop_path(x_shortcut)
        bs, h_dim = x.size(0), x.size(-1)
        device = x.device

        if expert_flag == 'modalities':
            end_index = special_toks_indices.get('<temporal>', special_toks_indices['<caption>'])
            spatial_feats = x[:, special_toks_indices['<spatial>']: end_index, :]
            spatial_feats = spatial_feats + self.drop_path(self.mlp_spatial(self.norm_spatial(spatial_feats)))
            spatial_index = torch.arange(special_toks_indices['<spatial>'], end_index, device=device)
            spatial_index = spatial_index.unsqueeze(0).unsqueeze(-1)
            spatial_index = spatial_index.repeat(bs, 1, h_dim)
            x = x.scatter(1, spatial_index, spatial_feats)
            # x[:, special_toks_indices['<spatial>']: special_toks_indices['<temporal>'], :] = spatial_feats

            end_index = special_toks_indices.get('<history>', special_toks_indices['</s>'])
            caption_feats = x[:, special_toks_indices['<caption>']: end_index, :]
            caption_feats = caption_feats + self.drop_path(self.mlp_cap(self.norm_cap(caption_feats)))
            caption_index = torch.arange(special_toks_indices['<caption>'], end_index, device=device)
            caption_index = caption_index.unsqueeze(0).unsqueeze(-1)
            caption_index = caption_index.repeat(bs, 1, h_dim)
            x = x.scatter(1, caption_index, caption_feats)

            # x[:, special_toks_indices['<caption>']: special_toks_indices['</s>'], :] = caption_feats

            if '<temporal>' in special_toks_indices:
                temporal_feats = x[:, special_toks_indices['<temporal>']: special_toks_indices['<caption>'], :]
                temporal_feats = temporal_feats + self.drop_path(self.mlp_temp(self.norm_temp(temporal_feats)))
                temporal_index = torch.arange(special_toks_indices['<temporal>'], special_toks_indices['<caption>'], device=device)
                temporal_index = temporal_index.unsqueeze(0).unsqueeze(-1)
                temporal_index = temporal_index.repeat(bs, 1, h_dim)
                x = x.scatter(1, temporal_index, temporal_feats)

                # x[:, special_toks_indices['<temporal>']: special_toks_indices['<caption>'], :] = temporal_feats

                vis_feats = x[:, special_toks_indices['<vis>']: special_toks_indices['<caption>'], :]
                vis_feats = vis_feats + self.drop_path(self.mlp_vis(self.norm_vis(vis_feats)))
                vis_index = torch.arange(special_toks_indices['<vis>'], special_toks_indices['<caption>'], device=device)
                vis_index = vis_index.unsqueeze(0).unsqueeze(-1)
                vis_index = vis_index.repeat(bs, 1, h_dim)
                x = x.scatter(1, vis_index, vis_feats)

                # x[:, special_toks_indices['<vis>']: special_toks_indices['<caption>'], :] = vis_feats

            if '<history>' in special_toks_indices:
                history_feats = x[:, special_toks_indices['<history>']: special_toks_indices['</s>'], :]
                history_feats = history_feats + self.drop_path(self.mlp_hist(self.norm_hist(history_feats)))
                history_index = torch.arange(special_toks_indices['<history>'], special_toks_indices['</s>'], device=device)
                history_index = history_index.unsqueeze(0).unsqueeze(-1)
                history_index = history_index.repeat(bs, 1, h_dim)
                x = x.scatter(1, history_index, history_feats)

        elif expert_flag == 'fusion':
            x = x + self.drop_path(self.mlp_fusion(self.norm_fusion(x)))

        return x, attn

        # if expert_flag == 2:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        # elif expert_flag == 0:
        #     x = (x[:, -it_split:])
        #     x = x + self.drop_path(self.sentence_mlp(self.sentence_norm(x)))
        # elif expert_flag == 1:
        #     x = (x[:, :-it_split ])
        #     x = x + self.drop_path(self.image_mlp(self.image_norm(x)))
        # elif expert_flag == 3:
        #     text, image = (x[:, :it_split], x[:, it_split:],)
        #     text = text + self.drop_path(self.sentence_mlp(self.sentence_norm(text)))
        #     image = image + self.drop_path(self.image_mlp(self.image_norm(image)))
        #     x = torch.cat([text, image], dim=1)
        # elif expert_flag == 4:
        #     x = x + self.drop_path(self.generation_mlp(self.generation_norm(x)))
        # return x, attn