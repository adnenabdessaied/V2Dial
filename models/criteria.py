from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import allgather_wgrad
from utils.dist import get_rank, get_world_size
from utils.easydict import EasyDict


def get_sim(
    x_proj: torch.Tensor,
    y_proj: torch.Tensor,
    temp=1.0,
):
    """calculate pair-wise similarity between two modalities x and y.

    Args:
        x_proj (torch.Tensor): The representation of modality x. Shape: [B,T,C] or [B,C].
        y_proj (torch.Tensor): The representation of modality y. Shape: [B,C].
        temp (torch.Tensor): The temperature. Shape: [].

    Returns: The similarity between modality x and y. Shape: [B,B].

    """
    x_proj = F.normalize(x_proj, dim=-1)
    y_proj = F.normalize(y_proj, dim=-1)
    assert x_proj.dim() in [2, 3]
    assert y_proj.dim() == 2
    if x_proj.dim() == 2:
        sim_x2y = torch.einsum("md,nd->mn", x_proj, y_proj) / temp  # (B,B)
    else:
        sim_x2y = torch.einsum("mld,nd->mln", x_proj, y_proj).mean(1) / temp  # (B,B)
    sim_y2x = sim_x2y.T
    return sim_x2y, sim_y2x


class ContMatchLoss(nn.Module):
    def __init__(self):
        super(ContMatchLoss, self).__init__()

    @torch.no_grad()
    def get_mask(self, sim, idx=None, normalize=False):
        """
        Args:
            sim (torch.Tensor): The similarity between videos and texts. shape: (B, B).
            idx (torch.Tensor): The index for each video. Shape: [B].
            normalize (bool): If true, make row sum equal to 1
        """
        if idx is not None:
            idx = idx.view(-1, 1)
            mask = torch.eq(idx, idx.T).to(sim.dtype)
            if normalize:
                mask = mask / mask.sum(1, keepdim=True)
        else:
            mask = torch.zeros_like(sim)
            mask.fill_diagonal_(1)
        return mask  # `1` mark valid/matched location

    @lru_cache(maxsize=16)
    def get_gather_args(self):
        """obtain the args for all_gather
        Returns: dict.

        """
        return EasyDict({"world_size": get_world_size(), "rank": get_rank()})


class STC_STM_Loss(ContMatchLoss):
    """Contrastive and matching losses"""

    def __init__(self):
        super(STC_STM_Loss, self).__init__()

    def stc_loss(
        self,
        temporal_proj: torch.Tensor,
        spatial_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True
    ):
        
        """forward to calculate the loss

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            idx (torch.Tensor): The index for each example. Shape: [B,].
            temp (torch.Tensor): The temperature. Shape: [].
            all_gather (bool): If true, will gather samples across all the GPUs and calculate loss across the gathered samples.

        Returns: loss_vtc (torch.Tensor): The video-text contrastive loss. Shape: [].

        """
        if all_gather:
            gather_args = self.get_gather_args()
            temporal_proj = allgather_wgrad(temporal_proj, gather_args)
            spatial_proj = allgather_wgrad(spatial_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_t2s, sim_s2t = get_sim(temporal_proj, spatial_proj, temp)

        with torch.no_grad():
            sim_t2s_targets = self.get_mask(sim_t2s, idx=idx, normalize=True)
            sim_s2t_targets = sim_t2s_targets

        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * sim_t2s_targets, dim=1).mean()
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * sim_s2t_targets, dim=1).mean()

        loss_stc = (loss_t2s + loss_s2t) / 2
        return loss_stc
    
    def stm_loss(
        self,
        grounding_expert,
        stm_head,
        # temp,
        spatial_embeds_orig,
        temporal_embeds_orig,
        temporal_proj,
        spatial_proj,
        idx,
        generation=False,
        temp=1.0
    ):
        spatial_embeds = spatial_embeds_orig.clone()
        temporal_embeds = temporal_embeds_orig.clone()
        with torch.no_grad():
            sim_s2t, sim_t2s = get_sim(temporal_proj, spatial_proj, temp)
            spatial_atts = torch.ones(
                spatial_embeds.size()[:-1], dtype=torch.long, device=spatial_embeds.device
            )
            temporal_atts = torch.ones(
                temporal_embeds.size()[:-1], dtype=torch.long, device=temporal_embeds.device
            )
            weights_s2t = F.softmax(sim_s2t + 1e-4, dim=1)  # (N, N)
            weights_t2s = F.softmax(sim_t2s + 1e-4, dim=1)

            mask = self.get_mask(sim_s2t, idx=idx).bool()
            weights_s2t.masked_fill_(mask, 0)
            weights_t2s.masked_fill_(mask, 0)
            weights_s2t = torch.nan_to_num_(weights_s2t, nan=1e-2, posinf=1e-2, neginf=1e-2)
            weights_t2s = torch.nan_to_num_(weights_t2s, nan=1e-2, posinf=1e-2, neginf=1e-2)

        if generation:
            with torch.no_grad():
                output = grounding_expert(
                    encoder_embeds=temporal_embeds,
                    attention_mask=temporal_atts,
                    encoder_hidden_states=spatial_embeds,
                    encoder_attention_mask=spatial_atts,
                    return_dict=True,
                )
            pos_feats = output.last_hidden_state
            return pos_feats

        else:
            # select a hard negatives within the batch
            spatial_neg_indices = torch.multinomial(weights_s2t, 1).squeeze()
            temporal_neg_indices = torch.multinomial(weights_t2s, 1).squeeze()


            spatial_embeds_neg = spatial_embeds[spatial_neg_indices]  # [B, L, c]
            temporal_embeds_neg = temporal_embeds[temporal_neg_indices]  # [B, L, d]
            # temporal_atts_neg = temporal_atts[temporal_neg_indices]

            # concat embeddings
            spatial_embeds_all = torch.cat([spatial_embeds, spatial_embeds_neg, spatial_embeds], dim=0)
            temporal_embeds_all = torch.cat([temporal_embeds, temporal_embeds, temporal_embeds_neg], dim=0)
            spatial_atts_all = torch.cat([spatial_atts, spatial_atts, spatial_atts], dim=0)
            temporal_atts_all = torch.cat([temporal_atts, temporal_atts, temporal_atts], dim=0)

            output = grounding_expert(
                inputs_embeds=temporal_embeds_all,
                attention_mask=temporal_atts_all,
                cross_embeds=spatial_embeds_all,
                cross_attention_mask=spatial_atts_all,
            )

            stm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

            stm_logits = stm_head(stm_embeds)  # [3*B, 2]

            bs = stm_logits.shape[0] // 3
            stm_labels = stm_logits.new_ones(3 * bs, dtype=torch.long)
            stm_labels[bs:] = 0
            loss_stm = F.cross_entropy(stm_logits, stm_labels)
            pos_feats = output.last_hidden_state[:bs]

            return loss_stm, pos_feats
        

class VCC_VCM_Loss(ContMatchLoss):
    """Contrastive and matching losses"""

    def __init__(self):
        super(VCC_VCM_Loss, self).__init__()

    def vcc_loss(
        self,
        vis_proj: torch.Tensor,
        cap_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True
    ):
        
        """forward to calculate the loss

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            idx (torch.Tensor): The index for each example. Shape: [B,].
            temp (torch.Tensor): The temperature. Shape: [].
            all_gather (bool): If true, will gather samples across all the GPUs and calculate loss across the gathered samples.

        Returns: loss_vtc (torch.Tensor): The video-text contrastive loss. Shape: [].

        """
        if all_gather:
            gather_args = self.get_gather_args()
            vis_proj = allgather_wgrad(vis_proj, gather_args)
            cap_proj = allgather_wgrad(cap_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_v2c, sim_c2v = get_sim(vis_proj, cap_proj, temp)

        with torch.no_grad():
            sim_v2c_targets = self.get_mask(sim_v2c, idx=idx, normalize=True)
            sim_c2v_targets = sim_v2c_targets

        loss_v2c = -torch.sum(F.log_softmax(sim_v2c, dim=1) * sim_v2c_targets, dim=1).mean()
        loss_c2v = -torch.sum(F.log_softmax(sim_c2v, dim=1) * sim_c2v_targets, dim=1).mean()

        loss_vcc = (loss_v2c + loss_c2v) / 2
        return loss_vcc
    
    def vcm_loss(
        self,
        grounding_expert,
        vcm_head,
        vis_embeds_orig,
        cap_embeds_orig,
        vis_proj,
        cap_proj,
        cap_atts,
        idx,
        generation=False,
        temp=1.0
    ):
        vis_embeds = vis_embeds_orig.clone()
        cap_embeds = cap_embeds_orig.clone()

        with torch.no_grad():
            sim_v2c, sim_c2v = get_sim(vis_proj, cap_proj, temp)
            vis_atts = torch.ones(
                vis_embeds.size()[:-1], dtype=torch.long, device=vis_embeds.device
            )

            weights_v2c = F.softmax(sim_v2c + 1e-4, dim=1)  # (N, N)
            weights_c2v = F.softmax(sim_c2v + 1e-4, dim=1)

            mask = self.get_mask(weights_v2c, idx=idx).bool()
            weights_v2c.masked_fill_(mask, 0)
            weights_c2v.masked_fill_(mask, 0)
            weights_v2c = torch.nan_to_num_(weights_v2c, nan=1e-2, posinf=1e-2, neginf=1e-2)
            weights_c2v = torch.nan_to_num_(weights_c2v, nan=1e-2, posinf=1e-2, neginf=1e-2)

        if generation:
            with torch.no_grad():
                output = grounding_expert(
                    encoder_embeds=cap_embeds,
                    attention_mask=cap_atts,
                    encoder_hidden_states=vis_embeds,
                    encoder_attention_mask=vis_atts,
                    return_dict=True,
                )
            pos_feats = output.last_hidden_state
            return pos_feats

        else:
            
            
            # select a hard negatives within the batch
            vis_neg_indices = torch.multinomial(weights_v2c, 1).squeeze()
            cap_neg_indices = torch.multinomial(weights_c2v, 1).squeeze()


            vis_embeds_neg = vis_embeds[vis_neg_indices]  # [B, L, c]
            cap_embeds_neg = cap_embeds[cap_neg_indices]  # [B, L, d]
            cap_atts_neg = cap_atts[cap_neg_indices]

            # concat embeddings
            vis_embeds_all = torch.cat([vis_embeds, vis_embeds_neg, vis_embeds], dim=0)
            cap_embeds_all = torch.cat([cap_embeds, cap_embeds, cap_embeds_neg], dim=0)
            vis_atts_all = torch.cat([vis_atts, vis_atts, vis_atts], dim=0)
            cap_atts_all = torch.cat([cap_atts, cap_atts, cap_atts_neg], dim=0)

            output = grounding_expert(
                inputs_embeds=cap_embeds_all,
                attention_mask=cap_atts_all,
                cross_embeds=vis_embeds_all,
                cross_attention_mask=vis_atts_all,
            )

            vcm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

            vcm_logits = vcm_head(vcm_embeds)  # [3*B, 2]

            bs = vcm_logits.shape[0] // 3
            vcm_labels = vcm_logits.new_ones(3 * bs, dtype=torch.long)
            vcm_labels[bs:] = 0
            loss_vcm = F.cross_entropy(vcm_logits, vcm_labels)
            pos_feats = output.last_hidden_state[:bs]
            return loss_vcm, pos_feats


class VHC_VHM_Loss(ContMatchLoss):
    """Contrastive and matching losses"""

    def __init__(self):
        super(VHC_VHM_Loss, self).__init__()

    def vhc_loss(
        self,
        vis_proj: torch.Tensor,
        hist_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True
    ):
        
        """forward to calculate the loss

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            idx (torch.Tensor): The index for each example. Shape: [B,].
            temp (torch.Tensor): The temperature. Shape: [].
            all_gather (bool): If true, will gather samples across all the GPUs and calculate loss across the gathered samples.

        Returns: loss_vtc (torch.Tensor): The video-text contrastive loss. Shape: [].

        """
        if all_gather:
            gather_args = self.get_gather_args()
            vis_proj = allgather_wgrad(vis_proj, gather_args)
            hist_proj = allgather_wgrad(hist_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_v2h, sim_h2v = get_sim(vis_proj, hist_proj, temp)

        with torch.no_grad():
            sim_v2h_targets = self.get_mask(sim_v2h, idx=idx, normalize=True)
            sim_h2v_targets = sim_v2h_targets

        loss_v2h = -torch.sum(F.log_softmax(sim_v2h, dim=1) * sim_v2h_targets, dim=1).mean()
        loss_h2v = -torch.sum(F.log_softmax(sim_h2v, dim=1) * sim_h2v_targets, dim=1).mean()

        loss_vhc = (loss_v2h + loss_h2v) / 2
        return loss_vhc
    
    def vhm_loss(
        self,
        grounding_expert,
        vhm_head,
        vis_embeds_orig,
        hist_embeds_orig,
        vis_proj,
        hist_proj,
        hist_atts,
        idx,
        generation=False,
        temp=1.0,
    ):
        vis_embeds = vis_embeds_orig.clone()
        hist_embeds = hist_embeds_orig.clone()
        with torch.no_grad():
            sim_v2h, sim_h2v = get_sim(vis_proj, hist_proj, temp)
            vis_atts = torch.ones(
                vis_embeds.size()[:-1], dtype=torch.long, device=vis_embeds.device
            )

            weights_v2h = F.softmax(sim_v2h + 1e-4, dim=1)  # (N, N)
            weights_h2v = F.softmax(sim_h2v + 1e-4, dim=1)

            mask = self.get_mask(weights_v2h, idx=idx).bool()
            weights_v2h.masked_fill_(mask, 0)
            weights_h2v.masked_fill_(mask, 0)
            weights_v2h = torch.nan_to_num_(weights_v2h, nan=1e-2, posinf=1e-2, neginf=1e-2)
            weights_h2v = torch.nan_to_num_(weights_h2v, nan=1e-2, posinf=1e-2, neginf=1e-2)

        if generation:
            with torch.no_grad():
                output = grounding_expert(
                    encoder_embeds=hist_embeds,
                    attention_mask=hist_atts,
                    encoder_hidden_states=vis_embeds,
                    encoder_attention_mask=vis_atts,
                    return_dict=True,
                # mode="fusion",
            )
            pos_feats = output.last_hidden_state
            return pos_feats

        else:
            # select a hard negatives within the batch
            vis_neg_indices = torch.multinomial(weights_v2h, 1).squeeze()
            hist_neg_indices = torch.multinomial(weights_h2v, 1).squeeze()

            vis_embeds_neg = vis_embeds[vis_neg_indices]  # [B, L, c]
            hist_embeds_neg = hist_embeds[hist_neg_indices]  # [B, L, d]
            hist_atts_neg = hist_atts[hist_neg_indices]

            # concat embeddings
            vis_embeds_all = torch.cat([vis_embeds, vis_embeds_neg, vis_embeds], dim=0)
            hist_embeds_all = torch.cat([hist_embeds, hist_embeds, hist_embeds_neg], dim=0)
            vis_atts_all = torch.cat([vis_atts, vis_atts, vis_atts], dim=0)
            hist_atts_all = torch.cat([hist_atts, hist_atts, hist_atts_neg], dim=0)

            output = grounding_expert(
                inputs_embeds=hist_embeds_all,
                attention_mask=hist_atts_all,
                cross_embeds=vis_embeds_all,
                cross_attention_mask=vis_atts_all,
            )

            vhm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

            vhm_logits = vhm_head(vhm_embeds)  # [3*B, 2]

            bs = vhm_logits.shape[0] // 3
            vhm_labels = vhm_logits.new_ones(3 * bs, dtype=torch.long)
            vhm_labels[bs:] = 0
            loss_vhm = F.cross_entropy(vhm_logits, vhm_labels)
            pos_feats = output.last_hidden_state[:bs]

            return loss_vhm, pos_feats


class CHC_CHM_Loss(ContMatchLoss):
    """Contrastive and matching losses"""

    def __init__(self):
        super(CHC_CHM_Loss, self).__init__()

    def chc_loss(
        self,
        cap_proj: torch.Tensor,
        hist_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True
    ):
        
        """forward to calculate the loss

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            idx (torch.Tensor): The index for each example. Shape: [B,].
            temp (torch.Tensor): The temperature. Shape: [].
            all_gather (bool): If true, will gather samples across all the GPUs and calculate loss across the gathered samples.

        Returns: loss_vtc (torch.Tensor): The video-text contrastive loss. Shape: [].

        """
        if all_gather:
            gather_args = self.get_gather_args()
            cap_proj = allgather_wgrad(cap_proj, gather_args)
            hist_proj = allgather_wgrad(hist_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_c2h, sim_h2c = get_sim(cap_proj, hist_proj, temp)

        with torch.no_grad():
            sim_c2h_targets = self.get_mask(sim_c2h, idx=idx, normalize=True)
            sim_h2c_targets = sim_c2h_targets

        loss_c2h = -torch.sum(F.log_softmax(sim_c2h, dim=1) * sim_c2h_targets, dim=1).mean()
        loss_h2c = -torch.sum(F.log_softmax(sim_h2c, dim=1) * sim_h2c_targets, dim=1).mean()

        loss_chc = (loss_c2h + loss_h2c) / 2
        return loss_chc
    
    def chm_loss(
        self,
        grounding_expert,
        chm_head,
        cap_embeds_orig,
        hist_embeds_orig,
        cap_proj,
        hist_proj,
        cap_atts,
        hist_atts,
        idx,
        generation=False,
        temp=1.0
    ):
        cap_embeds = cap_embeds_orig.clone()
        hist_embeds = hist_embeds_orig.clone()
        with torch.no_grad():
            sim_c2h, sim_h2c = get_sim(cap_proj, hist_proj, temp)

            weights_c2h = F.softmax(sim_c2h + 1e-4, dim=1)  # (N, N)
            weights_h2c = F.softmax(sim_h2c + 1e-4, dim=1)

            mask = self.get_mask(weights_c2h, idx=idx).bool()
            weights_c2h.masked_fill_(mask, 0)
            weights_h2c.masked_fill_(mask, 0)
            weights_c2h = torch.nan_to_num_(weights_c2h, nan=1e-2, posinf=1e-2, neginf=1e-2)
            weights_h2c = torch.nan_to_num_(weights_h2c, nan=1e-2, posinf=1e-2, neginf=1e-2)

        if generation:
            with torch.no_grad():
                output = grounding_expert(
                    encoder_embeds=hist_embeds,
                    attention_mask=hist_atts,
                    encoder_hidden_states=cap_embeds,
                    encoder_attention_mask=cap_atts,
                    return_dict=True,
                )
            pos_feats = output.last_hidden_state
            return pos_feats
        else:
            # select a hard negatives within the batch
            cap_neg_indices = torch.multinomial(weights_c2h, 1).squeeze()
            hist_neg_indices = torch.multinomial(weights_h2c, 1).squeeze()

            cap_embeds_neg = cap_embeds[cap_neg_indices]  # [B, L, c]
            cap_atts_neg = cap_atts[cap_neg_indices]
            hist_embeds_neg = hist_embeds[hist_neg_indices]  # [B, L, d]
            hist_atts_neg = hist_atts[hist_neg_indices]

            # concat embeddings
            cap_embeds_all = torch.cat([cap_embeds, cap_embeds_neg, cap_embeds], dim=0)
            hist_embeds_all = torch.cat([hist_embeds, hist_embeds, hist_embeds_neg], dim=0)
            cap_atts_all = torch.cat([cap_atts, cap_atts_neg, cap_atts], dim=0)
            hist_atts_all = torch.cat([hist_atts, hist_atts, hist_atts_neg], dim=0)

            output = grounding_expert(
                inputs_embeds=hist_embeds_all,
                attention_mask=hist_atts_all,
                cross_embeds=cap_embeds_all,
                cross_attention_mask=cap_atts_all,
            )

            chm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

            chm_logits = chm_head(chm_embeds)  # [3*B, 2]

            bs = chm_logits.shape[0] // 3
            chm_labels = chm_logits.new_ones(3 * bs, dtype=torch.long)
            chm_labels[bs:] = 0
            loss_chm = F.cross_entropy(chm_logits, chm_labels)
            pos_feats = output.last_hidden_state[:bs]
            return loss_chm, pos_feats


class MLMLoss(nn.Module):
    """masked language modeling loss."""

    def __init__(self, masking_prob, tokenizer):
        super(MLMLoss, self).__init__()
        self.tokenizer = tokenizer
        self.masking_prob = masking_prob

    def mlm_loss(
        self,
        text_encoder,
        text,
        text_embeds,
        vision_embeds,
        vision_atts,
    ):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.masking_prob)
        input_ids, labels = self.mask(
            input_ids,
            text_encoder.config.vocab_size,
            input_ids.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        # intermediate_mlm_output = text_encoder.bert(
        #     input_ids,
        #     attention_mask=text.attention_mask,
        #     encoder_hidden_states=vision_embeds,
        #     encoder_attention_mask=vision_atts,
        #     return_dict=True,
        #     # mode="text",
        # )

        # text_embeds = intermediate_mlm_output.last_hidden_state

        mlm_output = text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            # mode="fusion",
        )
        return mlm_output.loss

    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
