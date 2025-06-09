import logging
import torch
from models.utils import (interpolate_pos_relative_bias_beit,
                          load_temp_embed_with_mismatch)

logger = logging.getLogger(__name__)


def interpolate_pos_embed_beit(state_dict, new_model):
    """interpolate the positional embeddings.
    The spatial pe is relative and temporal pe is absolute.
    additional temporal pe is padded with 0.

    Args:
        state_dict (dict): The state_dict.
        new_model (nn.Module): The created model.

    Returns: dict. The state_dict with updated positional embeddings.

    """
    state_dict = interpolate_pos_relative_bias_beit(
        state_dict_old=state_dict,
        state_dict_new=new_model.state_dict(),
        patch_shape_new=new_model.beit.embeddings.patch_embeddings.patch_shape,
    )
    # absolute temporal pos bias
    temporal_pe_key = "beit.embeddings.temporal_position_embeddings"
    if temporal_pe_key in state_dict:
        logger.info(f"interpolate temporal positional embeddings: {temporal_pe_key}")
        state_dict[temporal_pe_key] = load_temp_embed_with_mismatch(
            temp_embed_old=state_dict[temporal_pe_key],
            temp_embed_new=new_model.state_dict()[temporal_pe_key],
        )
    return state_dict

def extract_beit_from_vindlu(vindlu_state_dict):
    beit_state_dict = {}
    beit_param_names = [k for k in vindlu_state_dict if k.startswith('vision_encoder.') and 'temp_model' not in k]
    for param_name in beit_param_names:
        new_name = param_name.replace('vision_encoder.', '')
        beit_state_dict[new_name] = vindlu_state_dict[param_name]
    
    return beit_state_dict

def build_beit(model_config, image_res, checkpoint=False):
    """build beit with configuration.

    Args:
        config (dict): The configs for beit.
        image_res (int): The image resolution.
        checkpoint (bool): Whether to enable gradient checkpointing.

    Returns: nn.Module

    """
    from .st_beit import BeitConfig as config_cls
    from .st_beit import BeitModel as model_cls


    vindlu_state_dict = torch.load(model_config['vindlu_path'])['model']
    state_dict = extract_beit_from_vindlu(vindlu_state_dict)
    model_config = model_config['beit_config_json']

    logger.info(
        f"Loading vit pre-trained weights from huggingface {model_config['pretrained']}."
    )
    # BEiT uses average pooled tokens instead of [CLS] used by other models
    aux_kwargs = {"add_pooling_layer": True}
    # tmp_model = model_cls.from_pretrained(model_config['beit_pretrained'], **aux_kwargs)


    # tmp_model = model_cls.from_pretrained(model_config['pretrained'], **aux_kwargs)
    # state_dict = tmp_model.state_dict()

    # del tmp_model

    logger.info(f"Init new model with new image size {image_res}, and load weights.")

    # other_cfg = model_config.temporal_modeling
    other_cfg = {}

    vit_config = config_cls.from_pretrained(
        model_config['pretrained'], image_size=image_res, **other_cfg
    )

    # vit_config.update(model_config)

    model = model_cls(config=vit_config, **aux_kwargs)

    if checkpoint:
        model.gradient_checkpointing_enable()

    # interpolate relative pos bias
    state_dict = interpolate_pos_relative_bias_beit(
        state_dict_old=state_dict,
        state_dict_new=model.state_dict(),
        patch_shape_new=model.embeddings.patch_embeddings.patch_shape,
    )

    # del prompt_bias_table
    for k in list(state_dict.keys()):
        if "prompt_bias_table" in k:
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    return model
