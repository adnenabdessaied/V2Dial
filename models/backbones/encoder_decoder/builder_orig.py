from .xflan_t5 import T5Config, T5ForConditionalGeneration
from .xbart_original import BartConfig, BartForConditionalGeneration, BartEncoder

import glog as logger


def build_encoder_decoder(model_config):
    """build (encoder-) decoder model for answer generation.

    Args:
        model_config (dict): model config.

    Returns: TODO

    """
    logger.info('[INFO] Loading Encoder Decoder: {}'.format(model_config['enc_dec_name']))

    if model_config['enc_dec_family'] == 'flan_t5':
        config_cls = T5Config
        model_cls  = T5ForConditionalGeneration
    elif model_config['enc_dec_family'] == 'bart':
        config_cls = BartConfig
        model_cls  = BartForConditionalGeneration
    else:
        raise ValueError('{} is not supported'.format(model_config['enc_dec_family']))
    config = config_cls.from_pretrained(model_config['enc_dec_name'])
    model_config['enc_dec_dim'] = config.d_model
    enc_dec = model_cls.from_pretrained(
        model_config['enc_dec_name'],
        config=config
    )

    return enc_dec


def build_encoder(model_config):
    """build (encoder-) decoder model for answer generation.

    Args:
        model_config (dict): model config.

    Returns: TODO

    """
    logger.info('[INFO] Loading Expert as Encoder of {}'.format(model_config['enc_dec_name']))

    if model_config['enc_dec_family'] == 'flan_t5':
        config_cls = T5Config
        model_cls  = T5ForConditionalGeneration
    elif model_config['enc_dec_family'] == 'bart':
        config_cls = BartConfig
        model_cls  = BartEncoder
    else:
        raise ValueError('{} is not supported'.format(model_config['enc_dec_family']))

    config = config_cls.from_pretrained(model_config['enc_dec_name'])
    model_config['enc_dec_dim'] = config.d_model
    config.encoder_layers = model_config['num_layers_modality_expert']

    expert = model_cls.from_pretrained(
        model_config['enc_dec_name'],
        config=config
    )

    return expert
