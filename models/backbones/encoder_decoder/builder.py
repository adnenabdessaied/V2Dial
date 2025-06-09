
import glog as logger
import re
import json

from peft import LoraConfig, get_peft_model

from .xflan_t5 import T5Config, T5ForConditionalGeneration
from .xbart import BartConfig, BartForConditionalGeneration, BartEncoder, BartForCausalLM


def build_encoder_decoder(model_config):
    """build (encoder-) decoder model for answer generation.

    Args:
        model_config (dict): model config.

    Returns: TODO

    """
    logger.info('[INFO] Loading Encoder Decoder [Type = {}]'.format(model_config['enc_dec_name']))

    if model_config['enc_dec_family'] == 'flan_t5':
        config_cls = T5Config
        model_cls  = T5ForConditionalGeneration
    elif model_config['enc_dec_family'] == 'bart':
        config_cls = BartConfig
        if model_config['use_decoder_only']:
            model_cls = BartForCausalLM
        else:
            model_cls = BartForConditionalGeneration
    else:
        raise ValueError('{} is not supported'.format(model_config['enc_dec_family']))
    enc_dec_config = config_cls.from_pretrained(model_config['enc_dec_name'])
    model_config['enc_dec_dim'] = enc_dec_config.d_model
    # enc_dec_config.encoder_layers = enc_dec_config.encoder_layers - model_config['num_layers_modality_expert_{}'.format(model_config['enc_dec_family'])]
    enc_dec = model_cls.from_pretrained(
        model_config['enc_dec_name'],
        config=enc_dec_config
    )

    # first_k = model_config['num_layers_modality_expert_{}'.format(model_config['enc_dec_family'])]
    # enc_dec.model.encoder.remove_first_k_layers(first_k)
    # get the last encoder layers 
    # enc_dec.


    if model_config['use_lora_enc_dec']:
        # load the lora config
        with open(model_config['lora_config'], 'r') as f: 
            lora_config = json.load(f)

        # get the linear layer to perform LoRA on
        model_modules = str(enc_dec.modules)
        pattern = r'\((\w+)\): Linear'
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))

        lora_config['target_modules'] = target_modules
        
        lora_config = LoraConfig(**lora_config)

        enc_dec = get_peft_model(enc_dec, lora_config)

    return enc_dec


def build_encoder(model_config, expert_type, modality=None):
    """build (encoder-) decoder model for answer generation.

    Args:
        model_config (dict): model config.

    Returns: TODO

    """
    log_txt = '[INFO] Loading {} Expert'.format(expert_type)
    if modality is not None:
            log_txt += ' [Modality = {}]'.format(modality)
    log_txt += ' [Type = {}]'.format(model_config['enc_dec_name'])

    logger.info(log_txt)

    if model_config['enc_dec_family'] == 'flan_t5':
        config_cls = T5Config
        model_cls  = T5ForConditionalGeneration
    elif model_config['enc_dec_family'] == 'bart':
        config_cls = BartConfig
        model_cls  = BartEncoder
    else:
        raise ValueError('{} is not supported'.format(model_config['enc_dec_family']))

    config = config_cls.from_pretrained(model_config['enc_dec_name'])
    config.modality_expert_layers = model_config['num_layers_modality_expert_{}'.format(model_config['enc_dec_family'])]
    config.grounding_expert_layers = model_config['num_layers_grounding_expert_{}'.format(model_config['enc_dec_family'])]

    model_config['enc_dec_dim'] = config.d_model

    expert = model_cls.from_pretrained(
        model_config['enc_dec_name'],
        config=config,
        expert_type=expert_type,
        modality=modality
    )

    if model_config['use_lora_expert']:
        # load the lora config
        with open(model_config['lora_config'], 'r') as f: 
            lora_config = json.load(f)

        # get the linear layer to perform LoRA on
        model_modules = str(expert.modules)
        pattern = r'\((\w+)\): Linear'
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))

        lora_config['target_modules'] = target_modules
        
        lora_config = LoraConfig(**lora_config)

        expert = get_peft_model(expert, lora_config)

    # expert = model_cls(
    #     config=config,
    #     expert_type=expert_type,
    #     modality=modality
    # )

    return expert


