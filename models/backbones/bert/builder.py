from .xbert import BertConfig, BertForMaskedLM, BertLMHeadModel, BertModel


def build_bert(model_config, pretrain, checkpoint, expert_type, modality_type='text'):
    """build text encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_size = model_config['expert_size']
    bert_config = BertConfig.from_json_file(model_config[f'bert_config_{bert_size}'])
    # bert_config.encoder_width = model_config.vision_encoder.d_model
    bert_config.gradient_checkpointing = checkpoint
    bert_config.num_hidden_layers = model_config['num_layers_{}_expert'.format(expert_type)]
    if expert_type=='modality':
        if modality_type == 'vis':
            bert_config.cross_attention_freq = 2
        else:
            bert_config.cross_attention_freq = -1
    else:
        bert_config.cross_attention_freq = 1

    if pretrain:
        text_encoder, loading_info = BertForMaskedLM.from_pretrained(
            f'bert-{bert_size}-uncased',
            config=bert_config,
            output_loading_info=True,
        )
    else:
        text_encoder, loading_info = BertModel.from_pretrained(
            f'bert-{bert_size}-uncased',
            config=bert_config,
            add_pooling_layer=True,
            output_loading_info=True,
        )

    return text_encoder


def build_bert_decoder(model_config, checkpoint):
    """build text decoder the same as the multimodal encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    bert_config.encoder_width = model_config.vision_encoder.d_model
    bert_config.gradient_checkpointing = checkpoint

    bert_config.fusion_layer = 0
    bert_config.num_hidden_layers = (
        bert_config.num_hidden_layers - model_config.text_encoder.fusion_layer
    )

    text_decoder, loading_info = BertLMHeadModel.from_pretrained(
        model_config.text_encoder.pretrained,
        config=bert_config,
        output_loading_info=True,
    )

    return text_decoder
