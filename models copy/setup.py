import copy
import os.path as osp
import glog as logger

import torch
from torch.utils.data import ConcatDataset
from models.backbones.beit.builder import interpolate_pos_embed_beit
from models.backbones.bert.tokenization_bert import BertTokenizer
from transformers import T5Tokenizer, BartTokenizer, LlamaTokenizer
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from datasets.dataloader import load_dataloaders
from datasets.pretraining import load_datasets as load_datasets_stage_1
from datasets.visdial_dataset import load_visdial_dataset
from datasets.champagne_dataset import load_champagne_dataset
from datasets.nextqa_dataset import load_nextqa_dataset
from datasets.avsd_dataset import load_avsd_dataset
# from datasets.avsd_dataset_like_mixer import load_avsd_dataset 

from processors.blip_processors import Blip2ImageTrainProcessor
from processors.blip_processors import BlipCaptionProcessor, BlipDialogProcessor

from utils.init import set_training_steps
# from models.v2dial import V2Dial, V2DialBase
from models.v2dial import V2DialBase, V2Dial, V2DialNoMoes

# from datasets.avsd_dataset import get_dataset, AVSDDataSet
from torch.utils.data import DataLoader


def setup_model(
    config, has_decoder=False, pretrain=False, find_unused_parameters=True
):
    logger.info("Creating model")

    if config['stage'] == 'stage_1':
        config = copy.deepcopy(config)
        
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = V2DialBase(config=config, expert_tokenizer=tokenizer)
        model = V2DialBase(config)
        model = model.to(torch.device('cuda'))
        model_without_ddp = model
        optimizer = create_optimizer(config, model)
        scheduler = create_scheduler(config, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

        if config['distributed']:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config['gpu']],
                find_unused_parameters=find_unused_parameters,  # `False` for image-only task
            )

        start_epoch = 0
        global_step = 0
        webvid_step = 0
        cc3m_step = 0
        
        if osp.isfile(config['pretrained_path']):
            logger.info(f"Loading checkpoint from {config['pretrained_path']}")
            checkpoint = torch.load(config['pretrained_path'], map_location="cpu")
            state_dict = checkpoint["model"]

            if config.resume:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                scaler.load_state_dict(checkpoint["scaler"])
                start_epoch = checkpoint["epoch"] + 1
                global_step = checkpoint["global_step"]
            elif not pretrain:  # downstream init from pretrained ckpt

                # interpolate positional embeddings.
                state_dict = interpolate_pos_embed_beit(state_dict, model_without_ddp)


                #TODO Might need to update to match the MoEs
                if not config.evaluate:  # finetuning from a pretarined weights.
                    for key in list(state_dict.keys()):
                        if "bert" in key:
                            encoder_key = key.replace("bert.", "")
                            state_dict[encoder_key] = state_dict[key]
                            if not has_decoder:
                                del state_dict[key]

                        # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                        # only for generation tasks like VQA
                        if has_decoder and "text_encoder" in key:
                            if "layer" in key:
                                encoder_keys = key.split(".")
                                layer_num = int(encoder_keys[4])
                                if layer_num < config.model.text_encoder.fusion_layer:
                                    del state_dict[key]
                                    continue
                                else:
                                    decoder_layer_num = layer_num - 9
                                    encoder_keys[4] = str(decoder_layer_num)
                                    encoder_key = ".".join(encoder_keys)
                            else:
                                encoder_key = key
                            decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                            state_dict[decoder_key] = state_dict[key]
                            del state_dict[key]

            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info(f"Loaded checkpoint from {config.pretrained_path}")
        else:
            logger.warning("No pretrained checkpoint provided, training from scratch")

        return (
            model,
            model_without_ddp,
            optimizer,
            scheduler,
            scaler,
            start_epoch,
            global_step,
            webvid_step,
            cc3m_step,
            config
        )
    else:
        # config = copy.deepcopy(config)
        # if config['use_original_feats']:
        #     model = AVSDBart(config)
        # else:
        #     # model = V2Dial(config, tokenizer_experts, tokenizer_enc_dec)
        # if config.use_moes:
        model = V2Dial(config)
        # else:
            # model = V2DialNoMoes(config)

        model = model.to(torch.device('cuda'))
        model_without_ddp = model

        optimizer = None
        scheduler = None
        scaler    = None
              
        start_epoch  = 0
        global_step  = 0
        if config['stage'] == 'stage_3':
            visdial_step = 0
            avsd_step    = 0
            nextqa_step  = 0

        ckpt_path = config.pretrained_path_resume if config.resume else config.pretrained_path_prev_stage
        if config.generating:
            ckpt_path = config.best_ckpt_path

        if osp.isfile(ckpt_path):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint["model"]

            if config.resume:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                scaler.load_state_dict(checkpoint["scaler"])
                start_epoch = checkpoint["epoch"] + 1
                global_step = checkpoint["global_step"]
                if config['stage'] == 'stage_3':
                    visdial_step = checkpoint['visdial_step']
                    avsd_step = checkpoint['avsd_step']
                    next_step = checkpoint['nextqa_step']


            if config['stage'] in ['stage_2', 'stage_3'] and config.use_moes:
                # Init. the history expert erights with the caption expert weights
                p_names = [
                    'moe_layers.{}.norm_hist.weight',
                    'moe_layers.{}.mlp_hist.fc1.weight',
                    'moe_layers.{}.mlp_hist.fc1.bias',
                    'moe_layers.{}.mlp_hist.fc2.weight',
                    'moe_layers.{}.mlp_hist.fc2.bias',                 
                ]

                for moe_layer_idx in range(config.num_moe_modality_layers):
                    for p_name in p_names:
                        p_hist_name = p_name.format(moe_layer_idx)
                        if p_hist_name not in state_dict:
                            p_cap_name = p_hist_name.replace('hist', 'cap')
                            state_dict[p_hist_name] = state_dict[p_cap_name].clone()

            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            logger.info(msg)

            logger.info(f"Loaded checkpoint from {ckpt_path}")
        else:
            logger.warning("No pretrained checkpoint provided, training from scratch")

        if config['training']:
            optimizer = create_optimizer(config, model_without_ddp)
            scheduler = create_scheduler(config, optimizer)
            scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

        elif config['generating']:
            model.llm.set_input_embeddings(model.text_embedding)

        if config['distributed']:
            
            static_graph=config.stage!='stage_1'
            if len(config.media_train) > 0:
                static_graph = False

            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp,
                device_ids=[config['gpu']],
                find_unused_parameters=find_unused_parameters,  # `False` for image-only task
                static_graph=static_graph
            )

        if config['stage'] == 'stage_3':
            return (
                model,
                model_without_ddp,
                optimizer,
                scheduler,
                scaler,
                start_epoch,
                global_step,
                visdial_step,
                avsd_step,
                nextqa_step,
                config
            )
        return (
                model,
                model_without_ddp,
                optimizer,
                scheduler,
                scaler,
                start_epoch,
                global_step,
                config
            )


def setup_data(config):
    logger.info("[INFO] Creating datasets")

    # define the processors 
    vis_processor  = Blip2ImageTrainProcessor(image_size=config.image_res)

    if config['stage'] == 'stage_1':
        text_processor = BlipCaptionProcessor(max_words=config.max_cap_len)

        if config['debugging']:
            train_datasets = load_datasets_stage_1(config, vis_processor, text_processor, 'val')
        else:
            train_datasets = load_datasets_stage_1(config, vis_processor, text_processor, 'train')

        val_datasets = load_datasets_stage_1(config, vis_processor, text_processor, 'val')

        # cc3m_dataset = ConcatDataset([train_datasets['cc3m'], val_datasets['cc3m']])

        # webvid_dataset = ConcatDataset([train_datasets['webvid'], val_datasets['webvid']])

        # train_datasets = [cc3m_dataset, webvid_dataset]
        train_datasets = list(train_datasets.values())
        val_datasets = list(val_datasets.values())

        batch_sizes = [config['batch_size_cc3m'], config['batch_size_webvid']]
        num_samples = [len(d) for d in train_datasets]
        config = set_training_steps(config, num_samples, batch_sizes)

        train_dataloaders = load_dataloaders(config, train_datasets, 'train', output_dict=True)
        val_dataloaders = load_dataloaders(config, val_datasets, 'val', output_dict=True)

        # val_datasets = load_datasets_stage_1(config, vis_processor, text_processor, 'test')

        # val_dataloader = load_dataloaders(config, val_datasets, 'test', output_dict=True)

    if config['stage'] == 'stage_2':
        text_processor = BlipDialogProcessor(max_words=config.max_text_len)  # max_words = 50
        train_datasets = [load_champagne_dataset(config, vis_processor, text_processor, 'train')]
        val_datasets = [load_champagne_dataset(config, vis_processor, text_processor, 'val')]
        batch_sizes = [config['batch_size_champagne']]
        num_samples = [len(d) for d in train_datasets]
        config = set_training_steps(config, num_samples, batch_sizes)

        train_dataloaders = load_dataloaders(config, train_datasets, 'train', output_dict=True)
        val_dataloaders = load_dataloaders(config, val_datasets, 'val', output_dict=True)


    if config['stage'] == 'stage_3':
        text_processor = BlipDialogProcessor(max_words=config.max_text_len)  # max_words = 50
        train_datasets = []
        val_datasets = []
        for medium in config['media_train']:
            if medium == 'visdial':
                load_dataset_fn = load_visdial_dataset
            elif medium == 'avsd':
                load_dataset_fn = load_avsd_dataset
            elif medium == 'nextqa':
                load_dataset_fn = load_nextqa_dataset
            # elif medium == 'champagne':
            #     load_dataset_fn = load_champagne_dataset

            train_datasets.append(load_dataset_fn(config, vis_processor, text_processor, 'train'))

        for medium in config['media_val']:
            if medium == 'visdial':
                load_dataset_fn = load_visdial_dataset
            elif medium == 'avsd':
                load_dataset_fn = load_avsd_dataset
            elif medium == 'nextqa':
                load_dataset_fn = load_nextqa_dataset
            # elif medium == 'champagne':
            #     load_dataset_fn = load_champagne_dataset

            val_datasets.append(load_dataset_fn(config, vis_processor, text_processor, 'val'))
        
        batch_sizes = [d.batch_size for d in train_datasets]
        num_samples = [len(d) for d in train_datasets]
        config = set_training_steps(config, num_samples, batch_sizes)

        train_dataloaders = load_dataloaders(config, train_datasets, 'train', output_dict=True)
        
        val_dataloaders = load_dataloaders(config, val_datasets, 'val', output_dict=True)
    
    return train_dataloaders, val_dataloaders


def setup_data_test(config):
    vis_processor  = Blip2ImageTrainProcessor(image_size=config.image_res)
    text_processor = BlipDialogProcessor(max_words=config.max_text_len)  # max_words = 50
    
    if config.media_test == 'visdial':
        load_dataset_fn = load_visdial_dataset
    elif config.media_test == 'avsd':
        load_dataset_fn = load_avsd_dataset
    elif config.media_test == 'nextqa':
        load_dataset_fn = load_nextqa_dataset
    test_dataset = load_dataset_fn(config, vis_processor, text_processor, 'test')

    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=test_dataset.batch_size)

    return test_dataloader


# def setup_data_test(config, args):
#     tokenizer_experts = BertTokenizer.from_pretrained('bert-base-uncased')
#     tokenizer_enc_dec = None
#     if config.enc_dec_family == 'flan_t5':
#         tokenizer_enc_dec = T5Tokenizer.from_pretrained(config.enc_dec_name)
#     elif config.enc_dec_family == 'bart':
#         tokenizer_enc_dec = BartTokenizer.from_pretrained(config.enc_dec_name)
#     if config['tie_embeddings']:
#         tokenizer_experts = tokenizer_enc_dec

#     if config['medium'] == 'avsd':
#         test_dataset = AVSDDataSet(config, 'avsd', tokenizer_experts, tokenizer_enc_dec, 'test')
#         test_dataloader = DataLoader(
#             test_dataset, shuffle=False, batch_size=test_dataset.batch_size, collate_fn=test_dataset.collate_fn)
#     return test_dataloader
