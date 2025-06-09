
import argparse


import torch

import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
# from transformers import BartTokenizer
from torch.utils.data import ConcatDataset

from utils.init import initialize_from_env
# from datasets.pretraining import load_datasets, VideoTextRetDataset
# from datasets.utils import get_datasets_media
from models.setup import setup_model, setup_data, setup_data_test
from tasks.pre_train import pre_train
# from tasks.ft_avsd import ft_avsd, generate
# from tasks.stage_2_3 import pretrain
# from tasks.stage_2 import train as train_stage_2

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Main script for v2dial')
parser.add_argument(
    '--model',
    type=str,
    default='v2dial/stage_1',
    help='model name to train or test')

parser.add_argument(
    '--mode',
    type=str,
    default='train',
    help='train, generate or debug'
    )

parser.add_argument(
    '--eval_dir',
    type=str,
    default='/scratch/abdessaied/projects/V2Dial_TU/logs/stage_4/v2dial-flant5_large_bert_experts_4_only_gen_AVSD'
)

parser.add_argument(
    '--wandb_mode',
    type=str,
    default='online',
    choices=['online', 'offline', 'disabled', 'run', 'dryrun']
)

parser.add_argument(
    '--wandb_project',
    type=str,
    default='V2Dial'
)

parser.add_argument(
    '--tag',
    type=str,
    # default='V2dial-bart_large-Experts_from_scratch-gen-modalityLayers_4-without_residuals-AVSD',
    # default='Q_base_bart_base_from_modality_experts_c3m_webvid2mToVisdialToAVSD_num_hist3_with_fc_embed',
    # default='like_mst_mixer_Q_base_bart_large_from_modality_experts_c3m_webvid2mToavsd_12_frames_without_temp_fp16',
    default='without_sep_spatial_temporal_experts',
    # default='flant5_large_bert_experts_4_only_gen_AVSD_24epochs',
    help="Tag to differentiate the models"
)

parser.add_argument(
    '--medium',
    type=str,
    default='avsd',
    help="Medium of the test dataset"
)

parser.add_argument(
    '--start_idx_gen',
    type=int,
    default=0,
    help="The start index for generation"
)

parser.add_argument(
    '--end_idx_gen',
    type=int,
    default=10,
    help="The end index for generation"
)

parser.add_argument(
    '--gen_subset_num',
    type=int,
    default=1,
    help="The index of the test split for generation"
)

parser.add_argument('--ssh', action='store_true',
                    help='whether or not we are executing command via ssh. '
                         'If set to True, we will not log.info anything to screen and only redirect them to log file')


def main(gpu, config, args):
    
    config['gpu'] = gpu
    if config['distributed']:
        dist.init_process_group(
            backend='nccl',
            world_size=config['num_gpus'],
            rank=gpu
        )
        torch.cuda.set_device(gpu)
    
    device = torch.device(f'cuda:{gpu}')
    if config.use_cpu:
        device = torch.device('cpu')
    config['device'] = device
    # model = V2Dial(config)
        
    # config['num_training_steps'] = num_step_per_epoch *  config['epochs']
    # config['num_warmup_steps'] = num_step_per_epoch * config['warmup_epochs']
    if config['training']:
        train_dataloaders, val_dataloaders = setup_data(config)

    (
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
    ) = setup_model(config, pretrain=True)

    pre_train(
        model,
        model_without_ddp,
        train_dataloaders,
        val_dataloaders,
        optimizer,
        global_step,
        webvid_step,
        cc3m_step,
        scheduler,
        scaler,
        start_epoch,
        config
        )

    if config['distributed']:
        dist.destroy_process_group()

if __name__ == '__main__':
    args = parser.parse_args()
    
    # initialization
    model, stage = args.model.split('/')
    config = initialize_from_env(model, args.mode, stage, args.eval_dir, tag=args.tag)
    config['wandb_enabled'] = args.wandb_mode == 'online'
    config['training'] = args.mode == 'train'
    config['generating'] = args.mode == 'generate'
    config['debugging'] = args.mode == 'debug'

    config['wandb_mode'] = args.wandb_mode
    config['medium'] = args.medium
    config['start_idx_gen'] = args.start_idx_gen
    config['end_idx_gen'] = args.end_idx_gen

    # config['wandb_project'] 
    # if config['accelerator'] == 'ddp':
    if config['num_gpus'] > 1:
        config['distributed'] = True
        mp.spawn(main, nprocs=config['num_gpus'], args=(config, args))
    else:
        config['distributed'] = False
        main(0, config, args)
