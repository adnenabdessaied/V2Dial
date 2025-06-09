
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
# from tasks.ft_avsd import ft_avsd, generate
from tasks.stage_3 import ft_avsd, generate, generate_nextqa, generate_visdial

parser = argparse.ArgumentParser(description='Main script for v2dial')
parser.add_argument(
    '--model',
    type=str,
    default='v2dial/stage_3',
    help='model name to train or test')

parser.add_argument(
    '--mode',
    type=str,
    default='generate',
    help='train, generate or debug'
    )

parser.add_argument(
    '--eval_dir',
    type=str,
    default='/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/logs/stage_3/v2dial-google_flan-t5-large-finetune_without_stc_stm_only_visdial'
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
    default="finetuned_visdial_without_stm_stc",
    # default='V2dial-bart_large-Experts_from_scratch-gen-modalityLayers_4-without_residuals-AVSD',
    # default='Q_base_bart_base_from_modality_experts_c3m_webvid2mToVisdialToAVSD_num_hist3_with_fc_embed',
    # default='like_mst_mixer_Q_base_bart_large_from_modality_experts_c3m_webvid2mToavsd_12_frames_without_temp_fp16',
    # default='from_stage1_after_avsd_only_visdial_4_frames_10_rounds_ft',
    # default='from_scratch_visdial',
    # default='no_moes_div_st_from_scratch_only_avsd_4_frames_3_rounds_ft_fp16',
    # default='flant5_large_bert_experts_4_only_gen_AVSD_24epochs',
    help="Tag to differentiate the models"
)

# parser.add_argument(
#     '--medium',
#     type=str,
#     default='avsd',
#     help="Medium of the test dataset"
# )

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
        model, model_without_ddp, optimizer, scheduler, scaler, start_epoch, global_step, visdial_step, avsd_step, nextqa_step, config
    ) = setup_model(config)

    if config['training']:
        ft_avsd(
            model,
            model_without_ddp,
            train_dataloaders,
            val_dataloaders,
            optimizer,
            global_step,
            visdial_step,
            avsd_step,
            nextqa_step,
            scheduler,
            scaler,
            start_epoch,
            config
        )
    elif config['generating']:
        test_dataloader = setup_data_test(config)
        if config.media_test == 'avsd':
            generate(model, test_dataloader, args.tag, config, gen_subset_num=args.gen_subset_num)
        if config.media_test == 'visdial':
            generate_visdial(model, test_dataloader, args.tag, config, gen_subset_num=args.gen_subset_num)
        elif config.media_test == 'nextqa':
            generate_nextqa(model, test_dataloader, args.tag, config, gen_subset_num=args.gen_subset_num)
        
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
    # config['medium'] = args.medium
    config['start_idx_gen'] = args.start_idx_gen
    config['end_idx_gen'] = args.end_idx_gen
    config['expert_permutation'] = None
    # config['expert_permutation'] = {
    #     'spatial': 'history',
    #     'temporal': 'temporal',
    #     'caption': 'caption',
    #     'history': 'spatial'
    # }

    # config['wandb_project'] 
    # if config['accelerator'] == 'ddp':
    if config['num_gpus'] > 1:
        config['distributed'] = True
        mp.spawn(main, nprocs=config['num_gpus'], args=(config, args))
    else:
        config['distributed'] = False
        main(0, config, args)
