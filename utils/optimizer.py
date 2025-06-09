""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import re
import torch
from torch import optim as optim
from utils.dist import is_main_process
import glog as logger
# from transformers import create_optimizer
# from transformers import AdamW
# import math


def create_optimizer(config, model):
    lr_scale = config.get('lr_layer_decay', 1)
    weight_decay = config.get('weight_decay', 0.01)

    optim_params = model.get_optimizer_params(weight_decay, lr_scale)

    num_parameters = 0
    for p_group in optim_params:
        for p in p_group['params']:
            num_parameters += p.data.nelement()    
    logger.info('number of trainable parameters: {}'.format(num_parameters))      
    
    lr = config.get('lr', 1e-4)
    betas = config.get('opt_betas', [0.9, 0.999])

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(lr),
        betas=betas
    )    

    return optimizer
