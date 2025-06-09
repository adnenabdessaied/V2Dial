""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
import math


# class LinearWarmupStepLRScheduler:
#     def __init__(
#         self,
#         optimizer,
#         max_epoch,
#         min_lr,
#         init_lr,
#         decay_rate=1,
#         warmup_start_lr=-1,
#         warmup_steps=0,
#         **kwargs
#     ):
#         self.optimizer = optimizer

#         self.max_epoch = max_epoch
#         self.min_lr = min_lr

#         self.decay_rate = decay_rate

#         self.init_lr = init_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

#     def step(self, cur_epoch, cur_step):
#         if cur_epoch == 0:
#             warmup_lr_schedule(
#                 step=cur_step,
#                 optimizer=self.optimizer,
#                 max_step=self.warmup_steps,
#                 init_lr=self.warmup_start_lr,
#                 max_lr=self.init_lr,
#             )
#         else:
#             step_lr_schedule(
#                 epoch=cur_epoch,
#                 optimizer=self.optimizer,
#                 init_lr=self.init_lr,
#                 min_lr=self.min_lr,
#                 decay_rate=self.decay_rate,
#             )


# class LinearWarmupCosineLRScheduler:
#     def __init__(
#         self,
#         optimizer,
#         max_epoch,
#         min_lr,
#         init_lr,
#         warmup_steps=0,
#         warmup_start_lr=-1,
#         **kwargs
#     ):
#         self.optimizer = optimizer

#         self.max_epoch = max_epoch
#         self.min_lr = min_lr

#         self.init_lr = init_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

#     def step(self, cur_epoch, cur_step):
#         # assuming the warmup iters less than one epoch
#         if cur_epoch == 0:
#             warmup_lr_schedule(
#                 step=cur_step,
#                 optimizer=self.optimizer,
#                 max_step=self.warmup_steps,
#                 init_lr=self.warmup_start_lr,
#                 max_lr=self.init_lr,
#             )
#         else:
#             cosine_lr_schedule(
#                 epoch=cur_epoch,
#                 optimizer=self.optimizer,
#                 max_epoch=self.max_epoch,
#                 init_lr=self.init_lr,
#                 min_lr=self.min_lr,
#             )


# class ConstantLRScheduler:
#     def __init__(self, optimizer, init_lr, warmup_start_lr=-1, warmup_steps=0, **kwargs):
#         self.optimizer = optimizer
#         self.lr = init_lr
#         self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
#         self.warmup_steps = warmup_steps
    
#     def step(self, cur_epoch, cur_step):
#         if cur_epoch == 0:
#             warmup_lr_schedule(
#                 step=cur_step,
#                 optimizer=self.optimizer,
#                 max_step=self.warmup_steps,
#                 init_lr=self.warmup_start_lr,
#                 max_lr=self.lr,
#             )
#         else:
#             for param_group in self.optimizer.param_groups:
#                 param_group["lr"] = self.lr


# schedulers = {
#     'constant_lr': ConstantLRScheduler,
#     'linear_warmup_cosine_lr': LinearWarmupCosineLRScheduler,
#     'linear_warmup_step_lr': LinearWarmupStepLRScheduler
# }


# def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
#     """Decay the learning rate"""
#     lr = (init_lr - min_lr) * 0.5 * (
#         1.0 + math.cos(math.pi * epoch / max_epoch)
#     ) + min_lr
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


# def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
#     """Warmup the learning rate"""
#     lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


# def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
#     """Decay the learning rate"""
#     lr = max(min_lr, init_lr * (decay_rate**epoch))
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


# def create_scheduler(config, optimizer):
#     scheduler_cls = schedulers[config.get('scheduler', 'constant_lr')]
#     max_epoch = config.epochs
#     min_lr = config.min_lr
#     init_lr = config.lr
#     warmup_start_lr = config.get('warmup_lr', -1)
#     warmup_steps = config.get('warmup_steps', 0)

#     scheduler = scheduler_cls(
#         optimizer=optimizer,
#         max_epoch=max_epoch,
#         min_lr=min_lr,
#         init_lr=init_lr,
#         decay_rate=None,
#         warmup_start_lr=warmup_start_lr,
#         warmup_steps=warmup_steps
#     )

#     return scheduler



class WarmupLinearScheduleNonZero(_LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
        Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, min_lr=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_factor = float(step) / float(max(1, self.warmup_steps))
        else:
            lr_factor = max(0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

        return [base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr for base_lr in self.base_lrs]


def create_scheduler(config, optimizer):
    lr_scheduler = None
    if config['scheduler'] == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['num_warmup_steps'],
            num_training_steps=config['num_training_steps'],
            num_cycles=0.5,
            min_lr_multi=config['min_lr_multi']
        )
    elif config['scheduler'] == 'linear':
        lr_scheduler = WarmupLinearScheduleNonZero(
            optimizer,
            config['num_warmup_steps'],
            config['num_training_steps'],
            min_lr = config['min_lr']
        )
    return lr_scheduler


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, min_lr_multi: float = 0., last_epoch: int = -1
):
    """
    Modified from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        min_lr_multi (`float`, *optional*, defaults to 0):
            The minimum learning rate multiplier. Thus the minimum learning rate is base_lr * min_lr_multi.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr_multi, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_multi, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
