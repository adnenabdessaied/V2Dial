import os
import datetime
import wandb
import torch
import pandas as pd
from time import time

import torch.distributed as dist
from torch.distributed import ReduceOp

from torch.nn.utils.clip_grad import clip_grad_value_
from utils.basic import MetricLogger, SmoothedValue, setup_seed, average_dicts
from datasets.utils import get_datasets_media
from datasets.dataloader import MetaLoader
from utils.dist import is_main_process, get_rank, get_world_size
from utils.logger import setup_wandb, log_dict_to_wandb
from .retrieval_utils import evaluation_wrapper
import glog as logger


def run_epoch(
    model,
    train_dataloaders,
    optimizer,
    epoch,
    global_step,
    webvid_step,
    cc3m_step,
    device,
    scheduler,
    scaler,
    config
):
    model.train()
    media_types = list(train_dataloaders.keys())

    log_freq = config['log_freq']
    # metric_logger = MetricLogger(delimiter=' ')
    # metric_logger.add_meter('lr', SmoothedValue(window=log_freq, fmt='{value:.6f}'))
    # metric_logger.add_meter("temperature", SmoothedValue(window=log_freq, fmt="{value:.4f}"))

    loss_names = ['loss_' + k for k in config['loss_dict'].keys()]
    # for l in loss_names:
    #     for m in media_types:
    #         metric_logger.add_meter(
    #             f'{m}/{l}', SmoothedValue(window=log_freq, fmt="{value:.4f}")
    #         )
    
    
    # header = '{} | Epoch = {}'.format(config['stage'], epoch)

    model_without_ddp = model
    if config['distributed']:
        model_without_ddp = model.module
        for k in train_dataloaders:
            train_dataloaders[k].sampler.set_epoch(epoch)

    train_dataloader = MetaLoader(name2loader=train_dataloaders)

    log_text_template = '\n' + '-' * 25 + '\n[Epoch {}/{}][Iter. {}/{}][Media-type {}]\n' 
    log_text_template += '[Losses] mlm (x{}) = {:.4f} | vcc (x{}) = {:.4f} | vcm (x{}) = {:.4f} | stc (x{}) = {:.4f} | stm (x{}) = {:.4f}\n'
    log_text_template += '[Other] lr = {:.4f} | temp = {:.4f} | eta = {}\n'

    # iterator = metric_logger.log_every(train_dataloader, log_freq, header)
    local_step = 0
    for media_type, (vis, caption, neg_vis) in train_dataloader:
        start = time()
        # loss_dict = {}
        vis = vis.to(device)
        neg_vis = neg_vis.to(device)
        # idx = idx.to(device)
       
        with torch.cuda.amp.autocast(enabled=config.fp16):
            loss_dict = model(vis, caption, neg_vis, media_type)
            # loss_dict.update(losses)
            loss = sum(loss_dict.values())
            loss_accum_grad = loss / config.accum_grad_every

        scaler.scale(loss_accum_grad).backward()

        # Perfrom gradient clipping: unscale --> clip
        if config['clip_grad_value'] > 0:
            # scaler.unscale_(optimizer)
            clip_grad_value_(model.parameters(), config.clip_grad_value)

        if local_step % config.accum_grad_every == 0:
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step(epoch, global_step)
            scheduler.step()
            optimizer.zero_grad()

        time_iter = time() - start
        eta = (len(train_dataloader) - local_step - 1) * time_iter
        eta = str(datetime.timedelta(seconds=eta))
        # log
        log_dict_webvid = {}
        log_dict_cc3m = {}
        log_dict_rest = {}
        for loss_name in loss_names:
            value = loss_dict[loss_name]
            value = value if isinstance(value, float) else value.item()
            # metric_logger.update(**{f"{media_type}/{loss_name}": value})
            if media_type == "cc3m":
                log_dict_cc3m[f"train/{media_type}/{loss_name}"] = value
            else:
                log_dict_webvid[f"train/{media_type}/{loss_name}"] = value
        
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(temperature=model_without_ddp.temp.item())
        log_dict_rest['train/other/lr'] = optimizer.param_groups[0]["lr"]
        log_dict_rest['train/other/temperature'] = model_without_ddp.temp.item()

        if is_main_process() and global_step % log_freq == 0 and local_step % config.accum_grad_every == 0:
            log_dict_rest['train/other/step'] = global_step
            if media_type == 'cc3m':
                log_dict_cc3m['train/cc3m/step'] = cc3m_step

                log_text = log_text_template.format(
                    epoch, config.epochs-1, local_step, len(train_dataloader) , media_type,
                    config.loss_dict['mlm'], log_dict_cc3m['train/cc3m/loss_mlm'],
                    config.loss_dict['vcc'], log_dict_cc3m['train/cc3m/loss_vcc'],
                    config.loss_dict['vcm'], log_dict_cc3m['train/cc3m/loss_vcm'],
                    config.loss_dict['stc'], log_dict_cc3m['train/cc3m/loss_stc'],
                    config.loss_dict['stm'], log_dict_cc3m['train/cc3m/loss_stc'],
                    log_dict_rest['train/other/lr'], log_dict_rest['train/other/temperature'], eta
                    )
                logger.info(log_text)

                if config['wandb_enabled']:
                    wandb.log(log_dict_rest)
                    wandb.log(log_dict_cc3m)
                # log_text_template = '[Epoch {}/{}][Iter. {}/{}][Media-type {}]\n' 
                # log_text_template += '[losses: mlm = {:.4f} | vcc = {:4f} | vcm = {:.4f} | stc = {:.4f} | stm = {:.4f}]\n'
                # log_text_template += '[Other: lr = {:.4f} | temp = {:4f}]\n'

            else:
                log_dict_webvid['train/webvid/step'] = webvid_step
                log_text = log_text_template.format(
                    epoch, config.epochs-1, local_step, len(train_dataloader) , media_type,
                    config.loss_dict['mlm'], log_dict_webvid['train/webvid/loss_mlm'],
                    config.loss_dict['vcc'], log_dict_webvid['train/webvid/loss_vcc'],
                    config.loss_dict['vcm'], log_dict_webvid['train/webvid/loss_vcm'],
                    config.loss_dict['stc'], log_dict_webvid['train/webvid/loss_stc'],
                    config.loss_dict['stm'], log_dict_webvid['train/webvid/loss_stm'],
                    log_dict_rest['train/other/lr'], log_dict_rest['train/other/temperature'], eta
                    )
                logger.info(log_text)

                if config['wandb_enabled']:
                    wandb.log(log_dict_rest)
                    wandb.log(log_dict_webvid)
                    

        if media_type == "cc3m":
            cc3m_step += 1
        else:
            webvid_step += 1 
        global_step += 1
        local_step += 1
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # logger.info(f"Averaged stats: {metric_logger.global_avg()}")

    return global_step, webvid_step, cc3m_step


def eval(model, val_dataloader, device, epoch, config):

    model.eval()

    log_text_template = '\n' + '-' * 25 + '\n[Val Epoch{}][Iter. {}/{}][Media-type {}]\n' 
    log_text_template += '[Losses] mlm = {:.4f} | vcc = {:.4f} | vcm = {:.4f} | stc = {:.4f} | stm = {:.4f} \n'

    # log_text_template += '[Losses] vcc = {:.4f} | vcm = {:.4f} | stc = {:.4f} | stm = {:.4f} | mlm = {:.4f} \n'
    # log_text_template += '[Losses] vhc = {:.4f} | vhm = {:.4f} | chc = {:.4f} | chm = {:.4f} | gen = {:.4f} \n'

    cum_loss_stc = 0
    cum_loss_stm = 0
    cum_loss_vcc = 0
    cum_loss_vcm = 0
    cum_loss_mlm = 0
    cum_loss_tot = 0
    val_step = 0

    # val_dataloader = MetaLoader(name2loader=val_dataloaders)
    media_type = val_dataloader.dataset.medium

    if is_main_process():
        start_time = time()

    # for vis, cap_ids, hist_ids, ques_ids, label_ids, enc_dec_input_ids, idx, _ in val_dataloader:
    for vis, caption, neg_vis in val_dataloader:
    # for vis, cap_ids, hist_ids, label_ids, enc_dec_input_ids, idx, _ in val_dataloader:
        vis = vis.to(device)
        neg_vis = neg_vis.to(device)
        # idx = idx.to(device)
        
        with torch.cuda.amp.autocast(enabled=config['fp16']):
            with torch.no_grad():
                # loss_dict, _ = model(vis, cap_ids, hist_ids, ques_ids, label_ids, enc_dec_input_ids, media_type)
                # loss_dict = model(vis, caption, neg_vis, neg_caption, media_type, file, neg_file)
                loss_dict = model(vis, caption, neg_vis, media_type)

                loss = sum(loss_dict.values())
                loss_stc = loss_dict['loss_stc']
                loss_stm = loss_dict['loss_stm']
                loss_vcc = loss_dict['loss_vcc']
                loss_vcm = loss_dict['loss_vcm']
                loss_mlm = loss_dict['loss_mlm']

                if config['distributed']:
                    dist.all_reduce(loss, op=ReduceOp.AVG)
                    if config.loss_dict['stc'] != 0:
                        dist.all_reduce(loss_stc, op=ReduceOp.AVG)
                    if config.loss_dict['stm'] != 0:
                        dist.all_reduce(loss_stm, op=ReduceOp.AVG)
                    if config.loss_dict['vcc'] != 0:
                        dist.all_reduce(loss_vcc, op=ReduceOp.AVG)
                    if config.loss_dict['vcm'] != 0:
                        dist.all_reduce(loss_vcm, op=ReduceOp.AVG)
                    if config.loss_dict['mlm'] != 0:
                        dist.all_reduce(loss_mlm, op=ReduceOp.AVG)

        if is_main_process():
            cum_loss_tot += loss.item() 
            cum_loss_stc += loss_stc.item()
            cum_loss_stm += loss_stm.item()
            cum_loss_vcc += loss_vcc.item()
            cum_loss_vcm += loss_vcm.item()
            cum_loss_mlm += loss_mlm.item()

            if val_step % config.log_freq == 0:
                log_text = log_text_template.format(
                    epoch, val_step, len(val_dataloader), media_type,
                    loss_mlm, loss_vcc, loss_vcm, loss_stc, loss_stm)
                # log_text_template = '\n' + '-' * 25 + '\n[Val Eoch{}][Iter. {}/{}][Media-type {}]\n' 
                # log_text_template += '[Losses] vcc = {:.4f} | vcm = {:.4f} | stc = {:.4f} | stm = {:.4f} | mlm = {:.4f} \n'
                # log_text_template += '[Losses] vhc = {:.4f} | vhm = {:.4f} | chc = {:.4f} | chm = {:.4f} | gen = {:.4f} \n'
                # log_text = log_text_template.format(
                #     epoch, val_step, len(val_dataloader), media_type,
                #     loss_vcc, loss_vcm, loss_stc, loss_stm, 0,
                #     loss_vhc, loss_vhm, loss_chc, loss_chm, loss_gen
                # )

                logger.info(log_text)
                # logger.info('[INFO] [Eval. Epoch {}][Iter. {}/{}][Losses] gen = {:.4f} | total = {:.4f}'.format(
                #     epoch, val_step, len(val_dataloader), gen_loss, loss
                # ))
            val_step += 1

    if config['distributed']:
        dist.barrier()

    if is_main_process():
        duration = time() - start_time

        cum_loss_tot /=  len(val_dataloader)
        cum_loss_stc /=  len(val_dataloader)
        cum_loss_stm /=  len(val_dataloader)
        cum_loss_vcc /=  len(val_dataloader)
        cum_loss_vcm /=  len(val_dataloader)
        cum_loss_mlm /=  len(val_dataloader)

        # cum_loss_vhc /=  len(val_dataloader)
        # cum_loss_vhm /=  len(val_dataloader)
        # cum_loss_chc /=  len(val_dataloader)
        # cum_loss_chm /=  len(val_dataloader)
        # cum_loss_gen /=  len(val_dataloader)
        logger.info('\n' + '-' * 25 + '\n' + 'Eval. took {}\n[Losses] cum_total = {:.4f}'.format(
           datetime.timedelta(seconds=int(duration)), cum_loss_tot
        ))

        # logger.info('\n' + '-' * 25 + '\n' + 'Eval. took {}\n[Losses] cum_gen = {:.4f} | cum_total = {:.4f}'.format(
        #    datetime.timedelta(seconds=int(duration)), cum_loss_gen, cum_loss_tot
        # ))

    loss_dict = {
        'stc': cum_loss_stc,
        'stm': cum_loss_stm,
        'vcc': cum_loss_vcc,
        'vcm': cum_loss_vcm,
        # 'vhc': cum_loss_vhc,
        # 'vhm': cum_loss_vhm,
        # 'chc': cum_loss_chc,
        # 'chm': cum_loss_chm,
        'mlm': cum_loss_mlm,
        # 'gen': cum_loss_gen,
        'tot': cum_loss_tot
    }
    return loss_dict


def pre_train(
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
):
    if is_main_process() and config['wandb_enabled']:
        run = setup_wandb(config)
    setup_seed(config['seed'] + get_rank())
    device = torch.device('cuda:{}'.format(config['gpu']))

    if is_main_process() and config['wandb_enabled']:
        wandb.watch(model)
    
    best = float('inf')
    best_epoch = 0

    logger.info('[INFO] Start training...')
    start_time_all = time()
    for epoch in range(start_epoch, config['epochs']):
        if not config['evaluate']:
            start_time_epoch = time()
            global_step, webvid_step, cc3m_step = run_epoch(
                model,
                train_dataloaders,
                optimizer,
                epoch,
                global_step,
                webvid_step,
                cc3m_step,
                device,
                scheduler,
                scaler,
                config
            )
            end_time_epoch = time()
            epoch_time = end_time_epoch - start_time_epoch
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            logger.info(f'[INFO] Epoch took {epoch_time_str}')

        if not config['debugging']:
            with torch.cuda.amp.autocast(enabled=config['fp16']):
                # # TODO
                # eval_res = {}
                # for val_name, val_loader in val_dataloaders_dict.items():
                #     res = evaluation_wrapper(
                #         model_without_ddp, val_loader, tokenizer, device, config, prefix=val_name
                #     )
                #     eval_res.update(res)
                val_res = {}

                for medium in val_dataloaders:
                    res = eval(
                        model,
                        val_dataloaders[medium],
                        device,
                        epoch,
                        config
                    )
                    val_res[medium] = res

            if is_main_process():
                # Average across all datasets
                avg_val_res = average_dicts(val_res)
                # log to wandb
                if config.wandb_enabled:
                    for medium in val_res:
                        log_dict_val = {}
                        # log_dict_val[f'val/{medium}/step'] = epoch
                        for l in val_res[medium]:
                            log_dict_val[f'val/{medium}/{l}'] = val_res[medium][l]
                        wandb.log(log_dict_val)
                    # for p, v in eval_res.items():
                    #     log_dict_to_wandb(v, step=global_step, prefix=p)
                if config.stop_key is not None and config.stop_key in avg_val_res:
                    cur_best = avg_val_res[config.stop_key]
                else:  # stop_key = None
                    cur_best = best - 1  # save the last as the best

                # Don't save vit weights as they are frozen
                state_dict = model_without_ddp.state_dict()
                state_dict = {k:v for k,v in state_dict.items() if 'visual_encoder' not in k}

                save_obj = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                torch.save(save_obj, os.path.join(config.log_dir, f"ckpt_{epoch:02d}.pth"))

                if not config.evaluate and cur_best < best:
                    torch.save(save_obj, os.path.join(config.log_dir, "ckpt_best.pth"))
                    # eval_file = "eval_res_best.json"
                    # eval_res.to_json(os.path.join(config.log_dir, eval_file))
                    best = cur_best

            if config.evaluate:
                break
        if config['distributed']:
            dist.barrier()
    
    total_time = time() - start_time_all
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'[INFO] Training took {total_time_str}')

    if is_main_process() and config['wandb_enabled']:
        run.finish()

