import os
import datetime
import wandb
import torch
import json
import numpy as np
from copy import deepcopy
from time import time
import torch.nn.functional as F

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
    # expert_tokenizer,
    # enc_dec_tokenizer,
    optimizer,
    epoch,
    global_step,
    visdial_step,
    avsd_step,
    nextqa_step,
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

    # if len(train_dataloaders) == 1:
    #     train_dataloader = list(train_dataloaders.values())[0]
    # else:
    train_dataloader = MetaLoader(name2loader=train_dataloaders)

    log_text_template = '\n' + '-' * 25 + '\n[Epoch {}/{}][Iter. {}/{}][Media-type {}]\n' 
    log_text_template += '[Loss] tot = {:.4f} | gen = {:.4f} \n'
    log_text_template += '[Other] lr = {:.6f} | iter_time = {:.2f} | eta = {}\n'

    # iterator = metric_logger.log_every(train_dataloader, log_freq, header)
    local_step = 0
    # vis, cap, hist, ques, ans, enc_dec_input, index, vid_id_list
    for media_type, batch in train_dataloader:
        vis, caption, history, answer = batch[0], batch[1], batch[2], batch[3]

        start = time()
        vis = vis.to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16):
            loss_dict = model(vis, caption, history, answer, media_type)
            loss = sum(loss_dict.values())
            loss = loss / config['accum_grad_every']
    
        scaler.scale(loss).backward()

        # Perfrom gradient clipping: unscale --> clip
        if config['clip_grad_value'] > 0:
            # scaler.unscale_(optimizer)
            clip_grad_value_(model.parameters(), config['clip_grad_value'])

        if local_step % config.accum_grad_every == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        time_iter = time() - start
        eta = (len(train_dataloader) - local_step - 1) * time_iter
        eta = str(datetime.timedelta(seconds=eta))
        # log
        log_dict_visdial = {}
        log_dict_avsd    = {}
        log_dict_nextqa  = {}

        log_dict_rest    = {}

        for loss_name in loss_names:
            value = loss_dict[loss_name]
            value = value if isinstance(value, float) else value.item()
            # metric_logger.update(**{f"{media_type}/{loss_name}": value})
            if media_type == 'visdial':
                log_dict_visdial[f"train/{media_type}/{loss_name}"] = value
            elif media_type == 'avsd':
                log_dict_avsd[f"train/{media_type}/{loss_name}"] = value
            elif media_type == 'nextqa':
                log_dict_nextqa[f"train/{media_type}/{loss_name}"] = value

        log_dict_rest['train/other/lr'] = optimizer.param_groups[0]["lr"]

        if is_main_process() and local_step % log_freq == 0 and local_step % config['accum_grad_every'] == 0:
            log_dict_rest['train/other/step'] = global_step

            if media_type == 'visdial':
                log_dict_visdial['train/visdial/step'] = visdial_step
                log_dict = log_dict_visdial

            elif media_type == 'avsd':
                log_dict_avsd['train/avsd/step'] = avsd_step
                log_dict = log_dict_avsd

            elif media_type == 'nextqa':
                log_dict_nextqa['train/nextqa/step'] = nextqa_step
                log_dict = log_dict_nextqa

            log_text = log_text_template.format(
                epoch, config.epochs-1, local_step, len(train_dataloader) , media_type, loss.item(),
                log_dict[f'train/{media_type}/loss_gen'], log_dict_rest['train/other/lr'],
                time_iter, eta
                )
            logger.info(log_text)
            
            if config['wandb_enabled']:
                wandb.log(log_dict_rest)
                wandb.log(log_dict)
        
        if media_type == 'visdial':
            visdial_step += 1
        elif media_type == 'avsd':
            avsd_step += 1
        elif media_type == 'nextqa':
            nextqa_step += 1


        local_step += 1
        global_step += 1

    return global_step, visdial_step, avsd_step, nextqa_step

            
        # if is_main_process() and local_step % config['log_model_outputs_every'] == 0 and config['log_model_outputs']:
        #     predictions = []
        #     labels = []
        #     probs = F.softmax(logits, dim=-1)
        #     preds = torch.topk(probs, 1)[1].squeeze(-1)
        #     preds = preds.tolist()
        #     lm_labels_list = label_ids['input_ids'].tolist()
        #     lm_labels_list = [[s for s in label if s != 1] for label in lm_labels_list] 
        #     # reponses = ''
        #     # labels = ''
        #     model_pred_text = ''
        #     for pred, label in zip(preds, lm_labels_list):
        #         predictions.append('\n' + 'Pred: ' +  tokenizer_enc_dec.decode(pred) + '\n')
        #         labels.append('\n' + 'GT: ' + tokenizer_enc_dec.decode(label) + '\n')
            
        #     if len(predictions) < 4:
        #         predictions = predictions[:4]
        #         labels = labels[:4]


        #     for label, pred in zip(labels, predictions):
        #         model_pred_text += label + pred
        #         model_pred_text += "---------------------"
        #     logger.info(model_pred_text)

        #         # output['reponses'] = reponses
        #         # output['gt'] = labels




def eval(model, val_dataloader, device, epoch, config):

    model.eval()

    log_text_template = '\n' + '-' * 25 + '\n[Val Epoch {}][Iter. {}/{}][Media-type {}]\n' 
    # log_text_template += '[Losses] vcc = {:.4f} | vcm = {:.4f} | stc = {:.4f} | stm = {:.4f} | mlm = {:.4f} \n'
    # log_text_template += '[Losses] vhc = {:.4f} | vhm = {:.4f} | chc = {:.4f} | chm = {:.4f} | gen = {:.4f} \n'

    log_text_template += '[Losses] gen = {:.4f} \n'

    # cum_loss_stc = 0
    # cum_loss_stm = 0
    # cum_loss_vcc = 0
    # cum_loss_vcm = 0
    # cum_loss_vhc = 0
    # cum_loss_vhm = 0
    # cum_loss_chc = 0
    # cum_loss_chm = 0
    # cum_loss_mlm = 0
    cum_loss_gen = 0
    cum_loss_tot = 0
    val_step = 0
    media_type = val_dataloader.dataset.medium
    if is_main_process():
        start_time = time()

    # for vis, cap_ids, hist_ids, ques_ids, label_ids, enc_dec_input_ids, idx, _ in val_dataloader:
    for batch in val_dataloader:

        vis, caption, history, answer = batch[0], batch[1], batch[2], batch[3]

        vis = vis.to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16):
            with torch.no_grad():
                # loss_dict, _ = model(vis, cap_ids, hist_ids, ques_ids, label_ids, enc_dec_input_ids, media_type)
                # loss_dict, _ = model(vis, cap_ids, hist_ids, label_ids, enc_dec_input_ids, media_type)
                loss_dict = model(vis, caption, history, answer, media_type)

                # loss_dict = model(vis, cap_ids, hist_ids, ques_ids, label_ids, media_type)
                loss = sum(loss_dict.values())
                # loss_stc = loss_dict['loss_stc']
                # loss_stm = loss_dict['loss_stm']
                # loss_vcc = loss_dict['loss_vcc']
                # loss_vcm = loss_dict['loss_vcm']
                # loss_vhc = loss_dict['loss_vhc']
                # loss_vhm = loss_dict['loss_vhm']
                # loss_chc = loss_dict['loss_chc']
                # loss_chm = loss_dict['loss_chm']
                # loss_mlm = loss_dict['loss_mlm']
                loss_gen = loss_dict['loss_gen']

                if config['distributed']:
                    dist.all_reduce(loss, op=ReduceOp.AVG)
                    # if config.loss_dict['stc'] != 0:
                    #     dist.all_reduce(loss_stc, op=ReduceOp.AVG)
                    # if config.loss_dict['stm'] != 0:
                    #     dist.all_reduce(loss_stm, op=ReduceOp.AVG)
                    # if config.loss_dict['vcc'] != 0:
                    #     dist.all_reduce(loss_vcc, op=ReduceOp.AVG)
                    # if config.loss_dict['vcm'] != 0:
                    #     dist.all_reduce(loss_vcm, op=ReduceOp.AVG)
                    # if config.loss_dict['vhc'] != 0:
                    #     dist.all_reduce(loss_vhc, op=ReduceOp.AVG)
                    # if config.loss_dict['vhm'] != 0:
                    #     dist.all_reduce(loss_vhm, op=ReduceOp.AVG)
                    # if config.loss_dict['chc'] != 0:
                    #     dist.all_reduce(loss_chc, op=ReduceOp.AVG)
                    # if config.loss_dict['chm'] != 0:
                    #     dist.all_reduce(loss_chm, op=ReduceOp.AVG)
                    # if config.loss_dict['mlm'] != 0:
                    #     dist.all_reduce(loss_mlm, op=ReduceOp.AVG)
                    if config.loss_dict['gen'] != 0:
                        dist.all_reduce(loss_gen, op=ReduceOp.AVG)

        if is_main_process():
            cum_loss_tot += loss.item() 
            # cum_loss_stc += loss_stc.item()
            # cum_loss_stm += loss_stm.item()
            # cum_loss_vcc += loss_vcc.item()
            # cum_loss_vcm += loss_vcm.item()
            # cum_loss_vhc += loss_vhc.item()
            # cum_loss_vhm += loss_vhm.item()
            # cum_loss_chc += loss_chc.item()
            # cum_loss_chm += loss_chm.item()
            # cum_loss_mlm += loss_mlm.item()
            cum_loss_gen += loss_gen.item()

            if val_step % config.log_freq == 0:
                # log_text_template = '\n' + '-' * 25 + '\n[Val Eoch{}][Iter. {}/{}][Media-type {}]\n' 
                # log_text_template += '[Losses] vcc = {:.4f} | vcm = {:.4f} | stc = {:.4f} | stm = {:.4f} | mlm = {:.4f} \n'
                # log_text_template += '[Losses] vhc = {:.4f} | vhm = {:.4f} | chc = {:.4f} | chm = {:.4f} | gen = {:.4f} \n'
                log_text = log_text_template.format(
                    epoch, val_step, len(val_dataloader), media_type,
                    # loss_vcc, loss_vcm, loss_stc, loss_stm, 0,
                    # loss_vhc, loss_vhm, loss_chc, loss_chm, 
                    loss_gen
                )

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
        # cum_loss_stc /=  len(val_dataloader)
        # cum_loss_stm /=  len(val_dataloader)
        # cum_loss_vcc /=  len(val_dataloader)
        # cum_loss_vcm /=  len(val_dataloader)
        # cum_loss_vhc /=  len(val_dataloader)
        # cum_loss_vhm /=  len(val_dataloader)
        # cum_loss_chc /=  len(val_dataloader)
        # cum_loss_chm /=  len(val_dataloader)
        # cum_loss_mlm /=  len(val_dataloader)
        cum_loss_gen /=  len(val_dataloader)

        logger.info('\n' + '-' * 25 + '\n' + 'Eval. took {}\n[Losses] cum_gen = {:.4f} | cum_total = {:.4f}'.format(
           datetime.timedelta(seconds=int(duration)), cum_loss_gen, cum_loss_tot
        ))
    loss_dict = {
        # 'stc': cum_loss_stc,
        # 'stm': cum_loss_stm,
        # 'vcc': cum_loss_vcc,
        # 'vcm': cum_loss_vcm,
        # 'vhc': cum_loss_vhc,
        # 'vhm': cum_loss_vhm,
        # 'chc': cum_loss_chc,
        # 'chm': cum_loss_chm,
        # 'mlm': cum_loss_mlm,
        'gen': cum_loss_gen,
        'tot': cum_loss_tot
    }
    return loss_dict


def ft_avsd(
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
):
    if is_main_process() and config['wandb_enabled']:
        run = setup_wandb(config)
    setup_seed(config['seed'] + get_rank())
    # device = torch.device('cuda:{}'.format(config['gpu']))
    device = config.device
    # expert_tokenizer = model_without_ddp.expert_tokenizer
    # enc_dec_tokenizer = model_without_ddp.enc_dec_tokenizer

    if is_main_process() and config['wandb_enabled']:
        wandb.watch(model)
    
    best = float('inf')

    logger.info('[INFO] Start training...')
    start_time_all = time()
    for epoch in range(start_epoch, config['epochs']):
        if not config['evaluate']:
            if is_main_process():
                start_time_epoch = time()

            global_step, visdial_step, avsd_step, nextqa_step = run_epoch(
                model,
                train_dataloaders,
                # expert_tokenizer,
                # enc_dec_tokenizer,
                optimizer,
                epoch,
                global_step,
                visdial_step,
                avsd_step,
                nextqa_step,
                device,
                scheduler,
                scaler,
                config
            )
            if is_main_process():
                end_time_epoch = time()
                epoch_time = end_time_epoch - start_time_epoch
                epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
                logger.info(f'[INFO] Epoch took {epoch_time_str}')

        if not config['debugging']:
            with torch.cuda.amp.autocast(enabled=config['fp16']):
                val_res = {}

                for medium in val_dataloaders:
                    res = eval(
                        model,
                        val_dataloaders[medium],
                        # expert_tokenizer,
                        # enc_dec_tokenizer,
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

                # Don't save vit and llm weights as they are frozen
                state_dict = model_without_ddp.state_dict()
                if config.freeze_vit:
                    state_dict = {k:v for k,v in state_dict.items() if 'visual_encoder' not in k}

                if config.freeze_llm:
                    state_dict = {k:v for k,v in state_dict.items() if 'llm' not in k}

                save_obj = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "visdial_step": visdial_step,
                    "avsd_step": avsd_step,
                    "nextqa_step": nextqa_step
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


def generate(model, dataloader, tag, config, gen_subset_num=None):

    model.eval()
    responses = {}
    # tokenizer_enc_dec = dataloader.dataset.tokenizer_enc_dec
    device = next(model.parameters()).device  # Assumes all model parameters are on the same device
    # Generate the repsonse for each round
    logger.info('[INFO] Generating responses for {} samples'.format(len(dataloader)))
    with torch.no_grad():
        # for counter, (vis, cap_ids, hist_ids, ques_ids, _, enc_dec_input_ids, _, vid_id) in enumerate(dataloader):
        for counter, (vis, cap, hist, ans, vis_ids) in enumerate(dataloader):

            start_time = time()
            vis        = vis.to(device, non_blocking=True)
            is_vid     = config.media_test in ['webvid', 'champagne', 'avsd', 'nextqa']

            # First get the visual features depending on the media type
            with torch.cuda.amp.autocast(enabled=config.fp16):
                cap_ids, cap_mask = model.tokenize_text(cap, device, max_len=None)
                hist_ids, hist_mask = model.tokenize_text(hist, device, max_len=None)
                
                if config.use_moes:
                    if config.use_sep_spatial_temp_experts:
                        vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = model.encode_vis(vis, device, is_vid=is_vid)
                    else:
                        vis_embed, vis_mask = model.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)

                    # construct the global input tensor --> use place holder for vis features
                    
                    if config.use_sep_spatial_temp_experts:
                        moe_outputs = model.moe_forward(
                            vis_embed_spatial, vis_spatial_mask,
                            vis_embed_temporal, vis_temporal_mask,
                            cap_ids, cap_mask,
                            hist_ids, hist_mask,
                            is_vid, device
                        )
                        spatial_embeds  = model.moe_to_llm(moe_outputs['spatial_embeds'])
                        temporal_embeds = model.moe_to_llm(moe_outputs['temporal_embeds']) if is_vid else None
                    
                    else:
                        moe_outputs = model.moe_forward_no_sep_spatial_temporal(
                            vis_embed, vis_mask,
                            cap_ids, cap_mask,
                            hist_ids, hist_mask,
                            is_vid, device
                        )
                        vis_embeds  = model.moe_to_llm(moe_outputs['vis_embeds'])
                    
                    cap_embeds      = model.moe_to_llm(moe_outputs['cap_embeds'])
                    hist_embeds     = model.moe_to_llm(moe_outputs['hist_embeds'])
                else:
                    cap_embeds  = model.llm_to_moe(model.text_embedding(cap_ids))
                    hist_embeds = model.llm_to_moe(model.text_embedding(hist_ids))
                    vis_embeds, vis_mask = model.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)


                if config.llm_family in ['llama', 'mistral']:
                    bos = torch.ones_like(cap_ids[:, :1]) * model.tokenizer.bos_token_id
                    bos_embeds = model.text_embedding(bos)
                    bos_mask = cap_mask[:, :1]

                    inputs_embeds, attention_mask =  model.pad_to_right_dec_only_gen_mode(cap_embeds, cap_mask, hist_embeds, hist_mask, device)
                    if is_vid:
                        inputs_embeds = torch.cat([bos_embeds, spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([bos_mask, vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([bos_embeds, spatial_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([bos_mask, vis_spatial_mask, attention_mask], dim=1)
                
                else:
                    inputs_embeds, attention_mask = model.pad_to_right_enc_dec(cap_embeds, cap_mask, hist_embeds, hist_mask, device)
                    if config.use_moes:
                        if not config.drop_vis_features:
                            if config.use_sep_spatial_temp_experts:
                                if is_vid:
                                    inputs_embeds = torch.cat([spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                                    attention_mask = torch.cat([vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                                else:
                                    inputs_embeds = torch.cat([spatial_embeds, inputs_embeds], dim=1)
                                    attention_mask = torch.cat([vis_spatial_mask, attention_mask], dim=1)
                            else:
                                inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
                                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
                
                decoded_ids = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=False,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    num_beams=config.beam_depth,
                    length_penalty=config.length_penalty,
                    max_length=config.max_generation_length,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    # use_cache=True
                    )
                
                response_batch = [model.tokenizer.decode(decoded_id, skip_special_tokens=True) for decoded_id in decoded_ids]
            
            for vis_id, response in zip(vis_ids, response_batch):
                responses[vis_id] = response

            time_elapsed = int(time() - start_time)
            print('Generating resonse {} / {} -- took {}s'.format(counter + 1, len(dataloader), time_elapsed))
            
    # Create a file with all responses
    with open(config['anno_avsd_test_dstc_{}'.format(config['dstc'])], 'r') as f:
        test_data = json.load(f)
    test_dialogs = deepcopy(test_data['dialogs'])
    # Filter the predicted dialogs
    test_dialogs = list(filter(lambda diag: diag['image_id'] in responses, test_dialogs))

    for i, dialog in enumerate(test_dialogs):
        vid_id = dialog['image_id']
        gen_response = responses[vid_id]
        round_num_to_answer = len(dialog['dialog'])-1
        assert dialog['dialog'][round_num_to_answer]['answer'] == '__UNDISCLOSED__'
        dialog['dialog'][round_num_to_answer]['answer'] = gen_response
        test_dialogs[i] = dialog

    # Log the file
    file_name = '{}_results_dstc{}_beam_depth_{}_lenPen_{}'.format(config['llm_name'].replace('/', '-'), config['dstc'], config['beam_depth'], config['length_penalty'])
    if gen_subset_num is not None:
        file_name += f'-part_{gen_subset_num}'
    file_name = f'{tag}_' + file_name
    output_path = os.path.join(config['output_dir_avsd_{}'.format(config['dstc'])], file_name + '.json')
    with open(output_path, 'w') as f:
        json.dump({'dialogs': test_dialogs}, f, indent=4)
    logger.info('Results logged to {}'.format(output_path))    
    # Switch back to training mode
    model.train()


def generate_visdial(model, dataloader, tag, config, gen_subset_num=None):

    model.eval()
    responses = {}
    # tokenizer_enc_dec = dataloader.dataset.tokenizer_enc_dec
    device = next(model.parameters()).device  # Assumes all model parameters are on the same device
    # Generate the repsonse for each round
    logger.info('[INFO] Generating responses for {} samples'.format(len(dataloader)))
    with torch.no_grad():
        # for counter, (vis, cap_ids, hist_ids, ques_ids, _, enc_dec_input_ids, _, vid_id) in enumerate(dataloader):
        for counter, (vis, cap, hist, ans, vis_ids, d_rounds) in enumerate(dataloader):

            start_time = time()
            vis        = vis.to(device, non_blocking=True)
            is_vid     = config.media_test in ['webvid', 'champagne', 'avsd', 'nextqa']

            # First get the visual features depending on the media type
            with torch.cuda.amp.autocast(enabled=config.fp16):
                # construct the global input tensor --> use place holder for vis features
                cap_ids, cap_mask = model.tokenize_text(cap, device, max_len=None)
                hist_ids, hist_mask = model.tokenize_text(hist, device, max_len=None)

                if config.use_moes:
                    if config.use_sep_spatial_temp_experts:
                        vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = model.encode_vis(vis, device, is_vid=is_vid)
                    else:
                        vis_embed, vis_mask = model.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)
                    
                
                    if config.use_sep_spatial_temp_experts:
                        moe_outputs = model.moe_forward(
                            vis_embed_spatial, vis_spatial_mask,
                            vis_embed_temporal, vis_temporal_mask,
                            cap_ids, cap_mask,
                            hist_ids, hist_mask,
                            is_vid, device
                        )
                        spatial_embeds  = model.moe_to_llm(moe_outputs['spatial_embeds'])
                        temporal_embeds = model.moe_to_llm(moe_outputs['temporal_embeds']) if is_vid else None
                    else:
                        moe_outputs = model.moe_forward_no_sep_spatial_temporal(
                            vis_embed, vis_mask,
                            cap_ids, cap_mask,
                            hist_ids, hist_mask,
                            is_vid, device
                        )
                        vis_embeds  = model.moe_to_llm(moe_outputs['vis_embeds'])

                    cap_embeds      = model.moe_to_llm(moe_outputs['cap_embeds'])
                    hist_embeds     = model.moe_to_llm(moe_outputs['hist_embeds'])
                else:
                    cap_embeds  = model.llm_to_moe(model.text_embedding(cap_ids))
                    hist_embeds = model.llm_to_moe(model.text_embedding(hist_ids))
                    vis_embeds, vis_mask = model.encode_vis_with_seq_spa_temp_att(vis, device, is_vid=is_vid)

                if config.llm_family in ['llama', 'mistral']:
                    bos = torch.ones_like(cap_ids[:, :1]) * model.tokenizer.bos_token_id
                    bos_embeds = model.text_embedding(bos)
                    bos_mask = cap_mask[:, :1]

                    inputs_embeds, attention_mask =  model.pad_to_right_dec_only_gen_mode(cap_embeds, cap_mask, hist_embeds, hist_mask, device)
                    if is_vid:
                        inputs_embeds = torch.cat([bos_embeds, spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([bos_mask, vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([bos_embeds, spatial_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([bos_mask, vis_spatial_mask, attention_mask], dim=1)
                
                else:
                    inputs_embeds, attention_mask = model.pad_to_right_enc_dec(cap_embeds, cap_mask, hist_embeds, hist_mask, device)
                    if config.use_moes:
                        if not config.drop_vis_features:
                            if config.use_sep_spatial_temp_experts:
                                if is_vid:
                                    inputs_embeds = torch.cat([spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                                    attention_mask = torch.cat([vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                                else:
                                    inputs_embeds = torch.cat([spatial_embeds, inputs_embeds], dim=1)
                                    attention_mask = torch.cat([vis_spatial_mask, attention_mask], dim=1)
                            else:
                                inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
                                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
                    
                decoded_ids = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=False,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    num_beams=config.beam_depth,
                    length_penalty=config.length_penalty,
                    max_length=config.max_generation_length,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    # use_cache=True
                    )
                
                response_batch = [model.tokenizer.decode(decoded_id, skip_special_tokens=True) for decoded_id in decoded_ids]
            
            for vis_id, d_round, response in zip(vis_ids.tolist(), d_rounds.tolist(), response_batch):
                responses[str(vis_id) + '_' + str(d_round)] = response

            time_elapsed = time() - start_time
            print('Generating resonse {} / {} -- eta = {} '.format(counter + 1, len(dataloader), str(datetime.timedelta(seconds=time_elapsed * (len(dataloader)-counter)))
            ))
            
    # # Create a file with all responses
    # with open(config['anno_avsd_test_dstc_{}'.format(config['dstc'])], 'r') as f:
    #     test_data = json.load(f)
    # test_dialogs = deepcopy(test_data['dialogs'])
    # # Filter the predicted dialogs
    # test_dialogs = list(filter(lambda diag: diag['image_id'] in responses, test_dialogs))

    # for i, dialog in enumerate(test_dialogs):
    #     vid_id = dialog['image_id']
    #     gen_response = responses[vid_id]
    #     round_num_to_answer = len(dialog['dialog'])-1
    #     assert dialog['dialog'][round_num_to_answer]['answer'] == '__UNDISCLOSED__'
    #     dialog['dialog'][round_num_to_answer]['answer'] = gen_response
    #     test_dialogs[i] = dialog

    # Log the file
    file_name = '{}_results_dstc{}_beam_depth_{}_lenPen_{}'.format(config['llm_name'].replace('/', '-'), config['dstc'], config['beam_depth'], config['length_penalty'])
    if gen_subset_num is not None:
        file_name += f'-part_{gen_subset_num}'
    file_name = f'{tag}_' + file_name
    output_path = os.path.join(config['output_dir_visdial'], file_name + '.json')
    with open(output_path, 'w') as f:
        json.dump(responses, f, indent=4)
    logger.info('Results logged to {}'.format(output_path))    
    # Switch back to training mode
    model.train()

def generate_nextqa(model, dataloader, tag, config, gen_subset_num=None):

    model.eval()
    responses = {}
    # tokenizer_enc_dec = dataloader.dataset.tokenizer_enc_dec
    device = next(model.parameters()).device  # Assumes all model parameters are on the same device
    # Generate the repsonse for each round
    logger.info('[INFO] Generating responses for {} samples'.format(len(dataloader)))
    with torch.no_grad():
        # for counter, (vis, cap_ids, hist_ids, ques_ids, _, enc_dec_input_ids, _, vid_id) in enumerate(dataloader):
        for counter, (vis, cap, hist, _, vid_ids, qid) in enumerate(dataloader):

            start_time = time()
            vis        = vis.to(device, non_blocking=True)
            is_vid     = config.media_test in ['webvid', 'champagne', 'avsd', 'nextqa']

            vid_id = vid_ids[0]
            qid = qid[0]
            if vid_id not in responses:
                responses[vid_id] = {}

            # First get the visual features depending on the media type
            with torch.cuda.amp.autocast(enabled=config.fp16):
                vis_embed_spatial, vis_spatial_mask, vis_embed_temporal, vis_temporal_mask = model.encode_vis(vis, device, is_vid=is_vid)

                # construct the global input tensor --> use place holder for vis features
                cap_ids, cap_mask = model.tokenize_text(cap, device, max_len=None)
                hist_ids, hist_mask = model.tokenize_text(hist, device, max_len=None)

                moe_outputs = model.moe_forward(
                    vis_embed_spatial, vis_spatial_mask,
                    vis_embed_temporal, vis_temporal_mask,
                    cap_ids, cap_mask,
                    hist_ids, hist_mask,
                    is_vid, device
                )
                spatial_embeds  = model.moe_to_llm(moe_outputs['spatial_embeds'])
                temporal_embeds = model.moe_to_llm(moe_outputs['temporal_embeds']) if is_vid else None
                cap_embeds      = model.moe_to_llm(moe_outputs['cap_embeds'])
                hist_embeds     = model.moe_to_llm(moe_outputs['hist_embeds'])

                if config.llm_family in ['llama', 'mistral']:
                    bos = torch.ones_like(cap_ids[:, :1]) * model.tokenizer.bos_token_id
                    bos_embeds = model.text_embedding(bos)
                    bos_mask = cap_mask[:, :1]

                    inputs_embeds, attention_mask =  model.pad_to_right_dec_only_gen_mode(cap_embeds, cap_mask, hist_embeds, hist_mask, device)
                    if is_vid:
                        inputs_embeds = torch.cat([bos_embeds, spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([bos_mask, vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([bos_embeds, spatial_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([bos_mask, vis_spatial_mask, attention_mask], dim=1)
                
                else:
                    inputs_embeds, attention_mask = model.pad_to_right_enc_dec(cap_embeds, cap_mask, hist_embeds, hist_mask, device)

                    if is_vid:
                        inputs_embeds = torch.cat([spatial_embeds, temporal_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([vis_spatial_mask, vis_temporal_mask, attention_mask], dim=1)
                    else:
                        inputs_embeds = torch.cat([spatial_embeds, inputs_embeds], dim=1)
                        attention_mask = torch.cat([vis_spatial_mask, attention_mask], dim=1)
                
                decoded_ids = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=False,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    num_beams=config.beam_depth,
                    length_penalty=config.length_penalty,
                    max_length=config.max_generation_length,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    # use_cache=True
                    )
                
                response = model.tokenizer.decode(decoded_ids[0], skip_special_tokens=True)
                responses[vid_id][qid] = response

            # for vis_id, response in zip(vis_ids, response_batch):
            #     responses[vis_id] = response

            time_elapsed = int(time() - start_time)
            print('Generating resonse {} / {} -- took {}s'.format(counter + 1, len(dataloader), time_elapsed))
            
    # Create a file with all responses
    file_name = 'results_nextqa_beam_depth_{}'.format(config['beam_depth'])
    if gen_subset_num is not None:
        file_name += f'-part_{gen_subset_num}'
    file_name = f'{tag}_' + file_name
    output_path = os.path.join(config['output_dir_nextqa'], file_name + '.json')
    with open(output_path, 'w') as f:
        json.dump(responses, f, indent=4)
    print('Results logged to {}'.format(output_path))
    print(os.getcwd())
    # Switch back to training mode
    model.train()


def generate_enc_dec(model, dataloader, tag, config, gen_subset_num=None):

    model.eval()
    responses = {}
    tokenizer_enc_dec = dataloader.dataset.tokenizer_enc_dec
    device = next(model.parameters()).device  # Assumes all model parameters are on the same device
    # Generate the repsonse for each round
    logger.info('[INFO] Generating responses for {} samples'.format(len(dataloader)))
    with torch.no_grad():
        # for counter, (vis, cap_ids, hist_ids, ques_ids, _, enc_dec_input_ids, _, vid_id) in enumerate(dataloader):
        for counter, (vis, cap_ids, hist_ids, _, enc_dec_input_ids, _, vid_id) in enumerate(dataloader):

            start_time = time()
            vis = vis.to(device, non_blocking=True)

            for k in cap_ids:
                if isinstance(cap_ids[k], torch.Tensor): 
                    cap_ids[k] = cap_ids[k].to(device)

            for k in hist_ids:
                if isinstance(hist_ids[k], torch.Tensor): 
                    hist_ids[k] = hist_ids[k].to(device)

            # for k in ques_ids:
            #     if isinstance(ques_ids[k], torch.Tensor): 
            #         ques_ids[k] = ques_ids[k].to(device)
            
            for k in enc_dec_input_ids:
                if isinstance(enc_dec_input_ids[k], torch.Tensor): 
                    enc_dec_input_ids[k] = enc_dec_input_ids[k].to(device)

            # response = beam_search_generation(
            #     model, vis, cap_ids, hist_ids, ques_ids, enc_dec_input_ids, tokenizer_enc_dec, config
            # )

            response = beam_search_generation(
                model, vis, cap_ids, hist_ids, enc_dec_input_ids, tokenizer_enc_dec, config
            )

            # Decode the response
            response = tokenizer_enc_dec.decode(response)
            responses[vid_id[0]] = response
            # all_graphs[vid] = graphs
            time_elapsed = int(time() - start_time)
            print('Generating resonse {} / {} -- took {}s'.format(counter + 1, len(dataloader), time_elapsed))
            
    # Create a file with all responses
    with open(config['anno_avsd_test_{}'.format(config['dstc'])], 'r') as f:
        test_data = json.load(f)
    test_dialogs = deepcopy(test_data['dialogs'])
    # Filter the predicted dialogs
    test_dialogs = list(filter(lambda diag: diag['image_id'] in responses, test_dialogs))

    for i, dialog in enumerate(test_dialogs):
        vid_id = dialog['image_id']
        gen_response = responses[vid_id]
        round_num_to_answer = len(dialog['dialog'])-1
        assert dialog['dialog'][round_num_to_answer]['answer'] == '__UNDISCLOSED__'
        dialog['dialog'][round_num_to_answer]['answer'] = gen_response
        test_dialogs[i] = dialog

    # Log the file
    file_name = 'results_dstc{}_beam_depth_{}'.format(config['dstc'], config['beam_depth'])
    if gen_subset_num is not None:
        file_name += f'-part_{gen_subset_num}'
    file_name = f'{tag}_' + file_name
    output_path = os.path.join(config['output_dir_avsd_{}'.format(config['dstc'])], file_name + '.json')
    with open(output_path, 'w') as f:
        json.dump({'dialogs': test_dialogs}, f, indent=4)
    logger.info('Results logged to {}'.format(output_path))    
    # Switch back to training mode
    model.train()


def beam_search_generation_decoder_only(model, vis, caption, history, enc_dec_input, tokenizer_enc_dec, config):
    
    # gen_ans = [bos_token]
    hyplist = [([], 0.0, [])]
    best_state = None
    comp_hyplist = []

    # drop_caption = self.config['dstc'] == 10
    # instance = build_input_from_segments(caption, history, gen_ans, tokenizer, drop_caption=drop_caption)

    encoder_outputs = None

    for i in range(config['max_generation_length']):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            decoder_input_ids = torch.tensor(st).long().cuda().unsqueeze(0)

            # output = model.generate(vis, caption, history, ques, decoder_input_ids, enc_dec_input, encoder_outputs, 'avsd')
            output = model.generate(vis, caption, history, decoder_input_ids, enc_dec_input, encoder_outputs, 'avsd')

            if encoder_outputs is None:
                encoder_outputs = output.encoder_outputs

            logits = output['logits'][:,-1,:].squeeze()  # get the logits of the last token
            logp = F.log_softmax(logits, dim=0)
            lp_vec = logp.cpu().data.numpy() + lp
            if i >= config['min_generation_length']:
                new_lp = lp_vec[eos_token] + config['length_penalty'] * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:  # reverse the order
                if o in [eos_token, unk_token]:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == config['beam_depth']:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == config['beam_depth']:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist
    
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        res = maxhyps[0][0] 
        if res[0] == bos_token:
            res = res[1:]
        if res[-1] == eos_token:
            res = res[:-1]
        return res
    else:
        return []


# def beam_search_generation(model, vis, caption, history, ques, enc_dec_input, tokenizer_enc_dec, config):
def beam_search_generation(model, vis, caption, history, enc_dec_input, tokenizer_enc_dec, config):

    if config['enc_dec_family'] == 'flan_t5':
        bos_token = tokenizer_enc_dec.pad_token_id
        eos_token = tokenizer_enc_dec.eos_token_id
    else:
        bos_token = tokenizer_enc_dec.bos_token_id
        eos_token = tokenizer_enc_dec.eos_token_id

    unk_token = tokenizer_enc_dec.unk_token_id

    # gen_ans = [bos_token]
    hyplist = [([], 0.0, [bos_token])]
    best_state = None
    comp_hyplist = []

    # drop_caption = self.config['dstc'] == 10
    # instance = build_input_from_segments(caption, history, gen_ans, tokenizer, drop_caption=drop_caption)

    encoder_outputs = None

    for i in range(config['max_generation_length']):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            decoder_input_ids = torch.tensor(st).long().cuda().unsqueeze(0)

            # output = model.generate(vis, caption, history, ques, decoder_input_ids, enc_dec_input, encoder_outputs, 'avsd')
            output = model.generate(vis, caption, history, decoder_input_ids, enc_dec_input, encoder_outputs, 'avsd')

            if encoder_outputs is None:
                encoder_outputs = output.encoder_outputs

            logits = output['logits'][:,-1,:].squeeze()  # get the logits of the last token
            logp = F.log_softmax(logits, dim=0)
            lp_vec = logp.cpu().data.numpy() + lp
            if i >= config['min_generation_length']:
                new_lp = lp_vec[eos_token] + config['length_penalty'] * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:  # reverse the order
                if o in [eos_token, unk_token]:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == config['beam_depth']:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == config['beam_depth']:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist
    
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        res = maxhyps[0][0] 
        if res[0] == bos_token:
            res = res[1:]
        if res[-1] == eos_token:
            res = res[:-1]
        return res
    else:
        return []