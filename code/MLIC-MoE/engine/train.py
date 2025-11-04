import torch
import time
import torch
import time
import os
import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import voc_mAP
from utils.util import cal_gpu
from utils.misc import concat_all_gather, MetricLogger, SmoothedValue, reduce_dict
from utils.hpc import pin_workers_iterator
from engine.validate import validate
from models.builder_network import do_forward_and_criterion

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, cfg, logger, warmup_scheduler=None):
    if cfg.TRAIN.amp:
        # scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.amp)
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.TRAIN.amp_dtype]
        scaler = torch.amp.GradScaler(enabled=cfg.TRAIN.amp)
    
    if hasattr(model, 'module'):
        model.module.switch_mode_train()
        model.module.training = True
    else:
        model.switch_mode_train()
        model.training = True

    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6e}'))
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.TRAIN.epochs)

    the_iterator = iter(train_loader)

    for it, data_full in enumerate(metric_logger.log_every(the_iterator, cfg.INPUT_OUTPUT.print_freq, header, logger=logger)):

        data_full['image'] = torch.stack([image.cuda(non_blocking=True) for image in data_full['image']], dim=0)
        data_full['target'] = torch.stack([target.cuda(non_blocking=True) for target in data_full['target']], dim=0)


        # do_forward_and_criterion(cfg, epoch, data_full, model, criterion, is_val=False)
        # compute output
        if cfg.TRAIN.amp:
            # with torch.amp.autocast(enabled=cfg.TRAIN.amp):
            with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                _, losses, loss_dict, weight_dict = do_forward_and_criterion(cfg, epoch, data_full, model, criterion, False)
        else:
            _, losses, loss_dict, weight_dict = do_forward_and_criterion(cfg, epoch, data_full, model, criterion, False)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # backward function
        if cfg.TRAIN.amp:
            # amp backward function
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if cfg.OPTIMIZER.max_clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.max_clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if cfg.OPTIMIZER.max_clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.max_clip_grad_norm)
            optimizer.step()

        # record learning_rate
        if epoch >= cfg.TRAIN.ema_epoch:
            ema_m.update(model)

        # update learning rate
        if cfg.OPTIMIZER.lr_scheduler in ('OneCycleLR', 'cosine'):
            scheduler.step()
        if warmup_scheduler != None and epoch < cfg.OPTIMIZER.warmup_epoch and cfg.OPTIMIZER.lr_scheduler != 'OneCycleLR':
            warmup_scheduler.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(all_loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return resstat