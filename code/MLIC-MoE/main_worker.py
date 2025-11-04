
import math
import os, sys
import random
import time
import json
import numpy as np
import os.path as osp
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dataset.get_data import get_datasets, distributedsampler
from models.builder_network import build_network
from losses.builder_criterion import build_criterion
from optim.builder_optim import build_optim
from engine import train, validate, validate_metric

from utils.util import ModelEma, save_checkpoint, kill_process, load_model, cal_gpu, get_ema_co
import utils.misc as misc_utils


best_mAP = 0.0
def main_worker(args, cfg, logger):
    global best_mAP

    # build model
    device = torch.device(cfg.TRAIN.device)
    model = build_network(cfg, logger)
    model = model.to(device)

    cfg.TRAIN.ema_decay = get_ema_co(cfg)
    logger.info('ema_decay: {ema_decay:.7f}'.format(ema_decay=cfg.TRAIN.ema_decay))
    ema_m = ModelEma(model, cfg.TRAIN.ema_decay) # 0.9997^641 = 0.82503551031

    #use_BN
    use_batchnorm = cfg.MODEL.use_BN
    if use_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DDP.local_rank], broadcast_buffers=False, find_unused_parameters=True)  # find_unused_parameters=True # find_unused_parameters=True 

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # criterion
    criterion = build_criterion(cfg, model)
    cfg.OPTIMIZER.lr_mult = 1.0
    logger.info("lr: {}".format(cfg.OPTIMIZER.lr_mult * cfg.OPTIMIZER.lr))
    # logger.info("lr: {}".format(cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr))

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=cfg.INPUT_OUTPUT.output)
    else:
        summary_writer = None

    # optionally resume from a checkpoint
    if cfg.INPUT_OUTPUT.resume:
        load_model(cfg, logger, model)

    # Data loading and Distributed Sampler
    train_dataset, val_dataset = get_datasets(cfg, logger)
    train_loader, val_loader, train_sampler = distributedsampler(cfg, train_dataset, val_dataset)
    cfg.DATA.len_train_loader = len(train_loader)


    if cfg.TRAIN.evaluate:
        train_labels = train_dataset.Y
        # if dist.get_rank() == 0:
        pos_ratio = train_labels.sum(0) / train_labels.shape[0]
        test_stats, mAP = validate_metric(val_loader, model, criterion, 0, cfg, logger, pos_ratio)
        logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
        return
    
    # lr_scheduler and optimizer
    warmup_scheduler, lr_scheduler, optimizer = build_optim(cfg, model)

    # global value
    start_time = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []

    output_dir = Path(cfg.INPUT_OUTPUT.output)
    torch.cuda.empty_cache()
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.epochs):
        epoch_start_time = time.time()

        train_sampler.set_epoch(epoch)
        if cfg.TRAIN.ema_epoch == epoch:
            ema_m = ModelEma(model.module, cfg.TRAIN.ema_decay)
            torch.cuda.empty_cache()        
        torch.cuda.empty_cache()

        
        # train for one epoch
        train_stats = train(train_loader, model, ema_m, criterion, optimizer, lr_scheduler, epoch, cfg, logger, warmup_scheduler)
        
        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', train_stats['all_loss'], epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % cfg.EVAL.val_interval == 0 and epoch >= cfg.EVAL.val_epoch_start:

            # evaluate on validation set
            torch.cuda.empty_cache()
            test_stats, mAP = validate(val_loader, model, criterion, epoch, cfg, logger)
            if cfg.TRAIN.ema_epoch > epoch:
                test_stats_ema, mAP_ema = None, 0
            else:
                test_stats_ema, mAP_ema = validate(val_loader, ema_m.module, criterion, epoch, cfg, logger)

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            # log_stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }
            if test_stats_ema:
                log_stats.update({f'test_ema_{k}': v for k, v in test_stats_ema.items()})
            log_stats.update({'epoch': epoch, 'n_parameters': n_parameters})
    
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            # tensorboard logger
            if summary_writer:
                for k, v in log_stats.items():
                    if 'loss' in k and 'test' in k:
                        summary_writer.add_scalar(k, v, epoch)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)
            
            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)


            log_stats['best_mAP'] = best_mAP
            log_stats['best_regular_mAP'] = best_regular_mAP

            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

            if cfg.INPUT_OUTPUT.output and misc_utils.is_main_process():
                with (output_dir / 'log.txt').open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                with (output_dir / 'log_out.txt').open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.arch,
                    'state_dict': state_dict,
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(cfg.INPUT_OUTPUT.output, 'checkpoint.pth.tar'))

            # if test_stats_ema is not None and (math.isnan(test_stats['all_loss']) or math.isnan(test_stats_ema['all_loss'])) :
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': cfg.MODEL.arch,
            #         'state_dict': model.state_dict(),
            #         'best_mAP': best_mAP,
            #         'optimizer' : optimizer.state_dict(),
            #     }, is_best=is_best, filename=os.path.join(cfg.INPUT_OUTPUT.output, 'checkpoint_nan.pth.tar'))
            #     logger.info('Loss is NaN, break')
            #     sys.exit(1)


            # early stop
            if cfg.TRAIN.early_stop:
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                logger.info('Training time {}'.format(total_time_str))
                logger.info("Now time: {}".format(str(datetime.datetime.now())))
                # break
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 6:
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        # if dist.get_rank() == 0 and cfg.TRAIN.kill_stop:
                        #     filename = sys.argv[0].split(' ')[0].strip()
                        #     killedlist = kill_process(filename, os.getpid())
                        #     logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 
                        break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info("Now time: {}".format(str(datetime.datetime.now())))
    logger.info("Best mAP {}:".format(best_mAP))

    if summary_writer:
        summary_writer.close()
    
    return 0