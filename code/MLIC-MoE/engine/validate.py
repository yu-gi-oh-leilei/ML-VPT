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
from utils.misc import concat_all_gather, MetricLogger, SmoothedValue, reduce_dict
from utils.hpc import pin_workers_iterator
from utils.metric import voc_mAP, asl_mAP
from utils.metric_new import VOCmAP
from models.builder_network import do_forward_and_criterion

@torch.no_grad()
def validate(val_loader, model, criterion, epoch, cfg, logger):

    if cfg.TRAIN.amp:
        # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.TRAIN.amp_dtype]
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['float32']

    if hasattr(model, 'module'):
        model.module.switch_mode_eval()
        model.module.training = False
    else:
        model.switch_mode_eval()
        model.training = False
    criterion.eval()
    
    metric_map = VOCmAP(cfg.DATA.num_class, year='2012', ignore_path=cfg.INPUT_OUTPUT.ignore_path)
    metric_map.reset()

    saved_data = []
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    the_iterator = iter(val_loader)

    # _cnt = 0
    # output_state_dict = {} # for debug only

    with torch.no_grad():
        for it, data_full in enumerate(metric_logger.log_every(the_iterator, cfg.INPUT_OUTPUT.print_freq, header, logger=logger)):
            
            data_full['image'] = torch.stack([image.cuda(non_blocking=True) for image in data_full['image']], dim=0)
            data_full['target'] = torch.stack([target.cuda(non_blocking=True) for target in data_full['target']], dim=0)
            lable = data_full['target'].clone()

            # compute output
            if cfg.TRAIN.amp:
                with torch.amp.autocast(enabled=cfg.TRAIN.amp, device_type='cuda'):
                # with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                    output, losses, loss_dict, weight_dict = do_forward_and_criterion(cfg, epoch, data_full, model, criterion, True)
            else:
                output, losses, loss_dict, weight_dict  = do_forward_and_criterion(cfg, epoch, data_full, model, criterion, True)

            # output_sm = output
            output_sm = torch.sigmoid(output)
        
            #  save some data            
            lable[lable < 0] = 0
            _item = torch.cat((output_sm.detach().cpu().data, lable.detach().cpu().data), 1)
            saved_data.append(_item)


            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()


            output_sm_gather = concat_all_gather(output_sm)
            target_gather = concat_all_gather(data_full['target'])


            metric_map.update(output_sm_gather.detach().cpu().numpy(), target_gather.cpu().numpy())

        
        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)


        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()


        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(cfg.INPUT_OUTPUT.output, saved_name), saved_data)


        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            # logger.info("Calculating mAP:")
            # filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            # metric_func = voc_mAP
            # mAP, aps = metric_func([os.path.join(cfg.INPUT_OUTPUT.output, _filename) for _filename in filenamelist], cfg.DATA.num_class, return_each=True, logger=logger)
            # logger.info("  mAP: {}".format(mAP))
            # if cfg.INPUT_OUTPUT.out_aps:
            #     logger.info("  aux aps: {}".format(np.array2string(aps, precision=5)))
            
            logger.info("Calculating mAP:")
            ap, mAP = metric_map.compute()

            # logger.info("  ap: {}".format(ap*100))
            logger.info("  mAP: {}".format(mAP*100))


        else:
            mAP = 0
            mAP_aux = 0

        if dist.get_world_size() > 1:
            dist.barrier()

        # mAP = max(mAP, mAP_aux)
        mAP = mAP * 100

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return resstat, mAP
