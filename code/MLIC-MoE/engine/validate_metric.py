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
from utils.metric_xie import sl_mAP_cf1_of1
from utils.metric_new import VOCmAP
from models.builder_network import do_forward_and_criterion
import matplotlib.pyplot as plt


global pos_ratio
# pos_ratio = 0.0

def calculate_metric(preds, labels):

    n_correct_pos = (labels*preds).sum(0)
    n_pred_pos = ((preds==1)).sum(0)
    n_true_pos = labels.sum(0)
    OP = n_correct_pos.sum()/n_pred_pos.sum()
    CP = np.nanmean(n_correct_pos/n_pred_pos)
    OR = n_correct_pos.sum()/n_true_pos.sum()
    CR = np.nanmean(n_correct_pos/n_true_pos)

    CF1 = (2 * CP * CR) / (CP + CR)
    OF1 = (2 * OP * OR) / (OP + OR)

    return CP, CR, CF1, OP, OR, OF1


def thresholding(labels, probs, logger):

    for thre in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = (probs>thre).astype(np.float32)
        metrics = calculate_metric(preds, labels)
        logger.info(f'{thre}, {np.round(np.array(metrics)*100, decimals=1)}')

    return 0


@torch.no_grad()
def validate_metric(val_loader, model, criterion, epoch, cfg, logger, pos_ratio_=0.0):
    pos_ratio = pos_ratio_


    
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
    
    # metric_map = VOCmAP(cfg.DATA.num_class, year='2012', ignore_path=cfg.INPUT_OUTPUT.ignore_path)
    # metric_map.reset()

    probs_ori = []
    labels = []
    image_name = []
    
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

            #  save some data            
            lable[lable < 0] = 0

            # add list
            probs_ori.append(torch.sigmoid(output).detach().cpu())
            labels.append(data_full['target'].detach().cpu())
            # image_name.append(data_full['name'])
            image_name = image_name + data_full['name']

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

        # saved data
        labels = torch.cat(labels).numpy()
        probs_ori = torch.cat(probs_ori).numpy()

        
        print(os.path.join(cfg.INPUT_OUTPUT.output, 'tmpdata'))

        data_ori = np.concatenate((probs_ori, labels), axis=1)
        saved_name_ori = 'tmpdata/data_ori_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(cfg.INPUT_OUTPUT.output, saved_name_ori), data_ori)
        
        # save iamge name
        saved_iamge_name_text = 'tmpdata/image_name_tmp.{}.txt'.format(dist.get_rank())
        saved_iamge_name_text = os.path.join(cfg.INPUT_OUTPUT.output, saved_iamge_name_text)
        with open(saved_iamge_name_text, 'w') as f:
            for name in image_name:
                f.write(str(name) + '\n')


        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            logger.info("#########################   save image name   #########################")
            filenamelist_name = ['tmpdata/image_name_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            filenamelist_name = [os.path.join(cfg.INPUT_OUTPUT.output, _filename) for _filename in filenamelist_name]
            if isinstance(filenamelist_name, str):
                filenamelist_name = [filenamelist_name]
            lines = []
            for imagessetfile in filenamelist_name:
                with open(imagessetfile, 'r') as f:
                    lines.extend(f.readlines())

            # os.makedirs(os.path.dirname(os.path.join(cfg.INPUT_OUTPUT.output, 'image_name.txt')), exist_ok=True) 
            with open(os.path.join(cfg.INPUT_OUTPUT.output, 'image_name.txt'), 'w') as f:
                for name in lines:
                    f.write(str(name))

            logger.info("#########################     finish save     #########################")


            logger.info("Calculating mAP:")

            filenamelist_ori = ['tmpdata/data_ori_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            mAP_ori, APs_ori, all_list_ori, top3_list_ori, outputs_ori = sl_mAP_cf1_of1([os.path.join(cfg.INPUT_OUTPUT.output, _filename) for _filename in filenamelist_ori], cfg.DATA.num_class, pos_ratio)

            logger.info("######################################################################")

            logger.info("mAP ori {:.2f}".format(mAP_ori))

            logger.info("  ap: {}".format(APs_ori))

            logger.info("CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(all_list_ori[0], all_list_ori[1], all_list_ori[2], all_list_ori[3], all_list_ori[4], all_list_ori[5]))

            logger.info("Top-3 CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(top3_list_ori[0], top3_list_ori[1], top3_list_ori[2], top3_list_ori[3], top3_list_ori[4], top3_list_ori[5]))

            plt.figure(figsize=(16, 8))

            bar_width = 0.25
            index = np.arange(len(APs_ori))
            plt.bar(index, APs_ori, bar_width, label='Ori')
            plt.savefig(os.path.join(cfg.INPUT_OUTPUT.output, 'bar.jpg'), dpi=500)
            
            logger.info("outputs_ori shape {}, outputs_ori shape {}, outputs_ori shape {}".format(outputs_ori[0].shape, outputs_ori[1].shape, outputs_ori[2].shape))
            logger.info("outputs_ori type {}, outputs_ori type {}, outputs_ori type {}".format(type(outputs_ori[0]), type(outputs_ori[1]), type(outputs_ori[2])))
            
            # logger.info(type(outputs_ori[0]), type(outputs_ori[1]), type(outputs_ori[2]))
            np.save(os.path.join(cfg.INPUT_OUTPUT.output, 'probs.npy'), outputs_ori[0])
            np.save(os.path.join(cfg.INPUT_OUTPUT.output, 'preds.npy'), outputs_ori[1])
            np.save(os.path.join(cfg.INPUT_OUTPUT.output, 'labels.npy'), outputs_ori[2])

            thresholding(outputs_ori[2], outputs_ori[0], logger)


        else:
            mAP_ori = 0
            mAP_aux = 0

        if dist.get_world_size() > 1:
            dist.barrier()

        # mAP = max(mAP, mAP_aux)
        mAP = mAP_ori

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return resstat, mAP
