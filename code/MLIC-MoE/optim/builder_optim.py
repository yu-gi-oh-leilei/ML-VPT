from optim.optimizer import build_optimizer
from optim.lrscheduler import build_scheduler


def build_optim(cfg, model):
    if cfg.OPTIMIZER.pattern_parameters == 'mutil_lr':
        small, large = [], []

        # update prototype
        if hasattr(model, 'module'):
            for n, p in model.module.named_parameters():
                if 'visual_encoder' in n and p.requires_grad:
                    small.append(p)
                    # print('visual_encoder: ', n)
                elif 'classifier' in n and p.requires_grad:
                    large.append(p)
                    # print('classifier: ', n)
                else:
                    # print('other: ', n)
                    continue
        else:
            for n, p in model.named_parameters():
                if 'visual_encoder' in n and p.requires_grad:
                    small.append(p)
                    # print('visual_encoder: ', n)
                elif 'classifier' in n and p.requires_grad:
                    large.append(p)
                    # print('classifier: ', n)
                else:
                    # print('other: ', n)
                    continue

        # parameters = [
        #     {'params': small, 'lr': cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr * cfg.OPTIMIZER.lrp},
        #     {'params': large, 'lr': cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr}
        # ]
        
        parameters = [
            {'params': small, 'lr': cfg.OPTIMIZER.lr * cfg.OPTIMIZER.lrp},
            {'params': large, 'lr': cfg.OPTIMIZER.lr}
        ]


    elif cfg.OPTIMIZER.pattern_parameters == 'single_lr':
        if hasattr(model, 'module'):
            parameters = [
                {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
            ]
            parameters = filter(lambda p: p.requires_grad, model.module.parameters())
        else:
            parameters = [
                {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
            ]
            parameters = filter(lambda p: p.requires_grad, model.parameters())
        

    elif cfg.OPTIMIZER.pattern_parameters == 'add_weight':
        parameters = add_weight_decay(model, cfg.OPTIMIZER.weight_decay)
    else:
        raise NotImplementedError

    optimizer = build_optimizer(cfg, parameters)
    
    warmup_scheduler, scheduler  = build_scheduler(cfg, optimizer)

    return warmup_scheduler, scheduler, optimizer


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    # used for training only.
    # copy from: https://github.com/Alibaba-MIIL/ASL/blob/main/train.py
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    

    
    
        # print('===')
        # parameters = filter(lambda p: p.requires_grad, model.prompt_learner.parameters())
        # for k, v in parameters[0].items():
        #     print(k)
        # print(parameters)
        # print(type(parameters))
        # print(list(parameters))