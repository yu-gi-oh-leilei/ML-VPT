import torch
from .lars import LARS

def build_optimizer(cfg, parameters):
    """Build function of Optimizer.
    Args:
        cfg (dict): Config of optimizer wrapper, optimizer constructor and optimizer.
        parameters: 
    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    if cfg.OPTIMIZER.optim == 'AdamW':
        optimizer = getattr(torch.optim, cfg.OPTIMIZER.optim)(
            parameters,
            cfg.OPTIMIZER.lr_mult * cfg.OPTIMIZER.lr,
            # lr=cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.OPTIMIZER.weight_decay
        )
    elif cfg.OPTIMIZER.optim == 'Adam_twd':
        optimizer = torch.optim.Adam(
            parameters,
            cfg.OPTIMIZER.lr_mult * cfg.OPTIMIZER.lr,
            # lr=cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    elif cfg.OPTIMIZER.optim == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.OPTIMIZER.lr,
            # lr=cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr,
            momentum=cfg.OPTIMIZER.momentum,
            weight_decay=cfg.OPTIMIZER.weight_decay,
            dampening=cfg.OPTIMIZER.sgd_dampening,
            nesterov=cfg.OPTIMIZER.sgd_nesterov
        )
    elif cfg.OPTIMIZER.optim == 'LARS':
        optimizer = LARS(
            parameters,
            lr=cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr,
            momentum=cfg.OPTIMIZER.momentum,
            weight_decay=cfg.OPTIMIZER.weight_decay
        )
    else:
        raise NotImplementedError

    return optimizer