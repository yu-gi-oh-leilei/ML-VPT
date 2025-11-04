import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


class LinearWarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class ConstantWarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_lr, lr, warmup_epoch, total_iters, len_train_loader, last_epoch=-1):
        self.warmup_lr = warmup_lr
        self.lr = lr
        self.warmup_epoch = warmup_epoch
        self.total_iters = total_iters
        self.len_train_loader = len_train_loader
        # self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # print(self.last_epoch, self.len_train_loader * self.warmup_epoch)
        # print(self.last_epoch, [base_lr * self.last_epoch / (self.total_iters) for base_lr in self.base_lrs])
        if self.last_epoch >= self.len_train_loader * self.warmup_epoch:
            # self.last_epoch = 0
            return [base_lr * self.last_epoch / (self.total_iters) for base_lr in self.base_lrs]
        else:
            return [self.warmup_lr]

def build_scheduler(cfg, optimizer):
    """Build function of Optimizer.
    Args:
        cfg (dict): Config of optimizer wrapper, optimizer constructor and optimizer.
        parameters: 
    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    if cfg.OPTIMIZER.lr_scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=cfg.OPTIMIZER.lr * cfg.OPTIMIZER.lr_mult, 
                                            # max_lr=cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr,
                                            steps_per_epoch=cfg.DATA.len_train_loader, 
                                            epochs=cfg.TRAIN.epochs, 
                                            pct_start=0.2)
    
    elif cfg.OPTIMIZER.lr_scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                             cfg.OPTIMIZER.epoch_step, 
                                             gamma=0.1, 
                                             last_epoch=-1)
        
    elif cfg.OPTIMIZER.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   mode='max', 
                                                   patience=1, 
                                                   verbose=True)


    elif cfg.OPTIMIZER.lr_scheduler == 'cosine':
        # scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, float(cfg.TRAIN.epochs)
        # )
        scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        eta_min=0.000001,
        T_max=(cfg.TRAIN.epochs - cfg.OPTIMIZER.warmup_epoch) * cfg.DATA.len_train_loader)

        if cfg.OPTIMIZER.warmup_epoch > 0:
            scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.OPTIMIZER.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=cfg.OPTIMIZER.warmup_epoch * cfg.DATA.len_train_loader)
    else:
        raise Exception('lr scheduler {} not found!'.format(cfg.OPTIMIZER.lr_scheduler)) 

    # elif cfg.OPTIMIZER.lr_scheduler == 'cosine':
    #     scheduler = lr_scheduler.CosineAnnealingLR(
    #         optimizer, float(cfg.TRAIN.epochs)
    #     )
    # else:
    #     raise Exception('lr scheduler {} not found!'.format(cfg.OPTIMIZER.lr_scheduler))  

    if cfg.OPTIMIZER.warmup_scheduler == True:
        if cfg.OPTIMIZER.warmup_type == "linear":
            warmup_scheduler = LinearWarmUpLR(optimizer, cfg.DATA.len_train_loader * cfg.OPTIMIZER.warmup_epoch)
        if cfg.OPTIMIZER.warmup_type == "constant":
            warmup_scheduler = ConstantWarmUpLR(optimizer, cfg.OPTIMIZER.warmup_lr, 
                                                # cfg.OPTIMIZER.base_lr, 
                                                # cfg.OPTIMIZER.batch_size * cfg.DDP.world_size / 256 * cfg.OPTIMIZER.base_lr,
                                                cfg.OPTIMIZER.lr * cfg.OPTIMIZER.lr_mult, 
                                                cfg.OPTIMIZER.warmup_epoch, 
                                                cfg.DATA.len_train_loader * cfg.OPTIMIZER.warmup_epoch, 
                                                cfg.DATA.len_train_loader)

    else:
        warmup_scheduler = None
    

    return warmup_scheduler, scheduler

'''

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]

'''


        # return [self.warmup_lr]
        # lr_list = [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        # print(len(lr_list))
        # print(lr_list[-1])
        # print(lr_list[-2])
        # print(lr_list[-3])

        # print(self.base_lrs, self.last_epoch, [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs], len([base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]))
        # return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        # print(self.last_epoch, self.base_lrs)


# warmup_scheduler
    # warmup_scheduler = WarmUpLR(optimizer, if cfg.OPTIMIZER.warmup_epoch > 0:
    #     if not cfg.OPTIMIZER.warmup_recount:
    #         scheduler.last_epoch = cfg.OPTIMIZER.warmup_epoch

    #     if cfg.OPTIMIZER.warmup_type == "constant":
    #         scheduler = ConstantWarmupScheduler(
    #             optimizer, scheduler, cfg.OPTIMIZER.warmup_epoch,
    #             cfg.OPTIMIZER.warmup_cons_lr
    #         )

    #     elif cfg.OPTIMIZER.warmup_type == "linear":
    #         scheduler = LinearWarmupScheduler(
    #             optimizer, scheduler, cfg.OPTIMIZER.warmup_epoch,
    #             cfg.OPTIMIZER.warmup_min_lr
    #         )
    #     else:
    #         raise ValueError
    # else:
    #     raise Exception('lr scheduler {} not found!'.format(cfg.OPTIMIZER.lr_scheduler))  

'''
def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCHEDS
            )
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                "be an integer, but got {}".format(type(stepsize))
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    if optim_cfg.WARMUP_EPOCH > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_MIN_LR
            )

        else:
            raise ValueError

    return scheduler
'''



# class _BaseWarmupScheduler(_LRScheduler):

#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.successor = successor
#         self.warmup_epoch = warmup_epoch
#         super().__init__(optimizer, last_epoch, verbose)

#     def get_lr(self):
#         raise NotImplementedError

#     def step(self, epoch=None):
#         if self.last_epoch >= self.warmup_epoch:
#             self.successor.step(epoch)
#             self._last_lr = self.successor.get_last_lr()
#         else:
#             super().step(epoch)


# class ConstantWarmupScheduler(_BaseWarmupScheduler):

#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         cons_lr,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.cons_lr = cons_lr
#         super().__init__(
#             optimizer, successor, warmup_epoch, last_epoch, verbose
#         )

#     def get_lr(self):
#         if self.last_epoch >= self.warmup_epoch:
#             return self.successor.get_last_lr()
#         return [self.cons_lr for _ in self.base_lrs]
