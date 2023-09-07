import warnings

from lib.utils.defaults import (MODEL_NAMES, LOSS_NAMES)
from .backbones import MobileNetV3
from torch.optim import SGD, Adam, AdamW
from lib import HOOKS
from lib.utils.scheduler import IterBasedScheduler, ExponentialIterScheduler, MultiStepIterScheduler


def build_model(cfg):
    """Build model"""
    mode = cfg.MODE
    model_name = cfg.MODEL.NAME if cfg.MODEL.NAME else f'{mode.lower()}_predictor'
    assert model_name in MODEL_NAMES, "Polydefkis - Model name not in the supported Models "
    return HOOKS.build(dict(type=model_name, cfg=cfg))


def get_model_mobilenetv3(_):
    return MobileNetV3()


def build_loss(cfg):
    """Build loss."""
    loss_name = cfg.LOSS.NAME if cfg.LOSS.NAME else f'base_loss'
    return HOOKS.build(dict(type=loss_name, cfg=cfg, mode=cfg.MODE))


def build_scheduler(cfg, optimizer, logger, epoch, len_dataset):
    gamma = cfg.LR_SCHEDULER.LR_FACTOR
    start_epoch = cfg.TRAIN.BEGIN_EPOCH

    total_epochs = cfg.TRAIN.END_EPOCH
    batch_size = cfg.DATASET.BATCH_SIZE
    scheduler_name = cfg.LR_SCHEDULER.NAME if cfg.LR_SCHEDULER.NAME is not None else ''
    if scheduler_name.startswith('multi'):
        # multistep
        milestones = [milestone for milestone in cfg.LR_SCHEDULER.LR_STEP]
        gamma = cfg.LR_SCHEDULER.LR_FACTOR
        scheduler = MultiStepIterScheduler(milestones=milestones, gamma=gamma, total_epochs=total_epochs,
                                           start_epoch=start_epoch,
                                           total_batches=int(len_dataset / batch_size) + 1)
    elif scheduler_name.startswith('exp'):

        scheduler = ExponentialIterScheduler(gamma=gamma, total_epochs=total_epochs, start_epoch=start_epoch,
                                             total_batches=int(len_dataset / batch_size) + 1)
    else:
        warnings.warn("LR scheduler is not defined. Currently running with no scheduler", RuntimeWarning)

        class Fake_Scheduler:
            def __init__(self):
                self.max_iters = 1

            @staticmethod
            def step():
                pass

        scheduler = Fake_Scheduler()

    scheduler = IterBasedScheduler(optimizer, scheduler=scheduler)
    if cfg.LR_SCHEDULER.WITH_WARMUP:
        # Careful when reseting training using warmup
        scheduler.with_warmup(warmup_iters=cfg.LR_SCHEDULER.WARMUP_ITERS,
                              warmup_ratio=cfg.LR_SCHEDULER.WARMUP_RATE)

    if scheduler_name != '':
        logger.info('Restarting at epoch {}, LR: {}, Scheduler epoch: {}'.
                    format(epoch, scheduler.get_last_lr(), scheduler.last_epoch))

    return scheduler


def build_optimizer(cfg, model):
    optimizer = None
    if cfg.OPTIMIZER.NAME.lower() == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=cfg.OPTIMIZER.LR,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.OPTIMIZER.WD,
            nesterov=cfg.OPTIMIZER.NESTEROV
        )
    elif cfg.OPTIMIZER.NAME.lower() == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=cfg.OPTIMIZER.LR,
            betas=(cfg.OPTIMIZER.BETA0, cfg.OPTIMIZER.BETA1),
            weight_decay=cfg.OPTIMIZER.WD
        )
    elif cfg.OPTIMIZER.NAME.lower() == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            # model.parameters(),
            lr=cfg.OPTIMIZER.LR,
            betas=(cfg.OPTIMIZER.BETA0, cfg.OPTIMIZER.BETA1),
            weight_decay=cfg.OPTIMIZER.WD
        )
    return optimizer
