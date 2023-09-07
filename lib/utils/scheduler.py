'''
Learning rate schedulers.
Written by Polydefkis Gkagkos
'''

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from .registry import Registry

Schedulers = Registry('Schedulers')
class LinearWarmup:
    def __init__(self, warmup_iters, warmup_ratio):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

    def __call__(self, where):
        return 1 - ((1 - where / self.warmup_iters) * (1 - self.warmup_ratio))

@Schedulers.register_module()
class ExponentialIterScheduler:
    def __init__(self, gamma, start_epoch, total_epochs, total_batches):
        self.gamma = gamma
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.max_iters = total_epochs * total_batches
        self.warmup_iters = 0

    def __call__(self, cur_iter):
        epoch_num = ((cur_iter - self.warmup_iters) * self.total_epochs) / self.max_iters

        return self.gamma ** (epoch_num + self.start_epoch)  # ((cur_iter - self.warmup_iters) / self.total_epochs))

@Schedulers.register_module()
class StepIterScheduler:
    def __init__(self, gamma, start_epoch, total_epochs, total_batches):
        self.gamma = gamma
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.max_iters = total_epochs * total_batches
        self.warmup_iters = 0
        self.current = 1.

    def __call__(self, cur_iter):
        epoch_num = ((cur_iter - self.warmup_iters) * self.total_epochs) / self.max_iters
        if (round(epoch_num + self.start_epoch, 5) % 1.) == 0 and cur_iter != 0.:
            self.current *= self.gamma

        return self.current

@Schedulers.register_module()
class MultiStepIterScheduler:
    def __init__(self, milestones, gamma, start_epoch, total_epochs, total_batches):
        self.milestones = milestones
        self.gamma = gamma
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.total_batches = total_batches
        self.max_iters = total_epochs * total_batches
        self.current = 1
        self.warmup_iters = 0

    def __call__(self, cur_iter) -> float:
        epoch_num = ((cur_iter - self.warmup_iters) * self.total_epochs) / self.max_iters
        if epoch_num + self.start_epoch in self.milestones:
            self.current *= self.gamma
        return self.current


class Compose:
    WHERE_EPSILON = 1e-6

    def __init__(self, schedulers, warmup_iters, max_iters):
        self.schedulers = schedulers
        total_percentage_warmup = warmup_iters / max_iters
        self.max_iters = max_iters
        self.lengths_v2 = np.cumsum([total_percentage_warmup, (1 - total_percentage_warmup)])

    def __call__(self, cur_iter):
        idx = len(self.lengths_v2) - 1
        for i, length in enumerate(self.lengths_v2):
            if (cur_iter + 1e-6) / self.max_iters <= length:
                idx = i
                break
        return self.schedulers[idx](cur_iter)


class IterBasedScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 scheduler,
                 last_iter=-1
                 ):
        self._scheduler = scheduler
        self._max_iters = scheduler.max_iters
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self):
        # we assume the last_epoch from _LRScheduler to be an iteration
        multiplier = self._scheduler(self.last_epoch)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def with_warmup(self, warmup_iters, warmup_ratio):
        warmup_func = LinearWarmup(warmup_iters, warmup_ratio)
        self._scheduler.warmup_iters = warmup_iters
        self._scheduler = Compose([warmup_func, self._scheduler], warmup_iters, self._max_iters)
