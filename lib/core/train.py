import logging
import time

import torch
import torch.nn as nn

from ..utils.metrics import AverageMeter
from ..utils.vis import *

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, device, epoch, writer_dict, lr_scheduler):
    
    # setup timer
    batch_time = AverageMeter()
    # switch to train mode
    model.train()
    n_losses_opt = 4 if config.MODE == 'vertex' else 1
    loss_names_opt = criterion.loss_names_opt

    # training loop
    end = time.time()
    for i, (input_data, meta) in enumerate(train_loader):
        input_data = input_data[0]
        meta = meta[0]
        # compute model output
        output = model(input_data.to(device))
        # compute loss
        losses_out = criterion(output, meta)
        losses_out_opt = {name: losses_out[name] for name in loss_names_opt}
        loss = sum(losses_out_opt.values())
        # compute gradients and do update step
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print training losses
        if i % config.PRINT_FREQ == 0:
            print_losses(epoch, i, len(train_loader), input_data.size(0) / batch_time.val, 
                        losses_out_opt, writer_dict, logger)