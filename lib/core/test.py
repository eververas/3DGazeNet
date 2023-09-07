import os
import time
import tqdm
import torch
import logging
import _pickle as cPickle

from ..utils import AverageMeter, get_final_log_dir

logger = logging.getLogger(__name__)


def test(config, test_loader, model, criterion, device, epoch=None):

    # setup timer
    batch_time = AverageMeter()
    # switch to eval mode
    model.eval()
    # setup loss dicts
    loss_names = criterion.loss_names
    loss_name_performance = criterion.loss_name_performance
    losses = dict.fromkeys(loss_names)
    for loss_name in loss_names:
        losses[loss_name] = AverageMeter()

    # test loop
    end = time.time()
    for i, (input_data, meta) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            # compute model output
            output = model(input_data.to(device))
            # compute losses
            losses_out = criterion(output, meta)
        # gather losses
        for loss_name, l_ in losses_out.items():
            losses[loss_name].update(l_.item(), input_data.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # report overall losses
    loss_string = ' '.join([f"{loss_name}: {losses[loss_name].avg:.4f}" for loss_name in losses.keys()])
    msg = f"Overall Losses: {loss_string}" 
    logger.info(msg)

    # write log file
    if epoch is not None:
        path_log = get_final_log_dir(config)
        os.makedirs(str(path_log), exist_ok=True)
        msg = f"Epoch: {epoch}"
        for k, v in losses.items():
            msg += f", {k}= {v.avg:.5f}"
        msg += '\n'
        with open(f"{path_log}/test_log.txt", 'a') as f:
            f.write(msg)

    # loss to track performance
    loss_track_performance = losses[loss_name_performance].avg
    return loss_track_performance
