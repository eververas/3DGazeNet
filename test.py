import os
import gc
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from lib.utils import load_from_checkpoint, create_logger, parse_args, update_config, update_dict, config
from lib.core import test, test_synthetic
from lib.dataset import build_dataset
from lib.models import build_model, build_loss


def main():
    # setup config
    args = parse_args()
    update_config(args.cfg)
    update_dict(config, vars(args))

    # device
    device = torch.device(f'cuda:{config.GPUS[0]}' if torch.cuda.is_available() else 'cpu')
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # logger
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'test')
    # build model + loss
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    # load checkpoint
    checkpoint = (args.checkpoint or os.path.join(final_output_dir, 'model_best.pth'))
    if os.path.isfile(checkpoint):
        logger.info(f'=> loading model from {checkpoint}')
        load_from_checkpoint(checkpoint, model, skip_optimizer=args.skip_optimizer)
    else:
        raise f'No valid checkpoints file {checkpoint}'

    with torch.cuda.device(device):
        gc.collect()
        torch.cuda.empty_cache()

    # setup data loader
    test_loader = torch.utils.data.DataLoader(
        build_dataset(config, args, is_train=False),
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    # test
    test(config, test_loader, model, criterion, device)

if __name__ == '__main__':
    main()
