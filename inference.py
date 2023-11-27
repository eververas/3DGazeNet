import os
import gc
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from lib.utils import load_from_checkpoint, get_final_output_dir, \
                    parse_args, update_config, update_dict, update_dataset_info, config
from lib.core import inference
from lib.dataset import build_dataset
from lib.models import build_model


def main():
    # setup config
    args = parse_args()
    update_config(args.cfg)
    update_dict(config, vars(args))
    update_dataset_info(args)

    # device
    device = torch.device(f'cuda:{config.GPUS[0]}' if torch.cuda.is_available() else 'cpu')
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = build_model(config).to(device)    
    # load checkpoint
    final_output_dir = get_final_output_dir(config)
    checkpoint = (args.checkpoint or os.path.join(final_output_dir, 'model_best.pth'))
    print(checkpoint)
    if os.path.isfile(checkpoint):
        print(f'=> loading model from {checkpoint}')
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
        drop_last=False
    )
    # run inference on test dataset
    inference(config, test_loader, model, device)

if __name__ == '__main__':
    main()
