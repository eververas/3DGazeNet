import os
import shutil
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.utils import parse_args, update_config, update_dict, config, \
    save_checkpoint, load_from_checkpoint, create_logger
from lib.core import train, test
from lib.dataset import build_dataset
from lib.models import build_model, build_loss, build_optimizer, build_scheduler


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

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
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')

    # define model + loss + optimizer
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(config, model)

    # copy model and config files to output dir
    shutil.copy2(os.path.join(os.path.dirname(__file__), args.cfg), final_output_dir)
    # load checkpoint
    epoch = config.TRAIN.BEGIN_EPOCH
    checkpoint = args.checkpoint if args.checkpoint is not None else ''
    if config.TRAIN.RESUME:
        checkpoint = f'{final_output_dir}/checkpoint.pth'
    if os.path.isfile(checkpoint):
        logger.info(f'=> loading model from {checkpoint}')
        epoch, optimizer_state_dict = load_from_checkpoint(
            checkpoint, model, skip_optimizer=args.skip_optimizer)
        if not args.skip_optimizer:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except ValueError:
                print('Optimizer state is not compatible, skipping optimizer loading.')

        if args.epoch is not None:
            epoch = args.epoch

    # summary writer
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0
    }
    # setup data laoders
    train_loader = torch.utils.data.DataLoader(
        build_dataset(config, args, is_train=True),
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = torch.utils.data.DataLoader(
        build_dataset(config, args, is_train=False),
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # scheduler
    lr_scheduler = build_scheduler(config, optimizer, logger, epoch, len_dataset=len(train_loader.dataset))

    # train loop
    is_best_model, best_perf, start_epoch, best_epoch = False, 9999999, epoch, epoch
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, device, epoch, writer_dict, lr_scheduler)
        # test
        if (epoch + 1) % config.TEST_FREQ == 0 or epoch == 0:
            # test
            current_perf = test(config, test_loader, model, criterion, device, epoch=epoch)
            # track best model
            is_best_model = True if current_perf < best_perf else False
            best_perf = current_perf if is_best_model else best_perf
            best_epoch = epoch if is_best_model else best_epoch
            # report and save
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            if is_best_model:
                logger.info('=> saving best model, with info - epoch: {}, error: {:.5f}'.format(best_epoch, best_perf))
            logger.info('=> current model info - epoch: {}, error: {:.5f}'.format(epoch, current_perf))
            logger.info('=> best model info - epoch: {}, error: {:.5f}'.format(best_epoch, best_perf))
            save_checkpoint({
                'epoch': epoch,
                'perf': current_perf,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                is_best_model, final_output_dir)

    # save final model state
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('Saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
