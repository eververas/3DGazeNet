from .mixed_dataset import ConcatDataset


def build_dataset(cfg, args, is_train, dataset_name_list=None):
    if dataset_name_list is None:
        dataset_name_list = cfg.DATASET.TRAIN_DATASETS if is_train else cfg.DATASET.TEST_DATASETS
    len_datasets = len(dataset_name_list)
    dataset = ConcatDataset(
        dataset_name_list=dataset_name_list,
        do_synthetic_training=args.train_synthetic,
        dataset_cfg=cfg.DATASET,
        input_shape=[cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]],
        is_train=is_train,
        debug=args.debug,
        custom_set=args.custom_set,
        len_datasets=len_datasets
    )
    return dataset
