import argparse
import yaml
from easydict import EasyDict as edict
from .defaults import config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mesh Regression network')
    # general
    parser.add_argument('--cfg', help='Experiment configure file name', required=True, type=str)
    # training
    parser.add_argument('--gpus', help='The gpu ids to be used.', type=str)
    parser.add_argument('--test_idx', help='The identity index to be used for testing. Used for some datasets only',
                        default=None, type=int)
    parser.add_argument('--debug', help='Activate debug mode.', action='store_true')
    parser.add_argument('--custom_set',default=None, help='A custom version for training. This is dataset specific option.')
    parser.add_argument('--workers', help='Number of dataloader workers.', type=int)
    parser.add_argument('--checkpoint', help='Path to a checkpoint to initialize from.', default=None, type=str)
    parser.add_argument('--skip_optimizer', help='Skip optimizer restoration from checkpoint.', action='store_true')
    parser.add_argument('--epoch', help='Force set epoch to this number.', default=None, type=int)

    args = parser.parse_args()
    return args


def update_dict(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            source[key.upper()] = update_dict(source.get(key, {}), value)
        elif value is not None:
            source[key.upper()] = overrides[key]
    if 'TEST_IDX' in source.keys() and 'DATASET' in source.keys() and source['TEST_IDX'] is not None:
        source['DATASET']['TEST_IDX'] = source['TEST_IDX']
    return source


def update_config(config_file):
    with open(config_file) as f:
        this_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    update_dict(config, this_config)


def get_model_name(cfg):
    mode = cfg.MODE
    name = cfg.MODEL.NAME if cfg.MODEL.NAME else f'{mode.lower()}_predictor'
    num_layers = cfg.MODEL.NUM_LAYERS

    name = '{model}_{num_layers}'.format(
        model=name,
        num_layers=num_layers)

    full_name = '{height}x{width}_{name}'.format(
        height=cfg.MODEL.IMAGE_SIZE[1],
        width=cfg.MODEL.IMAGE_SIZE[0],
        name=name)

    return name, full_name
