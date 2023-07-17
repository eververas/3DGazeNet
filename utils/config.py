from easydict import EasyDict as edict
import yaml

config = edict()
config.LOG_DIR = ''
config.EXP_NAME = ''
config.DEVICE = 'cuda:0'

config.DETECTOR = edict()
config.DETECTOR.PRETRAINED = ''
config.DETECTOR.IMAGE_SIZE = None

config.PREDICTOR = edict()

config.PREDICTOR.NUM_LAYERS = None
config.PREDICTOR.BACKBONE_TYPE = None
config.PREDICTOR.BACKBONE_SIZE = ''
config.PREDICTOR.PRETRAINED = ''
config.PREDICTOR.MODE = ''

config.PREDICTOR.PREDICT_FACE = None
config.PREDICTOR.IMAGE_SIZE = None
config.PREDICTOR.NUM_POINTS_OUT_FACE = None
config.PREDICTOR.NUM_POINTS_OUT_EYES = None

config.PREDICTOR.OPTIMIZE_VECTORS = None
config.PREDICTOR.GAZE_IS_VECTOR = None
config.PREDICTOR.GAZE_IS_RELATIVE_TO_FACE = None
config.PREDICTOR.BOUNDED = False
config.PREDICTOR.EXPANSION = None

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True


def _update_dict(k, v):
    for vk, vv in v.items():
        if isinstance(vv, edict):
            for vvk, vvv in vv.items():
                config[k][vk][vvk] = vvv
        elif vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, edict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
