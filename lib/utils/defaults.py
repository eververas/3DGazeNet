from easydict import EasyDict as edict


MODEL_NAMES = [
    'vertex_predictor',
    'gaze_predictor',
]

BACKBONES = [
    'resnet',
    'mobilevit',
]

datasets_base_dir = '/storage/nfs2/evangelosv/databases/EyeReconstruction'

DATASET_INFO = {
    'XGazeDataset': {
        'dataset_name': 'XGaze',
        'bbox_eyes_center_func': 'get_pred_center',
        'data_file_train': datasets_base_dir + 'XGaze/xgaze_448_exports/train/data_for_model/train_gaze_eyes3D_face68.pkl',
        'data_file_test': datasets_base_dir + 'XGaze/xgaze_448_exports/train/data_for_model/train_gaze_eyes3D_face68.pkl',
        'img_prefix_train': datasets_base_dir + 'XGaze/xgaze_448_exports/train/images',
        'img_prefix_test': datasets_base_dir + 'XGaze/xgaze_448_exports/train/images'
    },
    'MPIIFaceGazeDataset': {
        'dataset_name': 'mpiifacegaze',
        'bbox_eyes_center_func': 'get_pred_center',
        'data_file_train': datasets_base_dir + 'MPIIFaceGaze/data_for_model/all_gaze_eyes3D_face68.pkl',
        'data_file_test': datasets_base_dir + 'MPIIFaceGaze/data_for_model/all_gaze_eyes3D_face68.pkl',
        'img_prefix_train': datasets_base_dir + 'MPIIFaceGaze/images',
        'img_prefix_test': datasets_base_dir + 'MPIIFaceGaze/images',
    },
    'GazeCaptureDataset': {
        'dataset_name': 'GazeCapture',
        'bbox_eyes_center_func': 'get_pred_center',
        # 'data_file_train': datasets_base_dir + 'GazeCapture/data_for_model/all_gaze_eyes3D_face68.pkl',
        # 'data_file_test': datasets_base_dir + 'GazeCapture/data_for_model/all_gaze_eyes3D_face68.pkl',
        'data_file_train': datasets_base_dir + 'GazeCapture/data_for_model/all_500K_gaze_eyes3D_face68.pkl',
        'data_file_test': datasets_base_dir + 'GazeCapture/data_for_model/all_500K_gaze_eyes3D_face68.pkl',
        'img_prefix_train': datasets_base_dir + 'GazeCapture/images_normalized',
        'img_prefix_test': datasets_base_dir + 'GazeCapture/images_normalized'
    },
    'Gaze360Dataset': {
        'dataset_name': 'Gaze360',
        'bbox_eyes_center_func': 'get_pred_center',
        'data_file_train': datasets_base_dir + 'gaze360/data_for_model/train_gaze_eyes3D_face68.pkl',
        'data_file_test': datasets_base_dir + 'gaze360/data_for_model/test_gaze_eyes3D_face68.pkl',
        'img_prefix_train': datasets_base_dir + 'gaze360/images',
        'img_prefix_test': datasets_base_dir + 'gaze360/images',
    },
}

DATASETS_WITH_SAME_TRAIN_TEST_FILES = [val['dataset_name'].lower() for val in DATASET_INFO.values()
                                       if val['data_file_train'] == val['data_file_test']]

config = edict()
config.MODEL_DIR = ''
config.OUTPUT_DIR = ''
config.FINAL_OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.CFG = ''
config.GPUS = '0'
config.WORKERS = None
config.PRINT_FREQ = None
config.TEST_FREQ = None
config.MODE = None

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = None
config.MODEL.IMAGE_SIZE = [128, 128]  # width * height, ex: 192 * 256
config.MODEL.NUM_POINTS_OUT_EYES = None
config.MODEL.NUM_POINTS_OUT_FACE = 68

config.MODEL.BACKBONE_TYPE = 'resent'  # Resnet, Mobile, Mobilevit
config.MODEL.BACKBONE_PRETRAINED = ''
config.MODEL.MOBILE_VIT_SIZE = ''
config.MODEL.NUM_LAYERS = None
config.MODEL.BOUNDED = False
config.MODEL.EXPANSION = None
config.MODEL.STRIDES = None
config.MODEL.BLOCK_FILTERS = None

config.LOSS = edict()
config.LOSS.NAME = None
config.LOSS.VERTEX_REGRESSION_WEIGHT_EYES = 0.1
config.LOSS.VERTEX_REGRESSION_WEIGHT_FACE = 0.1
config.LOSS.EDGE_LENGTH_LOSS_WEIGHT_EYES = 0.01
config.LOSS.EDGE_LENGTH_LOSS_WEIGHT_FACE = 0.01
config.LOSS.GAZE_ACOS_WEIGHT = 1

# DATASET related params
config.DATASET = edict()
config.DATASET.BATCH_SIZE = 16
config.DATASET.REPEAT = [1]
config.DATASET.TEST_IDX = None
config.DATASET.TRAIN_DATASETS = ['']
config.DATASET.TEST_DATASETS = ['']
config.DATASET.USE_GT_BBOX = None

# training data augmentation
config.DATASET.AUGMENTATION = edict()
config.DATASET.AUGMENTATION.FLIP = False
config.DATASET.AUGMENTATION.SCALE_FACTOR = 0.
config.DATASET.AUGMENTATION.ROT_FACTOR = 0.
config.DATASET.AUGMENTATION.SHIFT_FACTOR = 0.
config.DATASET.AUGMENTATION.COLOR_SCALE = 0.
config.DATASET.AUGMENTATION.EXTENT_TO_CROP_RATIO = 1.6
config.DATASET.AUGMENTATION.FLIP_AUG_RATE = 0.5
config.DATASET.AUGMENTATION.ROT_AUG_RATE = 0.6
config.DATASET.AUGMENTATION.SHIFT_AUG_RATE = 0.5
config.DATASET.AUGMENTATION.IMG_RESCALE_RATE = 0.

# train
config.TRAIN = edict()
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 10
config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''
config.TRAIN.SHUFFLE = True

config.LR_SCHEDULER = edict()
config.LR_SCHEDULER.NAME = None
config.LR_SCHEDULER.LR_FACTOR = None
config.LR_SCHEDULER.LR_STEP = []
config.LR_SCHEDULER.WITH_WARMUP = False
config.LR_SCHEDULER.WARMUP_ITERS = None
config.LR_SCHEDULER.WARMUP_RATE = 0.001

config.OPTIMIZER = edict()
config.OPTIMIZER.NAME = None
config.OPTIMIZER.LR = 1e-3
config.OPTIMIZER.BETA0 = 0.9
config.OPTIMIZER.BETA1 = 0.999
config.OPTIMIZER.MOMENTUM = 0.9
config.OPTIMIZER.WD = 0.0001
config.OPTIMIZER.NESTEROV = False
config.OPTIMIZER.GAMMA1 = 0.99
config.OPTIMIZER.GAMMA2 = 0.0

# testing
config.TEST = edict()
config.TEST.SHUFFLE = False
config.TEST.EXPORT = False
config.TEST.PATH_EXPORT = ''

