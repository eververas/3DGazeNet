import time

import numpy as np
import torch
import _pickle as cPickle

from lib.utils.config import get_model_name
import logging
import os
import sys
import io


class TqdmSystemLogger(io.StringIO):
    """ A tqdm wrapper for a logger. Works if for a loop on training or inference"""

    def __init__(self, logger, suppress_new_line=True):
        super(TqdmSystemLogger, self).__init__()
        self.logger = logger
        self.buf = '\r'
        if suppress_new_line:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.terminator = ""

    def write(self, buf):
        self.buf = buf.strip('\n\t\n')

    def flush(self):
        self.logger.log(self.logger.level, '\r' + self.buf)

    def info(self, message):
        self.logger.info(message + '\n')


def get_logger(name, save_dir=None, use_time=True, use_tqdm=False):
    # returns a logger and initializes the save dir if its given
    logger = logging.getLogger(name)
    if len(logger.handlers):
        return logger
    logger.propagate = False
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)

    text_format = "%(name)s %(levelname)s: %(message)s"
    if use_time:
        text_format = "[%(asctime)s] " + text_format

    formatter = logging.Formatter(text_format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        file = os.path.join(save_dir, name + '_logs.txt')
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(file):  # if previous logs existed remove it
            os.remove(file)

        fh = logging.FileHandler(file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if use_tqdm:
        logger = TqdmSystemLogger(logger)

    return logger

def get_cfg_name(cfg_name):
    return os.path.basename(cfg_name).split('.')[0]

def get_final_output_dir(cfg):
    final_output_dir = f'{cfg.OUTPUT_DIR}/{cfg.MODEL_DIR}'
    if cfg.DATASET.TEST_IDX is not None:
        final_output_dir = f'{final_output_dir}/with_test_idx_{cfg.DATASET.TEST_IDX}'
    return final_output_dir

def get_final_log_dir(cfg):
    final_log_dir = f'{cfg.LOG_DIR}/{cfg.MODEL_DIR}'
    if cfg.DATASET.TEST_IDX is not None:
        final_log_dir = f'{final_log_dir}/with_test_idx_{cfg.DATASET.TEST_IDX}'
    return final_log_dir

def get_final_results_dir(cfg, cfg_name):
    final_res_dir = f'{cfg.RESULTS_DIR}/{cfg.MODEL_DIR}'
    if cfg.DATASET.TEST_IDX is not None:
        final_res_dir = f'{final_res_dir}/with_test_idx_{cfg.DATASET.TEST_IDX}'
    _cfg_name = get_cfg_name(cfg_name)
    final_res_dir = f'{final_res_dir}/{_cfg_name}'
    return final_res_dir

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = f'{cfg.OUTPUT_DIR}/{cfg.MODEL_DIR}'
    final_output_dir = get_final_output_dir(cfg)
    cfg_name = get_cfg_name(cfg_name)

    # set up logger
    os.makedirs(root_output_dir, exist_ok=True)
    model, _ = get_model_name(cfg)
    print(f'=> Creating {final_output_dir}')
    os.makedirs(final_output_dir, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{phase}_{cfg_name}_{time_str}.log'
    final_log_file = f'{final_output_dir}/{log_file}'
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = f'{cfg.LOG_DIR}/{model}/{cfg_name}_{time_str}'
    print(f'=> Creating {tensorboard_log_dir}\n')
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    cfg.FINAL_OUTPUT_DIR = final_output_dir

    return logger, final_output_dir, tensorboard_log_dir


def save_checkpoint(states, is_best, output_dir, name=None):
    best_model_path = os.path.join(output_dir, 'model_best.pth')
    if name is not None:
        best_model_path = os.path.join(output_dir, f'model_best_{name}.pth')
    states.update({'timestamp': time.strftime('%d/%m - %H:%M')})
    torch.save(states, os.path.join(output_dir, 'checkpoint.pth'))
    if os.path.exists(best_model_path):
        exp_best = torch.load(best_model_path, map_location=lambda storage, loc: storage)['perf']
        if is_best and states['perf'] < exp_best:
            torch.save(states, best_model_path)
    else:
        torch.save(states, best_model_path)


def load_eyes3d(fname_eyes3d='data/eyes3d.pkl'):
    with open(fname_eyes3d, 'rb') as f:
        eyes3d = cPickle.load(f)
    iris_idxs = eyes3d['left_iris_lms_idx']
    idxs481 = eyes3d['mask481']['idxs']
    iris_idxs481 = eyes3d['mask481']['idxs_iris']
    idxs288 = eyes3d['mask288']['idxs']
    iris_idxs288 = eyes3d['mask288']['idxs_iris']
    trilist_eye = eyes3d['mask481']['trilist']
    eyel_template = eyes3d['left_points'][idxs481]
    eyer_template = eyes3d['right_points'][idxs481]
    eye_template = {
        'left': eyes3d['left_points'][idxs481],
        'right': eyes3d['right_points'][idxs481]
    }
    eye_template_homo = {
        'left': np.append(eye_template['left'], np.ones((eyel_template.shape[0], 1)), axis=1),
        'right': np.append(eye_template['right'], np.ones((eyer_template.shape[0], 1)), axis=1)
    }
    eyes3d_dict = {
        'iris_idxs': iris_idxs, 
        'idxs481': idxs481, 
        'iris_idxs481': iris_idxs481, 
        'idxs288': idxs288,
        'iris_idxs288': iris_idxs288,
        'trilist_eye': trilist_eye, 
        'eye_template': eye_template,
        'eye_template_homo': eye_template_homo
    }
    return eyes3d_dict


def load_from_checkpoint(checkpoint, model, strict=True, skip_optimizer=False):
    saved_data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    state_dict = saved_data['state_dict'] if 'state_dict' in saved_data else saved_data
    model.load_state_dict(state_dict, strict=strict)

    if skip_optimizer:
        optimizer_state_dict = None
        print('Skip loading optimizer state')
    else:
        optimizer_state_dict = saved_data.get('optimizer', None)
        print('Loaded optimizer state')

    epoch = saved_data.get('epoch', 0)
    return epoch, optimizer_state_dict


def get_static_center(*_):
    x_extent = 45.
    y_extent = 45.
    center_x = 60. / 2.
    center_y = 36. / 2.
    return center_x, center_y, x_extent, y_extent


def get_lms_diag_from_face(face_sample):
    if face_sample['xyz68'] is not None:
        lms5 = face_sample['xyz68'][[36, 45, 30, 48, 54]][:, [1, 0]].astype(np.float32)
        lms5[0] = face_sample['xyz68'][36:42].mean(axis=0)[[1, 0]].astype(np.float32)
        lms5[1] = face_sample['xyz68'][42:48].mean(axis=0)[[1, 0]].astype(np.float32)
    elif face_sample['xy5'] is not None:
        lms5 = face_sample['xy5'][:, [1, 0]].astype(np.float32)
    else:
        raise KeyError("The face sample does not contain xyz68 landmarks")
    # ready lms5
    diag1 = np.linalg.norm(lms5[0] - lms5[4])
    diag2 = np.linalg.norm(lms5[1] - lms5[3])
    diag = np.max([diag1, diag2])
    return lms5, diag


def get_bbox_from_lms(face_sample):
    lms = face_sample['xyz68'][:, [1, 0]].astype(np.float32)
    left = min(lms[:, 0])
    top = min(lms[:, 1])
    right = max(lms[:, 0])
    bottom = max(lms[:, 1])
    return left, top, right, bottom


def get_face_center_from_face_bbox(face_sample):
    left, top, right, bottom = get_bbox_from_lms(face_sample)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    return center_x, center_y, bottom - top, right - left


def get_face_center_from_face_diag(face_sample):
    lms5, diag = get_lms_diag_from_face(face_sample)
    width = 1.2 * diag
    height = 1.2 * diag
    cnt = lms5[2]
    center_x = cnt[0]
    center_y = cnt[1]
    return center_x, center_y, width, height


def get_gt_center(x, y):
    width = max(x) - min(x)
    height = max(y) - min(y)
    width *= 1.2
    height *= 1.2
    center_x = np.array(x).mean()
    center_y = np.array(y).mean()
    return center_x, center_y, width, height


def get_pred_center(data_sample, eye_str):
    lms5, diag = get_lms_diag_from_face(data_sample['face'])
    crop_len = int(diag / 5)
    width = crop_len
    height = crop_len
    cnt = lms5[1]
    if eye_str == 'right_eye':
        cnt = lms5[0]
    center_x = cnt[0]
    center_y = cnt[1]
    return center_x, center_y, width, height


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def points_to_vector(points, iris_lms_idx):
    back = points[:, np.arange(32)].mean(axis=1, keepdim=True) # (B, 1, 3)
    front = points[:, iris_lms_idx].mean(axis=1, keepdim=True) # (B, 1, 3)
    vec = front - back
    vec = vec / torch.norm(vec, dim=2, keepdim=True)  # (B, 1, 3)
    return torch.squeeze(vec)
