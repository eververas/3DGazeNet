import copy
import random
import cv2
import numpy as np
import torch

from lib.utils.defaults import DATASET_INFO
from lib.utils import *
from .base_dataset import BaseDataset

import time


class GenericDataset(BaseDataset):
    def __init__(self, dataset_cfg=None, input_shape=None, is_train=None, transform=None,
                 fname_eyes3d='data/eyes3d.pkl', debug=False, test_sbjs=None,
                 len_datasets=1, custom_set=None, do_synthetic_training=None):

        assert dataset_cfg is not None
        self.dataset_cfg = dataset_cfg
        self.len_datasets = len_datasets
        self.is_train = is_train
        self.do_synthetic_training = do_synthetic_training

        self.test_sbjs = test_sbjs
        self.transform = transform

        # TODO make it more elegant later, individualize the augmentations
        self.aug_config = self.dataset_cfg.AUGMENTATION if self.is_train else None
        self.crop_width, self.crop_height = input_shape[0], input_shape[1]

        self.mean = torch.as_tensor([0.485, 0.456, 0.406])
        self.std = torch.as_tensor([0.229, 0.224, 0.225])
        self.aspect_ratio = self.crop_width * 1.0 / self.crop_height

        self.roots = {}
        self.roots = DATASET_INFO[self.__class__.__name__]
        if custom_set is not None and isinstance(custom_set, dict):
            self.roots.update(**custom_set)

        self.img_prefix = self.roots['img_prefix_train'] if self.is_train else self.roots['img_prefix_test']

        self.gt_bbox_exists = True  # flag value could change
        self.get_bbox_eyes_center_func = eval(self.roots['bbox_eyes_center_func'])

        eyes3d_dict = self.load_eyes3d(fname_eyes3d)
        self.iris_idxs481 = eyes3d_dict['iris_idxs481']
        self.trilist_eye = eyes3d_dict['trilist_eye']
        self.eye_template_homo = eyes3d_dict['eye_template_homo']

        self.idx_nosetip_in_lms68 = 30
        self.face_elements = ['left_eye', 'right_eye', 'face']

        self.db, self.db_names = self.load_db(debug)

        self.dataset_len = len(self.db_names)
        self.unique_len = len(list(self.db.keys()))

    def do_augmentations(self, get_rot_flip=False):
        """
        Compute random augmentation parameters.
        Args:
            get_rot_flip (Bool): Boolean for getting only rotation and flip augmentations.
            This is because we want to apply the same augmentation on both eyes
        Returns:
            scale (float): Box rescaling factor.
            rot (float): Random image rotation.
            do_flip (bool): Whether to flip image or not.
            color_scale (List): Color rescaling factor
            tx (float): Random translation along the x axis.
            ty (float): Random translation along the y axis.
        """

        rot = np.clip(np.random.randn(), -2.0, 2.0) * self.aug_config.ROT_FACTOR \
            if random.random() <= self.aug_config.ROT_AUG_RATE else 0
        do_flip = self.aug_config.FLIP and random.random() <= self.aug_config.FLIP_AUG_RATE
        if get_rot_flip:
            return rot, do_flip

        tx = np.clip(np.random.randn(), -1.0, 1.0) * self.aug_config.SHIFT_FACTOR
        ty = np.clip(np.random.randn(), -1.0, 1.0) * self.aug_config.SHIFT_FACTOR
        scale = np.clip(np.random.randn(), -1.0, 1.0) * self.aug_config.SCALE_FACTOR + 1.0

        c_up = 1.0 + self.aug_config.COLOR_SCALE
        c_low = 1.0 - self.aug_config.COLOR_SCALE
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

        only_R = random.random() < self.aug_config.R_CHANNEL_ONLY_RATE
        do_rescale = random.random() < self.aug_config.IMG_RESCALE_RATE

        img_rescale = (do_rescale, np.random.choice([1, 3, 5, 7]))  # kernel for blurring
        # img_rescale = (do_rescale, np.random.choice([3, 5, 7, 9, 11]))  # kernel for blurring
        # img_rescale = (do_rescale, np.random.choice([3, 5, 7, 9, 11, 13]))  # kernel for blurring

        return scale, color_scale, tx, ty, rot, do_flip, only_R, img_rescale

    def __getitem__(self, idx):
        if not self.is_train:
            return self.get_test_data(idx)
        while True:
            data = self.get_train_data(idx)
            if data is None:
                print(f"Problem with data at index {idx} of train")
                continue
            return data

    def _get_bbox_center(self, x, y, element_str, data_sample):
        if element_str == 'face':
            return get_gt_center(x, y)
        while True:
            if not (self.dataset_cfg.USE_GT_BBOX and self.gt_bbox_exists):
                bbox_center = self.get_bbox_eyes_center_func(data_sample, element_str)
                if bbox_center is None:
                    self.gt_bbox_exists = False
                    print(f"Predicted center for bboxes of eyes does not exist for dataset {self.roots['dataset_name']}"
                          f"\nUsing gt bboxes instead..")
                    continue
                return bbox_center
            else:
                return get_gt_center(x, y)

    def get_test_data(self, idx):
        name_key = self.db_names[idx]
        db_rec = copy.deepcopy(self.db[name_key])
        meta = []
        input_list = []
        single_image_info = self.load_single_img_info(db_rec['face'])
        image_path, cv_img, image_shape = single_image_info

        for element_str in self.face_elements:
            cv_img_numpy = cv_img.copy()
            # load img info
            center = [db_rec[element_str]['center_x'], db_rec[element_str]['center_y']]
            height = db_rec[element_str]['height']
            width = height * self.aspect_ratio
            input_args = (cv_img_numpy, [self.crop_width, self.crop_height], 0, False)  # rot=0, Flip=False
            trans, img_patch_cv = get_input_and_transform(center, [width, height], *input_args)

            np_img_patch_copy = img_patch_cv.copy()
            np_img_patch_copy = np.transpose(np_img_patch_copy, (2, 0, 1)) / 255  # (C,H,W) and between 0,1
            img_patch_torch = torch.as_tensor(np_img_patch_copy, dtype=torch.float32)  # to torch and from int to float
            img_patch_torch.sub_(self.mean.view(-1, 1, 1)).div_(self.std.view(-1, 1, 1))
            input_list += [img_patch_torch]

            # Transform vertices
            verts = 0
            if db_rec[element_str]['verts'] is not None:
                xy = db_rec[element_str]['verts'][:, [0, 1]]
                z = db_rec[element_str]['verts'][:, 2]
                xy = affine_transform_array(xy, trans)
                z *= float(self.crop_height) / float(height)
                # set verts
                verts = np.zeros_like(db_rec[element_str]['verts'])
                verts[:, [0, 1]] = xy
                verts[:, 2] = z

            meta += [{
                'verts': verts, 
                'gaze': db_rec[element_str]['gaze'],
                'head_pose': db_rec[element_str]['head_pose'],
                'element': element_str,
                'image_path': image_path,
                'image_shape': image_shape,
                'scale_multiplier': 1.,
                'center': center,
                'width': width,
                'height': height,
                'flip': False,
                'rotation': 0,
                'trans': trans,
                'init_height': db_rec[element_str]['height']
            }]

        model_input = np.concatenate((input_list[0], input_list[1], input_list[2]), axis=0)

        return model_input, meta

    
    def get_train_data(self, idx):

        name_key = self.db_names[idx]
        db_rec = copy.deepcopy(self.db[name_key])
        meta = []
        input_list = []
        # get augmentations
        scale, color_scale, tx, ty, rot, do_flip, only_R, img_rescale = self.do_augmentations()
        # load img info
        single_image_info = self.load_single_img_info(db_rec['face'])
        image_path, cv_img, image_shape = single_image_info

        for element_str in self.face_elements:
            
            cv_img_numpy = cv_img.copy()
            center = [db_rec[element_str]['center_x'], db_rec[element_str]['center_y']]
            height = db_rec[element_str]['height']
            if height > image_shape[1]:
                height = image_shape[1]
            width = height * self.aspect_ratio

            # apply scale
            width *= scale
            height *= scale
            # apply shift
            center[0] += height * ty
            center[1] += width * tx
            # compute geometric augmentation transform
            input_args = (cv_img_numpy, [self.crop_width, self.crop_height], rot, do_flip)
            trans, img_patch_cv = get_input_and_transform(center.copy(), [width, height], *input_args)
            # blur
            if img_rescale[0]:
                img_patch_cv = cv2.GaussianBlur(img_patch_cv, (img_rescale[1], img_rescale[1]), cv2.BORDER_DEFAULT)
            # to tesnor
            img_patch_cv = np.transpose(img_patch_cv, (2, 0, 1)) / 255.  # (C,H,W) and between 0,1
            img_patch_torch = torch.as_tensor(img_patch_cv, dtype=torch.float32)  # to torch and from int to float
            # image colour normalization + augmentation
            for n_c in range(3):
                img_patch_torch[n_c, :, :] = torch.clamp(img_patch_torch[n_c, :, :] * color_scale[n_c], 0, 1.)
                img_patch_torch[n_c, :, :] = (img_patch_torch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
            # add img to input list
            input_list += [img_patch_torch]
            # adjust gaze to rotation augmentation
            # if rot != 0.:
            #     gaze_vector = db_rec[element_str]['gaze']['vector']
            #     gaze_vector = rotate(gaze_vector, (0., 0., rot))  # rot is in degrees
            #     db_rec[element_str]['gaze']['vector'] = gaze_vector
            #     db_rec[element_str]['gaze']['pitchyaws'] = vector_to_pitchyaw(gaze_vector[None, :])[0]

            # adjust gaze + headpose to flip
            if do_flip and 'vector' in db_rec[element_str]['gaze']:
                db_rec[element_str]['gaze']['vector'][0] *= -1
                db_rec[element_str]['gaze']['pitchyaws'][1] *= -1
            if do_flip:
                db_rec[element_str]['head_pose'][1] *= -1

            # Transform vertices
            verts = 0
            if db_rec[element_str]['verts'] is not None:
                xy = db_rec[element_str]['verts'][:, [0, 1]]
                z = db_rec[element_str]['verts'][:, 2]
                # horizontal flip
                if do_flip:
                    xy[:, 0] = image_shape[1] - xy[:, 0] - 1
                xy = affine_transform_array(xy, trans)
                z *= float(self.crop_height) / float(height)
                # set verts
                verts = np.zeros_like(db_rec[element_str]['verts'])
                verts[:, [0, 1]] = xy
                verts[:, 2] = z

            meta += [{
                'verts': verts, 
                'gaze': db_rec[element_str]['gaze'],
                'head_pose': db_rec[element_str]['head_pose'],
                'element': element_str,
                'image_path': image_path,
                'image_shape': image_shape,
                'scale_multiplier': scale,
                'center': center,
                'width': width,
                'height': height,
                'flip': do_flip,
                'rotation': rot,
                'trans': trans,
                'init_height': db_rec[element_str]['height']
            }]
            
        # left + right eye + face images stacked channel-wise
        if do_flip:
            model_input = np.concatenate((input_list[1], input_list[0], input_list[2]), axis=0)
            meta = [meta[1], meta[0], meta[2]]
        else:
            model_input = np.concatenate((input_list[0], input_list[1], input_list[2]), axis=0)
            meta = [meta[0], meta[1], meta[2]]

        return [model_input], [meta]
