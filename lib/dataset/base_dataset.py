import _pickle as cPickle

import os
import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset

from lib.utils.defaults import DATASETS_WITH_SAME_TRAIN_TEST_FILES
from lib.utils import load_eyes3d


class BaseDataset(Dataset):

    @staticmethod
    def load_eyes3d(fname_eyes3d):
        return load_eyes3d(fname_eyes3d)

    def load_data_file(self, data_file, debug=False):
        with open(data_file, 'rb') as fi:
            data_list = cPickle.load(fi)
        if debug:
            data_list = data_list[: 3 * self.dataset_cfg.BATCH_SIZE]
        return data_list

    def load_single_img_info(self, db_rec, element_str=None):
        if element_str:
            db_rec = db_rec[element_str]
        image_path = db_rec['image_path']
        cv_img_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        assert cv_img_numpy is not None, f"\n\nFile: {image_path} is None/Corrupted:\n"
        image_shape = np.array(cv_img_numpy.shape[0:2])
        return image_path, cv_img_numpy, image_shape

    def load_db(self, debug):
        db = {}
        db_names = []
        data_file = self.roots['data_file_train'] if self.is_train else self.roots['data_file_test']
        dataset_name = self.roots['dataset_name']
        # load dataset file
        data_list = self.load_data_file(data_file, debug)
        # load dataset elements
        set = 'Train' if self.is_train else 'Test'
        print(f"Loading set {set}: {dataset_name} - {data_file.split('/')[-1].split('.')[0]}, Size: {len(data_list)}")
        for ii, data_sample in enumerate(tqdm.tqdm(data_list)):
            name_key, subject = self.get_name_subject(dataset_name, data_sample)
            if dataset_name.lower() in DATASETS_WITH_SAME_TRAIN_TEST_FILES:
                if not ((self.is_train ^ (subject in self.test_sbjs)) or debug):
                    continue
            db_names += [name_key]
            db[name_key] = dict.fromkeys(self.face_elements, {})
            for element_str in self.face_elements:
                db[name_key][element_str] = self._prepare_data(data_sample, element_str)
        return db, db_names

    def get_name_subject(self, dataset_name, data_sample):
        name_key = dataset_name + '/' + data_sample['name']
        subject = data_sample['name'].split('/')[0]
        return name_key, subject

    def _prepare_data(self, data_sample: dict, element_str: str) -> dict:
        preparation_func = eval(f"self._prepare_{element_str}")
        center_x, center_y, width, height, verts, gaze = preparation_func(data_sample)
        image_path = f"{self.img_prefix}/{data_sample['name']}"
        assert os.path.exists(image_path), f"Unable to locate sample:\n{image_path}"
        head_pose = np.zeros(2)
        if 'head_pose' in data_sample['face']:
            head_pose = data_sample['face']['head_pose']
        annotation = {
            'verts': verts,
            'gaze': gaze,
            'head_pose': head_pose,
            'image_path': image_path,
            'center_x': center_x,
            'center_y': center_y,
            'height': height}
        return annotation

    def _load_eye_verts(self, data_sample, element_str):
        verts = None
        if data_sample['eyes'] is not None:
            verts = (data_sample['eyes'][element_str]['P'] @ self.eye_template_homo[element_str].T).T
            verts = verts[:, [1, 0, 2]].astype(np.float32)
            verts[:, 2] -= verts[:, 2][:32].mean(axis=0)
        return verts

    def _load_face_verts(self, data_sample):
        if data_sample['face']['xyz68'] is not None:
            verts = data_sample['face']['xyz68']
            verts = verts[:, [1, 0, 2]].astype(np.float32)
            verts[:, 2] -= verts[:, 2][-68:][self.idx_nosetip_in_lms68]
        elif data_sample['face']['xy5'] is not None:
            verts = data_sample['face']['xy5']
            verts = verts[:, [1, 0]].astype(np.float32)
            verts = np.concatenate((verts, np.zeros((5, 1))), axis=1)
        else: 
            raise KeyError("The face sample does not contain xyz68 landmarks")
        return verts

    def _load_face_gaze(self, data_sample):
        return data_sample['gaze']['face'] if data_sample['gaze'] is not None else 0

    def _prepare_eyes(self, data_sample, element_str):
        element_str_short = element_str.split('_')[0]
        # coords (might be None in inference datasets)
        verts = self._load_eye_verts(data_sample, element_str_short)
        # gaze (might be None in inference datasets)
        gaze = self._load_face_gaze(data_sample)
        # bbox
        center_x, center_y, width, height = self._get_bbox_center(None, None, element_str, data_sample)
        width *= self.dataset_cfg.AUGMENTATION.EXTENT_TO_CROP_RATIO
        height *= self.dataset_cfg.AUGMENTATION.EXTENT_TO_CROP_RATIO
        return center_x, center_y, width, height, verts, gaze

    def _prepare_face(self, data_sample):
        # coords
        verts = self._load_face_verts(data_sample)
        # gaze (might be None in inference datasets)
        gaze = self._load_face_gaze(data_sample)
        # bbox
        center_x, center_y, width, height = self._get_bbox_center(verts[:, 0], verts[:, 1], 'face', data_sample)
        return center_x, center_y, width, height, verts, gaze

    def _prepare_left_eye(self, data_sample):
        return self._prepare_eyes(data_sample, 'left_eye')

    def _prepare_right_eye(self, data_sample):
        return self._prepare_eyes(data_sample, 'right_eye')

    def __len__(self, ):
        return len(self.db_names)
