import os
import numpy as np
from lib.dataset.generic_dataset import GenericDataset
from lib.utils import show_result

from lib import HOOKS


@HOOKS.register_module('gaze360')
class Gaze360Dataset(GenericDataset):
    def __init__(self, *args, **kwargs):
        super(Gaze360Dataset, self).__init__(*args, **kwargs)


@HOOKS.register_module('gazecapture')
class GazeCaptureDataset(GenericDataset):
    def __init__(self, data_split='data/gazecapture_data_split.pkl', *args,
                 **kwargs):
        kwargs['test_sbjs'] = self.load_data_file(data_split)['ids_test']
        super(GazeCaptureDataset, self).__init__(*args, **kwargs)


@HOOKS.register_module('mpiiface')
class MPIIFaceGazeDataset(GenericDataset):
    def __init__(self, *args, **kwargs):
        temp_idx = kwargs['dataset_cfg'].TEST_IDX
        test_idx = temp_idx if temp_idx is not None else 0
        test_ids = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13',
                    'p14']
        kwargs['test_sbjs'] = [test_ids[test_idx]]
        super(MPIIFaceGazeDataset, self).__init__(*args, **kwargs)


@HOOKS.register_module('xgaze')
class XGazeDataset(GenericDataset):
    def __init__(self, *args, **kwargs):
        kwargs['test_sbjs'] = ['subject0076', 'subject0104', 'subject0028', 'subject0100', 'subject0038', 'subject0080',
                               'subject0072', 'subject0052', 'subject0005', 'subject0106', 'subject0003', 'subject0099',
                               'subject0063', 'subject0088', 'subject0062', 'subject0039']
        super(XGazeDataset, self).__init__(*args, **kwargs)

@HOOKS.register_module('inference')
class InferenceDataset(GenericDataset):
    def __init__(self, *args, **kwargs):
        super(InferenceDataset, self).__init__(*args, **kwargs)
