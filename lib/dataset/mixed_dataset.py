import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from lib import HOOKS

import warnings


class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


class ConcatDataset(_ConcatDataset):
    r"""A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    builds the datasets as well.

    Args:
        dataset_name_list (list[:str:`Dataset`]): A list of datasets.
        args: dict of arguments used when building the individual
        datasets.
        kwargs: dict of keyword arguments used when building
        the individual datasets.
    """

    def __init__(self, dataset_name_list, **kwargs):
        datasets = []
        self.is_train = kwargs['is_train']
        self.repeat = kwargs['dataset_cfg'].REPEAT

        if len(self.repeat) != len(dataset_name_list):
            warnings.warn(f"=====> Unequal number of repeat values and number of datasets. <=====\n"
                          f"=====> Setting all datasets with the repeat value of the first dataset. " f"Which is {self.repeat[0]} <=====")
            self.repeat = [self.repeat[0] for _ in dataset_name_list]

        for i, class_name in enumerate(dataset_name_list):
            dataset = HOOKS.build(dict(type=class_name), custom_args=kwargs)
            if self.repeat[i] > 1 and self.is_train:
                dataset = RepeatDataset(dataset, times=self.repeat[i])
            datasets.append(dataset)
        super(ConcatDataset, self).__init__(datasets)

    def bisect_sizes(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return dataset_idx

    def __getitem__(self, idx):
        dataset_idx = self.bisect_sizes(idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.cumulative_sizes[-1]
