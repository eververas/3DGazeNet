import collections
from collections import deque, defaultdict
import torch
from typing import Any, Callable, Dict, TypeVar, Deque, Union

TDeque = TypeVar('TDeque', bound=Deque[Union[torch.Tensor, float]])


class BaseMetrics(collections.UserDict):
    """Custom dictionary that stores information about metrics"""

    def __init__(self, window_size: int = 30, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dict: Dict[str, TDeque] = defaultdict(lambda: deque(maxlen=window_size))
        self.global_count: Dict[str, int] = dict()

    def apply(self, func: Callable, name: str) -> float:
        """Apply a function to the results of one named """
        if name in self._dict:
            value = torch.tensor(list(self._dict[name]))
            return func(value)
        raise KeyError(name)

    def add(self, name: str, value: Union[torch.Tensor, float]) -> None:
        """Add a value to the named metric"""
        self._dict[name].append(float(value))

        self.data.setdefault(name, 0)
        self.data[name] += value

        self.global_count.setdefault(name, 0)
        self.global_count[name] += 1

    def count(self, name: str) -> float:
        """Number of elements"""
        return self.apply(len, name=name)

    def avg(self, name: str) -> float:
        """Mean value of elements"""
        return self.apply(lambda values: torch.mean(values).item(), name=name)

    def median(self, name: str) -> float:
        """Median value of timings"""
        return self.apply(lambda values: torch.median(values).item(), name=name)

    def global_avg(self, name: str) -> float:
        return self.data[name] / self.global_count[name]

    def count_avg_success(self, name: str) -> float:
        """For bool arrays only, Return a number between zero and one. One stands for all true"""
        n_falses = self.apply(lambda values: int((values != True).nonzero()), name=name)
        count = self.apply(len, name=name)
        n_trues = count - n_falses
        return n_trues / count


class MetricLogger(object):
    def __init__(self, delimiter="  ", writer=None):
        self.meters = BaseMetrics()
        self.delimiter = delimiter
        self.writer = writer

    def update(self, is_train=True, iteration=None, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if is_train:
                self.meters.add(k, v)
            if self.writer is not None:
                if k in ['time', 'data']:
                    tag = 'unclassified'
                    continue
                elif is_train:
                    tag = 'train'
                else:
                    tag = 'test'
                self.writer.add_scalar('/'.join([tag, k]), v, iteration)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.5f} ({:.5f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def get_all_avg(self):
        d = {}
        for name, meter in self.meters.items():
            d[name] = meter.global_avg
        return d


if __name__ == '__main__':
    metrics = BaseMetrics()

    value = torch.tensor(5)
    metrics.add(name='something', value=5)
    metrics.add(name='something', value=5)
    metrics.add(name='something', value=3)

    print(metrics.avg('something'))
    print(metrics.median('something'))
