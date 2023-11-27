# from .train import train, train_synthetic
# from .test import test, test_synthetic
# from .inference import inference

# __all__ = ['test', 'train', 'train_synthetic', 'test_synthetic', 'inference']

from .train import train
from .test import test
from .inference import inference

__all__ = ['test', 'train', 'inference']
