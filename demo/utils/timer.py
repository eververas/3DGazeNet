import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, ClassVar, Optional

import torch

from .base_metrics import BaseMetrics
from .logger import get_logger


def time_for_log(localtime: bool = True) -> str:
    ISOTIMEFORMAT = '%d-%h'
    f = time.localtime if localtime else time.gmtime
    return '{}'.format(time.strftime(ISOTIMEFORMAT, f(time.time())))


def time_synchronized() -> float:
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@dataclass
class Timer(ContextDecorator):
    """Time the code using a class context manager, or decorator"""

    metrics: ClassVar[BaseMetrics] = BaseMetrics()
    text: str = "{} elapsed time: {:0.5f} seconds"
    _start_time: Optional[float] = field(default=None, init=False, repr=False)
    _last: float = field(default=float("nan"), init=False, repr=False)
    name: Optional[str] = None
    logger: Optional[Logger] = None
    fps: Optional[bool] = None
    pprint: Optional[bool] = False
    save_path: Optional['str'] = None

    def __post_init__(self) -> None:
        time_synchronized()

        if self.fps:
            self.text = '{} running at {:.2f} FPS with {:.2f} avg FPS.'

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        self._last = (time.perf_counter() - self._start_time)
        self._start_time = None

        if self.name:
            self.metrics.add(self.name, 1 / self._last)

        if self.fps:
            log_msg = self.text.format(self.name, 1 / self._last, self.metrics.avg(self.name))
        else:
            log_msg = self.text.format(self.name, self._last)

        if self.pprint:
            self.logger.info(log_msg)

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.logger = get_logger(self.name)
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()

    @property
    def last(self):
        return self._last
