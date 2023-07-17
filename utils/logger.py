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


def get_logger(name,save=False, use_time=True, use_tqdm=False):
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
    if save:
        file = os.path.join(name, 'logs.txt')
        os.makedirs(name, exist_ok=True)
        if os.path.exists(file):  # if previous logs existed remove it
            os.remove(file)

        fh = logging.FileHandler(file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if use_tqdm:
        logger = TqdmSystemLogger(logger)

    return logger
