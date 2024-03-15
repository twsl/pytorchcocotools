from dataclasses import dataclass
import logging


def dataclass_dict(
    cls=None,
):
    cls = dataclass(cls)
    cls.__getitem__ = lambda self, key: self.__dict__[key]
    cls.__setitem__ = lambda self, key, value: self.__dict__.update({key: value})
    return cls


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger
