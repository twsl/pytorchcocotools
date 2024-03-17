from dataclasses import dataclass
import logging
from typing import TypeVar

_T = TypeVar("_T")


def dataclass_dict(cls=None) -> type[_T]:
    cls = dataclass(cls)
    cls.__len__ = lambda self: len(self.__dataclass_fields__)
    cls.__iter__ = lambda self: iter(self.__dataclass_fields__)
    cls.__getitem__ = lambda self, key: self.__dataclass_fields__[key]
    cls.__setitem__ = lambda self, key, value: self.__dataclass_fields__.update({key: value})
    cls.__delitem__ = lambda self, key: self.__dataclass_fields__.pop(key, None)
    cls.__contains__ = lambda self, key: key in self.__dataclass_fields__
    cls.keys = lambda self: self.__dataclass_fields__.keys()
    cls.values = lambda self: self.__dataclass_fields__.values()
    cls.items = lambda self: self.__dataclass_fields__.items()
    cls.get = lambda self, key, default=None: self.__dataclass_fields__.get(key, default)
    cls.pop = lambda self, key, default=None: self.__dataclass_fields__.pop(key, default)
    cls.clear = lambda self: self.__dataclass_fields__.clear()
    cls.update = lambda self, *args, **kwargs: self.__dataclass_fields__.update(*args, **kwargs)
    return cls  # type: ignore


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger
