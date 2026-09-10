from dataclasses import dataclass
from typing import TypeVar

_T = TypeVar("_T")


def dataclass_dict(cls=None) -> type[_T]:
    cls = dataclass(cls, order=True)
    cls.__len__ = lambda self: len(self.__dict__)
    cls.__iter__ = lambda self: iter(self.__dict__)
    cls.__getitem__ = lambda self, key: self.__dict__[key]
    cls.__setitem__ = lambda self, key, value: self.__dict__.update({key: value})
    cls.__delitem__ = lambda self, key: self.__dict__.pop(key, None)
    cls.__contains__ = lambda self, key: key in self.__dict__
    cls.keys = lambda self: self.__dict__.keys()
    cls.values = lambda self: self.__dict__.values()
    cls.items = lambda self: self.__dict__.items()
    cls.get = lambda self, key, default=None: self.__dict__.get(key, default)
    cls.pop = lambda self, key, default=None: self.__dict__.pop(key, default)
    cls.clear = lambda self: self.__dict__.clear()
    cls.update = lambda self, *args, **kwargs: self.__dict__.update(*args, **kwargs)
    return cls
