from dataclasses import dataclass
from typing import TypeVar

_T = TypeVar("_T")


def dataclass_dict(cls=None) -> type[_T]:  # pyright: ignore[reportInvalidTypeVarUse]
    cls = dataclass(cls)
    cls.__len__ = lambda self: len(self.__dict__)  # pyright: ignore[reportFunctionMemberAccess]
    cls.__iter__ = lambda self: iter(self.__dict__)  # pyright: ignore[reportFunctionMemberAccess]
    cls.__getitem__ = lambda self, key: self.__dict__[key]  # pyright: ignore[reportFunctionMemberAccess]
    cls.__setitem__ = lambda self, key, value: self.__dict__.update({key: value})  # pyright: ignore[reportFunctionMemberAccess]
    cls.__delitem__ = lambda self, key: self.__dict__.pop(key, None)  # pyright: ignore[reportFunctionMemberAccess]
    cls.__contains__ = lambda self, key: key in self.__dict__  # pyright: ignore[reportFunctionMemberAccess]
    cls.keys = lambda self: self.__dict__.keys()  # pyright: ignore[reportFunctionMemberAccess]
    cls.values = lambda self: self.__dict__.values()  # pyright: ignore[reportFunctionMemberAccess]
    cls.items = lambda self: self.__dict__.items()  # pyright: ignore[reportFunctionMemberAccess]
    cls.get = lambda self, key, default=None: self.__dict__.get(key, default)  # pyright: ignore[reportFunctionMemberAccess]
    cls.pop = lambda self, key, default=None: self.__dict__.pop(key, default)  # pyright: ignore[reportFunctionMemberAccess]
    cls.clear = lambda self: self.__dict__.clear()  # pyright: ignore[reportFunctionMemberAccess]
    cls.update = lambda self, *args, **kwargs: self.__dict__.update(*args, **kwargs)  # pyright: ignore[reportFunctionMemberAccess]
    return cls  # pyright: ignore[reportReturnType]
