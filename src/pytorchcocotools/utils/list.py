from typing import Any


def is_list_of_type(instance: Any, compare_type: type) -> bool:
    return isinstance(instance, list) and all(isinstance(item, compare_type) for item in instance)
